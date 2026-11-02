"""T-learner, X-learner and a response-model baseline, plus the Hillstrom uplift run.

Run with: python -m src.learners
"""

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.model_selection import train_test_split

from src import config, evaluate, load, simulate

FORBIDDEN_FEATURES = {
    *config.HILLSTROM_OUTCOMES,
    *config.CRITEO_OUTCOMES,
    config.HILLSTROM_ARM_COL,
    config.CRITEO_TREATMENT_COL,
    "exposure",
    config.LENTA_OUTCOME,
    config.LENTA_GROUP_COL,
    *config.LENTA_EXCLUDED_FEATURES,
}


def make_classifier(seed=config.SEED):
    return LGBMClassifier(random_state=seed, **config.LGBM_PARAMS)


def make_regressor(seed=config.SEED):
    return LGBMRegressor(random_state=seed, **config.LGBM_PARAMS)


def check_features(X):
    """Refuse to fit on outcomes or treatment, which would leak the answer into the model."""
    leaked = FORBIDDEN_FEATURES & set(getattr(X, "columns", []))
    if leaked:
        raise ValueError(f"These columns cannot be model inputs: {sorted(leaked)}")


def _prob(model, X):
    return model.predict_proba(X)[:, 1]


class ResponseModel:
    """Baseline: who buys when contacted? It fits treated customers only."""

    def __init__(self, seed=config.SEED):
        self.seed = seed

    def fit(self, X, y, t):
        check_features(X)
        y, t = np.asarray(y), np.asarray(t)
        self.model = make_classifier(self.seed).fit(X[t == 1], y[t == 1])
        return self

    def predict(self, X):
        return _prob(self.model, X)


class TLearner:
    """Two models: one predicts the outcome with treatment, one without. Uplift is the gap."""

    def __init__(self, seed=config.SEED):
        self.seed = seed

    def fit(self, X, y, t):
        check_features(X)
        y, t = np.asarray(y), np.asarray(t)
        self.treated_model = make_classifier(self.seed).fit(X[t == 1], y[t == 1])
        self.control_model = make_classifier(self.seed).fit(X[t == 0], y[t == 0])
        return self

    def predict(self, X):
        return _prob(self.treated_model, X) - _prob(self.control_model, X)


class XLearner:
    """Compare each person with the other group's model, then predict those effects directly."""

    def __init__(self, seed=config.SEED):
        self.seed = seed

    def fit(self, X, y, t):
        check_features(X)
        y, t = np.asarray(y), np.asarray(t)
        X_treated, y_treated = X[t == 1], y[t == 1]
        X_control, y_control = X[t == 0], y[t == 0]

        # Stage 1: the same two outcome models as the T-learner.
        treated_model = make_classifier(self.seed).fit(X_treated, y_treated)
        control_model = make_classifier(self.seed).fit(X_control, y_control)

        # Stage 2: estimate each person's effect as actual vs predicted from the other group.
        effect_on_treated = y_treated - _prob(control_model, X_treated)
        effect_on_control = _prob(treated_model, X_control) - y_control

        # Stage 3: learn to predict those effects from the features.
        self.treated_effect_model = make_regressor(self.seed).fit(X_treated, effect_on_treated)
        self.control_effect_model = make_regressor(self.seed).fit(X_control, effect_on_control)

        # Assignment was random, so the propensity is simply the treated share.
        self.treated_share = t.mean()
        return self

    def predict(self, X):
        # Give more weight to the effect model built from the larger group's outcome model.
        return self.treated_share * self.control_effect_model.predict(X) + (
            1 - self.treated_share
        ) * self.treated_effect_model.predict(X)


def hillstrom_features(df):
    """One-hot encode pre-treatment columns. This learns nothing from outcomes."""
    features = df[list(config.HILLSTROM_MODEL_FEATURES)]
    categorical = features.select_dtypes(exclude="number").columns.tolist()
    encoded = pd.get_dummies(features, columns=categorical, dtype=float)
    check_features(encoded)
    return encoded


def split_hillstrom(df, test_size=config.TEST_SIZE, seed=config.SEED):
    """Split once, so that all models share the same untouched hold-out customers."""
    strata = df[config.HILLSTROM_ARM_COL].astype(str) + "_" + df["conversion"].astype(str)
    return train_test_split(df, test_size=test_size, random_state=seed, stratify=strata)


def lenta_features(df):
    """Every Lenta column except the outcome, treatment and the columns excluded in config."""
    dropped = [config.LENTA_OUTCOME, config.LENTA_TREATMENT_COL, *config.LENTA_EXCLUDED_FEATURES]
    features = df.drop(columns=dropped)
    check_features(features)
    return features


def split_lenta(df, test_size=config.TEST_SIZE, seed=config.SEED):
    strata = df[config.LENTA_TREATMENT_COL].astype(str) + "_" + df[config.LENTA_OUTCOME].astype(str)
    return train_test_split(df, test_size=test_size, random_state=seed, stratify=strata)


def fit_and_score(X_train, y_train, t_train, X_holdout, seed=config.SEED):
    """Fit on the training rows only and return each model's hold-out scores."""
    models = {"response": ResponseModel(seed), "t_learner": TLearner(seed), "x_learner": XLearner(seed)}
    return {name: model.fit(X_train, y_train, t_train).predict(X_holdout) for name, model in models.items()}


def _email_rows(df, email):
    subset = df[df[config.HILLSTROM_ARM_COL].isin([email, config.HILLSTROM_CONTROL])]
    return subset, (subset[config.HILLSTROM_ARM_COL] == email).astype(int).to_numpy()


def run_hillstrom_uplift(df, outcomes=("conversion", "visit"), n_boot=config.N_BOOTSTRAP, seed=config.SEED):
    """Fit and evaluate each model for each email and outcome. Returns metrics and scores."""
    features = hillstrom_features(df)
    train, holdout = split_hillstrom(df, seed=seed)
    rows, all_scores = [], {}
    for email in (config.HILLSTROM_MENS, config.HILLSTROM_WOMENS):
        train_rows, t_train = _email_rows(train, email)
        holdout_rows, t_holdout = _email_rows(holdout, email)
        X_train, X_holdout = features.loc[train_rows.index], features.loc[holdout_rows.index]
        for outcome in outcomes:
            y_train = train_rows[outcome].to_numpy()
            y_holdout = holdout_rows[outcome].to_numpy()
            scores = fit_and_score(X_train, y_train, t_train, X_holdout, seed)
            all_scores[email, outcome] = {
                "index": holdout_rows.index,
                "y": y_holdout,
                "t": t_holdout,
                "scores": scores,
            }
            results = evaluate.bootstrap_metrics(
                y_holdout, t_holdout, scores, evaluate.default_metrics(), baseline="response", n_boot=n_boot, seed=seed
            )
            overall = evaluate.bootstrap_metrics(
                y_holdout,
                t_holdout,
                {"all_customers": np.zeros(len(y_holdout))},
                {"uplift_all": lambda y, t, s: evaluate.uplift_at_k(y, t, s, 1.0)},
                n_boot=n_boot,
                seed=seed,
            )
            rows += [{"email": email, "outcome": outcome, **row} for row in overall + results]
    return pd.DataFrame(rows), all_scores


def stability_check(df, seeds=range(1, 11), outcomes=("conversion", "visit")):
    """Repeat the split and fit with other seeds. Bootstrap intervals hold the models fixed, so
    this shows how much a result depends on which customers and seed trained the model."""
    features = hillstrom_features(df)
    rows = []
    for seed in seeds:
        train, holdout = split_hillstrom(df, seed=seed)
        for email in (config.HILLSTROM_MENS, config.HILLSTROM_WOMENS):
            train_rows, t_train = _email_rows(train, email)
            holdout_rows, t_holdout = _email_rows(holdout, email)
            X_train, X_holdout = features.loc[train_rows.index], features.loc[holdout_rows.index]
            for outcome in outcomes:
                y_holdout = holdout_rows[outcome].to_numpy()
                scores = fit_and_score(X_train, train_rows[outcome].to_numpy(), t_train, X_holdout, seed)
                for model, score in scores.items():
                    rows.append(
                        {
                            "email": email,
                            "outcome": outcome,
                            "model": model,
                            "seed": seed,
                            "qini_area": evaluate.qini_area(y_holdout, t_holdout, score),
                            "uplift_top_30pct": evaluate.uplift_at_k(y_holdout, t_holdout, score, 0.3),
                            "uplift_all": evaluate.uplift_at_k(y_holdout, t_holdout, score, 1.0),
                        }
                    )
    return pd.DataFrame(rows)


def summarize_stability(table):
    groups = table.groupby(["email", "outcome", "model"])
    summary = groups[["qini_area", "uplift_top_30pct", "uplift_all"]].agg(["mean", "min", "max"])
    summary[("seeds_qini_positive", "")] = groups["qini_area"].apply(lambda values: int((values > 0).sum()))
    return summary


class _ProbabilityOutput:
    """econml calls predict(), which on a classifier gives 0/1 labels. This returns probabilities."""

    def __init__(self, seed=config.SEED):
        self.seed = seed

    def fit(self, X, y):
        self.model = make_classifier(self.seed).fit(X, y)
        return self

    def predict(self, X):
        return _prob(self.model, X)


def library_t_scores(X_train, y_train, t_train, X_holdout, seed=config.SEED):
    from sklift.models import TwoModels

    model = TwoModels(make_classifier(seed), make_classifier(seed), method="vanilla")
    model.fit(X_train, pd.Series(y_train, index=X_train.index), pd.Series(t_train, index=X_train.index))
    return np.asarray(model.predict(X_holdout))


def library_x_scores(X_train, y_train, t_train, X_holdout, seed=config.SEED):
    from econml.metalearners import XLearner as EconmlXLearner
    from sklearn.dummy import DummyClassifier

    model = EconmlXLearner(
        models=_ProbabilityOutput(seed),
        cate_models=make_regressor(seed),
        # A constant propensity, like ours, because assignment was random.
        propensity_model=DummyClassifier(strategy="prior"),
    )
    model.fit(np.asarray(y_train), np.asarray(t_train), X=np.asarray(X_train))
    return np.asarray(model.effect(np.asarray(X_holdout))).ravel()


def compare_with_libraries(df, seed=config.SEED):
    """How far are our learners from the sklift and econml versions on the real hold-out?"""
    features = hillstrom_features(df)
    train, holdout = split_hillstrom(df, seed=seed)
    rows = []
    for email in (config.HILLSTROM_MENS, config.HILLSTROM_WOMENS):
        train_rows, t_train = _email_rows(train, email)
        holdout_rows, t_holdout = _email_rows(holdout, email)
        X_train, X_holdout = features.loc[train_rows.index], features.loc[holdout_rows.index]
        y_train, y_holdout = train_rows["conversion"].to_numpy(), holdout_rows["conversion"].to_numpy()
        pairs = {
            "T-learner vs sklift": (TLearner(seed), library_t_scores),
            "X-learner vs econml": (XLearner(seed), library_x_scores),
        }
        for name, (model, library) in pairs.items():
            ours = model.fit(X_train, y_train, t_train).predict(X_holdout)
            theirs = library(X_train, y_train, t_train, X_holdout, seed)
            rows.append(
                {
                    "email": email,
                    "comparison": name,
                    "correlation": np.corrcoef(ours, theirs)[0, 1],
                    "mean_abs_gap": np.abs(ours - theirs).mean(),
                    "max_abs_gap": np.abs(ours - theirs).max(),
                    "qini_ours": evaluate.qini_area(y_holdout, t_holdout, ours),
                    "qini_library": evaluate.qini_area(y_holdout, t_holdout, theirs),
                }
            )
    return pd.DataFrame(rows)


def simulated_recovery(n=40_000, seed=config.SEED):
    """How well do the models recover effects that are known, on simulated hold-out data?"""
    df = simulate.simulate_uplift(n, seed)
    train, holdout = train_test_split(df, test_size=config.TEST_SIZE, random_state=seed)
    X_train, X_holdout = train[list(simulate.FEATURES)], holdout[list(simulate.FEATURES)]
    truth = holdout["true_effect"].to_numpy()
    scores = fit_and_score(X_train, train["outcome"], train["treatment"], X_holdout, seed)
    scores["true_effect"] = truth
    rows = []
    for name, score in scores.items():
        is_effect_model = name in ("t_learner", "x_learner", "true_effect")
        rows.append(
            {
                "model": name,
                "correlation_with_truth": np.corrcoef(score, truth)[0, 1],
                "rmse": np.sqrt(((score - truth) ** 2).mean()) if is_effect_model else np.nan,
                "mean_prediction": score.mean(),
                "mean_truth": truth.mean(),
                "true_effect_of_top_30pct": truth[np.argsort(score)[::-1][: int(len(truth) * 0.3)]].mean(),
            }
        )
    return pd.DataFrame(rows)


if __name__ == "__main__":
    pd.set_option("display.width", 220, "display.max_columns", 30, "display.max_rows", 200)
    pd.set_option("display.float_format", lambda value: f"{value:.5f}")
    print("SIMULATED RECOVERY\n", simulated_recovery().to_string(index=False), "\n")
    hillstrom = load.load_hillstrom()
    print("LIBRARY COMPARISON\n", compare_with_libraries(hillstrom).to_string(index=False), "\n")
    table, _ = run_hillstrom_uplift(hillstrom)
    print("HOLD-OUT METRICS\n", table.to_string(index=False), "\n")
    print("STABILITY ACROSS 10 SEEDS\n", summarize_stability(stability_check(hillstrom)).to_string())
