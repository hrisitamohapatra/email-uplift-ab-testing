"""Compare targeting strategies on incremental conversions and profit under ASSUMED costs.

Run with: python -m src.roi
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from src import config, learners, load

FRACTIONS = np.array(config.ROI_FRACTIONS)

# Name: (learner, which outcome it trains on). "primary" is conversion, "secondary" is visit.
MODELS = {
    "response": (learners.ResponseModel, "primary"),
    "t_learner": (learners.TLearner, "primary"),
    "x_learner": (learners.XLearner, "primary"),
    "t_learner_visit_EXPLORATORY": (learners.TLearner, "secondary"),
    "x_learner_visit_EXPLORATORY": (learners.XLearner, "secondary"),
}


def uplift_by_fraction(y, t, score, fractions=FRACTIONS):
    """Treated rate minus control rate among the top share of customers, for every share at once."""
    y, t = np.asarray(y, dtype=float), np.asarray(t, dtype=float)
    order = np.argsort(np.asarray(score, dtype=float), kind="mergesort")[::-1]
    y, t = y[order], t[order]
    treated_outcomes = np.cumsum(y * t)
    control_outcomes = np.cumsum(y * (1 - t))
    treated_count = np.cumsum(t)
    ends = (len(y) * np.asarray(fractions)).astype(int) - 1
    customers = ends + 1
    with np.errstate(divide="ignore", invalid="ignore"):
        treated_rate = treated_outcomes[ends] / treated_count[ends]
        control_rate = control_outcomes[ends] / (customers - treated_count[ends])
    return treated_rate - control_rate


def profit_per_1000_customers(uplift, fraction, cost=config.ASSUMED_COST_PER_CONTACT_USD, margin=config.ASSUMED_MARGIN_PER_CONVERSION_USD):
    """Added profit vs emailing no one, under ASSUMED costs: 1000 * k * (uplift * margin - cost)."""
    return 1000 * np.asarray(fraction) * (np.asarray(uplift) * margin - cost)


def choose_fraction(uplift_curve, fractions=FRACTIONS, cost=config.ASSUMED_COST_PER_CONTACT_USD, margin=config.ASSUMED_MARGIN_PER_CONVERSION_USD):
    """The most profitable share to contact, or 0 (email no one) if no share makes money."""
    profit = profit_per_1000_customers(uplift_curve, fractions, cost, margin)
    if np.all(np.isnan(profit)) or np.nanmax(profit) <= 0:
        return 0.0
    return float(fractions[np.nanargmax(profit)])


def fit_all_models(X_train, y_primary, y_secondary, t_train, X_score, seed=config.SEED):
    """Fit every model on the training rows and score X_score."""
    outcomes = {"primary": np.asarray(y_primary), "secondary": np.asarray(y_secondary)}
    return {
        name: learner(seed).fit(X_train, outcomes[outcome], t_train).predict(X_score)
        for name, (learner, outcome) in MODELS.items()
    }


def out_of_fold_scores(X_train, y_primary, y_secondary, t_train, seed=config.SEED, n_folds=config.ROI_FOLDS):
    """Score each training customer with models that did not see them. Hold-out data is not used."""
    y_primary, y_secondary, t_train = np.asarray(y_primary), np.asarray(y_secondary), np.asarray(t_train)
    scores = {name: np.empty(len(X_train)) for name in MODELS}
    strata = t_train * 2 + y_primary
    folds = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for fit_rows, score_rows in folds.split(X_train, strata):
        fold_scores = fit_all_models(
            X_train.iloc[fit_rows], y_primary[fit_rows], y_secondary[fit_rows], t_train[fit_rows], X_train.iloc[score_rows], seed
        )
        for name, values in fold_scores.items():
            scores[name][score_rows] = values
    return scores


def _at_fraction(curve, fraction, fractions=FRACTIONS):
    """Uplift at a chosen share. At 0 nobody is contacted, so it counts as no added conversions."""
    if fraction == 0:
        return 0.0
    return curve[np.argmin(np.abs(fractions - fraction))]


def strategy_table(y, t, scores, chosen, n_boot=config.N_BOOTSTRAP, alpha=config.ALPHA, seed=config.SEED):
    """Hold-out results per strategy, with intervals and a paired gain against emailing everyone."""
    y, t = np.asarray(y, dtype=float), np.asarray(t, dtype=float)
    rng = np.random.default_rng(seed)
    zeros = np.zeros(len(y))
    everyone_draws = np.empty(n_boot)
    uplift_draws = {name: np.empty(n_boot) for name in scores}
    for i in range(n_boot):
        rows = rng.integers(0, len(y), len(y))
        everyone_draws[i] = uplift_by_fraction(y[rows], t[rows], zeros[rows], [1.0])[0]
        for name, score in scores.items():
            curve = uplift_by_fraction(y[rows], t[rows], score[rows])
            uplift_draws[name][i] = _at_fraction(curve, chosen[name])

    quantiles = [alpha / 2, 1 - alpha / 2]
    everyone_uplift = uplift_by_fraction(y, t, zeros, [1.0])[0]
    everyone_profit = profit_per_1000_customers(everyone_draws, 1.0)
    rows = [
        {"strategy": "email no one", "fraction": 0.0, "incremental_per_1000_contacts": 0.0, "profit_per_1000_customers": 0.0},
        {
            "strategy": "email everyone",
            "fraction": 1.0,
            "incremental_per_1000_contacts": 1000 * everyone_uplift,
            "incremental_low": 1000 * np.quantile(everyone_draws, quantiles[0]),
            "incremental_high": 1000 * np.quantile(everyone_draws, quantiles[1]),
            "profit_per_1000_customers": float(profit_per_1000_customers(everyone_uplift, 1.0)),
            "profit_low": np.quantile(everyone_profit, quantiles[0]),
            "profit_high": np.quantile(everyone_profit, quantiles[1]),
        },
    ]
    for name, score in scores.items():
        fraction = chosen[name]
        uplift = _at_fraction(uplift_by_fraction(y, t, score), fraction)
        profit = float(profit_per_1000_customers(uplift, fraction))
        profit_draws = profit_per_1000_customers(uplift_draws[name], fraction)
        gain_draws = profit_draws - everyone_profit
        rows.append(
            {
                "strategy": f"top k by {name}",
                "fraction": fraction,
                "incremental_per_1000_contacts": 1000 * uplift,
                "incremental_low": 1000 * np.nanquantile(uplift_draws[name], quantiles[0]),
                "incremental_high": 1000 * np.nanquantile(uplift_draws[name], quantiles[1]),
                "profit_per_1000_customers": profit,
                "profit_low": np.nanquantile(profit_draws, quantiles[0]),
                "profit_high": np.nanquantile(profit_draws, quantiles[1]),
                "gain_vs_everyone": profit - float(profit_per_1000_customers(everyone_uplift, 1.0)),
                "gain_low": np.nanquantile(gain_draws, quantiles[0]),
                "gain_high": np.nanquantile(gain_draws, quantiles[1]),
            }
        )
    return pd.DataFrame(rows)


def profit_curves(y, t, scores):
    """Hold-out profit at each share. Its peak is optimistic, so it is never used to choose k."""
    rows = []
    for name, score in scores.items():
        uplift = uplift_by_fraction(y, t, score)
        profit = profit_per_1000_customers(uplift, FRACTIONS)
        rows += [
            {"model": name, "fraction": f, "incremental_per_1000_contacts": 1000 * u, "profit_per_1000_customers": p}
            for f, u, p in zip(FRACTIONS, uplift, profit)
        ]
    return pd.DataFrame(rows)


def sensitivity_table(oof_curves, holdout_curves, everyone_uplift):
    """Re-choose k inside training data for each assumed cost and margin, then score on the hold-out."""
    rows = []
    for cost in config.ROI_COST_GRID_USD:
        for margin in config.ROI_MARGIN_GRID_USD:
            row = {
                "assumed_cost": cost,
                "assumed_margin": margin,
                "break_even_per_1000_contacts": 1000 * cost / margin,
                "everyone_profit": float(profit_per_1000_customers(everyone_uplift, 1.0, cost, margin)),
            }
            for name in oof_curves:
                fraction = choose_fraction(oof_curves[name], FRACTIONS, cost, margin)
                uplift = _at_fraction(holdout_curves[name], fraction)
                row[f"{name}_k"] = fraction
                row[f"{name}_profit"] = float(profit_per_1000_customers(uplift, fraction, cost, margin))
            rows.append(row)
    return pd.DataFrame(rows)


def run_experiment(X_train, y_train, y2_train, t_train, X_holdout, y_holdout, t_holdout, n_boot=config.N_BOOTSTRAP, seed=config.SEED):
    """Choose k inside the training data, then score every strategy on the hold-out."""
    oof = out_of_fold_scores(X_train, y_train, y2_train, t_train, seed)
    oof_curves = {name: uplift_by_fraction(y_train, t_train, score) for name, score in oof.items()}
    chosen = {name: choose_fraction(curve) for name, curve in oof_curves.items()}
    scores = fit_all_models(X_train, y_train, y2_train, t_train, X_holdout, seed)
    holdout_curves = {name: uplift_by_fraction(y_holdout, t_holdout, score) for name, score in scores.items()}
    everyone_uplift = uplift_by_fraction(y_holdout, t_holdout, np.zeros(len(y_holdout)), [1.0])[0]
    return {
        "chosen": chosen,
        "scores": scores,
        "strategies": strategy_table(y_holdout, t_holdout, scores, chosen, n_boot, seed=seed),
        "curves": profit_curves(y_holdout, t_holdout, scores),
        "sensitivity": sensitivity_table(oof_curves, holdout_curves, everyone_uplift),
    }


def _hillstrom_email(features, train, holdout, email):
    train_rows, t_train = learners._email_rows(train, email)
    holdout_rows, t_holdout = learners._email_rows(holdout, email)
    return (
        features.loc[train_rows.index],
        train_rows["conversion"].to_numpy(),
        train_rows["visit"].to_numpy(),
        t_train,
        features.loc[holdout_rows.index],
        holdout_rows["conversion"].to_numpy(),
        t_holdout,
    )


def run_hillstrom_roi(df, n_boot=config.N_BOOTSTRAP, seed=config.SEED):
    features = learners.hillstrom_features(df)
    train, holdout = learners.split_hillstrom(df, seed=seed)
    return {
        email: run_experiment(*_hillstrom_email(features, train, holdout, email), n_boot=n_boot, seed=seed)
        for email in (config.HILLSTROM_MENS, config.HILLSTROM_WOMENS)
    }


def roi_stability(df, seeds=range(1, 11)):
    """Repeat the whole selection and scoring with other seeds. Point estimates only."""
    features = learners.hillstrom_features(df)
    rows = []
    for seed in seeds:
        train, holdout = learners.split_hillstrom(df, seed=seed)
        for email in (config.HILLSTROM_MENS, config.HILLSTROM_WOMENS):
            X_train, y_train, y2_train, t_train, X_holdout, y_holdout, t_holdout = _hillstrom_email(features, train, holdout, email)
            oof = out_of_fold_scores(X_train, y_train, y2_train, t_train, seed)
            scores = fit_all_models(X_train, y_train, y2_train, t_train, X_holdout, seed)
            everyone = float(profit_per_1000_customers(uplift_by_fraction(y_holdout, t_holdout, np.zeros(len(y_holdout)), [1.0])[0], 1.0))
            for name in MODELS:
                fraction = choose_fraction(uplift_by_fraction(y_train, t_train, oof[name]))
                uplift = _at_fraction(uplift_by_fraction(y_holdout, t_holdout, scores[name]), fraction)
                profit = float(profit_per_1000_customers(uplift, fraction))
                rows.append({"email": email, "model": name, "seed": seed, "fraction": fraction, "profit": profit, "gain_vs_everyone": profit - everyone})
    return pd.DataFrame(rows)


def summarize_roi_stability(table):
    groups = table.groupby(["email", "model"])
    summary = groups[["fraction", "profit", "gain_vs_everyone"]].agg(["mean", "min", "max"])
    summary[("seeds_beating_everyone", "")] = groups["gain_vs_everyone"].apply(lambda values: int((values > 0).sum()))
    return summary


if __name__ == "__main__":
    pd.set_option("display.width", 240, "display.max_columns", 40, "display.max_rows", 200)
    pd.set_option("display.float_format", lambda value: f"{value:.3f}")
    print(f"ASSUMED COSTS: ${config.ASSUMED_COST_PER_CONTACT_USD} per contact, ${config.ASSUMED_MARGIN_PER_CONVERSION_USD} margin per conversion\n")
    hillstrom = load.load_hillstrom()
    for email_name, result in run_hillstrom_roi(hillstrom).items():
        print(f"=== {email_name}: STRATEGIES (hold-out; k chosen inside training data)\n", result["strategies"].to_string(index=False), "\n")
        print(f"=== {email_name}: PROFIT CURVES (hold-out, assumed costs)\n", result["curves"].pivot(index="fraction", columns="model", values="profit_per_1000_customers").to_string(), "\n")
        print(f"=== {email_name}: SENSITIVITY (hold-out, assumed costs)\n", result["sensitivity"].to_string(index=False), "\n")
    print("=== STABILITY ACROSS 10 SEEDS (assumed costs)\n", summarize_roi_stability(roi_stability(hillstrom)).to_string())
