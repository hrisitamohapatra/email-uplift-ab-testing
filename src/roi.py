"""Compare targeting strategies on incremental conversions and profit under ASSUMED costs.

Run with: python -m src.roi
Run Lenta with: python -m src.roi lenta
"""

import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from src import config, evaluate, experiment, learners, load

FRACTIONS = np.array(config.ROI_FRACTIONS)

# Name: (learner, which outcome it trains on). "primary" is conversion, "secondary" is visit.
MODELS = {
    "response": (learners.ResponseModel, "primary"),
    "t_learner": (learners.TLearner, "primary"),
    "x_learner": (learners.XLearner, "primary"),
    "t_learner_visit_EXPLORATORY": (learners.TLearner, "secondary"),
    "x_learner_visit_EXPLORATORY": (learners.XLearner, "secondary"),
}
# Lenta has only one outcome, so it has no exploratory second-outcome models.
PRIMARY_MODELS = {name: spec for name, spec in MODELS.items() if spec[1] == "primary"}


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


def fit_all_models(X_train, y_primary, y_secondary, t_train, X_score, seed=config.SEED, models=MODELS):
    """Fit every model on the training rows and score X_score."""
    outcomes = {"primary": np.asarray(y_primary), "secondary": np.asarray(y_secondary)}
    return {
        name: learner(seed).fit(X_train, outcomes[outcome], t_train).predict(X_score)
        for name, (learner, outcome) in models.items()
    }


def out_of_fold_scores(X_train, y_primary, y_secondary, t_train, seed=config.SEED, n_folds=config.ROI_FOLDS, models=MODELS):
    """Score each training customer with models that did not see them. Hold-out data is not used."""
    y_primary, y_secondary, t_train = np.asarray(y_primary), np.asarray(y_secondary), np.asarray(t_train)
    scores = {name: np.empty(len(X_train)) for name in models}
    strata = t_train * 2 + y_primary
    folds = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for fit_rows, score_rows in folds.split(X_train, strata):
        fold_scores = fit_all_models(
            X_train.iloc[fit_rows], y_primary[fit_rows], y_secondary[fit_rows], t_train[fit_rows], X_train.iloc[score_rows], seed, models
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


def run_experiment(X_train, y_train, y2_train, t_train, X_holdout, y_holdout, t_holdout, n_boot=config.N_BOOTSTRAP, seed=config.SEED, models=MODELS):
    """Choose k inside the training data, then score every strategy on the hold-out."""
    oof = out_of_fold_scores(X_train, y_train, y2_train, t_train, seed, models=models)
    oof_curves = {name: uplift_by_fraction(y_train, t_train, score) for name, score in oof.items()}
    chosen = {name: choose_fraction(curve) for name, curve in oof_curves.items()}
    scores = fit_all_models(X_train, y_train, y2_train, t_train, X_holdout, seed, models)
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


def _lenta_arrays(df, features, seed):
    train, holdout = learners.split_lenta(df, seed=seed)
    return (
        features.loc[train.index],
        train[config.LENTA_OUTCOME].to_numpy(),
        train[config.LENTA_TREATMENT_COL].to_numpy(),
        features.loc[holdout.index],
        holdout[config.LENTA_OUTCOME].to_numpy(),
        holdout[config.LENTA_TREATMENT_COL].to_numpy(),
    )


def run_lenta_roi(df, n_boot=config.N_BOOTSTRAP, seed=config.SEED):
    """Same method as Hillstrom. The outcome is a store visit, so money figures are illustrative only."""
    X_train, y_train, t_train, X_holdout, y_holdout, t_holdout = _lenta_arrays(df, learners.lenta_features(df), seed)
    result = run_experiment(X_train, y_train, y_train, t_train, X_holdout, y_holdout, t_holdout, n_boot, seed, PRIMARY_MODELS)
    result["metrics"] = pd.DataFrame(
        evaluate.bootstrap_metrics(y_holdout, t_holdout, result["scores"], evaluate.default_metrics(), baseline="response", n_boot=n_boot, seed=seed)
    )
    return result


def lenta_stability(df, chosen, seeds=range(1, 6)):
    """Refit with other seeds, keeping each model's chosen share fixed. Point estimates only."""
    features = learners.lenta_features(df)
    rows = []
    for seed in seeds:
        X_train, y_train, t_train, X_holdout, y_holdout, t_holdout = _lenta_arrays(df, features, seed)
        scores = fit_all_models(X_train, y_train, y_train, t_train, X_holdout, seed, PRIMARY_MODELS)
        everyone = float(profit_per_1000_customers(uplift_by_fraction(y_holdout, t_holdout, np.zeros(len(y_holdout)), [1.0])[0], 1.0))
        for name, score in scores.items():
            uplift = _at_fraction(uplift_by_fraction(y_holdout, t_holdout, score), chosen[name])
            profit = float(profit_per_1000_customers(uplift, chosen[name]))
            rows.append(
                {
                    "model": name,
                    "seed": seed,
                    "qini_area": evaluate.qini_area(y_holdout, t_holdout, score),
                    "uplift_top_30pct": evaluate.uplift_at_k(y_holdout, t_holdout, score, 0.3),
                    "profit": profit,
                    "gain_vs_everyone": profit - everyone,
                }
            )
    return pd.DataFrame(rows)


def fixed_budget_table(y, t, scores, share, n_boot=config.N_BOOTSTRAP, alpha=config.ALPHA, seed=config.SEED, budget=config.MEMO_BUDGET):
    """Lift among the top and bottom share by each score, against a random group of the same size.

    A random group's expected lift is the lift across all customers. Every comparison uses
    the same resampled rows. Profit is for contacting `budget` customers, under ASSUMED costs.
    """
    y, t = np.asarray(y, dtype=float), np.asarray(t, dtype=float)
    groups = {"random": (np.zeros(len(y)), 1.0)}
    for name, score in scores.items():
        groups[f"top {name}"] = (np.asarray(score, dtype=float), share)
        groups[f"bottom {name}"] = (-np.asarray(score, dtype=float), share)

    rng = np.random.default_rng(seed)
    draws = {name: np.empty(n_boot) for name in groups}
    for i in range(n_boot):
        rows = rng.integers(0, len(y), len(y))
        for name, (score, fraction) in groups.items():
            draws[name][i] = uplift_by_fraction(y[rows], t[rows], score[rows], [fraction])[0]

    quantiles = [alpha / 2, 1 - alpha / 2]
    results = []
    for name, (score, fraction) in groups.items():
        uplift = uplift_by_fraction(y, t, score, [fraction])[0]
        profit_draws = budget * (draws[name] * config.ASSUMED_MARGIN_PER_CONVERSION_USD - config.ASSUMED_COST_PER_CONTACT_USD)
        gain_draws = draws[name] - draws["random"]
        results.append(
            {
                "group": name,
                "added_per_1000_contacts": 1000 * uplift,
                "added_low": 1000 * np.nanquantile(draws[name], quantiles[0]),
                "added_high": 1000 * np.nanquantile(draws[name], quantiles[1]),
                "profit_for_budget": budget * (uplift * config.ASSUMED_MARGIN_PER_CONVERSION_USD - config.ASSUMED_COST_PER_CONTACT_USD),
                "profit_low": np.nanquantile(profit_draws, quantiles[0]),
                "profit_high": np.nanquantile(profit_draws, quantiles[1]),
                "gain_vs_random_per_1000": 1000 * (uplift - uplift_by_fraction(y, t, groups["random"][0], [1.0])[0]),
                "gain_low": 1000 * np.nanquantile(gain_draws, quantiles[0]),
                "gain_high": 1000 * np.nanquantile(gain_draws, quantiles[1]),
            }
        )
    return pd.DataFrame(results)


def t_learner_vs_response(y, t, scores, share, n_boot=config.N_BOOTSTRAP, alpha=config.ALPHA, seed=config.SEED):
    """T-learner's top group minus the response model's, per 1,000 contacts, on the same resamples."""
    y, t = np.asarray(y, dtype=float), np.asarray(t, dtype=float)
    rng = np.random.default_rng(seed)
    gaps = np.empty(n_boot)
    for i in range(n_boot):
        rows = rng.integers(0, len(y), len(y))
        gaps[i] = (
            uplift_by_fraction(y[rows], t[rows], scores["t_learner"][rows], [share])[0]
            - uplift_by_fraction(y[rows], t[rows], scores["response"][rows], [share])[0]
        )
    estimate = uplift_by_fraction(y, t, scores["t_learner"], [share])[0] - uplift_by_fraction(y, t, scores["response"], [share])[0]
    low, high = np.nanquantile(gaps, [alpha / 2, 1 - alpha / 2])
    return {"difference": 1000 * estimate, "low": 1000 * low, "high": 1000 * high}


def group_profile(features, score, share):
    """Average features among the highest and lowest scored customers, and across everyone."""
    size = int(len(features) * share)
    order = np.argsort(np.asarray(score, dtype=float), kind="mergesort")[::-1]
    return pd.DataFrame(
        {
            "top_group": features.iloc[order[:size]].mean(),
            "bottom_group": features.iloc[order[-size:]].mean(),
            "all_customers": features.mean(),
        }
    )


def _men_vs_none_scores(features, df, seed):
    train, holdout = learners.split_hillstrom(df, seed=seed)
    X_train, y_train, _, t_train, X_holdout, y_holdout, t_holdout = _hillstrom_email(features, train, holdout, config.HILLSTROM_MENS)
    scores = {
        "response": learners.ResponseModel(seed).fit(X_train, y_train, t_train).predict(X_holdout),
        "t_learner": learners.TLearner(seed).fit(X_train, y_train, t_train).predict(X_holdout),
    }
    return X_holdout, y_holdout, t_holdout, scores


def run_memo(df, n_boot=config.N_BOOTSTRAP, seed=config.SEED):
    """The memo's question: which 10,000 to email with the men's email, and which to leave out."""
    share = config.MEMO_BUDGET / len(df)
    features = learners.hillstrom_features(df)
    X_holdout, y_holdout, t_holdout, scores = _men_vs_none_scores(features, df, seed)
    return {
        "share": share,
        "holdout_rows": len(y_holdout),
        "budget_table": fixed_budget_table(y_holdout, t_holdout, scores, share, n_boot, seed=seed),
        "t_vs_response": t_learner_vs_response(y_holdout, t_holdout, scores, share, n_boot, seed=seed),
        "profile": group_profile(X_holdout, scores["t_learner"], share),
    }


def memo_stability(df, seeds=range(1, 11)):
    """Repeat the fixed-budget comparison with other seeds. Point estimates only."""
    share = config.MEMO_BUDGET / len(df)
    features = learners.hillstrom_features(df)
    rows = []
    for seed in seeds:
        _, y_holdout, t_holdout, scores = _men_vs_none_scores(features, df, seed)
        overall = uplift_by_fraction(y_holdout, t_holdout, np.zeros(len(y_holdout)), [1.0])[0]
        row = {"seed": seed, "random": 1000 * overall}
        for name, score in scores.items():
            row[f"top {name}"] = 1000 * uplift_by_fraction(y_holdout, t_holdout, score, [share])[0]
            row[f"bottom {name}"] = 1000 * uplift_by_fraction(y_holdout, t_holdout, -score, [share])[0]
        rows.append(row)
    return pd.DataFrame(rows)


def full_sample_profit(df, cost=config.ASSUMED_COST_PER_CONTACT_USD, margin=config.ASSUMED_MARGIN_PER_CONVERSION_USD):
    """Profit per 1,000 customers from emailing everyone, using all customers in each arm."""
    rows = []
    for email in (config.HILLSTROM_MENS, config.HILLSTROM_WOMENS):
        treated = df.loc[df[config.HILLSTROM_ARM_COL] == email, "conversion"]
        control = df.loc[df[config.HILLSTROM_ARM_COL] == config.HILLSTROM_CONTROL, "conversion"]
        lift = experiment.two_proportion_test(treated.sum(), len(treated), control.sum(), len(control))
        rows.append(
            {
                "email": email,
                "added_per_1000": 1000 * lift["difference"],
                "profit": float(profit_per_1000_customers(lift["difference"], 1.0, cost, margin)),
                "profit_low": float(profit_per_1000_customers(lift["ci_low"], 1.0, cost, margin)),
                "profit_high": float(profit_per_1000_customers(lift["ci_high"], 1.0, cost, margin)),
            }
        )
    return pd.DataFrame(rows)


if __name__ == "__main__":
    pd.set_option("display.width", 240, "display.max_columns", 40, "display.max_rows", 200)
    pd.set_option("display.float_format", lambda value: f"{value:.5f}" if abs(value) < 0.01 else f"{value:.3f}")
    print(f"ASSUMED COSTS: ${config.ASSUMED_COST_PER_CONTACT_USD} per contact, ${config.ASSUMED_MARGIN_PER_CONVERSION_USD} margin per conversion\n")
    if "memo" in sys.argv:
        hillstrom = load.load_hillstrom()
        print("FULL-SAMPLE PROFIT FROM EMAILING EVERYONE\n", full_sample_profit(hillstrom).to_string(index=False), "\n")
        memo = run_memo(hillstrom)
        print(f"SHARE {memo['share']:.5f}, HOLD-OUT ROWS {memo['holdout_rows']}\n")
        print("FIXED BUDGET OF 10,000, MEN'S EMAIL (hold-out)\n", memo["budget_table"].to_string(index=False), "\n")
        print("T-LEARNER MINUS RESPONSE, TOP GROUP, PER 1,000 CONTACTS", memo["t_vs_response"], "\n")
        print("PROFILE BY T-LEARNER SCORE (hold-out)\n", memo["profile"].to_string(), "\n")
        stability = memo_stability(hillstrom)
        print("STABILITY ACROSS 10 SEEDS\n", stability.drop(columns="seed").describe().loc[["mean", "min", "max"]].to_string())
        print("T-learner top beats random in", int((stability["top t_learner"] > stability["random"]).sum()), "of 10")
        print("Response top beats random in", int((stability["top response"] > stability["random"]).sum()), "of 10")
        print("T-learner top beats response top in", int((stability["top t_learner"] > stability["top response"]).sum()), "of 10")
        sys.exit()
    if "lenta" in sys.argv:
        config.LGBM_PARAMS["n_jobs"] = 4  # This dataset is about 10 times larger.
        lenta = load.load_lenta()
        outcome = run_lenta_roi(lenta)
        print("LENTA: outcome is a store visit, so money figures are ILLUSTRATIVE ONLY\n")
        print("CHOSEN SHARES", outcome["chosen"], "\n")
        print("STRATEGIES (hold-out)\n", outcome["strategies"].to_string(index=False), "\n")
        print("RANKING METRICS (hold-out)\n", outcome["metrics"].to_string(index=False), "\n")
        print("PROFIT CURVES (hold-out)\n", outcome["curves"].pivot(index="fraction", columns="model", values="profit_per_1000_customers").to_string(), "\n")
        print("SENSITIVITY (hold-out)\n", outcome["sensitivity"].to_string(index=False), "\n")
        stability = lenta_stability(lenta, outcome["chosen"])
        print("STABILITY ACROSS 5 SEEDS, SHARE HELD FIXED\n", stability.groupby("model")[["qini_area", "uplift_top_30pct", "profit", "gain_vs_everyone"]].agg(["mean", "min", "max"]).to_string())
        sys.exit()
    hillstrom = load.load_hillstrom()
    for email_name, result in run_hillstrom_roi(hillstrom).items():
        print(f"=== {email_name}: STRATEGIES (hold-out; k chosen inside training data)\n", result["strategies"].to_string(index=False), "\n")
        print(f"=== {email_name}: PROFIT CURVES (hold-out, assumed costs)\n", result["curves"].pivot(index="fraction", columns="model", values="profit_per_1000_customers").to_string(), "\n")
        print(f"=== {email_name}: SENSITIVITY (hold-out, assumed costs)\n", result["sensitivity"].to_string(index=False), "\n")
    print("=== STABILITY ACROSS 10 SEEDS (assumed costs)\n", summarize_roi_stability(roi_stability(hillstrom)).to_string())
