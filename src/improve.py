"""Exploratory improvements: averaging across seeds and choosing the best email per customer.

These came after the main results, so they cannot replace the headline.
Run with: python -m src.improve
"""

import numpy as np
import pandas as pd
from scipy import stats

from src import config, evaluate, learners, load

SPLIT_SEEDS = range(101, 111)
MODEL_SEEDS_PER_SPLIT = 5
INTERVAL_SPLIT = 101

NO_EMAIL, MENS, WOMENS = 0, 1, 2
ARM_CODES = {config.HILLSTROM_CONTROL: NO_EMAIL, config.HILLSTROM_MENS: MENS, config.HILLSTROM_WOMENS: WOMENS}
EMAILS = {"mens": config.HILLSTROM_MENS, "womens": config.HILLSTROM_WOMENS}
LEARNERS = {"response": learners.ResponseModel, "t_learner": learners.TLearner, "x_learner": learners.XLearner}


def model_seeds(split_seed, n=MODEL_SEEDS_PER_SPLIT):
    return [split_seed * 10 + i for i in range(n)]


def seed_score_list(learner, X_train, y_train, t_train, X_score, seeds):
    """One set of scores per model seed, all on the same training rows."""
    return [learner(seed).fit(X_train, y_train, t_train).predict(X_score) for seed in seeds]


def _top_uplift(y, t, score, share):
    return evaluate.uplift_at_k(y, t, score, share)


def compare_seed_averaging(df, split_seeds=SPLIT_SEEDS, n_model_seeds=MODEL_SEEDS_PER_SPLIT):
    """Compare one model seed with the average of several, on the same split and customers."""
    share = config.MEMO_BUDGET / len(df)
    features = learners.hillstrom_features(df)
    rows = []
    for split_seed in split_seeds:
        train, holdout = learners.split_hillstrom(df, seed=split_seed)
        for email_name, email in EMAILS.items():
            train_rows, t_train = learners._email_rows(train, email)
            holdout_rows, t_holdout = learners._email_rows(holdout, email)
            X_train, X_holdout = features.loc[train_rows.index], features.loc[holdout_rows.index]
            y_train, y_holdout = train_rows["conversion"].to_numpy(), holdout_rows["conversion"].to_numpy()
            for name, learner in LEARNERS.items():
                seed_scores = seed_score_list(learner, X_train, y_train, t_train, X_holdout, model_seeds(split_seed, n_model_seeds))
                single, averaged = seed_scores[0], np.mean(seed_scores, axis=0)
                rows.append(
                    {
                        "split": split_seed,
                        "email": email_name,
                        "model": name,
                        "qini_single": evaluate.qini_area(y_holdout, t_holdout, single),
                        "qini_averaged": evaluate.qini_area(y_holdout, t_holdout, averaged),
                        "top_single": 1000 * _top_uplift(y_holdout, t_holdout, single, share),
                        "top_averaged": 1000 * _top_uplift(y_holdout, t_holdout, averaged, share),
                        "seed_agreement": np.corrcoef(seed_scores[0], seed_scores[1])[0, 1],
                    }
                )
    return pd.DataFrame(rows)


def summarize_seed_averaging(table):
    groups = table.groupby(["email", "model"])
    summary = groups[["qini_single", "qini_averaged", "top_single", "top_averaged", "seed_agreement"]].mean()
    summary["qini_averaged_better"] = groups.apply(lambda g: int((g["qini_averaged"] > g["qini_single"]).sum()), include_groups=False)
    summary["top_averaged_better"] = groups.apply(lambda g: int((g["top_averaged"] > g["top_single"]).sum()), include_groups=False)
    return summary


def make_policies(uplift_mens, uplift_womens, share, cost=config.ASSUMED_COST_PER_CONTACT_USD, margin=config.ASSUMED_MARGIN_PER_CONVERSION_USD):
    """Turn predicted uplift into a choice per customer: 0 no email, 1 men's, 2 women's."""
    uplift_mens, uplift_womens = np.asarray(uplift_mens), np.asarray(uplift_womens)
    n = len(uplift_mens)
    best_email = np.where(uplift_mens >= uplift_womens, MENS, WOMENS)
    best_uplift = np.maximum(uplift_mens, uplift_womens)
    size = int(n * share)

    def top_only(score, email):
        policy = np.full(n, NO_EMAIL)
        top = np.argsort(score, kind="mergesort")[::-1][:size]
        policy[top] = email[top] if isinstance(email, np.ndarray) else email
        return policy

    return {
        "no one": np.full(n, NO_EMAIL),
        "men's to everyone": np.full(n, MENS),
        "women's to everyone": np.full(n, WOMENS),
        "best email to everyone": best_email,
        "best email if predicted profitable": np.where(best_uplift > cost / margin, best_email, NO_EMAIL),
        "budget: men's only": top_only(uplift_mens, MENS),
        "budget: best email": top_only(best_uplift, best_email),
    }


def policy_conversion(y, arm, policy):
    """Expected conversion rate if everyone followed the policy, estimated from random arms.

    Customers whose random arm matches the policy's choice stand in for everyone assigned
    that arm. Each arm's rate is weighted by the share of customers the policy assigns to it.
    """
    y, arm, policy = np.asarray(y, dtype=float), np.asarray(arm), np.asarray(policy)
    total = 0.0
    for choice in np.unique(policy):
        matching = (policy == choice) & (arm == choice)
        if not matching.any():
            return float("nan")
        total += (policy == choice).mean() * y[matching].mean()
    return float(total)


def policy_results(y, arm, policy, cost=config.ASSUMED_COST_PER_CONTACT_USD, margin=config.ASSUMED_MARGIN_PER_CONVERSION_USD):
    """Added conversions and profit per 1,000 customers, compared with emailing no one."""
    y, arm = np.asarray(y, dtype=float), np.asarray(arm)
    added = policy_conversion(y, arm, policy) - y[arm == NO_EMAIL].mean()
    contacted = (np.asarray(policy) != NO_EMAIL).mean()
    return {"added_per_1000": 1000 * added, "profit_per_1000": 1000 * (added * margin - contacted * cost), "share_contacted": contacted}


def policy_table(y, arm, policies, benchmarks, n_boot=config.N_BOOTSTRAP, alpha=config.ALPHA, seed=config.SEED):
    """Hold-out policy results with intervals, and a paired gain against each policy's benchmark."""
    y, arm = np.asarray(y, dtype=float), np.asarray(arm)
    rng = np.random.default_rng(seed)
    added = {name: np.empty(n_boot) for name in policies}
    profit = {name: np.empty(n_boot) for name in policies}
    for i in range(n_boot):
        rows = rng.integers(0, len(y), len(y))
        for name, policy in policies.items():
            result = policy_results(y[rows], arm[rows], policy[rows])
            added[name][i], profit[name][i] = result["added_per_1000"], result["profit_per_1000"]

    quantiles = [alpha / 2, 1 - alpha / 2]
    table = []
    for name, policy in policies.items():
        point = policy_results(y, arm, policy)
        benchmark = benchmarks.get(name)
        row = {
            "policy": name,
            **point,
            "added_low": np.nanquantile(added[name], quantiles[0]),
            "added_high": np.nanquantile(added[name], quantiles[1]),
            "profit_low": np.nanquantile(profit[name], quantiles[0]),
            "profit_high": np.nanquantile(profit[name], quantiles[1]),
        }
        if benchmark is not None:
            gap = profit[name] - profit[benchmark]
            row["benchmark"] = benchmark
            row["gain"] = point["profit_per_1000"] - policy_results(y, arm, policies[benchmark])["profit_per_1000"]
            row["gain_low"], row["gain_high"] = np.nanquantile(gap, quantiles)
        table.append(row)
    return pd.DataFrame(table)


BENCHMARKS = {
    "women's to everyone": "men's to everyone",
    "best email to everyone": "men's to everyone",
    "best email if predicted profitable": "men's to everyone",
    "budget: best email": "budget: men's only",
}


def averaged_uplift_scores(train, features, score_index, split_seed):
    """Averaged T-learner uplift for each email, fitted on training rows only."""
    scores = {}
    for email_name, email in EMAILS.items():
        rows, t_train = learners._email_rows(train, email)
        seed_scores = seed_score_list(
            learners.TLearner, features.loc[rows.index], rows["conversion"].to_numpy(), t_train, features.loc[score_index], model_seeds(split_seed)
        )
        scores[email_name] = np.mean(seed_scores, axis=0)
    return scores


def compare_policies(df, split_seeds=SPLIT_SEEDS, interval_split=INTERVAL_SPLIT, n_boot=config.N_BOOTSTRAP):
    """Score each policy on every split, with intervals on the one split chosen in advance."""
    share = config.MEMO_BUDGET / len(df)
    features = learners.hillstrom_features(df)
    all_rows, interval_table = [], None
    for split_seed in split_seeds:
        train, holdout = learners.split_hillstrom(df, seed=split_seed)
        scores = averaged_uplift_scores(train, features, holdout.index, split_seed)
        policies = make_policies(scores["mens"], scores["womens"], share)
        y = holdout["conversion"].to_numpy()
        arm = holdout[config.HILLSTROM_ARM_COL].map(ARM_CODES).to_numpy()
        for name, policy in policies.items():
            all_rows.append({"split": split_seed, "policy": name, **policy_results(y, arm, policy)})
        if split_seed == interval_split:
            interval_table = policy_table(y, arm, policies, BENCHMARKS, n_boot, seed=split_seed)
    return pd.DataFrame(all_rows), interval_table


def summarize_policies(table):
    wide = table.pivot(index="split", columns="policy", values="profit_per_1000")
    rows = []
    for name in wide.columns:
        benchmark = BENCHMARKS.get(name)
        row = {
            "policy": name,
            "mean_share_contacted": table.loc[table["policy"] == name, "share_contacted"].mean(),
            "mean_added_per_1000": table.loc[table["policy"] == name, "added_per_1000"].mean(),
            "mean_profit": wide[name].mean(),
            "min_profit": wide[name].min(),
            "max_profit": wide[name].max(),
        }
        if benchmark is not None:
            gap = wide[name] - wide[benchmark]
            row["benchmark"] = benchmark
            row["mean_gain"] = gap.mean()
            row["min_gain"] = gap.min()
            row["max_gain"] = gap.max()
            row["splits_beating_benchmark"] = int((gap > 0).sum())
        rows.append(row)
    return pd.DataFrame(rows)


def targeting_test_size(email_rate, control_rate, extra_lift, alpha=config.ALPHA, power=config.POWER):
    """Customers per group to test whether targeting adds extra_lift over random selection.

    Measuring lift in each selection needs an emailed and a no-email group, so four groups
    in total. n = (z_alpha + z_power)^2 * sum of p(1 - p) across the four / extra_lift^2
    """
    rates = np.array([email_rate + extra_lift, control_rate, email_rate, control_rate])
    z_total = stats.norm.ppf(1 - alpha / 2) + stats.norm.ppf(power)
    return int(np.ceil(z_total**2 * (rates * (1 - rates)).sum() / extra_lift**2))


if __name__ == "__main__":
    pd.set_option("display.width", 240, "display.max_columns", 40, "display.max_rows", 100)
    pd.set_option("display.float_format", lambda value: f"{value:.5f}" if abs(value) < 0.01 else f"{value:.3f}")
    hillstrom = load.load_hillstrom()
    print("EXPLORATORY. ASSUMED COSTS: $0.10 per contact, $30 margin per conversion\n")
    print("SEED AVERAGING ACROSS 10 NEW SPLITS\n", summarize_seed_averaging(compare_seed_averaging(hillstrom)).to_string(), "\n")
    by_split, intervals = compare_policies(hillstrom)
    print(f"POLICIES ON SPLIT {INTERVAL_SPLIT}, WITH INTERVALS\n", intervals.to_string(index=False), "\n")
    print("POLICIES ACROSS 10 NEW SPLITS\n", summarize_policies(by_split).to_string(index=False))
