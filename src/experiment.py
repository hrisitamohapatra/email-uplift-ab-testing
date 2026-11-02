"""Experiment analysis for Hillstrom: checks, lift, multiple testing, adjustment and power.

Run the full analysis with: python -m src.experiment
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import optimize, stats

from src import config, load


def sample_ratio_check(counts, expected_shares=None):
    """Chi-square test: do the arm sizes match the split the experiment intended?"""
    counts = np.asarray(counts, dtype=float)
    if expected_shares is None:
        expected_shares = np.full(len(counts), 1 / len(counts))
    expected = counts.sum() * np.asarray(expected_shares)
    chi_square, p_value = stats.chisquare(counts, expected)
    return {"chi_square": float(chi_square), "p_value": float(p_value)}


def covariate_balance(df, features, arm_col, arms):
    """Mean of each feature per arm, plus the largest standardized difference between arms."""
    categorical = df[list(features)].select_dtypes(exclude="number").columns.tolist()
    encoded = pd.get_dummies(df[list(features)], columns=categorical, dtype=float)
    groups = {arm: encoded[df[arm_col] == arm] for arm in arms}
    rows = []
    for column in encoded.columns:
        row = {"feature": column}
        for arm in arms:
            row[arm] = groups[arm][column].mean()
        differences = []
        for i, arm_a in enumerate(arms):
            for arm_b in arms[i + 1:]:
                a, b = groups[arm_a][column], groups[arm_b][column]
                pooled_sd = np.sqrt((a.var() + b.var()) / 2)
                differences.append(abs(a.mean() - b.mean()) / pooled_sd if pooled_sd > 0 else 0.0)
        row["max_abs_smd"] = max(differences)
        rows.append(row)
    return pd.DataFrame(rows)


def two_proportion_test(successes_a, n_a, successes_b, n_b, alpha=config.ALPHA):
    """Difference in rates (a minus b), with a confidence interval and a two-sided z-test."""
    rate_a, rate_b = successes_a / n_a, successes_b / n_b
    difference = rate_a - rate_b
    z_critical = stats.norm.ppf(1 - alpha / 2)

    # The interval does not assume the rates are equal.
    se = np.sqrt(rate_a * (1 - rate_a) / n_a + rate_b * (1 - rate_b) / n_b)

    # The test does assume they are equal: that is the null hypothesis.
    pooled = (successes_a + successes_b) / (n_a + n_b)
    se_null = np.sqrt(pooled * (1 - pooled) * (1 / n_a + 1 / n_b))
    p_value = 2 * stats.norm.sf(abs(difference) / se_null) if se_null > 0 else 1.0

    # Relative lift uses the log of the ratio, whose sampling error is closer to normal.
    if successes_a > 0 and successes_b > 0:
        ratio = rate_a / rate_b
        se_log = np.sqrt(1 / successes_a - 1 / n_a + 1 / successes_b - 1 / n_b)
        relative = ratio - 1
        relative_low = ratio * np.exp(-z_critical * se_log) - 1
        relative_high = ratio * np.exp(z_critical * se_log) - 1
    else:
        relative = relative_low = relative_high = np.nan

    return {
        "mean_a": rate_a,
        "mean_b": rate_b,
        "difference": difference,
        "ci_low": difference - z_critical * se,
        "ci_high": difference + z_critical * se,
        "relative_lift": relative,
        "relative_low": relative_low,
        "relative_high": relative_high,
        "p_value": float(p_value),
    }


def bootstrap_mean_difference(a, b, n_boot=config.N_BOOTSTRAP, alpha=config.ALPHA, seed=config.SEED):
    """Percentile bootstrap interval for mean(a) - mean(b)."""
    rng = np.random.default_rng(seed)
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    differences = np.empty(n_boot)
    for i in range(n_boot):
        resample_a = a[rng.integers(0, len(a), len(a))]
        resample_b = b[rng.integers(0, len(b), len(b))]
        differences[i] = resample_a.mean() - resample_b.mean()
    low, high = np.quantile(differences, [alpha / 2, 1 - alpha / 2])
    return float(low), float(high)


def spend_test(a, b, n_boot=config.N_BOOTSTRAP, alpha=config.ALPHA, seed=config.SEED):
    """Difference in mean spend: bootstrap interval, Welch t-test p-value."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    low, high = bootstrap_mean_difference(a, b, n_boot, alpha, seed)
    mean_a, mean_b = a.mean(), b.mean()
    p_value = stats.ttest_ind(a, b, equal_var=False).pvalue
    return {
        "mean_a": mean_a,
        "mean_b": mean_b,
        "difference": mean_a - mean_b,
        "ci_low": low,
        "ci_high": high,
        "relative_lift": mean_a / mean_b - 1 if mean_b > 0 else np.nan,
        "relative_low": np.nan,
        "relative_high": np.nan,
        "p_value": float(p_value),
    }


def holm_adjust(p_values):
    """Holm correction: smallest p times m, next times m-1, never decreasing as p grows."""
    p_values = np.asarray(p_values, dtype=float)
    m = len(p_values)
    order = np.argsort(p_values)
    scaled = p_values[order] * (m - np.arange(m))
    adjusted_sorted = np.minimum(np.maximum.accumulate(scaled), 1.0)
    adjusted = np.empty(m)
    adjusted[order] = adjusted_sorted
    return adjusted


def compare_arms(df, metric, comparisons, arm_col, alpha=config.ALPHA, n_boot=config.N_BOOTSTRAP, seed=config.SEED):
    """Run every comparison for one metric, with Holm correction across them."""
    rows = []
    for index, (arm_a, arm_b) in enumerate(comparisons):
        a = df.loc[df[arm_col] == arm_a, metric]
        b = df.loc[df[arm_col] == arm_b, metric]
        if metric == "spend":
            result = spend_test(a, b, n_boot, alpha, seed + index)
        else:
            result = two_proportion_test(a.sum(), len(a), b.sum(), len(b), alpha)
        rows.append({"metric": metric, "comparison": f"{arm_a} vs {arm_b}", **result})
    table = pd.DataFrame(rows)
    table["p_holm"] = holm_adjust(table["p_value"])
    table["significant_holm"] = table["p_holm"] < alpha
    return table


def regression_adjusted_effect(df, metric, arm_a, arm_b, arm_col, covariate="history", alpha=config.ALPHA):
    """Effect of arm_a vs arm_b from a regression controlling for a pre-treatment covariate.

    The same model without the covariate gives the unadjusted standard error, so the
    two can be compared. HC2 standard errors stay valid when the outcome is binary or skewed.
    """
    subset = df[df[arm_col].isin([arm_a, arm_b])]
    outcome = subset[metric].to_numpy(dtype=float)
    treated = (subset[arm_col] == arm_a).to_numpy(dtype=float)
    centered = subset[covariate].to_numpy(dtype=float) - subset[covariate].mean()

    unadjusted = sm.OLS(outcome, sm.add_constant(treated)).fit(cov_type="HC2")
    adjusted = sm.OLS(outcome, sm.add_constant(np.column_stack([treated, centered]))).fit(cov_type="HC2")
    low, high = adjusted.conf_int(alpha)[1]
    return {
        "metric": metric,
        "comparison": f"{arm_a} vs {arm_b}",
        "unadjusted_effect": unadjusted.params[1],
        "unadjusted_se": unadjusted.bse[1],
        "adjusted_effect": adjusted.params[1],
        "adjusted_se": adjusted.bse[1],
        "ci_low": low,
        "ci_high": high,
        "p_value": adjusted.pvalues[1],
        "se_reduction": 1 - adjusted.bse[1] / unadjusted.bse[1],
    }


def _z_total(alpha, power):
    """The two-sided significance z-value plus the power z-value."""
    return stats.norm.ppf(1 - alpha / 2) + stats.norm.ppf(power)


def required_n_per_arm(baseline, delta, alpha=config.ALPHA, power=config.POWER):
    """Sample size per arm to detect baseline vs baseline + delta.

    n = (z_alpha + z_power)^2 * [p1(1 - p1) + p2(1 - p2)] / delta^2
    """
    treated = baseline + delta
    variance = baseline * (1 - baseline) + treated * (1 - treated)
    return int(np.ceil(_z_total(alpha, power) ** 2 * variance / delta**2))


def mde_proportion(baseline, n_per_arm, alpha=config.ALPHA, power=config.POWER):
    """Smallest absolute lift detectable with n_per_arm: the sample-size formula solved for delta."""

    def gap(delta):
        treated = baseline + delta
        variance = baseline * (1 - baseline) + treated * (1 - treated)
        return _z_total(alpha, power) ** 2 * variance / delta**2 - n_per_arm

    return float(optimize.brentq(gap, 1e-9, 1 - baseline - 1e-9))


def mde_mean(sd, n_per_arm, alpha=config.ALPHA, power=config.POWER):
    """Smallest detectable difference in a mean: (z_alpha + z_power) * sd * sqrt(2 / n)."""
    return float(_z_total(alpha, power) * sd * np.sqrt(2 / n_per_arm))


def achieved_power(baseline, delta, n_per_arm, alpha=config.ALPHA):
    """Chance of a significant result if the true lift is delta."""
    treated = baseline + delta
    se = np.sqrt((baseline * (1 - baseline) + treated * (1 - treated)) / n_per_arm)
    return float(stats.norm.cdf(abs(delta) / se - stats.norm.ppf(1 - alpha / 2)))


def power_table(df, comparisons, arm_col, alpha=config.ALPHA, power=config.POWER):
    """Minimum detectable effect per comparison, at alpha and at Holm's strictest step (alpha / m)."""
    rows = []
    strictest_alpha = alpha / len(comparisons)
    for arm_a, arm_b in comparisons:
        a, b = df[df[arm_col] == arm_a], df[df[arm_col] == arm_b]
        n = min(len(a), len(b))
        for metric in ("conversion", "visit", "spend"):
            baseline = b[metric].mean()
            if metric == "spend":
                mde = mde_mean(b[metric].std(), n, alpha, power)
                mde_strict = mde_mean(b[metric].std(), n, strictest_alpha, power)
            else:
                mde = mde_proportion(baseline, n, alpha, power)
                mde_strict = mde_proportion(baseline, n, strictest_alpha, power)
            rows.append(
                {
                    "metric": metric,
                    "comparison": f"{arm_a} vs {arm_b}",
                    "baseline": baseline,
                    "n_per_arm": n,
                    "mde": mde,
                    "mde_strictest_holm": mde_strict,
                    "observed_difference": a[metric].mean() - baseline,
                }
            )
    return pd.DataFrame(rows)


def run_hillstrom(df, seed=config.SEED):
    """Run the whole Stage 2 analysis and return each table."""
    arm_col = config.HILLSTROM_ARM_COL
    comparisons = config.HILLSTROM_COMPARISONS
    counts = df[arm_col].value_counts().reindex(config.HILLSTROM_ARMS)
    metrics = ("conversion", "visit", "spend")
    return {
        "arm_counts": counts,
        "sample_ratio": sample_ratio_check(counts.to_numpy()),
        "balance": covariate_balance(df, config.HILLSTROM_FEATURES, arm_col, config.HILLSTROM_ARMS),
        "lift": pd.concat(
            [compare_arms(df, metric, comparisons, arm_col, seed=seed) for metric in metrics],
            ignore_index=True,
        ),
        "regression": pd.DataFrame(
            [
                regression_adjusted_effect(df, metric, arm_a, arm_b, arm_col)
                for metric in metrics
                for arm_a, arm_b in comparisons
            ]
        ),
        "power": power_table(df, comparisons, arm_col),
    }


if __name__ == "__main__":
    pd.set_option("display.width", 220, "display.max_columns", 30, "display.max_rows", 100)
    pd.set_option("display.float_format", lambda value: f"{value:.5f}")
    results = run_hillstrom(load.load_hillstrom())
    print("ARM COUNTS\n", results["arm_counts"].to_string(), "\n")
    print("SAMPLE RATIO CHECK\n", results["sample_ratio"], "\n")
    print("COVARIATE BALANCE\n", results["balance"].to_string(index=False), "\n")
    print("LIFT\n", results["lift"].to_string(index=False), "\n")
    print("REGRESSION ADJUSTMENT\n", results["regression"].to_string(index=False), "\n")
    print("POWER\n", results["power"].to_string(index=False))
