"""Experiment tests. Simulations use fixed seeds, and each allowed band is about three
standard errors of the simulated rate, so a failure points to a real problem."""

import numpy as np
import pandas as pd
import pytest
from statsmodels.stats.multitest import multipletests

from src import config, experiment, load

ARM_COL = "arm"
ARM_PAIRS = (("a", "c"), ("b", "c"), ("a", "b"))


def band(rate, sims):
    return 3 * np.sqrt(rate * (1 - rate) / sims)


def spend_sample(rng, n, buyer_share=0.02, multiplier=1.0):
    """Many zeros and a heavy tail, like real spend."""
    buyers = rng.random(n) < buyer_share
    return buyers * rng.lognormal(mean=3, sigma=1, size=n) * multiplier


# Sample-ratio check


def test_sample_ratio_matches_hand_calculation():
    result = experiment.sample_ratio_check([110, 90])
    assert result["chi_square"] == pytest.approx(2.0)
    assert result["p_value"] == pytest.approx(0.1573, abs=1e-4)


def test_sample_ratio_flags_a_broken_split():
    assert experiment.sample_ratio_check([21_000, 21_000, 22_000])["p_value"] < 0.001


def test_sample_ratio_supports_unequal_split():
    assert experiment.sample_ratio_check([850, 150], [0.85, 0.15])["chi_square"] == pytest.approx(0.0)


# Covariate balance


def test_balance_finds_a_planted_imbalance_only():
    rng = np.random.default_rng(1)
    n = 6_000
    df = pd.DataFrame({ARM_COL: np.resize(["a", "b", "c"], n), "fair": rng.normal(size=n)})
    df["shifted"] = rng.normal(size=n) + (df[ARM_COL] == "a") * 0.5
    df["city"] = rng.choice(["x", "y"], n)
    table = experiment.covariate_balance(df, ["fair", "shifted", "city"], ARM_COL, ["a", "b", "c"]).set_index("feature")
    assert table.loc["shifted", "max_abs_smd"] == pytest.approx(0.5, abs=0.1)
    assert table.loc["fair", "max_abs_smd"] < 0.1
    assert {"city_x", "city_y"} <= set(table.index)


# Holm correction


def test_holm_matches_hand_calculation():
    np.testing.assert_allclose(experiment.holm_adjust([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])


def test_holm_matches_statsmodels_and_caps_at_one():
    rng = np.random.default_rng(2)
    for _ in range(100):
        p_values = rng.random(3)
        expected = multipletests(p_values, method="holm")[1]
        np.testing.assert_allclose(experiment.holm_adjust(p_values), expected)


# A/A tests: both groups are identical, so about 5% of tests should be significant


@pytest.mark.parametrize("rate, n", [(0.15, 5_000), (0.009, 21_000)])
def test_aa_proportions_false_positive_rate(rate, n):
    rng = np.random.default_rng(3)
    sims = 3_000
    significant = 0
    for _ in range(sims):
        result = experiment.two_proportion_test(rng.binomial(n, rate), n, rng.binomial(n, rate), n)
        significant += result["p_value"] < config.ALPHA
    assert abs(significant / sims - config.ALPHA) < band(config.ALPHA, sims)


def test_aa_spend_bootstrap_and_welch_false_positive_rate():
    rng = np.random.default_rng(4)
    sims, n = 200, 5_000
    ci_excludes_zero = 0
    p_significant = 0
    for sim in range(sims):
        a, b = spend_sample(rng, n), spend_sample(rng, n)
        result = experiment.spend_test(a, b, n_boot=300, seed=sim)
        ci_excludes_zero += result["ci_low"] > 0 or result["ci_high"] < 0
        p_significant += result["p_value"] < config.ALPHA
    assert abs(ci_excludes_zero / sims - config.ALPHA) < band(config.ALPHA, sims)
    assert abs(p_significant / sims - config.ALPHA) < band(config.ALPHA, sims)


def test_aa_regression_false_positive_rate():
    rng = np.random.default_rng(5)
    sims, n = 500, 2_000
    significant = 0
    for _ in range(sims):
        history = rng.lognormal(mean=5, sigma=1, size=n)
        df = pd.DataFrame(
            {
                ARM_COL: np.resize(["a", "c"], n),
                "history": history,
                "visit": (rng.random(n) < 0.15).astype(int),
            }
        )
        result = experiment.regression_adjusted_effect(df, "visit", "a", "c", ARM_COL)
        significant += result["p_value"] < config.ALPHA
    assert abs(significant / sims - config.ALPHA) < band(config.ALPHA, sims)


def test_aa_holm_keeps_family_wise_error_at_or_below_alpha():
    rng = np.random.default_rng(6)
    sims, n = 2_000, 5_000
    any_raw = 0
    any_holm = 0
    for _ in range(sims):
        successes = rng.binomial(n, 0.15, 3)
        p_values = [
            experiment.two_proportion_test(successes[i], n, successes[j], n)["p_value"]
            for i, j in ((0, 2), (1, 2), (0, 1))
        ]
        any_raw += min(p_values) < config.ALPHA
        any_holm += experiment.holm_adjust(p_values).min() < config.ALPHA
    assert any_holm / sims < config.ALPHA + band(config.ALPHA, sims)
    # Without the correction, the chance of at least one false alarm is clearly above alpha.
    assert any_raw / sims > config.ALPHA + band(config.ALPHA, sims)


def test_aa_on_real_control_arm_when_available():
    """Split the real no-email customers in half at random: no true difference can exist."""
    if not (config.DATA_DIR / "hillstorm_no_indices.csv.gz").exists():
        pytest.skip("Hillstrom file not downloaded")
    control = load.load_hillstrom().query("segment == @config.HILLSTROM_CONTROL")
    conversion = control["conversion"].to_numpy()
    spend = control["spend"].to_numpy(dtype=float)
    half = len(control) // 2
    rng = np.random.default_rng(7)
    sims = 1_000
    conversion_hits = 0
    spend_hits = 0
    for _ in range(sims):
        order = rng.permutation(len(control))
        a, b = order[:half], order[half:]
        conversion_hits += (
            experiment.two_proportion_test(conversion[a].sum(), len(a), conversion[b].sum(), len(b))["p_value"]
            < config.ALPHA
        )
        spend_hits += experiment.stats.ttest_ind(spend[a], spend[b], equal_var=False).pvalue < config.ALPHA
    assert abs(conversion_hits / sims - config.ALPHA) < band(config.ALPHA, sims)
    assert abs(spend_hits / sims - config.ALPHA) < band(config.ALPHA, sims)


# Planted effects of a known size


def test_planted_conversion_effect_is_recovered():
    rng = np.random.default_rng(8)
    n = 20_000
    result = experiment.two_proportion_test(rng.binomial(n, 0.12), n, rng.binomial(n, 0.10), n)
    assert result["ci_low"] < 0.02 < result["ci_high"]
    assert result["ci_low"] > 0
    assert result["relative_low"] < 0.20 < result["relative_high"]


def test_planted_effect_interval_covers_truth_about_95_percent_of_the_time():
    rng = np.random.default_rng(9)
    sims, n = 3_000, 5_000
    covered = 0
    for _ in range(sims):
        result = experiment.two_proportion_test(rng.binomial(n, 0.12), n, rng.binomial(n, 0.10), n)
        covered += result["ci_low"] < 0.02 < result["ci_high"]
    assert abs(covered / sims - 0.95) < band(0.95, sims)


def test_planted_spend_effect_is_recovered():
    rng = np.random.default_rng(10)
    n, buyer_share, multiplier = 20_000, 0.10, 1.2
    lognormal_mean = np.exp(3 + 1**2 / 2)
    true_difference = buyer_share * lognormal_mean * (multiplier - 1)
    a = spend_sample(rng, n, buyer_share, multiplier)
    b = spend_sample(rng, n, buyer_share)
    result = experiment.spend_test(a, b, n_boot=1_000)
    assert result["ci_low"] < true_difference < result["ci_high"]
    assert result["ci_low"] > 0


def test_regression_recovers_planted_effect_and_reduces_noise():
    rng = np.random.default_rng(11)
    n = 10_000
    df = pd.DataFrame({ARM_COL: np.resize(["a", "c"], n), "history": rng.normal(size=n)})
    df["outcome"] = (df[ARM_COL] == "a") * 1.0 + 2 * df["history"] + rng.normal(size=n)
    result = experiment.regression_adjusted_effect(df, "outcome", "a", "c", ARM_COL)
    assert result["ci_low"] < 1.0 < result["ci_high"]
    assert result["se_reduction"] > 0.5


# Comparison table


def test_compare_arms_applies_holm_and_uses_the_right_test_per_metric():
    rng = np.random.default_rng(12)
    n = 3_000
    df = pd.DataFrame(
        {
            ARM_COL: np.resize(["a", "b", "c"], n),
            "visit": (rng.random(n) < 0.2).astype(int),
            "spend": spend_sample(rng, n, 0.1),
        }
    )
    visit = experiment.compare_arms(df, "visit", ARM_PAIRS, ARM_COL)
    spend = experiment.compare_arms(df, "spend", ARM_PAIRS, ARM_COL, n_boot=100)
    np.testing.assert_allclose(visit["p_holm"], experiment.holm_adjust(visit["p_value"]))
    assert visit["relative_low"].notna().all()
    assert spend["relative_low"].isna().all()
    assert len(visit) == len(spend) == 3


def test_bootstrap_is_repeatable_with_a_seed():
    rng = np.random.default_rng(13)
    a, b = spend_sample(rng, 1_000), spend_sample(rng, 1_000)
    first = experiment.bootstrap_mean_difference(a, b, n_boot=100, seed=1)
    second = experiment.bootstrap_mean_difference(a, b, n_boot=100, seed=1)
    assert first == second


# Plain-English reading


def test_reading_describes_a_significant_increase():
    text = experiment.plain_english_reading(experiment.two_proportion_test(600, 5_000, 500, 5_000))
    assert "12.00%" in text and "10.00%" in text
    assert "2.00 percentage points higher" in text
    assert "statistically significant (p" in text
    assert "not statistically significant" not in text


def test_reading_does_not_claim_no_effect_when_not_significant():
    text = experiment.plain_english_reading(experiment.two_proportion_test(49, 1_000, 50, 1_000))
    assert "lower" in text
    assert "not statistically significant" in text
    assert "does not prove" in text


def test_reading_uses_the_chosen_confidence_level():
    result = experiment.two_proportion_test(600, 5_000, 500, 5_000, alpha=0.10)
    assert "90% confidence interval" in experiment.plain_english_reading(result, alpha=0.10)


# Assignment check


def test_assignment_check_finds_random_assignment_looks_random():
    rng = np.random.default_rng(16)
    n = 30_000
    features = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n)})
    treatment = (rng.random(n) < 0.75).astype(int)
    outcome = (rng.random(n) < 0.10 + 0.02 * treatment).astype(int)
    result = experiment.assignment_check(features, treatment, outcome)
    assert result["auc"] == pytest.approx(0.5, abs=0.03)
    assert result["adjusted_low"] < 0.02 < result["adjusted_high"]


def test_assignment_check_detects_and_reduces_confounding():
    """Treatment goes to likely buyers, so the raw lift is biased upward. The true effect is 0.02."""
    rng = np.random.default_rng(17)
    n = 60_000
    confounder = rng.normal(size=n)
    features = pd.DataFrame({"a": confounder, "b": rng.normal(size=n)})
    treatment = (rng.random(n) < 1 / (1 + np.exp(-1.5 * confounder))).astype(int)
    baseline = np.clip(0.10 + 0.08 * confounder, 0.01, 0.9)
    outcome = (rng.random(n) < baseline + 0.02 * treatment).astype(int)
    result = experiment.assignment_check(features, treatment, outcome)
    assert result["auc"] > 0.7
    assert result["raw_low"] > 0.02
    assert abs(result["adjusted_lift"] - 0.02) < abs(result["raw_lift"] - 0.02)


# Power and minimum detectable effect


def test_required_n_matches_hand_calculation():
    # (1.95996 + 0.84162)^2 * (0.10 * 0.90 + 0.12 * 0.88) / 0.02^2 = 3838.1, rounded up
    assert experiment.required_n_per_arm(0.10, 0.02, alpha=0.05, power=0.80) == 3_839


def test_mde_and_required_n_are_the_same_formula_reversed():
    mde = experiment.mde_proportion(0.10, 3_839, alpha=0.05, power=0.80)
    assert mde == pytest.approx(0.02, abs=1e-4)


def test_mde_mean_matches_hand_calculation():
    assert experiment.mde_mean(10, 100, alpha=0.05, power=0.80) == pytest.approx(3.9616, abs=1e-3)


def test_formula_power_matches_simulated_power():
    rng = np.random.default_rng(14)
    n = experiment.required_n_per_arm(0.10, 0.02)
    sims = 3_000
    significant = 0
    for _ in range(sims):
        result = experiment.two_proportion_test(rng.binomial(n, 0.12), n, rng.binomial(n, 0.10), n)
        significant += result["p_value"] < config.ALPHA
    assert experiment.achieved_power(0.10, 0.02, n) == pytest.approx(0.80, abs=0.001)
    assert abs(significant / sims - 0.80) < band(0.80, sims)


def test_power_table_is_stricter_at_holm_alpha():
    rng = np.random.default_rng(15)
    n = 3_000
    df = pd.DataFrame(
        {
            ARM_COL: np.resize(["a", "b", "c"], n),
            "conversion": (rng.random(n) < 0.05).astype(int),
            "visit": (rng.random(n) < 0.2).astype(int),
            "spend": spend_sample(rng, n, 0.1),
        }
    )
    table = experiment.power_table(df, ARM_PAIRS, ARM_COL)
    assert len(table) == 9
    assert (table["mde_strictest_holm"] > table["mde"]).all()
