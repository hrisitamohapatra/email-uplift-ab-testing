"""Improvement tests: hand-worked policy values, known effects, and no hold-out use."""

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

from src import config, improve, learners, simulate

NO_EMAIL, MENS, WOMENS = improve.NO_EMAIL, improve.MENS, improve.WOMENS


def three_arm_experiment(n=30_000, seed=1):
    """Men's email helps only when x > 0, women's only when x < 0, and no email is a flat 10%."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    arm = rng.integers(0, 3, n)
    rate = 0.10 + 0.05 * ((arm == MENS) & (x > 0)) + 0.05 * ((arm == WOMENS) & (x < 0))
    return x, arm, (rng.random(n) < rate).astype(int)


def test_policy_conversion_matches_hand_calculation():
    y = [1, 0, 1, 1, 0, 0]
    arm = [MENS, MENS, WOMENS, WOMENS, NO_EMAIL, NO_EMAIL]
    policy = [MENS, MENS, MENS, WOMENS, WOMENS, NO_EMAIL]
    # Men's: 3 of 6 assigned, matching rate 1/2. Women's: 2 of 6, rate 1. None: 1 of 6, rate 0.
    assert improve.policy_conversion(y, arm, policy) == pytest.approx(0.5 * 0.5 + (2 / 6) * 1 + 0)


def test_policy_conversion_for_one_arm_is_that_arms_rate():
    y = [1, 0, 1, 0]
    arm = [MENS, MENS, NO_EMAIL, NO_EMAIL]
    assert improve.policy_conversion(y, arm, [MENS] * 4) == pytest.approx(0.5)


def test_policy_conversion_is_nan_when_no_customers_match():
    assert np.isnan(improve.policy_conversion([1, 0], [NO_EMAIL, NO_EMAIL], [MENS, MENS]))


def test_policy_profit_matches_hand_calculation():
    y = [1, 0, 0, 0]
    arm = [MENS, MENS, NO_EMAIL, NO_EMAIL]
    result = improve.policy_results(y, arm, [MENS] * 4, cost=0.10, margin=30)
    # Lift 0.5 with everyone contacted: 1000 * (0.5 * 30 - 0.10) = 14,900
    assert result["added_per_1000"] == pytest.approx(500)
    assert result["profit_per_1000"] == pytest.approx(14_900)


def test_no_one_policy_has_zero_profit():
    y = [1, 0, 1, 0]
    arm = [NO_EMAIL, NO_EMAIL, MENS, MENS]
    result = improve.policy_results(y, arm, [NO_EMAIL] * 4)
    assert result["profit_per_1000"] == 0
    assert result["share_contacted"] == 0


def test_make_policies_chooses_correctly():
    policies = improve.make_policies([0.02, 0.0, 0.001, 0.05], [0.01, 0.03, 0.001, 0.0], share=0.5, cost=0.10, margin=30)
    np.testing.assert_array_equal(policies["best email to everyone"], [MENS, WOMENS, MENS, MENS])
    # Break-even is 0.1 / 30 = 0.0033, so the third customer gets nothing.
    np.testing.assert_array_equal(policies["best email if predicted profitable"], [MENS, WOMENS, NO_EMAIL, MENS])
    np.testing.assert_array_equal(policies["budget: men's only"], [MENS, NO_EMAIL, NO_EMAIL, MENS])
    np.testing.assert_array_equal(policies["budget: best email"], [NO_EMAIL, WOMENS, NO_EMAIL, MENS])


def test_true_best_email_policy_beats_either_single_email():
    x, arm, y = three_arm_experiment()
    best = np.where(x > 0, MENS, WOMENS)
    value_best = improve.policy_conversion(y, arm, best)
    assert value_best == pytest.approx(0.15, abs=0.01)
    assert value_best > improve.policy_conversion(y, arm, np.full(len(y), MENS)) + 0.015
    assert value_best > improve.policy_conversion(y, arm, np.full(len(y), WOMENS)) + 0.015


def test_policy_table_reports_a_paired_gain_against_the_benchmark():
    x, arm, y = three_arm_experiment(n=9_000)
    policies = {"men's to everyone": np.full(len(y), MENS), "best email to everyone": np.where(x > 0, MENS, WOMENS)}
    table = improve.policy_table(y, arm, policies, improve.BENCHMARKS, n_boot=100).set_index("policy")
    row = table.loc["best email to everyone"]
    assert row["benchmark"] == "men's to everyone"
    assert row["gain_low"] > 0
    assert row["gain_low"] < row["gain"] < row["gain_high"]


def test_identical_policies_have_exactly_zero_gain():
    x, arm, y = three_arm_experiment(n=3_000)
    same = np.full(len(y), MENS)
    table = improve.policy_table(y, arm, {"men's to everyone": same, "best email to everyone": same}, improve.BENCHMARKS, n_boot=50).set_index("policy")
    assert table.loc["best email to everyone", "gain"] == 0
    assert table.loc["best email to everyone", "gain_low"] == 0
    assert table.loc["best email to everyone", "gain_high"] == 0


def test_averaging_seeds_makes_scores_more_stable():
    df = simulate.simulate_uplift(8_000, seed=2)
    train, holdout = train_test_split(df, test_size=config.TEST_SIZE, random_state=2)
    columns = list(simulate.FEATURES)

    def scores(seeds):
        return improve.seed_score_list(learners.TLearner, train[columns], train["outcome"], train["treatment"], holdout[columns], seeds)

    single_a, single_b = scores([1])[0], scores([2])[0]
    averaged_a = np.mean(scores([11, 12, 13, 14, 15]), axis=0)
    averaged_b = np.mean(scores([21, 22, 23, 24, 25]), axis=0)
    assert np.corrcoef(averaged_a, averaged_b)[0, 1] > np.corrcoef(single_a, single_b)[0, 1]


def test_targeting_test_size_matches_hand_calculation():
    # (1.95996 + 0.84162)^2 * (0.15 * 0.85 + 0.05 * 0.95 + 0.10 * 0.90 + 0.05 * 0.95) / 0.05^2 = 981.1, rounded up
    assert improve.targeting_test_size(0.10, 0.05, 0.05) == 982


def test_targeting_test_size_grows_as_the_extra_lift_shrinks():
    assert improve.targeting_test_size(0.01, 0.005, 0.002) > improve.targeting_test_size(0.01, 0.005, 0.005)


def make_small_hillstrom(n=6_000, seed=3):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "recency": rng.integers(1, 13, n),
            "history_segment": "x",
            "history": rng.lognormal(5, 1, n),
            "mens": rng.integers(0, 2, n),
            "womens": rng.integers(0, 2, n),
            "zip_code": rng.choice(["Urban", "Rural"], n),
            "newbie": rng.integers(0, 2, n),
            "channel": rng.choice(["Web", "Phone"], n),
            config.HILLSTROM_ARM_COL: np.resize(config.HILLSTROM_ARMS, n),
            "visit": (rng.random(n) < 0.3).astype(int),
            "conversion": (rng.random(n) < 0.1).astype(int),
            "spend": 0.0,
        }
    )


def test_compare_seed_averaging_covers_each_email_and_model():
    table = improve.compare_seed_averaging(make_small_hillstrom(), split_seeds=[101], n_model_seeds=2)
    assert len(table) == 2 * 3
    assert len(improve.summarize_seed_averaging(table)) == 2 * 3


def test_hold_out_outcomes_cannot_affect_predicted_uplift():
    df = make_small_hillstrom()
    train, holdout = learners.split_hillstrom(df, seed=101)
    features = learners.hillstrom_features(df)
    original = improve.averaged_uplift_scores(train, features, holdout.index, 101)
    changed = df.copy()
    changed.loc[holdout.index, "conversion"] = 1 - changed.loc[holdout.index, "conversion"]
    altered = improve.averaged_uplift_scores(changed.loc[train.index], features, holdout.index, 101)
    for email in improve.EMAILS:
        np.testing.assert_array_equal(original[email], altered[email])


def test_compare_policies_runs_and_marks_benchmarks():
    by_split, intervals = improve.compare_policies(make_small_hillstrom(), split_seeds=[101], n_boot=10)
    assert set(by_split["policy"]) == set(intervals["policy"])
    summary = improve.summarize_policies(by_split).set_index("policy")
    assert summary.loc["budget: best email", "benchmark"] == "budget: men's only"
