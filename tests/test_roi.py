"""ROI tests: hand-worked profit, correct choice of k, and no use of hold-out outcomes to choose it."""

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

from src import config, evaluate, learners, roi, simulate

COLUMNS = list(simulate.FEATURES)


def split_simulation(n=6_000, seed=1):
    df = simulate.simulate_uplift(n, seed)
    train, holdout = train_test_split(df, test_size=config.TEST_SIZE, random_state=seed)
    return (
        train[COLUMNS],
        train["outcome"].to_numpy(),
        train["prob_with"].gt(0.5).astype(int).to_numpy(),
        train["treatment"].to_numpy(),
        holdout[COLUMNS],
        holdout["outcome"].to_numpy(),
        holdout["treatment"].to_numpy(),
    )


def test_profit_matches_hand_calculation():
    # 1000 * 0.5 * (0.01 * 30 - 0.10) = 100
    assert roi.profit_per_1000_customers(0.01, 0.5, cost=0.10, margin=30) == pytest.approx(100)


def test_profit_is_zero_at_break_even():
    assert roi.profit_per_1000_customers(0.10 / 30, 1.0, cost=0.10, margin=30) == pytest.approx(0)


def test_uplift_by_fraction_matches_evaluate_module():
    rng = np.random.default_rng(1)
    y, t, score = rng.integers(0, 2, 2_000), rng.integers(0, 2, 2_000), rng.normal(size=2_000)
    curve = roi.uplift_by_fraction(y, t, score)
    for fraction, value in zip(roi.FRACTIONS, curve):
        assert value == pytest.approx(evaluate.uplift_at_k(y, t, score, fraction))


def test_uplift_by_fraction_is_nan_when_top_group_has_one_arm():
    curve = roi.uplift_by_fraction([1, 0, 1, 0], [1, 1, 0, 0], [4, 3, 2, 1], [0.5, 1.0])
    assert np.isnan(curve[0])
    assert curve[1] == pytest.approx(0)


def test_choose_fraction_picks_the_most_profitable_share():
    fractions = np.array([0.5, 1.0])
    # 1000 * 0.5 * (0.02 * 30 - 0.1) = 250 beats 1000 * 1 * (0.005 * 30 - 0.1) = 50
    assert roi.choose_fraction(np.array([0.02, 0.005]), fractions, cost=0.10, margin=30) == 0.5


def test_choose_fraction_emails_no_one_when_nothing_is_profitable():
    assert roi.choose_fraction(np.array([0.001, 0.001]), np.array([0.5, 1.0]), cost=0.10, margin=30) == 0.0


def test_choose_fraction_ignores_shares_with_no_estimate():
    assert roi.choose_fraction(np.array([np.nan, 0.01]), np.array([0.5, 1.0]), cost=0.10, margin=30) == 1.0
    assert roi.choose_fraction(np.array([np.nan, np.nan]), np.array([0.5, 1.0]), cost=0.10, margin=30) == 0.0


def test_true_effect_ranking_targets_the_right_people_when_contact_is_costly():
    """Known truth: profitable customers are exactly those with effect * margin > cost."""
    df = simulate.simulate_uplift(200_000, seed=2)
    cost, margin = 3.0, 30.0
    curve = roi.uplift_by_fraction(df["outcome"], df["treatment"], df["true_effect"])
    chosen = roi.choose_fraction(curve, roi.FRACTIONS, cost, margin)
    truly_profitable = (df["true_effect"] * margin > cost).mean()
    assert chosen == pytest.approx(truly_profitable, abs=0.1)
    best = roi.profit_per_1000_customers(curve, roi.FRACTIONS, cost, margin)[roi.FRACTIONS == chosen][0]
    assert best > roi.profit_per_1000_customers(curve[-1], 1.0, cost, margin)


def test_strategy_table_has_all_strategies_and_correct_zero_row():
    y, t = np.array([1, 0] * 500), np.array([1, 1, 0, 0] * 250)
    scores = {"m": np.arange(1_000, dtype=float)}
    table = roi.strategy_table(y, t, scores, {"m": 0.5}, n_boot=50).set_index("strategy")
    assert set(table.index) == {"email no one", "email everyone", "top k by m"}
    assert table.loc["email no one", "profit_per_1000_customers"] == 0
    assert table.loc["email everyone", "fraction"] == 1


def test_choosing_everyone_gives_exactly_zero_gain():
    rng = np.random.default_rng(3)
    y, t, score = rng.integers(0, 2, 1_000), rng.integers(0, 2, 1_000), rng.normal(size=1_000)
    row = roi.strategy_table(y, t, {"m": score}, {"m": 1.0}, n_boot=100).iloc[-1]
    assert row["gain_vs_everyone"] == pytest.approx(0)
    assert row["gain_low"] == pytest.approx(0)
    assert row["gain_high"] == pytest.approx(0)


def test_choosing_no_one_gives_zero_profit_with_no_interval_width():
    rng = np.random.default_rng(4)
    y, t, score = rng.integers(0, 2, 1_000), rng.integers(0, 2, 1_000), rng.normal(size=1_000)
    row = roi.strategy_table(y, t, {"m": score}, {"m": 0.0}, n_boot=100).iloc[-1]
    assert row["profit_per_1000_customers"] == 0
    assert row["profit_low"] == row["profit_high"] == 0


def test_hold_out_outcomes_cannot_affect_the_chosen_fraction():
    X_train, y_train, y2_train, t_train, X_holdout, y_holdout, t_holdout = split_simulation()
    first = roi.run_experiment(X_train, y_train, y2_train, t_train, X_holdout, y_holdout, t_holdout, n_boot=10)
    second = roi.run_experiment(X_train, y_train, y2_train, t_train, X_holdout, 1 - y_holdout, t_holdout, n_boot=10)
    assert first["chosen"] == second["chosen"]
    for name in first["scores"]:
        np.testing.assert_array_equal(first["scores"][name], second["scores"][name])
    assert not first["strategies"]["profit_per_1000_customers"].equals(second["strategies"]["profit_per_1000_customers"])


def test_out_of_fold_scores_fill_every_row_and_differ_from_in_sample():
    X_train, y_train, y2_train, t_train, *_ = split_simulation()
    oof = roi.out_of_fold_scores(X_train, y_train, y2_train, t_train)
    in_sample = roi.fit_all_models(X_train, y_train, y2_train, t_train, X_train)
    for name in roi.MODELS:
        assert np.isfinite(oof[name]).all()
        assert not np.allclose(oof[name], in_sample[name])


def make_lenta_like(n=4_000, seed=1):
    df = simulate.simulate_uplift(n, seed, treatment_share=0.75)
    table = df[COLUMNS].assign(
        **{config.LENTA_OUTCOME: df["outcome"], config.LENTA_TREATMENT_COL: df["treatment"], "age": 40.0, "gender": "x"},
        response_sms=0.5,
        response_viber=0.5,
    )
    return table


def test_lenta_features_leave_out_outcome_treatment_and_excluded_columns():
    columns = learners.lenta_features(make_lenta_like()).columns
    assert list(columns) == COLUMNS


def test_learners_refuse_excluded_lenta_columns():
    df = make_lenta_like()
    X = df[COLUMNS + ["response_sms"]]
    with pytest.raises(ValueError, match="cannot be model inputs"):
        learners.TLearner().fit(X, df[config.LENTA_OUTCOME], df[config.LENTA_TREATMENT_COL])


def test_run_lenta_roi_uses_primary_models_only():
    result = roi.run_lenta_roi(make_lenta_like(), n_boot=10)
    assert set(result["chosen"]) == {"response", "t_learner", "x_learner"}
    assert set(result["metrics"]["model"]) == {"response", "t_learner", "x_learner"}


def test_lenta_stability_holds_the_chosen_share_fixed():
    chosen = {"response": 0.5, "t_learner": 0.5, "x_learner": 1.0}
    table = roi.lenta_stability(make_lenta_like(), chosen, seeds=[1, 2])
    assert len(table) == 2 * 3
    assert (table.loc[table["model"] == "x_learner", "gain_vs_everyone"].abs() < 1e-9).all()


def test_fixed_budget_finds_more_lift_at_the_top_of_a_true_ranking():
    df = simulate.simulate_uplift(40_000, seed=3)
    table = roi.fixed_budget_table(df["outcome"], df["treatment"], {"true": df["true_effect"]}, 0.15, n_boot=100).set_index("group")
    assert table.loc["top true", "added_per_1000_contacts"] > table.loc["random", "added_per_1000_contacts"]
    assert table.loc["bottom true", "added_per_1000_contacts"] < table.loc["random", "added_per_1000_contacts"]
    assert table.loc["top true", "gain_low"] > 0
    assert table.loc["random", "gain_low"] == table.loc["random", "gain_high"] == 0


def test_fixed_budget_profit_matches_hand_calculation():
    # Every treated customer converts and no control customer does, so uplift is 1.
    y, t = np.array([1, 0] * 100), np.array([1, 0] * 100)
    table = roi.fixed_budget_table(y, t, {"m": np.arange(200.0)}, 0.5, n_boot=10, budget=10_000).set_index("group")
    # 10,000 * (1 * 30 - 0.10) = 299,000
    assert table.loc["top m", "profit_for_budget"] == pytest.approx(299_000)


def test_same_ranking_gives_zero_gap_between_models():
    rng = np.random.default_rng(5)
    y, t, score = rng.integers(0, 2, 1_000), rng.integers(0, 2, 1_000), rng.normal(size=1_000)
    result = roi.t_learner_vs_response(y, t, {"t_learner": score, "response": score}, 0.2, n_boot=50)
    assert result == {"difference": 0, "low": 0, "high": 0}


def test_group_profile_separates_top_and_bottom():
    features = pd.DataFrame({"a": np.arange(100.0)})
    profile = roi.group_profile(features, features["a"], 0.1)
    assert profile.loc["a", "top_group"] == pytest.approx(94.5)
    assert profile.loc["a", "bottom_group"] == pytest.approx(4.5)
    assert profile.loc["a", "all_customers"] == pytest.approx(49.5)


def test_full_sample_profit_uses_the_assumed_formula():
    arms = [config.HILLSTROM_MENS] * 1_000 + [config.HILLSTROM_WOMENS] * 1_000 + [config.HILLSTROM_CONTROL] * 1_000
    conversion = [1] * 20 + [0] * 980 + [1] * 10 + [0] * 990 + [1] * 5 + [0] * 995
    table = roi.full_sample_profit(pd.DataFrame({config.HILLSTROM_ARM_COL: arms, "conversion": conversion}), cost=0.10, margin=30)
    # Lift of 0.015: 1000 * (0.015 * 30 - 0.10) = 350
    assert table.iloc[0]["profit"] == pytest.approx(350)
    assert table.iloc[0]["profit_low"] < 350 < table.iloc[0]["profit_high"]


def test_visit_trained_models_are_labelled_exploratory():
    for name, (_, outcome) in roi.MODELS.items():
        assert (outcome == "secondary") == ("EXPLORATORY" in name)


def test_sensitivity_covers_the_grid_and_moves_with_costs():
    curve = np.full(len(roi.FRACTIONS), 0.01)
    table = roi.sensitivity_table({"m": curve}, {"m": curve}, 0.01)
    assert len(table) == len(config.ROI_COST_GRID_USD) * len(config.ROI_MARGIN_GRID_USD)
    row = table[(table["assumed_cost"] == 0.5) & (table["assumed_margin"] == 15)].iloc[0]
    # 0.01 * 15 = 0.15 is below the 0.50 cost: lose money by emailing all, so choose no one.
    assert row["everyone_profit"] == pytest.approx(1000 * (0.15 - 0.5))
    assert row["m_k"] == 0
    assert row["m_profit"] == 0


def test_stability_summary_counts_seeds_beating_everyone():
    table = pd.DataFrame(
        {
            "email": ["a"] * 3,
            "model": ["m"] * 3,
            "seed": [1, 2, 3],
            "fraction": [0.5, 0.5, 1.0],
            "profit": [10.0, 5.0, 4.0],
            "gain_vs_everyone": [2.0, -1.0, 0.0],
        }
    )
    assert roi.summarize_roi_stability(table)[("seeds_beating_everyone", "")].iloc[0] == 1
