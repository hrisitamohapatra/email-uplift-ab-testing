"""Simulation tests, including the main check: do the learners recover a known effect?"""

import numpy as np
import pandas as pd
import pytest
from scipy import stats
from sklearn.model_selection import train_test_split

from src import config, evaluate, learners, simulate


@pytest.fixture(scope="module")
def fitted():
    df = simulate.simulate_uplift(40_000, seed=1)
    train, holdout = train_test_split(df, test_size=config.TEST_SIZE, random_state=1)
    columns = list(simulate.FEATURES)
    scores = learners.fit_and_score(train[columns], train["outcome"], train["treatment"], holdout[columns], seed=1)
    return holdout, scores


def test_same_seed_gives_same_data():
    pd.testing.assert_frame_equal(simulate.simulate_uplift(500, seed=3), simulate.simulate_uplift(500, seed=3))


def test_probabilities_are_valid_and_effect_is_their_gap():
    df = simulate.simulate_uplift(5_000, seed=2)
    assert df[["prob_with", "prob_without"]].stack().between(0, 1).all()
    np.testing.assert_allclose(df["true_effect"], df["prob_with"] - df["prob_without"])


def test_effects_include_harm_and_large_benefit():
    df = simulate.simulate_uplift(20_000, seed=2)
    assert df["true_effect"].min() < -0.02
    assert df["true_effect"].max() > 0.15


def test_treatment_is_random_and_unrelated_to_features():
    df = simulate.simulate_uplift(40_000, seed=4, treatment_share=0.85)
    assert df["treatment"].mean() == pytest.approx(0.85, abs=0.01)
    for column in simulate.FEATURES:
        p_value = stats.ttest_ind(df.loc[df["treatment"] == 1, column], df.loc[df["treatment"] == 0, column]).pvalue
        assert p_value > 0.001


def test_who_buys_anyway_is_separate_from_who_is_persuaded():
    df = simulate.simulate_uplift(40_000, seed=5)
    assert abs(np.corrcoef(df["prob_without"], df["true_effect"])[0, 1]) < 0.2


def test_simple_difference_recovers_average_effect():
    df = simulate.simulate_uplift(200_000, seed=6)
    treated, control = df[df["treatment"] == 1], df[df["treatment"] == 0]
    estimate = treated["outcome"].mean() - control["outcome"].mean()
    assert estimate == pytest.approx(df["true_effect"].mean(), abs=0.006)


@pytest.mark.parametrize("model", ["t_learner", "x_learner"])
def test_learners_recover_known_effects(fitted, model):
    holdout, scores = fitted
    truth = holdout["true_effect"].to_numpy()
    assert np.corrcoef(scores[model], truth)[0, 1] > 0.7
    assert scores[model].mean() == pytest.approx(truth.mean(), abs=0.015)
    # Better than the best possible single number for everybody.
    assert np.sqrt(((scores[model] - truth) ** 2).mean()) < truth.std()


@pytest.mark.parametrize("model", ["t_learner", "x_learner"])
def test_uplift_models_beat_response_model_at_finding_persuadable_people(fitted, model):
    holdout, scores = fitted
    truth = holdout["true_effect"].to_numpy()
    top = int(len(truth) * 0.3)
    uplift_top = truth[np.argsort(scores[model])[::-1][:top]].mean()
    response_top = truth[np.argsort(scores["response"])[::-1][:top]].mean()
    assert uplift_top > response_top + 0.03


def test_response_model_tracks_buying_not_persuasion(fitted):
    holdout, scores = fitted
    assert np.corrcoef(scores["response"], holdout["prob_with"])[0, 1] > 0.8
    assert abs(np.corrcoef(scores["response"], holdout["true_effect"])[0, 1]) < 0.4


def test_x_learner_recovers_effects_with_an_uneven_split():
    df = simulate.simulate_uplift(40_000, seed=7, treatment_share=0.85)
    train, holdout = train_test_split(df, test_size=config.TEST_SIZE, random_state=7)
    columns = list(simulate.FEATURES)
    model = learners.XLearner(seed=7).fit(train[columns], train["outcome"], train["treatment"])
    assert np.corrcoef(model.predict(holdout[columns]), holdout["true_effect"])[0, 1] > 0.6


def test_no_real_effect_gives_no_real_qini_signal():
    df = simulate.simulate_uplift(40_000, seed=8, effect_scale=0.0)
    train, holdout = train_test_split(df, test_size=config.TEST_SIZE, random_state=8)
    columns = list(simulate.FEATURES)
    score = learners.TLearner(seed=8).fit(train[columns], train["outcome"], train["treatment"]).predict(holdout[columns])
    results = evaluate.bootstrap_metrics(
        holdout["outcome"], holdout["treatment"], {"t": score}, {"qini_area": evaluate.qini_area}, n_boot=300, seed=8
    )
    assert results[0]["ci_low"] < 0 < results[0]["ci_high"]
