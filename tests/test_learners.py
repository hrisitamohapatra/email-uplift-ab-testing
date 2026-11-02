"""Learner tests: no leakage, no fitting on the hold-out, agreement with the libraries."""

import numpy as np
import pandas as pd
import pytest

from src import config, learners, simulate

COLUMNS = list(simulate.FEATURES)


@pytest.fixture(scope="module")
def data():
    df = simulate.simulate_uplift(6_000, seed=1)
    return df.iloc[:4_000], df.iloc[4_000:]


def make_hillstrom(n=6_000, seed=1):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "recency": rng.integers(1, 13, n),
            "history_segment": rng.choice(["low", "high"], n),
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


@pytest.mark.parametrize("name", ["visit", "conversion", "spend", "segment", "treatment", "exposure"])
@pytest.mark.parametrize("learner", [learners.ResponseModel, learners.TLearner, learners.XLearner])
def test_learners_refuse_outcomes_and_treatment_as_inputs(data, learner, name):
    train, _ = data
    X = train[COLUMNS].assign(**{name: 0})
    with pytest.raises(ValueError, match="cannot be model inputs"):
        learner().fit(X, train["outcome"], train["treatment"])


def test_hillstrom_features_contain_only_pre_treatment_columns():
    columns = learners.hillstrom_features(make_hillstrom()).columns
    assert not any(column.startswith(("visit", "conversion", "spend", "segment", "history_segment")) for column in columns)
    assert {"recency", "history", "mens", "womens", "newbie", "zip_code_Urban", "channel_Web"} <= set(columns)


def test_split_is_disjoint_and_keeps_arms_and_conversions_balanced():
    df = make_hillstrom()
    train, holdout = learners.split_hillstrom(df)
    assert not set(train.index) & set(holdout.index)
    assert len(holdout) == pytest.approx(len(df) * config.TEST_SIZE, abs=1)
    assert holdout["conversion"].mean() == pytest.approx(train["conversion"].mean(), abs=0.002)
    shares = holdout[config.HILLSTROM_ARM_COL].value_counts(normalize=True)
    assert shares.between(0.32, 0.35).all()


@pytest.mark.parametrize("learner", [learners.ResponseModel, learners.TLearner, learners.XLearner])
def test_same_seed_gives_same_scores(data, learner):
    train, holdout = data
    first = learner(seed=2).fit(train[COLUMNS], train["outcome"], train["treatment"]).predict(holdout[COLUMNS])
    second = learner(seed=2).fit(train[COLUMNS], train["outcome"], train["treatment"]).predict(holdout[COLUMNS])
    np.testing.assert_array_equal(first, second)


def test_response_model_ignores_control_outcomes(data):
    train, holdout = data
    changed = train["outcome"].where(train["treatment"] == 1, 1 - train["outcome"])
    original = learners.ResponseModel().fit(train[COLUMNS], train["outcome"], train["treatment"]).predict(holdout[COLUMNS])
    altered = learners.ResponseModel().fit(train[COLUMNS], changed, train["treatment"]).predict(holdout[COLUMNS])
    np.testing.assert_array_equal(original, altered)


def test_x_learner_weights_follow_treated_share(data):
    train, holdout = data
    model = learners.XLearner().fit(train[COLUMNS], train["outcome"], train["treatment"])
    share = train["treatment"].mean()
    expected = share * model.control_effect_model.predict(holdout[COLUMNS]) + (1 - share) * model.treated_effect_model.predict(holdout[COLUMNS])
    np.testing.assert_allclose(model.predict(holdout[COLUMNS]), expected)


def test_hold_out_outcomes_cannot_affect_the_scores(monkeypatch):
    """Change only the hold-out outcomes. If nothing was fitted on them, the scores cannot move."""
    monkeypatch.setattr(learners.evaluate, "bootstrap_metrics", lambda *args, **kwargs: [])
    df = make_hillstrom()
    train, holdout = learners.split_hillstrom(df)

    # Both runs must use the same rows in the same order, because LightGBM's seeded row
    # sampling depends on row order. Flipping outcomes would also change a stratified split.
    def same_split(frame, **kwargs):
        return frame.loc[train.index], frame.loc[holdout.index]

    monkeypatch.setattr(learners, "split_hillstrom", same_split)
    _, original = learners.run_hillstrom_uplift(df, outcomes=("conversion",))
    changed = df.copy()
    changed.loc[holdout.index, "conversion"] = 1 - changed.loc[holdout.index, "conversion"]
    _, altered = learners.run_hillstrom_uplift(changed, outcomes=("conversion",))

    for key in original:
        for model in ("response", "t_learner", "x_learner"):
            np.testing.assert_array_equal(original[key]["scores"][model], altered[key]["scores"][model])
        assert not np.array_equal(original[key]["y"], altered[key]["y"])


def test_stability_check_covers_every_seed_email_outcome_and_model():
    table = learners.stability_check(make_hillstrom(), seeds=[1, 2], outcomes=("conversion",))
    assert len(table) == 2 * 2 * 1 * 3
    summary = learners.summarize_stability(table)
    assert summary["seeds_qini_positive"].between(0, 2).all()


def test_t_learner_matches_sklift(data):
    train, holdout = data
    ours = learners.TLearner().fit(train[COLUMNS], train["outcome"], train["treatment"]).predict(holdout[COLUMNS])
    theirs = learners.library_t_scores(train[COLUMNS], train["outcome"], train["treatment"], holdout[COLUMNS])
    np.testing.assert_allclose(ours, theirs, atol=1e-9)


def test_x_learner_matches_econml(data):
    train, holdout = data
    ours = learners.XLearner().fit(train[COLUMNS], train["outcome"], train["treatment"]).predict(holdout[COLUMNS])
    theirs = learners.library_x_scores(train[COLUMNS], train["outcome"], train["treatment"], holdout[COLUMNS])
    assert np.corrcoef(ours, theirs)[0, 1] > 0.999
    assert np.abs(ours - theirs).max() < 0.01
