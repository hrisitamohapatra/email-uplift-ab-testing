"""Evaluation tests: hand-worked examples, agreement with sklift, and interval behaviour."""

import ast
from pathlib import Path

import numpy as np
import pytest
from sklearn.metrics import auc
from sklift import metrics as sklift_metrics

from src import config, evaluate, simulate

Y = [1, 0, 1, 0]
T = [1, 0, 0, 1]
SCORE = [0.9, 0.8, 0.7, 0.6]


def random_experiment(seed, n=4_000):
    rng = np.random.default_rng(seed)
    return (rng.random(n) < 0.2).astype(int), rng.integers(0, 2, n), rng.normal(size=n)


def test_qini_curve_matches_hand_calculation():
    customers, incremental = evaluate.qini_curve(Y, T, SCORE)
    np.testing.assert_allclose(customers, [0, 1, 2, 3, 4])
    np.testing.assert_allclose(incremental, [0, 1, 1, 0.5, 0])


def test_qini_area_matches_hand_calculation():
    # Trapezoids: 0.5 + 1 + 0.75 + 0.25 = 2.5. Random targeting has area 0 here. 2.5 / 4^2.
    assert evaluate.qini_area(Y, T, SCORE) == pytest.approx(0.15625)


def test_uplift_at_k_matches_hand_calculation():
    assert evaluate.uplift_at_k(Y, T, SCORE, 0.5) == pytest.approx(1.0)
    assert evaluate.uplift_at_k(Y, T, SCORE, 1.0) == pytest.approx(0.0)


def test_uplift_at_k_is_nan_when_one_group_is_absent():
    assert np.isnan(evaluate.uplift_at_k(Y, T, SCORE, 0.25))


def test_tied_scores_do_not_depend_on_row_order():
    rng = np.random.default_rng(1)
    y, t = rng.integers(0, 2, 500), rng.integers(0, 2, 500)
    score = rng.integers(0, 5, 500) / 5
    shuffled = rng.permutation(500)
    first = evaluate.qini_curve(y, t, score)
    second = evaluate.qini_curve(y[shuffled], t[shuffled], score[shuffled])
    np.testing.assert_allclose(first[0], second[0])
    np.testing.assert_allclose(first[1], second[1])


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_qini_curve_matches_sklift(seed):
    y, t, score = random_experiment(seed)
    ours = evaluate.qini_curve(y, t, score)
    theirs = sklift_metrics.qini_curve(y, score, t)
    np.testing.assert_allclose(ours[0], theirs[0])
    np.testing.assert_allclose(ours[1], theirs[1])


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_qini_area_matches_sklift_curve(seed):
    y, t, score = random_experiment(seed)
    x, values = sklift_metrics.qini_curve(y, score, t)
    theirs = (auc(x, values) - auc([0, x[-1]], [0, values[-1]])) / x[-1] ** 2
    assert evaluate.qini_area(y, t, score) == pytest.approx(theirs)


@pytest.mark.parametrize("k", [0.1, 0.3, 0.5])
def test_uplift_at_k_matches_sklift(k):
    y, t, score = random_experiment(4)
    theirs = sklift_metrics.uplift_at_k(y, score, t, strategy="overall", k=k)
    assert evaluate.uplift_at_k(y, t, score, k) == pytest.approx(theirs)


def test_evaluate_calculates_with_numpy_only():
    tree = ast.parse(Path(evaluate.__file__).read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {alias.name.split(".")[0] for alias in node.names}
        if isinstance(node, ast.ImportFrom):
            imported.add(node.module.split(".")[0])
    assert imported == {"functools", "numpy", "src"}


def test_true_effect_ranking_beats_random_ranking():
    df = simulate.simulate_uplift(40_000, seed=5)
    y, t = df["outcome"], df["treatment"]
    assert evaluate.qini_area(y, t, df["true_effect"]) > 0.005
    noise = np.random.default_rng(5).normal(size=len(df))
    assert abs(evaluate.qini_area(y, t, noise)) < 0.005


def test_bootstrap_is_repeatable_and_contains_estimate():
    y, t, score = random_experiment(6)
    metrics = {"top": lambda a, b, c: evaluate.uplift_at_k(a, b, c, 0.5)}
    first = evaluate.bootstrap_metrics(y, t, {"m": score}, metrics, n_boot=200, seed=1)
    second = evaluate.bootstrap_metrics(y, t, {"m": score}, metrics, n_boot=200, seed=1)
    assert first == second
    assert first[0]["ci_low"] < first[0]["value"] < first[0]["ci_high"]


def test_identical_models_have_zero_gap():
    y, t, score = random_experiment(7)
    metrics = evaluate.default_metrics()
    rows = evaluate.bootstrap_metrics(y, t, {"base": score, "copy": score}, metrics, baseline="base", n_boot=50)
    for row in [r for r in rows if r["model"] == "copy"]:
        assert row["gain_vs_baseline"] == 0
        assert row["gain_low"] == row["gain_high"] == 0


def test_bootstrap_interval_covers_known_top_group_effect():
    """The true effect of the top half is known in simulation: check how often the interval holds it."""
    sims = 100
    covered = 0
    metrics = {"top": lambda a, b, c: evaluate.uplift_at_k(a, b, c, 0.5)}
    for sim in range(sims):
        df = simulate.simulate_uplift(3_000, seed=sim)
        score = df["x1"].to_numpy()
        top = np.argsort(score, kind="mergesort")[::-1][: len(df) // 2]
        truth = df["true_effect"].to_numpy()[top].mean()
        row = evaluate.bootstrap_metrics(df["outcome"], df["treatment"], {"m": score}, metrics, n_boot=200, seed=sim)[0]
        covered += row["ci_low"] < truth < row["ci_high"]
    allowed = 3 * np.sqrt(0.95 * 0.05 / sims)
    assert abs(covered / sims - (1 - config.ALPHA)) < allowed
