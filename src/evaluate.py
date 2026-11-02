"""Qini curve and uplift@k with numpy only, plus bootstrap intervals.

A person's own effect can never be observed, so these metrics judge a ranking by a group
result: do the customers ranked highest show more lift than customers overall?
"""

from functools import partial

import numpy as np

from src import config


def qini_curve(y, t, score):
    """Incremental outcomes among the top-ranked customers, at each point in the ranking.

    Q(k) = treated outcomes - control outcomes * (treated count / control count).
    The ratio rescales control to the size of the treated group. Customers with tied
    scores are kept together, so the curve does not depend on an arbitrary tie order.
    """
    y, t, score = np.asarray(y, dtype=float), np.asarray(t, dtype=float), np.asarray(score, dtype=float)
    order = np.argsort(score, kind="mergesort")[::-1]
    y, t, score = y[order], t[order], score[order]

    last_of_each_score = np.r_[np.flatnonzero(np.diff(score)), len(score) - 1]
    treated_count = np.cumsum(t)[last_of_each_score]
    treated_outcomes = np.cumsum(y * t)[last_of_each_score]
    control_outcomes = np.cumsum(y * (1 - t))[last_of_each_score]
    customers = last_of_each_score + 1.0
    control_count = customers - treated_count

    ratio = np.divide(treated_count, control_count, out=np.zeros_like(treated_count), where=control_count != 0)
    incremental = treated_outcomes - control_outcomes * ratio
    return np.r_[0.0, customers], np.r_[0.0, incremental]


def qini_area(y, t, score):
    """Area between the model's Qini curve and random targeting, as a share of the full square.

    Zero means no better than random. Larger means the ranking finds lift earlier.
    """
    customers, incremental = qini_curve(y, t, score)
    random_targeting = incremental[-1] * customers / customers[-1]
    return float(np.trapezoid(incremental - random_targeting, customers) / customers[-1] ** 2)


def uplift_at_k(y, t, score, k):
    """Treated rate minus control rate among the top k share of customers by score."""
    y, t = np.asarray(y, dtype=float), np.asarray(t, dtype=float)
    order = np.argsort(np.asarray(score, dtype=float), kind="mergesort")[::-1]
    top = order[: int(len(y) * k)]
    y_top, t_top = y[top], t[top]
    if t_top.sum() == 0 or t_top.sum() == len(t_top):
        return float("nan")
    return float(y_top[t_top == 1].mean() - y_top[t_top == 0].mean())


def default_metrics(fractions=config.UPLIFT_TOP_FRACTIONS):
    metrics = {"qini_area": qini_area}
    for k in fractions:
        metrics[f"uplift_top_{int(k * 100)}pct"] = partial(uplift_at_k, k=k)
    return metrics


def bootstrap_metrics(y, t, scores, metrics, baseline=None, n_boot=config.N_BOOTSTRAP, alpha=config.ALPHA, seed=config.SEED):
    """Percentile intervals for each model and metric, using one set of resampled rows.

    Every model is scored on the same resamples, so a difference between two models is not
    blurred by which customers were drawn. Models stay fixed: the only randomness measured
    is which hold-out customers were sampled, not how the models were trained.
    """
    y, t = np.asarray(y, dtype=float), np.asarray(t, dtype=float)
    scores = {name: np.asarray(score, dtype=float) for name, score in scores.items()}
    rng = np.random.default_rng(seed)
    draws = {(model, metric): np.empty(n_boot) for model in scores for metric in metrics}
    for i in range(n_boot):
        rows = rng.integers(0, len(y), len(y))
        for model, score in scores.items():
            for metric, function in metrics.items():
                draws[model, metric][i] = function(y[rows], t[rows], score[rows])

    quantiles = [alpha / 2, 1 - alpha / 2]
    results = []
    for model, score in scores.items():
        for metric, function in metrics.items():
            low, high = np.nanquantile(draws[model, metric], quantiles)
            row = {"model": model, "metric": metric, "value": function(y, t, score), "ci_low": low, "ci_high": high}
            if baseline is not None and model != baseline:
                gap = draws[model, metric] - draws[baseline, metric]
                gap_low, gap_high = np.nanquantile(gap, quantiles)
                row["gain_vs_baseline"] = row["value"] - function(y, t, scores[baseline])
                row["gain_low"], row["gain_high"] = gap_low, gap_high
            results.append(row)
    return results
