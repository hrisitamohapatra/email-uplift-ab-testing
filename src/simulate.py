"""Synthetic data with a known individual treatment effect, which real data can never show."""

import numpy as np
import pandas as pd
from scipy.special import expit

from src import config

FEATURES = ("x0", "x1", "x2", "x3")


def simulate_uplift(n, seed=config.SEED, treatment_share=0.5, effect_scale=1.0):
    """Simulate a randomized experiment where each person's true effect is known.

    x0 drives who buys anyway. x1 drives who is persuaded. Keeping them separate means a
    model that only finds likely buyers will not find the people the email actually moves.
    Effects range from slightly negative to clearly positive. x2 and x3 are pure noise.
    """
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({column: rng.normal(size=n) for column in FEATURES})
    prob_without = expit(-1.5 + 1.5 * df["x0"])
    effect = effect_scale * (0.25 * expit(3 * df["x1"]) - 0.05)
    prob_with = np.clip(prob_without + effect, 0, 1)
    df["treatment"] = (rng.random(n) < treatment_share).astype(int)
    df["prob_without"] = prob_without
    df["prob_with"] = prob_with
    df["true_effect"] = prob_with - prob_without
    prob_observed = np.where(df["treatment"] == 1, prob_with, prob_without)
    df["outcome"] = (rng.random(n) < prob_observed).astype(int)
    return df
