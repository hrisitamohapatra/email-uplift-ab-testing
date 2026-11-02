"""Project-wide settings: paths, seed, statistical choices, columns and cost assumptions."""

import os
from pathlib import Path

# Paths
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
CACHE_DIR = ROOT / ".cache"

# Matplotlib writes a cache to the user profile by default. This keeps it in the
# project, and only for the running Python process.
os.environ.setdefault("MPLCONFIGDIR", str(CACHE_DIR / "matplotlib"))

# Reproducibility
SEED = 42

# Statistical choices (reasons are in docs/decisions.md)
ALPHA = 0.05
POWER = 0.80
N_BOOTSTRAP = 2_000
TEST_SIZE = 0.30

# Uplift models: small, slow-learning trees because positive cases are rare. Not tuned.
LGBM_PARAMS = {
    "n_estimators": 200,
    "learning_rate": 0.03,
    "num_leaves": 7,
    "min_child_samples": 100,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "n_jobs": 1,
    "verbose": -1,
}
UPLIFT_TOP_FRACTIONS = (0.1, 0.2, 0.3, 0.5)

# ASSUMPTIONS, not measurements: neither dataset contains costs or margins.
# Every money figure in this project depends on these two numbers.
ASSUMED_COST_PER_CONTACT_USD = 0.10
ASSUMED_MARGIN_PER_CONVERSION_USD = 30.00

# ROI grids. The sensitivity ranges are also assumptions; the base case is the pair above.
ROI_FRACTIONS = tuple(step / 100 for step in range(5, 101, 5))
ROI_COST_GRID_USD = (0.05, 0.10, 0.25, 0.50)
ROI_MARGIN_GRID_USD = (15.00, 30.00, 60.00)
ROI_FOLDS = 5
MEMO_BUDGET = 10_000

# Hillstrom columns
HILLSTROM_ROWS = 64_000
HILLSTROM_ARM_COL = "segment"
HILLSTROM_MENS = "Mens E-Mail"
HILLSTROM_WOMENS = "Womens E-Mail"
HILLSTROM_CONTROL = "No E-Mail"
HILLSTROM_ARMS = (HILLSTROM_MENS, HILLSTROM_WOMENS, HILLSTROM_CONTROL)
HILLSTROM_COMPARISONS = (
    (HILLSTROM_MENS, HILLSTROM_CONTROL),
    (HILLSTROM_WOMENS, HILLSTROM_CONTROL),
    (HILLSTROM_MENS, HILLSTROM_WOMENS),
)
HILLSTROM_FEATURES = (
    "recency",
    "history_segment",
    "history",
    "mens",
    "womens",
    "zip_code",
    "newbie",
    "channel",
)
HILLSTROM_MODEL_FEATURES = tuple(f for f in HILLSTROM_FEATURES if f != "history_segment")
HILLSTROM_OUTCOMES = ("visit", "conversion", "spend")

# Lenta columns
LENTA_ROWS = 687_029
LENTA_FEATURE_COLUMNS = 193
LENTA_GROUP_COL = "group"
LENTA_TREATED = "test"
LENTA_CONTROL = "control"
LENTA_TREATMENT_COL = "treatment"
LENTA_OUTCOME = "response_att"
LENTA_TREATMENT_SHARE = 0.75
LENTA_TREATMENT_SHARE_TOLERANCE = 0.01
LENTA_OUTCOME_RATE = 0.10
LENTA_OUTCOME_RATE_TOLERANCE = 0.02
LENTA_SMD_LIMIT = 0.10
# Personal attributes are left out by choice. Features failing the balance check are added here.
LENTA_EXCLUDED_FEATURES = ("age", "gender", "response_sms", "response_viber")
