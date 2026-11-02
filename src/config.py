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

# ASSUMPTIONS, not measurements: neither dataset contains costs or margins.
# Every money figure in this project depends on these two numbers.
ASSUMED_COST_PER_CONTACT_USD = 0.10
ASSUMED_MARGIN_PER_CONVERSION_USD = 30.00

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
HILLSTROM_OUTCOMES = ("visit", "conversion", "spend")

# Criteo columns
CRITEO_ROWS_FULL = 13_979_592
CRITEO_ROWS_10PCT = CRITEO_ROWS_FULL // 10
CRITEO_ROW_TOLERANCE = 0.01
CRITEO_TREATMENT_COL = "treatment"
CRITEO_TREATMENT_SHARE = 0.85
CRITEO_TREATMENT_SHARE_TOLERANCE = 0.01
CRITEO_FEATURES = tuple(f"f{i}" for i in range(12))
CRITEO_OUTCOMES = ("visit", "conversion")
