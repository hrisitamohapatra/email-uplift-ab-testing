"""Load the two experiments and refuse to return data that fails basic sanity checks."""

import pandas as pd

from src import config  # Must stay above sklift: it sets the Matplotlib cache folder.
from sklift.datasets import fetch_hillstrom, fetch_lenta


def _check_columns(df, expected):
    if set(df.columns) != set(expected):
        raise ValueError(f"Expected columns {sorted(expected)}, got {sorted(df.columns)}")


def _check_no_missing(df):
    missing = df.isna().sum()
    if missing.any():
        raise ValueError(f"Missing values found: {missing[missing > 0].to_dict()}")


def _check_binary(df, columns):
    for column in columns:
        if not df[column].isin([0, 1]).all():
            raise ValueError(f"Column {column} must contain only 0 and 1")


def validate_hillstrom(df):
    if len(df) != config.HILLSTROM_ROWS:
        raise ValueError(f"Expected {config.HILLSTROM_ROWS} rows, got {len(df)}")
    _check_columns(
        df,
        [*config.HILLSTROM_FEATURES, config.HILLSTROM_ARM_COL, *config.HILLSTROM_OUTCOMES],
    )
    _check_no_missing(df)
    if set(df[config.HILLSTROM_ARM_COL]) != set(config.HILLSTROM_ARMS):
        raise ValueError(f"Unexpected arms: {sorted(df[config.HILLSTROM_ARM_COL].unique())}")
    _check_binary(df, ["visit", "conversion"])
    if (df["spend"] < 0).any():
        raise ValueError("Spend cannot be negative")


def validate_lenta(df):
    if len(df) != config.LENTA_ROWS:
        raise ValueError(f"Expected {config.LENTA_ROWS} rows, got {len(df)}")
    # Features may have missing values, but the group and outcome may not.
    feature_count = df.shape[1] - 2
    if feature_count != config.LENTA_FEATURE_COLUMNS:
        raise ValueError(f"Expected {config.LENTA_FEATURE_COLUMNS} feature columns, got {feature_count}")
    _check_no_missing(df[[config.LENTA_GROUP_COL, config.LENTA_OUTCOME]])
    if set(df[config.LENTA_GROUP_COL]) != {config.LENTA_TREATED, config.LENTA_CONTROL}:
        raise ValueError(f"Unexpected groups: {sorted(df[config.LENTA_GROUP_COL].unique())}")
    _check_binary(df, [config.LENTA_OUTCOME])
    share = (df[config.LENTA_GROUP_COL] == config.LENTA_TREATED).mean()
    if abs(share - config.LENTA_TREATMENT_SHARE) > config.LENTA_TREATMENT_SHARE_TOLERANCE:
        raise ValueError(f"Expected treatment share near {config.LENTA_TREATMENT_SHARE}, got {share:.4f}")
    rate = df[config.LENTA_OUTCOME].mean()
    if abs(rate - config.LENTA_OUTCOME_RATE) > config.LENTA_OUTCOME_RATE_TOLERANCE:
        raise ValueError(f"Expected outcome rate near {config.LENTA_OUTCOME_RATE}, got {rate:.4f}")


def load_hillstrom(data_dir=config.DATA_DIR, download=False):
    """Return one row per customer. Uses the network only if download=True."""
    features, outcomes, arm = fetch_hillstrom(
        target_col="all",
        data_home=str(data_dir),
        download_if_missing=download,
        return_X_y_t=True,
    )
    df = pd.concat([features, arm.rename(config.HILLSTROM_ARM_COL), outcomes], axis=1)
    validate_hillstrom(df)
    return df


def load_lenta(data_dir=config.DATA_DIR, download=False):
    """Return one row per customer, with group turned into a 0/1 treatment column."""
    features, outcome, group = fetch_lenta(
        data_home=str(data_dir),
        download_if_missing=download,
        return_X_y_t=True,
    )
    df = pd.concat([features, group.rename(config.LENTA_GROUP_COL), outcome.rename(config.LENTA_OUTCOME)], axis=1)
    validate_lenta(df)
    treatment = (df[config.LENTA_GROUP_COL] == config.LENTA_TREATED).astype("int8").rename(config.LENTA_TREATMENT_COL)
    return pd.concat([df.drop(columns=config.LENTA_GROUP_COL), treatment], axis=1)
