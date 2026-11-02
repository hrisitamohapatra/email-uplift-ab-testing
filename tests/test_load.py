"""Loader tests. These never touch the network: most use small fake tables, and the
real-file test is skipped unless the dataset has already been downloaded."""

import numpy as np
import pandas as pd
import pytest

from src import config, load

CRITEO_TEST_ROWS = 2_000


def make_hillstrom(rows=config.HILLSTROM_ROWS):
    arms = np.resize(config.HILLSTROM_ARMS, rows)
    return pd.DataFrame(
        {
            "recency": 1,
            "history_segment": "1) $0 - $100",
            "history": 50.0,
            "mens": 1,
            "womens": 0,
            "zip_code": "Urban",
            "newbie": 0,
            "channel": "Web",
            config.HILLSTROM_ARM_COL: arms,
            "visit": 0,
            "conversion": 0,
            "spend": 0.0,
        },
        index=range(rows),
    )


def make_criteo(rows=CRITEO_TEST_ROWS):
    treated_rows = int(rows * config.CRITEO_TREATMENT_SHARE)
    df = pd.DataFrame({column: np.arange(rows, dtype=float) for column in config.CRITEO_FEATURES})
    df[config.CRITEO_TREATMENT_COL] = pd.array(
        [1] * treated_rows + [0] * (rows - treated_rows), dtype="Int8"
    )
    df["visit"] = pd.array([0] * rows, dtype="Int8")
    df["conversion"] = pd.array([0] * rows, dtype="Int8")
    return df


def fake_fetch_hillstrom(calls):
    def fetch(**kwargs):
        calls.append(kwargs)
        df = make_hillstrom()
        outcomes = list(config.HILLSTROM_OUTCOMES)
        return df[list(config.HILLSTROM_FEATURES)], df[outcomes], df[config.HILLSTROM_ARM_COL]

    return fetch


def fake_fetch_criteo(calls):
    def fetch(**kwargs):
        calls.append(kwargs)
        df = make_criteo()
        outcomes = list(config.CRITEO_OUTCOMES)
        return df[list(config.CRITEO_FEATURES)], df[outcomes], df[config.CRITEO_TREATMENT_COL]

    return fetch


@pytest.fixture
def small_criteo_10pct(monkeypatch):
    monkeypatch.setattr(config, "CRITEO_ROWS_10PCT", CRITEO_TEST_ROWS)


def test_valid_hillstrom_passes():
    load.validate_hillstrom(make_hillstrom())


@pytest.mark.parametrize(
    "break_table, message",
    [
        (lambda df: df.iloc[:-1], "rows"),
        (lambda df: df.drop(columns="channel"), "columns"),
        (lambda df: df.assign(extra=1), "columns"),
        (lambda df: df.assign(recency=np.nan), "Missing"),
        (lambda df: df.assign(segment="Other"), "arms"),
        (lambda df: df.assign(conversion=2), "conversion"),
        (lambda df: df.assign(spend=-1.0), "negative"),
    ],
)
def test_broken_hillstrom_is_rejected(break_table, message):
    with pytest.raises(ValueError, match=message):
        load.validate_hillstrom(break_table(make_hillstrom()))


def make_lenta(rows=2_000):
    treated_rows = int(rows * config.LENTA_TREATMENT_SHARE)
    positives = int(rows * config.LENTA_OUTCOME_RATE)
    df = pd.DataFrame({f"feature_{i}": 0.5 for i in range(3)}, index=range(rows))
    df[config.LENTA_GROUP_COL] = [config.LENTA_TREATED] * treated_rows + [config.LENTA_CONTROL] * (rows - treated_rows)
    df[config.LENTA_OUTCOME] = [1] * positives + [0] * (rows - positives)
    return df


@pytest.fixture
def small_lenta(monkeypatch):
    monkeypatch.setattr(config, "LENTA_ROWS", 2_000)
    monkeypatch.setattr(config, "LENTA_FEATURE_COLUMNS", 3)


def test_valid_lenta_passes_even_with_missing_features(small_lenta):
    load.validate_lenta(make_lenta().assign(feature_0=np.nan))


@pytest.mark.parametrize(
    "break_table, message",
    [
        (lambda df: df.iloc[:-1], "rows"),
        (lambda df: df.drop(columns="feature_0"), "feature columns"),
        (lambda df: df.assign(group="other"), "groups"),
        (lambda df: df.assign(response_att=np.nan), "Missing"),
        (lambda df: df.assign(group=df["group"].where(df.index > 5)), "Missing"),
        (lambda df: df.assign(response_att=2), "response_att"),
        (lambda df: df.assign(group=["test"] * 1_000 + ["control"] * 1_000), "treatment share"),
        (lambda df: df.assign(response_att=[1] * 1_000 + [0] * 1_000), "outcome rate"),
    ],
)
def test_broken_lenta_is_rejected(small_lenta, break_table, message):
    with pytest.raises(ValueError, match=message):
        load.validate_lenta(break_table(make_lenta()))


def test_load_lenta_does_not_download_and_builds_treatment(monkeypatch, tmp_path, small_lenta):
    calls = []

    def fetch(**kwargs):
        calls.append(kwargs)
        df = make_lenta()
        features = df.drop(columns=[config.LENTA_GROUP_COL, config.LENTA_OUTCOME])
        return features, df[config.LENTA_OUTCOME], df[config.LENTA_GROUP_COL]

    monkeypatch.setattr(load, "fetch_lenta", fetch)
    df = load.load_lenta(data_dir=tmp_path)
    assert calls[0]["download_if_missing"] is False
    assert calls[0]["data_home"] == str(tmp_path)
    assert config.LENTA_GROUP_COL not in df.columns
    assert df[config.LENTA_TREATMENT_COL].mean() == pytest.approx(config.LENTA_TREATMENT_SHARE)


@pytest.mark.skipif(
    not (config.DATA_DIR / "lenta_dataset.csv.gz").exists(),
    reason="Lenta file not downloaded",
)
def test_real_lenta_file_passes_validation():
    df = load.load_lenta()
    assert len(df) == config.LENTA_ROWS
    assert config.LENTA_TREATMENT_COL in df.columns


def test_valid_criteo_passes():
    load.validate_criteo(make_criteo(), CRITEO_TEST_ROWS)


@pytest.mark.parametrize(
    "break_table, message",
    [
        (lambda df: df.iloc[:1_000], "rows"),
        (lambda df: df.assign(exposure=1), "columns"),
        (lambda df: df.assign(f0=np.nan), "Missing"),
        (lambda df: df.assign(visit=2), "visit"),
        (lambda df: df.assign(treatment=1), "treatment share"),
    ],
)
def test_broken_criteo_is_rejected(break_table, message):
    with pytest.raises(ValueError, match=message):
        load.validate_criteo(break_table(make_criteo()), CRITEO_TEST_ROWS)


def test_stratified_sample_keeps_ratio_and_is_repeatable():
    df = make_criteo()
    first = load.stratified_sample(df, config.CRITEO_TREATMENT_COL, 1_000, seed=1)
    second = load.stratified_sample(df, config.CRITEO_TREATMENT_COL, 1_000, seed=1)
    assert len(first) == 1_000
    assert first[config.CRITEO_TREATMENT_COL].mean() == pytest.approx(config.CRITEO_TREATMENT_SHARE)
    pd.testing.assert_frame_equal(first, second)


def test_stratified_sample_rejects_too_many_rows():
    with pytest.raises(ValueError, match="Cannot sample"):
        load.stratified_sample(make_criteo(), config.CRITEO_TREATMENT_COL, CRITEO_TEST_ROWS + 1)


def test_load_hillstrom_does_not_download_by_default(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(load, "fetch_hillstrom", fake_fetch_hillstrom(calls))
    df = load.load_hillstrom(data_dir=tmp_path)
    assert len(df) == config.HILLSTROM_ROWS
    assert calls[0]["download_if_missing"] is False
    assert calls[0]["data_home"] == str(tmp_path)


def test_load_criteo_defaults_are_safe(monkeypatch, tmp_path, small_criteo_10pct):
    calls = []
    monkeypatch.setattr(load, "fetch_criteo", fake_fetch_criteo(calls))
    df = load.load_criteo(data_dir=tmp_path)
    assert calls[0]["download_if_missing"] is False
    assert calls[0]["percent10"] is True
    assert calls[0]["treatment_col"] == "treatment"
    assert calls[0]["data_home"] == str(tmp_path)
    assert "exposure" not in df.columns
    assert df[config.CRITEO_TREATMENT_COL].dtype == "int8"


def test_load_criteo_can_sample(monkeypatch, tmp_path, small_criteo_10pct):
    monkeypatch.setattr(load, "fetch_criteo", fake_fetch_criteo([]))
    df = load.load_criteo(data_dir=tmp_path, sample_rows=1_000)
    assert len(df) == 1_000


@pytest.mark.parametrize("loader", [load.load_hillstrom, load.load_lenta, load.load_criteo])
def test_missing_file_raises_instead_of_downloading(loader, tmp_path):
    with pytest.raises(OSError, match="Dataset missing"):
        loader(data_dir=tmp_path)


@pytest.mark.skipif(
    not (config.DATA_DIR / "hillstorm_no_indices.csv.gz").exists(),
    reason="Hillstrom file not downloaded",
)
def test_real_hillstrom_file_passes_validation():
    assert len(load.load_hillstrom()) == config.HILLSTROM_ROWS
