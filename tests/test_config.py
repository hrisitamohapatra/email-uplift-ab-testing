from src import config


def test_data_and_cache_stay_inside_project():
    assert config.DATA_DIR.is_relative_to(config.ROOT)
    assert config.CACHE_DIR.is_relative_to(config.ROOT)


def test_outcomes_are_never_listed_as_features():
    assert not set(config.HILLSTROM_FEATURES) & set(config.HILLSTROM_OUTCOMES)
    assert not set(config.CRITEO_FEATURES) & set(config.CRITEO_OUTCOMES)


def test_treatment_columns_are_never_listed_as_features():
    assert config.HILLSTROM_ARM_COL not in config.HILLSTROM_FEATURES
    assert config.CRITEO_TREATMENT_COL not in config.CRITEO_FEATURES
    assert "exposure" not in config.CRITEO_FEATURES


def test_comparisons_are_the_three_pairs_of_arms():
    pairs = {frozenset(pair) for pair in config.HILLSTROM_COMPARISONS}
    assert len(pairs) == 3
    assert {arm for pair in pairs for arm in pair} == set(config.HILLSTROM_ARMS)


def test_statistical_settings_are_valid():
    assert 0 < config.ALPHA < 1
    assert 0 < config.POWER < 1
    assert 0 < config.TEST_SIZE < 1


def test_cost_assumptions_are_positive():
    assert config.ASSUMED_COST_PER_CONTACT_USD > 0
    assert config.ASSUMED_MARGIN_PER_CONVERSION_USD > 0
