"""App tests. AppTest runs the script directly, so no web server is started."""

import pytest
from streamlit.testing.v1 import AppTest

from src import config, experiment

APP = str(config.ROOT / "app.py")


@pytest.fixture
def app():
    return AppTest.from_file(APP, default_timeout=120).run()


def inputs(app):
    return {widget.label: widget for widget in app.number_input}


def metrics(app):
    return {metric.label: metric.value for metric in app.metric}


def test_planning_mode_matches_the_formula(app):
    assert not app.exception
    assert metrics(app)["Customers needed per group"] == f"{experiment.required_n_per_arm(0.10, 0.02):,}"
    assert metrics(app)["Customers needed per group"] == "3,839"
    assert metrics(app)["Customers needed in total"] == "7,678"


def test_planning_mode_responds_to_inputs(app):
    inputs(app)["Baseline conversion rate (%)"].set_value(20.0).run()
    assert not app.exception
    assert metrics(app)["Customers needed per group"] == f"{experiment.required_n_per_arm(0.20, 0.02):,}"


def test_planning_mode_rejects_rates_above_100_percent(app):
    inputs(app)["Baseline conversion rate (%)"].set_value(99.0).run()
    inputs(app)["Smallest lift worth detecting (percentage points)"].set_value(5.0).run()
    assert not app.exception
    assert app.error
    assert not app.metric


def test_results_mode_matches_the_test_and_gives_a_reading(app):
    app.radio[0].set_value("Analyze results").run()
    assert not app.exception
    result = experiment.two_proportion_test(600, 5_000, 500, 5_000, 0.05)
    assert metrics(app)["Lift, A minus B (percentage points)"] == f"{result['difference'] * 100:+.2f}"
    assert metrics(app)["p-value"] == f"{result['p_value']:.4f}"
    reading = " ".join(block.value for block in app.markdown)
    assert experiment.plain_english_reading(result, 0.05) in reading


def test_results_mode_rejects_more_conversions_than_customers(app):
    app.radio[0].set_value("Analyze results").run()
    inputs(app)["Group A conversions"].set_value(6_000).run()
    assert not app.exception
    assert app.error
    assert not app.metric
