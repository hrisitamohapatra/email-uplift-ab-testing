"""Streamlit A/B test calculator. Run with: streamlit run app.py

All calculations live in src/experiment.py; this file only handles inputs and display.
"""

import streamlit as st

from src import experiment

PLAN = "Plan sample size"
RESULTS = "Analyze results"

st.title("A/B test calculator")
st.write("For a two-group test with a yes/no outcome, such as whether a customer converted.")
mode = st.radio("What do you want to do?", [PLAN, RESULTS], horizontal=True)
alpha = st.number_input("Significance level (%)", min_value=0.1, max_value=20.0, value=5.0, step=0.5) / 100

if mode == PLAN:
    baseline = st.number_input("Baseline conversion rate (%)", min_value=0.01, max_value=99.0, value=10.0, step=0.5) / 100
    mde = st.number_input("Smallest lift worth detecting (percentage points)", min_value=0.01, max_value=50.0, value=2.0, step=0.1) / 100
    power = st.number_input("Power (%)", min_value=50.0, max_value=99.0, value=80.0, step=1.0) / 100
    if baseline + mde >= 1:
        st.error("The baseline rate plus the lift must stay below 100%.")
    else:
        per_arm = experiment.required_n_per_arm(baseline, mde, alpha, power)
        st.metric("Customers needed per group", f"{per_arm:,}")
        st.metric("Customers needed in total", f"{2 * per_arm:,}")
        st.write(
            f"To have a {power:.0%} chance of detecting a change from {baseline:.2%} to {baseline + mde:.2%} "
            f"at the {alpha:.1%} significance level, plan for {per_arm:,} customers in each group. "
            "Set this before the test starts, and avoid stopping early just because the result looks good."
        )
else:
    left, right = st.columns(2)
    n_a = left.number_input("Group A customers", min_value=1, value=5_000, step=100)
    successes_a = left.number_input("Group A conversions", min_value=0, value=600, step=10)
    n_b = right.number_input("Group B customers", min_value=1, value=5_000, step=100)
    successes_b = right.number_input("Group B conversions", min_value=0, value=500, step=10)
    if successes_a > n_a or successes_b > n_b:
        st.error("Conversions cannot be more than customers.")
    else:
        result = experiment.two_proportion_test(successes_a, n_a, successes_b, n_b, alpha)
        st.metric("Lift, A minus B (percentage points)", f"{result['difference'] * 100:+.2f}")
        st.metric(f"{1 - alpha:.1%} confidence interval", f"{result['ci_low'] * 100:+.2f} to {result['ci_high'] * 100:+.2f}")
        st.metric("p-value", f"{result['p_value']:.4f}")
        st.write(experiment.plain_english_reading(result, alpha))
