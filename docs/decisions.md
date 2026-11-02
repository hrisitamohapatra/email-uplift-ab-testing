# Analysis decisions

These choices were written down before any data was downloaded or analysed, so the results cannot influence them. Any later change gets a dated note here saying what changed and why.

## The question

Who should a company email, and who would have bought anyway? Answering this needs a comparison of what each customer did with and without the email, which is why both datasets are randomized experiments.

## Hillstrom (64,000 customers, the main experiment)

| Decision | Choice | Reason |
|---|---|---|
| Primary metric | `conversion` (purchase within two weeks) | It is the outcome the question asks about. |
| Secondary metrics | `visit`, `spend` | Visits are common, so they show effects more clearly but are not revenue. Spend is the closest to revenue but heavy-tailed, with many zeros, so it uses a bootstrap interval. |
| Comparisons | Men's vs no email, women's vs no email, men's vs women's | These are the three possible pairs of arms. |
| Significance level | 0.05, two-sided | A conventional choice. Two-sided because an email could also reduce purchases. |
| Multiple testing | Holm correction across the 3 comparisons, applied separately within each metric | Three tests raise the chance of a false positive, and Holm controls that without being as strict as Bonferroni. |
| How to read the results | Only the conversion comparisons are confirmatory; visit and spend are supporting evidence | Picking one primary metric in advance stops us from choosing whichever metric looks best afterwards. |
| Intervals | 95%, shown unadjusted next to the Holm-adjusted p-values | An interval can exclude zero while the adjusted p-value is not significant. In that case, the adjusted p-value decides. |
| Power target | 80% | A conventional planning value, used for the minimum detectable effect. |
| Regression adjustment | Uses `history` (past-year spend) | It is measured before the email, so it can reduce noise without biasing the estimate. |

### Exact methods (Stage 2)

These were recorded before the analysis was run on the real data.

| Step | Method | Reason |
|---|---|---|
| Sample-ratio check | Chi-square test against an equal three-way split, at the 0.05 level | A mismatch would suggest a broken experiment, which would make the results unreliable. |
| Covariate balance | Standardized mean difference, worst pair of arms, flagged above 0.1 | Unlike a p-value, it does not shrink just because the sample is large. 0.1 is a common rule of thumb. |
| Conversion and visit | Unpooled interval, pooled two-proportion z-test | The interval should not assume the arms are equal, but the test's null hypothesis does. |
| Spend interval | Percentile bootstrap with 2,000 resamples | Spend is mostly zeros with a heavy tail, so a formula-based interval is less trustworthy. |
| Spend p-value | Welch t-test | Holm needs an exact p-value, and 2,000 resamples cannot resolve very small ones. |
| Regression adjustment | Outcome on arm and `history`, with HC2 robust standard errors | HC2 stays valid when the outcome is binary or skewed. This is a supporting check, so the unadjusted results above remain primary. |
| Minimum detectable effect | Sample-size formula solved for the effect, at 80% power, using the second-named arm's baseline | Shows the smallest effect this experiment could reliably find. |

## Criteo (the larger second experiment)

| Decision | Choice | Reason |
|---|---|---|
| Treatment | Assigned `treatment`, never `exposure` | Only assignment was randomized. `exposure` records whether the advertiser won the ad auction for that user, so comparing by it would not be a fair comparison. |
| Primary metric | `conversion` | Same as Hillstrom. Conversions are very rare here, so small differences will not be over-read. |
| Secondary metric | `visit` | It is more common, so its estimates are more precise. |
| Sample | 1–2 million rows, keeping the treatment ratio | Keeps the work manageable on a laptop. The 10% file is the default if it passes the ratio check. |

**Limitation:** Criteo measures advertising, not email. It can test whether the method works at scale, but its results say nothing direct about email customers.

## Pre-treatment features vs outcomes

Only pre-treatment columns may be model inputs. A column measured after the email could itself be affected by the email, and using it would leak the answer into the model.

| Dataset | Column | Role | Model input? |
|---|---|---|---|
| Hillstrom | `recency`, `history`, `mens`, `womens`, `zip_code`, `newbie`, `channel` | Pre-treatment feature | Yes |
| Hillstrom | `history_segment` | Pre-treatment feature, a binned copy of `history` | No, it duplicates `history` |
| Hillstrom | `segment` | Assigned arm | Only as the treatment, never as a feature |
| Hillstrom | `visit`, `conversion`, `spend` | Outcome | **Never** |
| Criteo | `f0` to `f11` | Pre-treatment feature | Yes |
| Criteo | `treatment` | Assigned arm | Only as the treatment, never as a feature |
| Criteo | `exposure` | Post-treatment; removed when the data is loaded | **Never** |
| Criteo | `visit`, `conversion` | Outcome | **Never** |

**Two cautions:**
- In Hillstrom, `mens` and `womens` are not the email arms. They record whether the customer bought that merchandise in the past year.
- Criteo's features are anonymised, so we cannot check for ourselves that they were recorded before treatment. We rely on the dataset's publisher for that.

## Assumptions

Neither dataset contains costs or margins, so these values are illustrative and not measurements. They are set in `src/config.py`, and any money figure must say "assumed costs".

| Assumption | Value |
|---|---|
| Cost per contact | $0.10 |
| Margin per conversion | $30.00 |

These stay the base case and will not be tuned after seeing results, because choosing costs by their profit would bias the answer. Stage 4 will instead show how the recommendation changes across a range of costs and margins, with that range recorded here before the profit curves are run.

## Modelling rules (for Stages 3 and 4)

- 30% of the data is held out, and nothing is fitted on it.
- The random seed is 42.
- Bootstrap intervals use 2,000 resamples.

## Uplift modelling choices (Stage 3)

These were recorded before any uplift model was fitted on the real data.

| Decision | Choice | Reason |
|---|---|---|
| Hillstrom arms | Model each email separately against no email | Each customer gets an expected lift from each email. Pooling would hide the difference between the emails. The cost is less data per model. |
| Outcomes | `conversion` is primary; `visit` is secondary | Conversion answers the question but has few positive cases, so `visit` gives a second, less noisy reading. |
| Hold-out | 30%, split once, stratified by arm and conversion | Every model is scored on the same customers, who are never used for fitting. |
| Features | Pre-treatment columns without `history_segment`; categories one-hot encoded | One-hot encoding learns nothing from the outcomes, so doing it before the split cannot leak. |
| Model settings | Fixed in `src/config.py` and not tuned | Tuning on the hold-out would inflate the results. Small, slow-learning trees suit so few positive cases. |
| Response baseline | Purchase model fitted on emailed customers only | It ranks by "who buys when emailed", the usual marketing approach, and it cannot separate people persuaded by the email from people who would have bought anyway. |
| Main evaluation | Qini area above random, and uplift in the top 10%, 20%, 30% and 50% | Individual effects can never be observed directly, so models are judged on whether high-scoring groups show more lift. |
| Intervals | Bootstrap the hold-out rows, 2,000 resamples, 95% percentile | Model comparisons use the same resamples for both models. These intervals do not include the randomness from training the model. |
| Stability check | Repeat the split and fit with 10 other seeds and report the spread | Covers the training randomness the intervals leave out. Added after a test showed that row order alone can move scores, and before any model was fitted on real data. |

## ROI choices (Stage 4)

These were recorded before any profit figure was computed on the real data. The Stage 3 model results were already known, which is why the visit-trained scores are labelled exploratory.

| Decision | Choice | Reason |
|---|---|---|
| Strategies | Email everyone, email no one, top k% by response model, top k% by uplift model (T and X both shown) | Reporting both uplift models avoids choosing the better one on the hold-out. |
| Exploratory strategy | Rank by uplift trained on `visit`, then measure conversion and profit | Visits were the only stable signal in Stage 3. This was proposed after seeing that, so it cannot be the headline result. |
| Incremental conversions | 1,000 × (treated rate − control rate) among those contacted, per 1,000 contacts | Comparing treated and control customers within the targeted group gives the added conversions, not total conversions. |
| Profit | Per 1,000 customers in the base: 1,000 × k × (uplift × margin − cost), relative to emailing no one | Customers who would have bought anyway add no profit, and every contact has a cost. |
| Choosing k | Grid from 5% to 100% in steps of 5%, chosen by 5-fold out-of-fold profit inside the training data; email no one if no fraction is profitable | Picking the best fraction on the hold-out and scoring it there would overstate profit. |
| Reporting | Hold-out profit at the chosen k, with 2,000 bootstrap resamples and a paired gain against emailing everyone | The gain answers the real question: does targeting beat the simple alternative? |
| Stability | Repeat with 10 other seeds | Covers the training randomness the intervals leave out. |
| Cost sensitivity | Cost per contact of $0.05, $0.10, $0.25 and $0.50; margin per conversion of $15, $30 and $60 | Shows how far the recommendation depends on assumptions the data cannot check. The base case stays $0.10 and $30. |

## Environment

- Python 3.14 on Windows, in a project-local `.venv`.
- **`econml` instead of `causalml`:** `causalml` 0.18.0 has no prebuilt wheel for Python 3.14 on Windows, and we install prebuilt packages only.
