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

## Still to decide

- **Which Hillstrom arms to use for uplift modelling (Stage 3):** pool both emails into "any email", or model each one separately. To be decided and recorded here before the models are fitted.

## Environment

- Python 3.14 on Windows, in a project-local `.venv`.
- **`econml` instead of `causalml`:** `causalml` 0.18.0 has no prebuilt wheel for Python 3.14 on Windows, and we install prebuilt packages only.
