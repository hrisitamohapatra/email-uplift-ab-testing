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

## Lenta replaces Criteo (changed 2026-11-02)

**What changed:** Lenta is now the second dataset. Criteo was not analysed.

**Why:** on 2026-11-02, both download links in `scikit-uplift` and the link on Criteo's own dataset page returned 404 to header requests. Lenta was chosen over X5 (needs features built from 45.8M purchase rows) and MegaFon (synthetic data). The `load_criteo` code remains in `src/load.py`, but it has only ever been run on fake data.

These choices were recorded before any model was fitted on Lenta.

| Decision | Choice | Reason |
|---|---|---|
| Role | Second test, kept separate from Hillstrom | Tests whether the methods do better with many more positive cases. Different customers, features and outcomes mean the data cannot be combined, and more data cannot be proven as the cause of any difference. |
| Treatment | `group`: `test` vs `control` | The assigned group. |
| Outcome | `response_att`, a store visit | Lenta has no purchase outcome, so the $30 margin does not really apply and Lenta money figures are illustrative only. Hillstrom remains the source of the main money figures. |
| Excluded by choice | `age`, `gender` | Personal attributes are left out of targeting by default. |
| Post-treatment check | Compare each feature and its missingness between groups, and exclude anything above 0.1 standardized difference | The docs do not state the timing of every feature. In a randomized test, a pre-treatment feature should look the same in both groups. This cannot prove a feature is pre-treatment, because a post-treatment feature that the campaign did not move would pass too. |
| Missing values | Left as they are | LightGBM handles missing values natively, and 153 columns have some. |
| Hold-out | 30%, stratified by group and outcome | Same approach as Hillstrom. |
| Stability | 5 other seeds, holding each model's chosen share fixed | Choosing the share again would take too long at this size, so this measures less than the Hillstrom check. |

**Licence:** the `scikit-uplift` docs state none for this data, so it stays out of git and the source is credited.

**Findings from the checks, before any model was fitted:**
- `response_sms` (share of past SMS messages responded to) failed the check at 0.198 and is excluded under the rule.
- `response_viber` passed at 0.068 but is the same kind of measure and the next largest difference, so it is also excluded. This is a judgement call beyond the written rule.
- `gender` differs by 0.026, about 9 times what chance alone would give at this sample size, and a campaign cannot change gender. Assignment may therefore not be purely random.

**Added because of that last finding:** predict assignment from the kept features on unseen rows, and compare the raw lift with a lift calculated within 5 groups of customers who had a similar predicted chance of treatment. An AUC near 0.5 would mean assignment looks random. This can only adjust for what the features measure.

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

## Memo method (Stage 5)

These were recorded before the fixed-budget figures were calculated.

| Decision | Choice | Reason |
|---|---|---|
| Question | Which 10,000 to email and which 10,000 to leave out | Set by the brief, as 10,000 of 64,000 customers (15.625%). Because the share is fixed in advance, it avoids choosing the best-looking share. |
| Email | Men's email vs no email | It had the larger average effect in Stage 2. |
| Main ranking | T-learner trained on conversion | It directly addresses who would have bought anyway, and it is the simpler uplift model to explain. |
| Compared with | Response model, and a random 10,000 | The random comparison uses the lift across all hold-out customers. The memo must say so if the T-learner does not win. |
| Evidence | Hold-out with 2,000 paired bootstrap resamples, plus 10 other seeds | Same approach as Stage 4. |
| Leave-out group | The lowest-ranked 15.625%, with the lift measured in that group | Shows whether leaving them out gives up real conversions. |
| Profile | Average features of the top and bottom groups on the hold-out | Gives a usable description. Hillstrom has no customer IDs, so the memo gives a rule rather than a list of names. |

## Improvements tried after the main results (exploratory)

These were recorded before the improvements were run. They came after the main results were known, so they are exploratory: they cannot replace the headline, and the recorded results stay unchanged.

| Decision | Choice | Reason |
|---|---|---|
| New splits | Split seeds 101 to 110, none used before | Less tied to the runs already seen, though they reuse the same customers, so this is not truly fresh data. |
| Seed averaging | Average 5 model seeds; compare with the first of those seeds alone on the same split | Tests whether averaging reduces instability without changing the training data. |
| Seed-averaging measures | Qini area and uplift in the top 15.625%, on conversion | The main ranking measure and the memo's fixed budget. |
| Best email per customer | Use averaged T-learner scores for each email and choose the higher | Answers which email each person should get. |
| Policies compared | No one, men's to everyone, women's to everyone, best email to everyone, best email only if predicted profitable, and best email vs men's only for the top 15.625% | Each policy is set from training data and the assumed costs, with no choice made on the hold-out. |
| Scoring a policy | Among hold-out customers whose random arm matches the policy's choice, weight each arm's conversion rate by the share assigned to it | Random assignment means those matching customers fairly represent everyone assigned to that arm. |
| Intervals | 2,000 paired bootstrap resamples on split 101, plus averages across all 10 splits | Split 101 was chosen in advance so that no split is picked for looking good. |
| Benchmarks | Men's to everyone; for the fixed budget, men's only to the top 15.625% | An improvement should beat the current recommendation. |

**Memo corrected as a result (2026-11-02):** the T-learner's top group averaged 5.9 added conversions per 1,000 across the new splits, against 6.7 for a random group, compared with 12.5 in the first run. The memo now says to budget as if the 10,000 were random. It also drops its claim that a small test would settle targeting: detecting an extra 5 per 1,000 needs about 51,000 customers across four groups. The earlier figures are unchanged and still reported.

## Environment

- Python 3.14 on Windows, in a project-local `.venv`.
- **`econml` instead of `causalml`:** `causalml` 0.18.0 has no prebuilt wheel for Python 3.14 on Windows, and we install prebuilt packages only.
