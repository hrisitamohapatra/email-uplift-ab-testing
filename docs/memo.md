# Who to email if we can only email 10,000 customers

**Recommendation:** send the men's email to everyone if the limit can be lifted, because at the assumed costs that was the best strategy tested. If the limit stays, send it to the 10,000 customers the uplift model ranks highest, but budget as if they were a random 10,000, because the ranking is not proven to do better. Do not treat the lowest-ranked 10,000 as customers the email fails to reach.

All money figures use **assumed costs**: $0.10 per contact and $30 margin per conversion. The data contains no real costs.

## What to expect from 10,000 emails

Plan on about **67 added conversions (95% interval 36 to 102), worth about $1,018 ($82 to $2,046) under assumed costs**. This is the result for a random 10,000.

The ranking's result depended heavily on which customers were used to train and test it:

| Runs | Model's top group, added conversions per 1,000 contacts | Random group |
|---|---|---|
| First run | 12.5 (2.9 to 22.2) | 6.7 (3.6 to 10.2) |
| 10 repeat runs, average | 8.9 | 6.7 |
| 10 further runs, average | 5.9 | 6.7 |

## Which 10,000 to email

The model's top group tends to have spent more last year and to buy through several channels or online. These traits come from the first run and describe the group, but they are not proven reasons to target it.

| Customer trait | Top group | Everyone | Lowest group |
|---|---|---|---|
| Spend in the past year | $411 | $243 | $249 |
| Bought men's items in the past year | 70% | 55% | 47% |
| Buys across several channels | 22% | 12% | 14% |
| Buys by phone | 28% | 43% | 56% |
| Lives in a rural area | 6% | 15% | 40% |
| New in the past year | 47% | 51% | 29% |

## Which 10,000 to leave out

Leave out the lowest-ranked group only because the budget requires it. Its lift was 5.9 per 1,000 (-4.0 to 14.6) in the first run and averaged 7.8 across repeat runs, above the 3.3 needed to break even at assumed costs. We found no evidence that the email fails to work for these customers.

## Risks

1. **Targeting is unproven.** The model's advantage disappeared in later runs, and a simple model of who is likely to buy did about as well. Testing it properly is not cheap: detecting an extra 5 conversions per 1,000 would take about 51,000 customers across four groups.
2. **The profit depends on assumed costs.** Emailing everyone only pays if margin is about 150 times the contact cost or more. The assumed figures give 300 times. Please replace them with real costs.
3. **The data is limited.** It comes from one 2008 test that measured two weeks. Long-term effects such as unsubscribes are not included, and today's customers may respond differently.
