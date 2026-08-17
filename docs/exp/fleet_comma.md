# FLEET M — comma-context pack

| feature | AUROC B | AUROC C | TPR@1e-2 B | TPR@1e-2 C |
|---|---|---|---|---|
| oxford_rate | 0.586 | 0.514 | 0.027 | 0.030 |
| splice_rate | 0.518 | 0.510 | 0.000 | 0.000 |
| comma_and_rate | 0.580 | 0.524 | 0.025 | 0.029 |
| comma_but_rate | 0.517 | 0.513 | 0.000 | 0.000 |
| comma_because_rate | 0.514 | 0.502 | 0.000 | 0.000 |
| comma_then_quote | 0.527 | 0.501 | 0.000 | 0.008 |
| period_in_quotes | 0.584 | 0.641 | 0.000 | 0.000 |
| comma_then_capital | 0.518 | 0.510 | 0.000 | 0.000 |
| comma_then_lower | 0.518 | 0.510 | 0.000 | 0.000 |
| end_variety | 0.566 | 0.552 | 0.030 | 0.018 |
| sent_comma_mean | 0.515 | 0.527 | 0.029 | 0.015 |
| sent_comma_stdev | 0.619 | 0.602 | 0.055 | 0.000 |

## HGB increment (train A, read C)

| arm | n | AUROC C | TPR@1e-2 C |
|---|---|---|---|
| panel157 | 157 | 0.993 | 0.915 [0.896,0.931] |
| panel+comma169 | 169 | 0.994 | 0.906 [0.886,0.923] |
| comma12 alone | 12 | 0.717 | 0.089 [0.073,0.108] |
