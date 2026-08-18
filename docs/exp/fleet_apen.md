# FLEET U — ApEn/SampEn of the byte stream (solo)

| feature | AUROC B | AUROC C | TPR@1e-2 B | TPR@1e-2 C |
|---|---|---|---|---|
| apen_char | 0.610 | 0.604 | 0.041 | 0.058 |
| sampen_char | 0.554 | 0.545 | 0.042 | 0.059 |

## HGB increment (train A, read C)

| arm | n | AUROC C | TPR@1e-2 C | TPR@1e-3 C |
|---|---|---|---|---|
| panel250 | 250 | 0.9932 | 0.933 [0.916,0.947] | 0.778 |
| panel+252 | 252 | 0.9930 | 0.923 [0.905,0.938] | 0.769 |
| apen2 alone | 2 | 0.6558 | 0.060 [0.047,0.076] | 0.004 |
