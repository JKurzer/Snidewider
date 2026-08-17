# FLEET J — function-word pack

| feature | AUROC B | AUROC C | TPR@1e-2 B | TPR@1e-2 C |
|---|---|---|---|---|
| fw_total | 0.502 | 0.507 | 0.029 | 0.025 |
| fw_article | 0.513 | 0.517 | 0.097 | 0.120 |
| fw_preposition | 0.524 | 0.534 | 0.038 | 0.071 |
| fw_pronoun | 0.517 | 0.513 | 0.004 | 0.014 |
| fw_first_person | 0.541 | 0.540 | 0.000 | 0.000 |
| fw_second_person | 0.509 | 0.512 | 0.000 | 0.009 |
| fw_auxiliary | 0.516 | 0.526 | 0.000 | 0.000 |
| fw_conj_coord | 0.505 | 0.510 | 0.039 | 0.053 |
| fw_conj_subord | 0.508 | 0.515 | 0.000 | 0.000 |
| fw_discourse | 0.508 | 0.518 | 0.027 | 0.017 |
| fw_hedge | 0.516 | 0.525 | 0.000 | 0.000 |
| fw_intensifier | 0.560 | 0.512 | 0.000 | 0.000 |
| fw_distinct | 0.566 | 0.571 | 0.059 | 0.110 |

## HGB increment (train A; read C)

| arm | n | AUROC C | TPR@1e-2 C |
|---|---|---|---|
| panel153 | 153 | 0.994 | 0.921 [0.903,0.936] |
| panel+fw166 | 166 | 0.993 | 0.917 [0.898,0.933] |
| fw13 alone | 13 | 0.803 | 0.161 [0.140,0.185] |
