# FINAL EXAM — holdout fold, contacted once
Protocol: A trains detectors, B trains meta, C sets threshold, holdout answers once.
Threshold (C-human max + eps): 0.999615
Holdout: 11371 human, 20000 AI (sampled from 386614)

## Headline
- AUROC: **0.7144**
- achieved FPR: **0.00053** (6/11371; target <= 0.001)
- TPR at that threshold: **0.0099** [0.0086, 0.0114]
- gate: AI fire rate 0.0727, human fire rate 0.00554

## Per-domain TPR
- abstracts: 24/2679 = 0.009
- books: 39/2665 = 0.015
- news: 80/2601 = 0.031
- poetry: 1/2575 = 0.000
- recipes: 1/2779 = 0.000
- reddit: 5/2638 = 0.002
- reviews: 3/1388 = 0.002
- wiki: 45/2675 = 0.017
