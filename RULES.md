# RULES — ai-text-detection (the pinned floor; keep <=20 lines)

1. Raw data is immutable: `data/raw/` is write-once; derived stuff goes to `data/derived/`.
2. Every detector ships with an eval or it doesn't ship.
3. Report TPR at fixed FPR (default 1e-3) with CIs — never bare accuracy.
4. No leakage: split by source document, never by chunk. Paraphrase sets stay out of tuning.
5. Features are pure functions; caching fine, hidden state not.
6. Baseline before neural. If the baseline ties, the neural one goes.
7. The paper's claims get extracted into a repro plan before we build on them.
8. Small diffs. Grep before read. One concept per read.
9. Edited this file? Re-pin it: `python %USERPROFILE%\.code_puppy\skills\context-pins\scripts\pin.py pin RULES.md`
