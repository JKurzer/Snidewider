# PERF-RULES — performance work (pinned; keep <=10 lines)

1. Measure, never guess. You don't have the kung fu to guess where a perf issue is. Never.
2. Profile before optimizing: instrument or sample first. No numbers, no changes.
3. Ablation is encouraged: remove or swap ONE piece, measure the delta.
4. Asymptotics on paper != performance on metal: constants, caches, allocators decide.
5. One variable at a time. Record before/after numbers in the commit or docs.
6. No empirical diagnosis in 15 min: stop, escalate with measurements, not theories.
7. No optimization ships without a benchmark that would have caught the original problem.
