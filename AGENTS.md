# Codex Working Rules

1. The research goal takes priority over framework generality.
2. Never modify the original AccelWattch TENSOR benchmark in place.
3. Make only one controlled experimental change per variant.
4. Generate variants reproducibly and record the exact diff and hashes.
5. Keep primary power runs and NCU profile runs separate.
6. Preserve all raw outputs; never overwrite or silently delete failed runs.
7. Every derived value must include its source run ID, unit, and formula.
8. Do not equate board energy with intrinsic Tensor Core energy.
9. Do not validate differential subtraction unless common execution activity is demonstrably matched.
10. Distinguish source-confirmed facts, SASS-dependent facts, runtime evidence, and hypotheses.
11. Do not create adapters or abstractions for workloads that are not part of the current study.
12. Prefer small, reviewable changes and one Git commit per controlled modification.
13. Do not make scientific conclusions without explicit researcher review.
14. The current priority is the V100 AccelWattch TENSOR controlled-variant study.
