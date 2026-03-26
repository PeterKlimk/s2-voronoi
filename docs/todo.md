# TODO

## Heuristics

- Revisit packed `r=2` expansion activation heuristics.
  The current design is intentionally simple: expansion is enabled by default and triggers whenever a query exhausts packed `r=1` without proving termination. Bench results suggest this is worthwhile on rougher distributions, but it can be slightly negative on smoother `--lloyd`-style inputs. If heuristic gating is added later, keep it in the neighbor-source layer rather than spreading policy across `process_cell`.

- Consolidate other heuristic decisions in one place.
  There are now multiple heuristic knobs across packed kNN, termination cadence, and fallback behavior. Future work should track these together rather than adding one-off local rules.
