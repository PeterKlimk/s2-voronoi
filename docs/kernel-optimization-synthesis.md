## Decision

Advance three normalized candidates to adversarial review:

1. Builder-demanded exact prefixes (`sol-high H1`)
2. Compact deferred high-key overflow (`sol-high H3`, preferred variant of the family shared with `cline H2`)
3. Builder-aware shell-cell rejection (`sol-high H2`)

Do not advance `cline H1` without first clearing its close collision with the rejected directional-certificate branch. Treat `cline H3` as a topology-cache premise, not yet as the cross-query SIMD mechanism described.

## Normalized candidate set

| Family | Report hypotheses | Material distinction |
|---|---|---|
| Demand-sized packed prefixes | `sol-high H1`; explicitly rejected in Cline’s considered list | Changes `PackedQuery`/builder feedback so `emit_run` produces fewer exact slots. |
| Deferred high-key overflow | `sol-high H3`, `cline H2` | Both cap eager `chunk0_keys`; Sol retains block membership/maxima for targeted reconstruction, while Cline relies on the existing full tail rescan after moving the split. These variants must not be conflated. |
| Shell-cell non-cutting rejection | `sol-high H2` | Moves an evolving builder certificate ahead of `ShellFrontier::scan_cell`, avoiding the entire candidate lifecycle for certified cells. |
| Exact batch-remainder termination | `cline H1` | Productionizes the timing-only exact sweep in `audit_directional_batch_skip`. |
| Shared shell topology | `cline H3`; rejected broadly by Sol | Reusing BFS layer topology is separable from batching query dots or cell emission. Only the former clearly avoids stitching changes. |

## Convergence, disagreements, and blind spots

The reviews genuinely converge on the joint-kernel model: fixed exact batches are produced by `PackedQuery::frontier` and `PackedKnnCellScratch::next_chunk`, consumed by `clip_batch_source`, and certified by `Topo2DBuilder::can_terminate`. They also agree that ordinary work is packed-path dominated, shell takeover is rare there, skewed inputs retain excessive high keys, and full lazy reconstruction has already lost badly.

The only real proposal-level convergence is the deferred-overflow family (`sol-high H3`/`cline H2`). It is evidence-based—`unused_chunk0_keys` and clustered key capacity are large—but both reports inherit those facts from the same ledger, so this is shared-premise convergence rather than independent empirical confirmation.

Key disagreements:

- `sol-high H1` versus Cline’s rejection of adaptive chunk sizing is unresolved. Source supports Sol’s mechanical premise: `emit_run` already accepts arbitrary `n_target`, but `PackedQuery::frontier` fixes requests through `PackedNeighborPolicy::{chunk0_size,chunk_size}` and caches the resulting exact frontier. Neither report has the emitted-versus-consumed telemetry needed to decide whether the API change removes meaningful work.
- Cline’s `H2` incorrectly claims the retained kth dot improves the exhaustion certificate. If excess keys exist, the kth retained dot is above the current model threshold `self.thresholds[qi]`; using it as the unseen upper bound is therefore looser, not tighter. It increases the chance of `ensure_tail_directed_for`, even though it can still reduce retained storage.
- Cline’s `H1` is closer to prior work than reported. Branch `d9d0975` implemented batch-level termination attempts at the same pre-/mid-batch sites, guarded by `can_terminate(batch.unseen_bound)`, and lost 3.9–4.5% instructions on ordinary inputs. The exact `candidate_would_be_unchanged` sweep has higher recall than that directional support predicate, but repeats much of the real unchanged-clip classification. This is a changed predicate, not a new cost model.
- On group-shared shells, source partially supports Cline: `DirectedEligibility::cell_mode` is group-invariant outside the center, and `emit_generator_group` could preserve serial cell completion while caching BFS cell lists. However, Cline’s cross-query SIMD dots require speculative work for queries not yet known to enter takeover, or deferred/concurrent construction that reintroduces the stitching problem identified by Sol and the ledger.

Shared blind spots include:

- No per-query retained-count, emitted-depth, consumed-depth, or exact-batch suffix distribution.
- No decomposition of mega/clustered inflation into packed tail, dense band, shell scan dots, pending sorting, and clips.
- Both retention designs understate work shifted into streaming top-k maintenance, metadata writes, or later rescans.
- Neither review establishes whether shell-cell enclosures are tight enough after accounting for cube-face boundaries and clipping tolerance direction.
- Code-layout and hot-state costs remain first-class risks; the ledger contains several structurally cleaner fused paths that lost despite reducing apparent work.

## Prior-work verification

- Demand-sized prefixes: genuinely untested. The rejected partial-selection and 2× whole-sort experiments changed how a fixed request was produced, not whether it was produced.
- Compact overflow (`sol-high H3`): related but not rejected. Full lazy recomputation rebuilt 15.75M keys and lost 28.3% instructions; block membership could materially change rescan scope if deferred blocks are sparse.
- Raw cap/full-tail variant (`cline H2`): much closer to that rejected lazy path. Retaining the nearest budget changes request incidence, but its certificate moves in the unfavorable direction.
- Shell-cell rejection (`sol-high H2`): untested changed-cost variant. Static adjacent-cell cap pruning lost, and `d9d0975` worked after candidate materialization; shell-only whole-cell rejection would operate earlier.
- Exact batch skip (`cline H1`): closely related to rejected `d9d0975`, not merely to the later selected-neighbor classifier experiments.
- Shared shell traversal (`cline H3`): topology caching is untested; per-query shell SIMD lost 6.5–8.5%, while true group-wide traversal/emission remains blocked on stitching redesign.

## Ranking

| Rank | Candidate | Expected removed work | Correctness risk | Breadth | Smallest falsification |
|---:|---|---|---|---|---|
| 1 | Demand-sized prefixes (`sol-high H1`) | Medium ceiling: partition/sort/scatter and abandoned exact slots | Low–medium | Broad packed workloads | Very small telemetry-only census |
| 2 | Compact deferred overflow (`sol-high H3`) | High on clustered pressure: keys, ordering, working set | High | Clustered/mega; gated neutral ordinary | Small histogram and mask simulation |
| 3 | Shell-cell rejection (`sol-high H2`) | High per hit: dots through clips eliminated upstream | Very high geometric proof burden | Narrow shell-heavy regimes | Medium shadow-cell oracle |
| 4 | Exact batch skip (`cline H1`) | Low but potentially broad unchanged-clip residue | Low–medium | Broad | Existing shadow counters, but prior negative dominates |
| 5 | Shared shell topology (`cline H3`, narrowed) | Low–medium traversal bookkeeping; dot work remains | Medium narrowed, high as proposed | Narrow | Existing shell-incidence counters |
| 6 | Raw cap plus full rescan (`cline H2` variant) | High retained-byte ceiling, but work shifts to rescans | High | Clustered/mega | Small per-query depth census |

This ordering is not a confidence average. Candidate 1 leads because it has broad potential and the cheapest decisive falsification. Candidate 2 has stronger evidence of waste but substantially greater implementation and correctness exposure.

## Adversarial-review assignments

- Demand-sized prefixes: attack arbitrary-prefix ordering/bounds, cached-frontier validity, deterministic request policy, and whether builder state can select a useful ask without adding comparable certificate work.
- Compact overflow: attack exact key coverage and tie order, conservative block maxima, dense-band exclusions, streaming metadata cost, and worst-case overflow density. Explicitly reject the claim that Cline’s moved floor tightens termination.
- Shell-cell rejection: attack the cell enclosure, f32 storage/f64 predicate seam, cube-face boundaries, epsilon direction, fallback disablement, and sparse-cell economics.

## Highest-information next experiment

Add a timing-only packed exact-batch production/consumption census—no behavior change—in `PackedQuery::frontier` and `clip_batch_source`. Record emitted slots, visited slots, and abandoned suffix length, split by `PackedChunk0` versus `PackedTail` and first versus later batch. Pair it with existing `select_partition`, `select_sort`, and `select_scatter` timings.

Run the established Fibonacci, uniform, clustered, and mega counter workloads only after the instrumentation exists.

- Success: both Fibonacci and uniform abandon at least 25% of emitted packed slots, average at least four abandoned first-batch slots per packed-finishing cell, and selection/sort/scatter account for at least 5% of packed-kNN time. Advance `sol-high H1`.
- Neutral: 10–25% abandonment, one to three abandoned slots per relevant cell, or a signal confined to one ordinary distribution. Keep demand sizing behind the overflow family.
- Rejection: below 10% abandonment or below one slot per packed-finishing cell, selection/scatter below 2% of packed time, or waste concentrated in shell-dominated cells. Drop demand sizing and make compact-overflow telemetry the next gate.

No files were modified and no benchmarks were run.
