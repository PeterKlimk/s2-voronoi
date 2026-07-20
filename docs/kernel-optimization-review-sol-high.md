### 0. Review metadata

- Git revision: `badfa647bf5c9688715b7693d15aace3bdef67c6`
- Branch: `agent/code-quality-plan`
- Reviewer/model label: `/root/kernel_fishing_sol` (Codex/Sol; exact deployed model version not exposed)
- Supporting files read beyond the minimal packet:
  - `AGENTS.md`
  - `docs/kernel-optimization-brief.md`
  - `docs/performance.md`
  - `docs/packed-preparation-inventory.md`
  - `src/knn_clipping/driver.rs`
  - `src/knn_clipping/cell_build/run/frontier.rs`
  - `src/cube_grid/query/stream.rs`
  - `src/cube_grid/query/shells.rs`
  - `src/cube_grid/packed_knn/mod.rs`
  - `src/cube_grid/packed_knn/scratch.rs`
  - `src/cube_grid/packed_knn/timing.rs`
  - `src/knn_clipping/topo2d/builder/clip.rs`
  - `src/knn_clipping/topo2d/builder/projection.rs`
  - `src/policy.rs`
  - Read-only Git history for directional-certificate commits, especially `d9d0975`
- No other reviewers’ reports were inspected.
- No benchmark campaign was run.
- The worktree was not modified and remained clean.

### 1. Kernel model

`build_cells_sharded_live_dedup` prepares one complete same-grid-cell generator group before emitting any cell in that group. `prepare_group_directed` collects the center plus neighboring grid-cell ranges, excluding same-bin-earlier ranges from directed work. It computes a per-query security floor covering everything outside the scanned neighborhood, derives a count-model high threshold, scans the directed center triangle and eligible ring ranges, and retains descending-key candidates above that threshold in per-query `chunk0_keys`. Center-tail candidates are only counted; tail keys are reconstructed on demand.

Each cell first clips incoming edge checks from earlier same-shard cells. `DirectedNeighborStream` then exposes either an exact ordered batch, a conservative bounded-but-unmaterialized frontier, or exhaustion. `PackedQuery` obtains exact prefixes through `next_chunk`/`emit_run`: the remainder is partitioned as necessary, the prefix is sorted, and slots are scattered. The fixed policy requests 16 slots for chunk zero and 8 thereafter.

`clip_batch_source` consumes candidates in descending-dot order. Shell candidates are insertion-deduplicated because takeover re-covers packed-served points; packed candidates are simply marked. Small bounded polygons normally use the N=3–5 SIMD clippers. A changed clip invalidates the cached termination threshold. An unchanged clip triggers a certificate check against the exact next-dot/beyond-batch maximum. After a batch, the next exact or bounded frontier can also be certified before consumption.

Gnomonic termination converts the current polygon’s conservative `max_r2` into a raw-dot threshold. It is unavailable while synthetic bounds remain and is always false in spherical fallback. If packed high and optional tail coverage cannot certify completion, the stream switches to `ShellFrontier`, which starts a Chebyshev BFS from the home cell, recomputes point dots, materializes and incrementally orders each layer, and relies on attempted-slot filtering during clipping.

Unverified premises include the distribution of consumed positions within packed batches, per-query high-key lengths rather than aggregates, and whether conservative whole-cell directional bounds are tight enough on actual shell frontiers.

### 2. Ranked hypotheses

| Rank | ID | Proposal | Primary reduced work | Expected regimes | Impact | Evidence | Risk | Experiment cost | Prior-work overlap |
|---|---|---|---|---|---:|---:|---:|---|---|
| 1 | H1 | Builder-demanded exact-prefix sizing | Partition/sort/scatter and emitted-but-unconsumed slots | Best on Fibonacci/uniform; possible clustered benefit; little shell effect | 4 | 3 | 2 | S | related |
| 2 | H2 | Shell-cell directional rejection before point scanning | Candidate dots, pending keys, sorting, emission, and clipping | Mega/great-circle and shell-heavy clustered; gated neutral on ordinary inputs | 5 | 2 | 4 | M | related |
| 3 | H3 | Capped exact high prefix plus compact deferred overflow | Retained `u64` keys, peak live state, and sorting of unused high candidates | Clustered/mega high-pressure groups; ordinary path should remain unchanged | 4 | 3 | 4 | M | related |

### 3. Hypothesis details

#### H1 — Builder-demanded exact prefixes

- **Mechanism:** Let the consumer request a prefix length reflecting current builder state instead of having `PackedQuery` always request 16 and then 8. An unbounded cell can request only enough candidates to plausibly become bounded; a bounded cell near certification can request one to four. Larger requests remain available after repeated progress. `emit_run` already accepts a runtime `n_target` and constructs a sound bound for whatever exact prefix it emits.
- **Current work removed:** Partitioning, sorting, and scattering slots that termination abandons partway through a fixed batch. Candidate dot evaluation and initial high-key materialization do not disappear under this hypothesis.
- **Affected symbols:** `PackedQuery::frontier`, `PackedNeighborPolicy`, `PreparedPackedGroup::next_chunk`, `PackedKnnCellScratch::next_chunk`, `emit_run`, `consume_stream`, `clip_batch_source`, `Topo2DBuilder::{is_bounded, can_terminate}`.
- **Correctness proof obligations:** Every requested prefix must remain globally descending with deterministic slot tie-breaking. Its `unseen_bound` must cover both the unselected remainder and later stages. Cached-frontier behavior must remain valid for arbitrary request lengths. Smaller batches must not alter directed ownership, fallback entry, or the set/order of constraints actually consumed before termination.
- **Empirical risks:** Repeated `select_nth_unstable` calls can cost more than sorting a 9–16-key remainder; the ledger already shows that excessive partitioning is branch-heavy. Very small requests can also add stream/frontier dispatch and certificate checks.
- **Expected workload behavior:** Fibonacci and uniform cells that close during chunk zero are the main target. Clustered cells benefit only if they still abandon substantial prefixes; cells that consume long runs may regress. Mega cells dominated by shell takeover see little benefit unless their packed prelude also oversupplies.
- **Prior-work relationship:** Related. Fixed prefix selection, the 2× whole-sort rule, and a partial-selection rewrite have been measured, but builder-controlled demand sizing has a different cost model: it aims to avoid producing a prefix rather than change how the same prefix is selected.
- **Smallest falsifying experiment:** Add timing-only counters for exact packed slots produced and consumed, split by chunk-zero/tail and first/subsequent batch, plus a histogram of the unused suffix at termination. Reject before changing behavior if oversupply is normally below one slot or is confined to workloads where selection time is negligible.
- **Telemetry:** Existing `select_partition`, `select_sort`, `select_scatter`, `unused_chunk0_keys`, `candidate_work`, and cell-stage counters provide context. Minimum new counters: `packed_batch_slots_emitted`, `packed_batch_slots_consumed`, and unused-suffix histograms by packed stage.
- **Why this is not merely a micro-optimization:** It changes the feedback protocol between clipping state and neighbor production so unnecessary ordered frontiers are never produced.

#### H2 — Shell-cell directional rejection

- **Mechanism:** During shell takeover, expose each traversed grid cell to a builder-owned conservative predicate before `scan_cell`. Bound every possible site in the cube-map cell and prove that its bisector contains every current polygon vertex. A certified cell remains part of BFS transit but contributes no point dots or pending keys. Since later clipping only shrinks the polygon, a certified non-cutting cell remains redundant.
- **Current work removed:** Per-point dot evaluation, `PendingKey` materialization, layer partition/sort work, emitted candidates, attempted-set accesses, and unchanged clips for all eligible sites in certified cells.
- **Affected symbols:** `ShellFrontier::{build_pending, scan_cell}`, `DirectedNeighborStream::frontier`, `CubeMapGrid::cell_min_dist_sq` or a corresponding conservative cell-cap primitive, `GnomonicBuilder`/`Topo2DBuilder` directional support state, and the shell portion of `consume_stream`.
- **Correctness proof obligations:** The cell enclosure must contain every canonical stored point despite f32 norms and cube-face boundaries. Its world-space support inequality must conservatively imply the clipper’s nominal all-inside result, including tolerance direction. Skipping must preserve directed eligibility and traversal discovery. The predicate must be disabled for unbounded and fallback builders unless separately proved.
- **Empirical risks:** Adjacent cell caps may be too broad, as earlier static cap pruning found. Evaluating several polygon vertices per cell may cost more than scanning sparsely populated cells. Adding builder interaction to traversal can enlarge hot state or damage code layout.
- **Expected workload behavior:** Ordinary Fibonacci/uniform normally never enters shell takeover, so a takeover-only and initially `grid_rebuilt`-gated form should be neutral. Mega and great-circle workloads have the strongest opportunity because candidate inflation and shell cost are large. Non-rebuilt clustered inputs could benefit, but broad cells and distribution-specific codegen are regression risks.
- **Prior-work relationship:** Related. Static per-ring-cell cap pruning lost because adjacent caps rarely pruned, and candidate-wise directional termination added 3.9–4.5% instructions on Fibonacci/uniform. The changed cost model is shell-only, dense-gated, and whole-cell rejection before dots rather than per-candidate certification after materialization. The ledger explicitly identifies dense-gated directional termination as untried.
- **Smallest falsifying experiment:** Add a timing-only shadow predicate at shell-cell scan boundaries. Continue scanning normally, but record certified cells and eligible resident slots, and verify every resident candidate with the exact non-mutating all-vertices test in checked/timing runs. Reject if certified resident slots are a small fraction of shell-scanned slots or if cap tests approach the cost of scanning them.
- **Telemetry:** Existing `shell_layer_batches`, `shell_layer_slots`, `shell_layer_prefix_consumed`, `candidate_work`, directional-shadow counters, and `grid_rebuilt`. Minimum new counters: cell-cap tests/hits, eligible resident slots in hit cells, and exact-oracle false positives.
- **Why this is not merely a micro-optimization:** It moves the clipping certificate across the traversal boundary and eliminates entire cell scans and their downstream candidate lifecycle.

#### H3 — Compact deferred high-key overflow

- **Mechanism:** For high-pressure groups, retain only a bounded nearest exact high-key prefix per query. Represent further above-threshold candidates compactly by range/block membership masks plus conservative maxima. Most queries finish from the prefix; a query that needs overflow reconstructs exact keys only from marked blocks. This avoids the previous lazy approach’s full neighborhood rescan while keeping ordinary groups on the current layout.
- **Current work removed:** Most unused 8-byte high keys, their allocator capacity, and sorting/partitioning of overflow never requested. Requested overflow pays targeted dot reconstruction only for marked candidate blocks.
- **Affected symbols:** `PackedKnnCellScratch` high-candidate storage, `prepare_group_directed` center/ring extraction, `PreparedPackedGroup`, `PackedKnnCellScratch::next_chunk`, `ensure_tail_directed_for`, `emit_run`, `PackedKnnTimings`, and pressure policy near `MAX_AGGREGATE_CANDIDATE_WORK`.
- **Correctness proof obligations:** The retained prefix must be the deterministic top keys under the exact existing total order. Every omitted high candidate must be represented once and reconstructed before a lower candidate can be emitted. Overflow maxima and stage bounds must cover all deferred candidates. Dense-band, directed-center, same-bin-earlier, and padded SIMD-tail semantics must remain exact.
- **Empirical risks:** Maintaining a bounded top set and masks can add comparisons and scattered stores during the already-efficient SIMD scan. Marked blocks may be dense enough that reconstruction repeats nearly all dots. Additional representations can worsen cache locality despite using less capacity.
- **Expected workload behavior:** The ledger’s 86.4% unused high keys and multi-megabyte clustered capacities indicate a substantial ceiling. Clustered and mega groups near aggregate-work limits are the target. Fibonacci/uniform should retain the present eager-key path because their peak storage is small and their current scan is efficient.
- **Prior-work relationship:** Related. Full lazy recomputation of retained high keys was previously tried and regressed roughly 28–30% because later requests rescanned 15.75M keys. This proposal changes that cost model by retaining compact candidate membership and rescanning only marked blocks, under a high-pressure gate. It is also distinct from the rejected partial-selection rewrite.
- **Smallest falsifying experiment:** Instrument per-query high-key count, maximum consumed high-key position, number of candidate blocks containing deferred highs, and simulated mask bytes for several prefix caps. Reject if compact metadata plus targeted block-rescan dots do not substantially undercut current key bytes, or if many high-pressure queries consume deep overflow.
- **Telemetry:** Existing `chunk0_keys`, `unused_chunk0_keys`, `packed_keys_materialized`, `packed_key_capacity_peak`, aggregate-work fallback incidence, and select-phase timings. Minimum new counter: a per-query high-count/consumed-depth histogram and simulated marked-block count.
- **Why this is not merely a micro-optimization:** It changes high-candidate ownership and lifetime, reducing the hot retained working set and deferring ordering work until demonstrated demand.

### 4. Considered but rejected

- **Static per-ring-cell cap pruning:** Already measured as a net loss because adjacent caps rarely prune. Without evolving builder state or shell-only gating, the cost model is unchanged.
- **Packed-to-shell attempted-slot filtering:** Already measured with low duplicate coverage and extra branching. Filtering by the packed coverage threshold describes essentially the same duplicate set after packed exhaustion.
- **Full lazy recomputation of `chunk0_keys`:** Previously regressed instructions, branches, and cycles by roughly 28–65% because requested queries rescanned too much candidate space. H3 is retained only because compact overflow membership changes that rescan cost.
- **Ungated candidate-wise directional termination:** The archived implementation added 3.9–4.5% instructions on ordinary workloads. Per-candidate support work after materialization does not remove enough upstream work.
- **Group-wide shell takeover batching alone:** Same-bin cells are serialized by live outgoing/incoming edge checks. Sharing traversal without redesigning stitching and scheduling assumes false independence.
- **Checking `can_terminate` after every changed clip:** A changed clip invalidates the cached radius-derived threshold, so this forces its square-root/trigonometric rebuild repeatedly, mainly to save the first unchanged clip after final progress. The likely reduction is too small relative to the added work.
- **Reordering a batch by cutting severity:** Determining severity requires clip-like classification, and non-dot order complicates exact suffix bounds. It risks moving work from clipping into scheduling rather than removing it.
- **Small-clip local rewrites:** Scalar N=3 distances, transition tables, interpolation reshuffling, and deferred radius maintenance are ledger collisions without a changed cost model. They do not reduce candidate or clip counts.

### 5. Cross-cutting unknowns

- Exact packed batch production versus prefix consumption is not measured. `unused_chunk0_keys` includes never-emitted retained keys and cannot establish H1’s ceiling.
- Aggregate high-key telemetry does not reveal per-query high counts, consumed depth, or whether pressure is concentrated in a few pathological queries. This materially affects H1 and H3.
- No current measurement reports whether a conservative whole-grid-cell directional bound certifies actual shell cells, only candidate-wise exact/support shadow results. This controls H2.
- Selection timings lack size/cardinality distributions, making it unclear whether H1 would remove expensive sorts or only cheap sorting-network cases.
- Peak key capacity is allocator capacity retained across groups, not purely live payload. H3 must distinguish reduced active traffic from capacity that remains resident in reusable inner vectors.
- The relative contribution of shell-scanned dots, pending-key ordering, attempted-set duplicates, and clipping within mega/great-circle candidate inflation is not separated finely enough to predict H2’s end-to-end ceiling.

### 6. Recommended first experiment

Choose H1, builder-demanded exact-prefix sizing. It has the best information value because its ceiling can be measured with a tiny timing-only change, its correctness mechanism already exists for arbitrary `n_target`, and the result directly says whether a larger stream/builder API change is warranted.

- **Reduced-work prediction:** Many packed-finishing cells terminate after consuming only a prefix of the fixed 16-slot first batch or 8-slot later batches. Demand sizing should eliminate a material fraction of packed selection, sorting, and scattering while leaving candidate scans and key retention unchanged.
- **Smallest implementation boundary:** Add timing-only counters in `clip_batch_source` and the packed frontier path for slots emitted and actually consumed, split into chunk-zero versus tail and first versus later exact batches. Do not change batch sizes in this first experiment.
- **Semantic validation required:** Counters must compile out with the zero-sized timing backend, must not alter frontier advancement, and must preserve byte-identical outputs. Run existing exact-bound, stream, packed-ordering, backend-fingerprint, correctness, and adversarial tests when implemented.
- **Workloads and counters:** Use ordinary 500k Fibonacci and uniform as the primary target, plus 100k clustered and mega counter-regimes. Record emitted slots, consumed slots, termination position, batch count, `select_partition`, `select_sort`, `select_scatter`, `unused_chunk0_keys`, `candidate_work`, and packed-versus-shell stage.
- **Success criteria:** At least roughly four unused emitted slots per packed-finishing cell, or at least 25% of emitted packed slots abandoned, with the oversupply broad across Fibonacci and uniform and selection/scatter taking measurable packed time.
- **Neutral criteria:** One to three unused slots per relevant cell, or a signal limited to clustered inputs where H3 offers a more direct storage reduction. H1 would remain possible but fall below H3.
- **Rejection criteria:** Fewer than one unused emitted slot per packed-finishing cell, most cells consuming complete batches, or oversupply occurring almost entirely in shell-dominated cells where packed selection is negligible.
- **Ranking update:** A strong ordinary-path oversupply signal keeps H1 first and supplies concrete demand sizes for a behavioral probe. A weak signal moves H3 to first because retained unused keys, not emitted prefixes, are then the better-supported waste source. If oversupply is strong only in mega/shell-heavy workloads, H2 becomes first because removing upstream shell scans has the larger plausible ceiling.
