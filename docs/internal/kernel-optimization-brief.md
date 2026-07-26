# Kernel Optimization Brief

**Status:** analysis handoff; propose and measure before editing

This brief scopes a deeper optimization pass over the computation kernels. The desired changes are
algorithmic or structural rather than local source reshuffling. Preserve correctness and the
existing performance frontier: reduced work is valuable, but a cleaner-looking loop that worsens
cache behavior, code layout, or cycles is not.

Use [`kernel-optimization-agent-prompt.md`](kernel-optimization-agent-prompt.md) as the identical
entry point for each independent fishing pass. It enforces read-only work and a common response
schema so reports from different models can be compared directly.

## Multi-model workflow

### 1. Independent fishing

Run three to five models against the same frozen revision, prompt, and minimal source packet. Each
model works independently and must not read another model's report. The agents return analysis only:
no patches, source edits, commits, or implementation-shaped pseudo-diffs.

The common schema is important, but agreement is not automatically correctness. Similar models may
share the same blind spot or be anchored by the same source organization. Preserve dissenting
hypotheses and explicit uncertainty for synthesis.

### 2. Optional diversity round

After the blind pass, use narrowly assigned reviewers only where the initial reports are thin. Useful
lenses include traversal/batching, clipping or constraint representation, data movement/cache
behavior, numerical geometry, and scheduling/stitching. These are supplements, not replacements for
the comparable broad reports.

### 3. Synthesis

A fresh reviewer receives all independent reports plus this brief and the performance ledger. It
should:

1. normalize differently worded versions of the same mechanism;
2. distinguish genuine independent convergence from a shared premise;
3. map proposals to accepted, rejected, or untested prior experiments;
4. identify contradictions and the observation that would resolve each one;
5. rank hypotheses by expected work reduction, correctness risk, experiment cost, and breadth of
   workload benefit; and
6. recommend no more than three candidates for adversarial review.

The synthesis must not average confidence scores into false precision. A minority proposal with a
clear mechanism and cheap falsification test can outrank a vague consensus.

### 4. Adversarial review

Give the shortlisted candidates to a reviewer whose job is to reject them. Check conservative
bounds, directed ownership, deterministic ordering, topology and rounding, fallback behavior,
state-size/code-layout costs, and distribution-specific regressions. Record whether each concern is
a proof obligation, a measurable risk, or merely speculative.

### 5. Experiment selection

Only after synthesis and adversarial review should an implementation agent receive one hypothesis.
The assignment should define one predicted reduced-work mechanism, one smallest falsifying
experiment, semantic gates, counter workloads, and an explicit revert condition. Do not bundle
multiple hypotheses into the first patch.

## Central framing

The primary kernel is the feedback loop between neighbor discovery, half-space clipping, and the
termination certificate. Treating nearest-neighbor search and clipping as independent phases hides
the most interesting optimization opportunities:

1. the neighbor path produces a descending exact batch or a conservative unseen-neighbor bound;
2. each accepted neighbor clips the evolving cell;
3. the changed polygon alters the termination threshold; and
4. termination may discard the remainder of the batch and all later neighbor work.

The first question should therefore be how to reduce total candidate production, ordering,
materialization, and clipping—not how to shave instructions from one isolated primitive.

## Minimal source packet

Read these four files first. Together they contain roughly 2,800 lines and most of the useful
computation.

1. [`cell_build/run.rs`](../src/knn_clipping/cell_build/run.rs)
   - Start with `clip_batch`, `clip_batch_source`, `consume_stream`, and `build_cell_into`.
   - This is the joint kernel: ordered candidates become clips, and the evolving builder decides
     whether the neighbor stream can terminate.
2. [`packed_knn/scratch/prepare.rs`](../src/cube_grid/packed_knn/scratch/prepare.rs)
   - Start with `PackedKnnCellScratch::prepare_group_directed`.
   - This owns group-level range discovery, security thresholds, dense-band handling, and the SIMD
     center/ring scans.
3. [`packed_knn/scratch/emit.rs`](../src/cube_grid/packed_knn/scratch/emit.rs)
   - Start with `ensure_tail_directed_for`, `next_chunk`, and `emit_run`.
   - This owns lazy tail materialization and the partition/sort/scatter path that turns candidate
     keys into exact descending batches with conservative remainder bounds.
4. [`topo2d/clippers/small.rs`](../src/knn_clipping/topo2d/clippers/small.rs)
   - Start with `eval_small_dists`, `clip_small_ptr`, and `clip_small_ptr_d`.
   - These are the dominant N=3–5 polygon-classification and output kernels.

If context is severely constrained, omit `emit.rs` initially. Do not omit `cell_build/run.rs`: it
contains the coupling that determines whether an NN or clipping proposal can reduce end-to-end
work.

## Focused supporting context

Read these regions only when a proposal crosses the corresponding boundary:

- [`driver.rs`](../src/knn_clipping/driver.rs): the same-grid-cell group preparation and
  `emit_generator_group` loop. This explains what work is currently shared across generators and
  why cells within a shard are ordered.
- [`query/stream.rs`](../src/cube_grid/query/stream.rs): `DirectedNeighborStream::frontier` and
  `advance_frontier`. This is the packed-to-shell protocol and the ownership of cached exact or
  bounded frontiers.
- [`topo2d/builder/clip.rs`](../src/knn_clipping/topo2d/builder/clip.rs): the `Topo2DBuilder`
  dispatch and the gnomonic/fallback clip implementations. Read this before changing builder state,
  fallback entry, or clip-result semantics.
- [`fp.rs`](../src/fp.rs): the explicit SIMD backend seam. Read this only for a proposal that
  changes dot-product, threshold-mask, or signed-distance arithmetic.
- [`policy.rs`](../src/policy.rs) and [`tolerances.rs`](../src/tolerances.rs): read before changing a
  budget, threshold, or comparison. Policy and numerical slack are deliberately separate.
- [`query/shells.rs`](../src/cube_grid/query/shells.rs): read for clustered, mega, or group-wide
  shell-takeover work. Ordinary well-distributed inputs mostly finish in the packed path.

Do not begin with `compute.rs`, assembly, reconciliation, or local rebuilding unless measurements
show that a proposed kernel change shifts work into those phases.

## Contracts that proposals must preserve

- Input sites are canonical unit-sphere f32 positions. Geometric predicates and clipping use f64
  where required by their error model.
- Every exact neighbor batch is ordered so its next dot and `unseen_bound` form a sound termination
  certificate. A bound must cover both the unconsumed batch suffix and every later frontier.
- Directed same-bin eligibility is part of deterministic ownership, not merely a duplicate filter.
- Shell takeover re-covers points already served by the packed path and therefore deduplicates
  attempted slots.
- Earlier cells can emit live edge checks consumed by later cells in the same shard. Group-wide
  traversal or cell batching must redesign this scheduling/stitching dependency rather than assume
  the cells are independent.
- Clip comparisons, epsilon direction, interpolation, polygon winding, neighbor attribution, and
  fallback handoff are correctness-sensitive. Bit changes require topology and adversarial testing,
  not just a microbenchmark.
- Most cells spend most of their clipping life at N=3–5. Large-polygon work and fallback paths matter
  on adversarial distributions but should not enlarge the ordinary live state without evidence.
- Determinism across thread counts, bin counts, and supported SIMD/scalar configurations is part of
  the contract.

## High-value architectural questions

Investigate these as hypotheses, not predetermined solutions:

1. Can the evolving clipping state request candidates more adaptively than the current fixed-size
   exact batches?
2. Can a sound certificate avoid materializing, partitioning, or sorting candidates that will be
   abandoned after one or two unchanged clips?
3. Can same-grid-cell generators share more range, dot-product, threshold, or traversal work without
   expanding hot per-cell state or violating directed ownership?
4. Is a different constraint or polygon representation materially better for the common N=3–5
   lifecycle, including termination and extraction—not just the isolated clip instruction count?
5. Can constraint ordering reduce the number of changed clips or reach a tight termination bound
   earlier while retaining determinism?
6. Can dense or shell-heavy workloads use a group-level traversal if edge-check production and
   consumption are scheduled differently?
7. Is there a useful dual or incremental formulation that avoids repeatedly solving nearly the same
   local geometry for neighboring generators?

For every proposal, identify which count it should reduce: candidate dot evaluations, retained
keys, partition/sort work, emitted candidates, changed clips, unchanged clips, shell layers, or
fallback incidence. A proposal without a predicted work reduction is probably another micro/codegen
experiment.

## Prior-work guardrail

Read [Source-pinned performance decisions](../performance.md#source-pinned-performance-decisions)
before implementing local changes. That ledger records accepted and rejected experiments involving:

- lazy candidate and tail materialization;
- shell specialization and group batching;
- small-clip dispatch, transition lookup tables, distance evaluation, and interpolation scheduling;
- builder layout, cached norms/radii, constraint storage, and explicit capacity checks;
- forced inlining/outlining and compile-time specialization; and
- several apparently cheaper fused passes that lost on cache behavior or code layout.

Do not treat a rejected source shape as permanently impossible. Revisit it only when a broader
algorithmic change alters its cost model, and state what changed.

## Expected first deliverable

The independent agents use the exact response format in
[`kernel-optimization-agent-prompt.md`](kernel-optimization-agent-prompt.md). In substance, each
report must explain the joint kernel, rank three to five structural hypotheses, identify the work
each would remove, state risks and unknowns, and define the smallest falsifying experiment. Prefer
one independently measurable mechanism at a time.

## Measurement and acceptance

The machine is shared, so use wall clock as supporting evidence rather than the first gate.

- Build optimized native comparison artifacts with `./scripts/bench_build.sh`.
- Use interleaved `perf stat` runs through `./scripts/bench_perf.sh`.
- Compare retired instructions and branches first, then paired cycles. Treat cache events as
  diagnostic unless they are large and repeatable.
- Cover at least Fibonacci and uniform ordinary inputs plus clustered or mega for density contrast.
- Use detailed `timing` output to confirm that the intended subphase changed and work was not merely
  displaced.
- Run `cargo test --release`, `cargo test --profile checked`, and the relevant adversarial and
  backend comparisons for any retained implementation.

An ordinary-path win that materially regresses clustered/mega work, or the reverse, needs a robust
workload classifier with a clear cost model. Do not add a distribution-sensitive heuristic solely
to rescue a mixed benchmark result.

## Agent entry point

Point every independent reviewer at
[`kernel-optimization-agent-prompt.md`](kernel-optimization-agent-prompt.md). Do not add leading
suggestions or model-specific hints during the blind pass; the shared prompt and frozen revision are
the experimental control.
