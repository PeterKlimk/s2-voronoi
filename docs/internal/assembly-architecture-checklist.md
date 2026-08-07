# Assembly-driven architecture experiment checklist

Status: active  
Opened: 2026-08-07  
Primary workload: native AVX2, 4M uniform points, 16 physical workers, `--no-preprocess`

This checklist tracks architectural experiments selected from production-release assembly. The
objective is to remove publication, control, or live-state work rather than optimize arithmetic.
Run the items in order unless an earlier result exposes a dependency that changes the next gate.
Detailed measurements and rejected implementation notes belong in
[`kernel-optimization-experiment-log.md`](kernel-optimization-experiment-log.md); keep only the
current status and concise disposition here.

## Required gate for every item

- [ ] Inspect native production-release assembly before benchmarking.
- [ ] Preserve exact clipping, ownership, output ordering, diagnostics, and deterministic results.
- [ ] Compare manually copied binaries through equal-length aliases.
- [ ] Check pinned 1M uniform and Fibonacci counters first.
- [ ] Use 4M uniform/16-worker cycles and wall time as the primary acceptance gate.
- [ ] Check 4M Fibonacci and 500k clustered controls for credible candidates.
- [ ] Confirm generic-target behavior and executable codegen isolation where target-specific.
- [ ] Run focused correctness tests, then the full release/checked/clippy/fmt gate before retaining.
- [ ] Record the disposition and remove rejected production code.

## Experiments

### ARCH-ASM-001 — Fuse extraction with dedup preparation

- **Status:** In progress
- **Hypothesis:** `to_vertex_data_full` materializes `CellOutputBuffer`; `emit_cell_output` rereads
  it and separately prepares `scratch.vertex_indices`. An emission-oriented extraction target may
  eliminate an intermediate publication/read pass across two leaves representing about 9.7% of
  whole-run IBS operation samples.
- **First step:** Map which complete-cycle fields `collect_and_resolve` requires before any output
  can be committed. Identify the smallest fused scratch representation without changing edge order,
  keys, fallback extraction, or cross-bin ownership.
- **Reopening boundary:** This is work removal, not another partitioned owner/local emission order.

### ARCH-ASM-002 — Separate gnomonic and fallback stream phases

- **Status:** Queued
- **Hypothesis:** Consume the ordinary stream through a gnomonic-only loop until a fallback request,
  then transfer the remainder to a cold fallback loop. Remove per-neighbor builder-mode/result
  machinery and shrink the dominant hot leaf.
- **Constraint:** The split must cover the stream phase, not merely outline one helper around the
  existing per-neighbor enum branch.

### ARCH-ASM-003 — Combine clipping and termination certification

- **Status:** Queued
- **Hypothesis:** Let the bounded clip operation consume the exact successor/unseen bound and return
  only continue, terminate, or a cold exceptional outcome. Avoid returning `ClipResult` and then
  reopening builder state for boundedness and termination.
- **Constraint:** Preserve the rule that mid-batch termination is checked only after an unchanged
  clip and uses the complete exact remainder bound.

### ARCH-ASM-004 — Split stream consumption permanently at boundedness

- **Status:** Queued
- **Hypothesis:** Boundedness is monotonic. Move from an initial unbounded stream loop into a
  bounded-only loop, removing repeated boundedness checks and bounding-reference tracking from the
  steady state.
- **Constraint:** This must specialize whole-stream control flow; the previously rejected isolated
  bounded/unbounded clip-function split is not a reason to retry the same mechanism.

### ARCH-ASM-005 — Fuse resolution with final emission

- **Status:** Queued
- **Hypothesis:** Avoid filling `scratch.vertex_indices` and then zipping it with the output buffer.
  Resolve and publish finalized endpoints directly if deferred and cross-bin contracts do not
  require complete-cycle index materialization.
- **First step:** Prove which resolution decisions are edge-local and which require the complete
  cycle.

### ARCH-ASM-006 — Separate hot success state from cold diagnostics

- **Status:** Queued
- **Hypothesis:** Represent the ordinary success path with compact state and enter a cold
  continuation only for fallback, allocation failure, or unexpected diagnostics. Reduce hot frame
  size and spills in `clip_batch_source`, `emit_generator_group`, and `emit_cell_output`.
- **Acceptance condition:** Assembly must show a smaller hot frame, fewer hot spills, or removed
  control work. Outlining alone is not sufficient.

## Closed adjacent probes

- Per-attempt diagnostic trace publication batching: rejected; boundary branches and short-batch
  fixed work erased instruction/store savings.
- Packed intersection division, flattened dispatch, transition tables, and isolated bounded clip
  splits: rejected and not part of this queue.
- Partitioned vertex emission: rejected; ARCH-ASM-001 and ARCH-ASM-005 must remove an intermediate
  representation or traversal rather than only reorder the same ownership work.
