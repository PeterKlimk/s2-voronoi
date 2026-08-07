# Assembly-driven architecture experiment checklist

Status: completed
Opened: 2026-08-07  
Primary workload: native AVX2, 4M uniform points, 16 physical workers, `--no-preprocess`

This checklist tracks architectural experiments selected from production-release assembly. The
objective is to remove publication, control, or live-state work rather than optimize arithmetic.
Run the items in order unless an earlier result exposes a dependency that changes the next gate.
Detailed measurements and rejected implementation notes belong in
[`kernel-optimization-experiment-log.md`](kernel-optimization-experiment-log.md); keep only the
current status and concise disposition here.

## Required gate for every item

- [x] Inspect native production-release assembly before benchmarking.
- [x] Preserve exact clipping, ownership, output ordering, diagnostics, and deterministic results.
- [x] Compare manually copied binaries through equal-length aliases.
- [x] Check pinned 1M uniform and Fibonacci counters first.
- [x] Use 4M uniform/16-worker cycles and wall time as the primary acceptance gate.
- [x] Check 4M Fibonacci and 500k clustered controls for credible candidates.
- [x] Confirm generic-target behavior and executable codegen isolation where target-specific.
- [x] Run focused correctness tests, then the full release/checked/clippy/fmt gate before retaining.
- [x] Record the disposition and remove rejected production code.

## Experiments

### ARCH-ASM-001 — Fuse extraction with dedup preparation

- **Status:** Completed — retained
- **Result:** On native AVX2, initialize the reusable resolution-index vector inside the existing
  extraction loop and carry it in `CellOutputBuffer`. This removes the separate per-cell
  `clear`/`resize` fill loop from edge collection without changing edge order, keys, fallback
  extraction, or ownership. Generic targets retain the former `EdgeScratch` path.
- **Production gate:** 4M uniform/16-worker cycles -1.00% over 20 physical pairs (18/20 favorable),
  instructions -0.22%, branches -0.98%; twelve three-build wall pairs were neutral. 4M Fibonacci
  cycles -0.44%; 500k clustered was neutral. Native pinned uniform improved strongly; pinned
  Fibonacci cycles were neutral despite lower instructions/branches.
- **Validation:** Full release and checked suites, clippy, formatting, and native wide/scalar
  fingerprints passed. Generic hot-path counters remained neutral after target gating.
- **Reopening boundary:** Further fusion must remove edge collection or final emission traversal;
  do not merely relocate this now-fused index initialization again.

### ARCH-ASM-002 — Separate gnomonic and fallback stream phases

- **Status:** Completed — retained
- **Result:** Select the concrete builder once per batch segment and monomorphize the stream loop for
  gnomonic and fallback builders. A fallback request ends the gnomonic segment, converts the
  builder, and resumes only the unconsumed suffix. Production assembly removes the builder-mode
  discriminant from the per-neighbor backedge while preserving one outer mode check.
- **Production gate:** 4M uniform/16-worker cycles -0.25% over 20 physical pairs (13/20 favorable),
  instructions -0.99%, branches -2.23%, and branch misses -0.43%. Twelve three-build wall pairs
  measured cycles/time -0.19%. Pinned uniform/Fibonacci, 4M Fibonacci, and clustered controls were
  favorable; clustered cycles fell 1.59%.
- **Validation:** Full release and checked suites, clippy, formatting, native wide/scalar matching
  fingerprints, and generic-target uniform/Fibonacci controls passed. The production hot leaf grew
  407 bytes and executable text grew 840 bytes, an accepted cost for removing repeated control.
- **Reopening boundary:** Explicitly outlining the fallback monomorph shrank the hot leaf by 184
  bytes but regressed pinned uniform cycles 0.54%; keep the compiler-integrated split unless a new
  phase representation removes more state or code rather than changing placement alone.

### ARCH-ASM-003 — Combine clipping and termination certification

- **Status:** Completed — rejected
- **Result:** A lazy exact-bound continuation let the gnomonic clip operation return continue,
  terminate, fallback, or failure directly. It preserved unchanged-only certification and shrank
  both packed and shell production leaves, but did not produce a robust primary-cycle win.
- **Production gate:** The packed leaf shrank 159 bytes and pinned uniform instructions fell 0.32%,
  but 4M uniform/16-worker cycles improved only 0.05% over 20 one-build pairs and 0.08% over 16
  two-build pairs. 4M Fibonacci cycles regressed 0.21% (5/12 favorable); clustered cycles were
  neutral. The candidate was removed.
- **Reopening boundary:** Do not merely move the existing unchanged/bounded/bound sequence across an
  interface again. A retry must eliminate certification work or state, not only combine its return
  enum with clipping.

### ARCH-ASM-004 — Split stream consumption permanently at boundedness

- **Status:** Completed — rejected
- **Result:** Separate unbounded and permanently bounded batch segments transferred immediately
  after the first changed clip removed the bounding-reference check from the bounded loop, but
  introduced another monomorphized loop and transition backedge.
- **Production gate:** The packed leaf grew 886 bytes and executable text grew 2,176 bytes. Pinned
  uniform/Fibonacci cycles regressed 0.52%/0.63%, instructions rose 0.13%/0.25%, and branches rose
  about 0.8%. The 4M uniform/16-worker primary regressed 0.60% (2/12 favorable). The candidate was
  removed before unnecessary large cross-distribution runs.
- **Reopening boundary:** Boundedness monotonicity alone does not pay for a segment transition and
  duplicated stream loop. Revisit only if bounded mode can also eliminate substantial polygon state
  or clipping work, not just its repeated predicate.

### ARCH-ASM-005 — Fuse resolution with final emission

- **Status:** Completed — rejected
- **Dependency result:** Incoming earlier-edge checks can patch either endpoint, including vertices
  visited before a later edge, so final owner emission cannot begin until the complete edge cycle
  has resolved. Deferred slots and outgoing checks also require final cell-local indices.
- **Experiment:** The legal partial fusion forwarded ordinary later-cell checks immediately after
  finalizing each endpoint pair, retaining the complete index cycle and overflow fallback. It
  removed the later-check readback traversal but added a cursor test and live first/previous
  endpoints to every vertex.
- **Production gate:** `emit_cell_output` grew 1,261 bytes and executable text grew 2,288 bytes.
  Pinned uniform/Fibonacci cycles regressed 2.96%/2.64%; 4M uniform/16-worker cycles regressed 2.89%
  (0/8 favorable). The candidate was removed.
- **Reopening boundary:** Complete-cycle resolution is mandatory. Revisit only with a representation
  that removes both index materialization and per-vertex cursor control; do not interleave outgoing
  forwarding with the current mixed owner loop again.

### ARCH-ASM-006 — Separate hot success state from cold diagnostics

- **Status:** Completed — retained
- **Result:** Replace three always-live `usize` fallback counters in build/stat state with one
  compact fallback code carried beside the already-required diagnostic trigger, plus one recovery
  flag. Decode the three telemetry counters only at the telemetry publication seam. The state
  machine can record at most one ordinary fallback: it either installs the fallback builder or
  terminates the stream.
- **Production gate:** `emit_generator_group`'s frame fell 16 bytes, the leaf shrank 44 bytes, and
  executable text fell 168 bytes. 4M uniform/16-worker cycles improved 0.62% over 20 pairs (17/20
  favorable) despite instructions +0.36%; 4M Fibonacci improved 0.63% and clustered improved 0.35%.
  Twelve three-build wall pairs were neutral in cycles and 0.24% favorable in elapsed time.
- **Validation:** Full release and checked suites, telemetry-feature unit tests, clippy, formatting,
  native wide/scalar matching fingerprints, and generic uniform/Fibonacci controls passed.
- **Reopening boundary:** Do not re-expand rare fallback accounting into hot per-cell words. Further
  cold separation must remove another live field or frame slot and pass the same latency gates;
  outlining without state removal remains insufficient.

## Closed adjacent probes

- Per-attempt diagnostic trace publication batching: rejected; boundary branches and short-batch
  fixed work erased instruction/store savings.
- Packed intersection division, flattened dispatch, transition tables, and isolated bounded clip
  splits: rejected and not part of this queue.
- Partitioned vertex emission: rejected; ARCH-ASM-001 and ARCH-ASM-005 must remove an intermediate
  representation or traversal rather than only reorder the same ownership work.
