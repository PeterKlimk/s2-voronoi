# Selected-neighbor constraint batches

**Status:** research plan, not an implementation commitment or authoritative queue item

This document explores a fused handoff between directed neighbor selection and per-cell polygon
clipping. The proposal is deliberately specific to the `knn_clipping` backend: it does not make the
general cube-grid index understand gnomonic geometry, and it does not change the public spatial-query
API.

The central hypothesis is that the current internal separation is too fine-grained for the hot path.
Packed neighbor selection produces a sorted batch with useful geometric information, but exposes only
slot ids to cell construction. Clipping then reloads each point, reconstructs a generator-local
bisector, and processes candidates one at a time. A selected-constraint preparation stage could keep
the correctness responsibilities clear while preserving enough batch structure for SIMD,
instruction-level parallelism, and redundant-constraint rejection.

[`work-log.md`](work-log.md) remains the authoritative queue. A candidate from this plan should move
there or into the narrow experiment queue in [`performance.md`](performance.md) only after it has a
scoped implementation and motivating measurements.

## Current seam and lost information

The ordinary path is approximately:

```text
packed coordinate scan
    -> f32 dot-bearing u64 sort keys
    -> selected Vec<u32> slot ids
    -> reload SlotPoint records
    -> scalar generator-local f64 bisector preparation
    -> one-constraint-at-a-time polygon clipping
```

The slot-only boundary discards or hides:

- the `f32` query-candidate dot already used for selection and ordering;
- the selected batch's width, ordering, and natural SIMD grouping;
- the opportunity to gather several `SlotPoint` records together;
- independent work in the generator-local projections for several constraints; and
- the opportunity to classify several constraints against the same current polygon.

This is more accurately a separation of information than a necessary separation of concerns. The
cube grid should continue to own candidate discovery and certified ordering. Cell construction
should continue to own the exact clipping representation and state transitions. Between them, a
consumer-specific stage can turn an exact selected-neighbor window into exact prepared constraints.

## Proposed pipeline

```text
candidate discovery and certified ordering
                  |
                  v
       exact selected-neighbor batch
                  |
                  v
  rolling exact constraint preparation (for example, four at a time)
                  |
                  v
 cheap batch redundancy certificate, then optional exact all-inside classification
                  |
                  v
 ordered scalar polygon mutation for the remaining candidates
```

The important new boundary is between selected-constraint processing and polygon mutation. Bisector
coefficients are independent within a selected window; polygon changes are not.

An illustrative reusable scratch layout is:

```rust,ignore
struct ConstraintWindow<const W: usize> {
    slots: [u32; W],
    indices: [u32; W],
    dots: [f32; W],
    a: [f64; W],
    b: [f64; W],
    c: [f64; W],
    ab2: [f64; W],
    len: usize,
}
```

This is an internal design sketch, not a proposed stable type. A structure-of-arrays window should
make across-constraint SIMD straightforward and avoid widening the long-lived packed key streams.
The first useful width is likely four `f64` lanes. Preparing a whole 16-entry chunk-zero batch is
unlikely to be appropriate: ordinary cells currently consume only about seven candidates, and
termination frequently occurs within the first batch.

## Exact constraint preparation

For generator `g`, tangent basis `(t1, t2, g)`, and selected neighbor `n`, preparation must reproduce
the current exact-promoted-`f64` formula:

```text
a   = -n.t1
b   = -n.t2
c   = scale(|n|^2, |g|^2) * generator_dot_g - n.g
ab2 = a*a + b*b
```

The implementation must preserve the existing FMA policy and lane-wise operation semantics through
the `src/fp.rs` backend seam. The neighbor source's stored `f32` dot is suitable for ordering and
termination bounds; it is not a replacement for the exact `f64 n.g` used by clipping.

The selected-neighbor window should be prepared only after exact prefix selection. Computing or
storing full coefficients during the broad packed candidate scan would multiply retained memory and
perform expensive work for candidates that are never consumed, especially on clustered inputs.

## Batch-level opportunities

### 1. Preserve the selection dot

Carry `(slot, dot)` through exact batch emission, potentially by retaining the existing packed `u64`
key representation in the caller-owned frontier scratch. This can remove the candidate reload and
`dot3_f32` currently used to build a mid-batch termination bound after an unchanged clip.

This is a low-risk control experiment, not the main expected win. It widens the emitted scratch
stream and saves a dot only at particular termination checkpoints.

### 2. Prepare a rolling constraint window

Gather and project a small number of selected neighbors together. Begin with a scalar implementation
that calls the same formula through a new prepared-constraint entry point; this isolates interface
and speculation costs. Then evaluate a four-lane `wide` implementation behind `src/fp.rs`.

A rolling window limits speculative work:

```text
prepare 4 -> classify 4 -> consume survivors in order -> check termination -> prepare next 4
```

Window width is a measured policy, not an architectural constant. Widths one, two, four, eight, and
the full exact batch should be compared using the actual number of prepared-but-unconsumed
constraints.

### 3. Vectorize the radial redundancy certificate

The current bounded-polygon fast path recognizes a sufficient all-inside condition:

```text
c >= 0 && c*c >= ab2 * polygon.max_r2
```

Apply this test across a prepared window before entering the polygon clipper. It shares one polygon
radius and requires no vertex loads. A successful certificate means the ordinary clip would return
`Unchanged`.

The initial implementation should preserve the current activation policy so its effect can be
attributed. Broader use, including small polygons, is a separate experiment.

### 4. Classify constraints across the current polygon

For constraints that fail the radial certificate, reverse the existing SIMD orientation:

```text
current kernel:  one constraint  x four/eight polygon vertices
batch classifier: four constraints x one polygon vertex at a time
```

Accumulate an all-inside bit for each constraint. If a constraint contains every vertex of the
current convex polygon, it is permanently redundant: later clips only shrink that polygon.

Constraints not proven redundant must remain in their original nearest-first order and use the
ordinary clipper. Their earlier classification is only "unknown", not evidence that they will still
change a polygon modified by a preceding candidate.

The classifier can initially use exact vertex evaluation. A later experiment may replace or precede
it with a smaller directional support envelope, informed by the existing timing-only directional
shadow telemetry.

### 5. Reclassify after progress only if measurements justify it

After an active constraint shrinks the polygon, more remaining constraints may become redundant.
Reclassifying the rest of the window could save clipper calls, but it repeats distance work and adds
control flow. Do not include it in the first fused experiment. Record the number of initially
unknown constraints that subsequently return `Unchanged` to estimate its ceiling.

## Correctness and behavioral invariants

The fused path must preserve all of the following:

1. **Exact point model.** Constraint coefficients use the same canonicalized `f32` coordinates
   promoted to `f64`, norm correction, basis, FMA policy, and strict half-plane convention as the
   current gnomonic builder.
2. **Certified candidate order.** Every constraint not proven redundant is presented to the
   mutating clipper in the existing deterministic nearest-first order, including dot/index ties.
3. **Ordered consumption semantics.** Speculatively preparing or classifying a later candidate does
   not make it consumed. Neighbor counters, attempted-slot state, termination checks, trace state,
   and stream advancement proceed in original order and stop at the same logical position.
4. **Termination coverage.** The retained dot for the next selected candidate is still combined
   with `batch.unseen_bound`. Skipping a proven-redundant constraint must perform the same ordered
   opportunity to check termination as an ordinary `Unchanged` result.
5. **Dynamic plane ids.** A prepared constraint does not permanently own a `plane_idx`. Assign the
   current accepted-constraint index immediately before a mutating clip, because preceding
   redundant constraints are not retained.
6. **Fallback safety.** Slots and indices remain available after preparation. If the builder enters
   spherical fallback, discard unconsumed gnomonic coefficients and classification results and
   continue through the existing neighbor-based fallback interface.
7. **Source coverage and deduplication.** Packed chunk zero, packed tail, and shell takeover retain
   their current eligibility and attempted-neighbor rules. The first prototype may optimize packed
   batches only, but the unoptimized sources must remain behaviorally identical.
8. **Seed behavior.** Incoming edge-check seeds remain on their current path initially. They are
   construction invariants expected to be active, not ordinary proximity-tail candidates.
9. **Failure behavior.** Projection, polygon-cap, clipped-away, exhaustion, and repair outcomes must
   remain valid under the existing correctness suites. Any deliberate change in the exact point at
   which a cold fallback triggers requires a separate correctness argument and experiment.

Ideal half-space intersection is order-independent, but the production algorithm's floating-point
state, metadata, fallback transitions, and diagnostics are not assumed to be. Batch preparation may
be speculative; polygon mutation may not be reordered.

## Ownership and module boundary

The coupling should be explicit and narrow:

- `cube_grid` and its packed query code own discovery, eligibility, exact ordering, and unseen
  bounds;
- `cell_build` owns the selected-neighbor window, rolling-consumption state, termination points, and
  fallback transition;
- `topo2d` owns exact coefficient preparation, redundancy predicates, and polygon mutation; and
- `fp` owns any new portable lane operations.

The general `CubeMapGrid` query API should not expose gnomonic types. A backend-private frontier may
emit compact dot-bearing selected entries, while a `topo2d` preparation function consumes their
positions and the current builder projection state. This is deliberate backend fusion, not a new
cross-crate abstraction.

## Experiment sequence

Implement each stage independently enough to attribute its result. Do not bundle all proposed
optimizations into the first comparison.

### Phase 0: establish the opportunity

Add timing-build telemetry for:

- time in coefficient construction versus polygon classification/output;
- exact batches and rolling windows prepared;
- constraints prepared, consumed, and prepared but not consumed;
- radial certificates attempted and accepted;
- exact batch-classifier attempts, candidates tested, and redundancies found;
- initially unknown candidates later reported `Unchanged`; and
- fallback transitions with an outstanding prepared window.

Use counters rather than fine-grained timers inside the production hot loop where possible. A
profiling-only outlined coefficient kernel or fixed-work microbenchmark can provide attribution
without permanently adding per-candidate clocks.

### Phase 1: dot-bearing frontier control

Retain the exact selection dot through batch emission and use it only for existing termination-bound
composition. Confirm unchanged work counters, results, and candidate order. This establishes the
cost of a richer frontier record before adding geometry.

### Phase 2: scalar prepared-window seam

Introduce the selected-constraint window and a prepared-constraint clip entry point, initially with
width one or scalar preparation. The goal is to measure abstraction, scratch-layout, and code-size
cost independently of SIMD savings.

### Phase 3: four-lane preparation

Prepare four constraints with exact lane-wise `f64` operations. Compare rolling widths and record
speculation. Keep ordinary sequential clipping unchanged.

### Phase 4: batch redundancy filters

Add the vectorized radial certificate first. Then add exact across-constraint all-inside
classification as a separate candidate. Preserve ordered termination checkpoints for every skipped
constraint.

### Phase 5: deeper fusion only after attribution

Possible follow-ups include adaptive window width, support-envelope classification, reclassification
after polygon progress, or direct handoff from packed selection storage. Each needs evidence that
the preceding stage leaves a material residual cost.

## Measurement plan

The primary outcome is end-to-end time/cycles, supported by retired instructions, branches, cache
behavior, code size, and exact work counters. Moving work from `clipping` into `packed_knn` is not a
win by itself.

At minimum measure:

- corrected Fibonacci and uniform inputs for the ordinary packed-dominated path;
- clustered and bimodal inputs for retained-key volume and tail behavior;
- `mega` and great-circle inputs for long streams, shell takeover, and high-degree polygons;
- one thread and the normal multithreaded configuration;
- default and high bin counts where scratch replication or instruction-cache pressure may matter;
- portable release and `target-cpu=native` builds if explicit SIMD changes code generation; and
- correctness/adversarial suites plus `--validate` at a practical size.

For orientation, one 100k corrected-Fibonacci, single-threaded timing run on 2026-07-16 reported
166.3 ms internal total, including 40.0 ms packed kNN, 58.5 ms clipping, and 8.6 ms certification.
Cells consumed 7.2 candidates and emitted 6.0 final edges on average. This identifies a meaningful
phase but is not baseline evidence for accepting a design.

Useful acceptance signals are:

- identical or intentionally explained candidate-consumption and final-edge counters;
- no correctness, validation, fallback, or output-fingerprint regression;
- a reduction in total instructions and cycles on ordinary workloads, not merely a phase shift;
- bounded prepared-but-unconsumed work across distributions;
- no material peak-memory or multithreaded cache regression; and
- a clear benefit over code-size and maintenance cost on both native and portable builds.

Reject or narrow a candidate if it widens hot streams without repaying the traffic, speculates over
most of chunk zero, worsens clustered peak storage, perturbs exact clipping signs, or wins only in
microbenchmarks while remaining neutral or negative end to end.

## Open questions

- Does coefficient construction occupy enough of clipping time to repay a new window and dispatch?
- Is four the best preparation width on the supported SIMD backends, or does scalar ILP already
  capture most of the gain?
- How often does the radial certificate reject candidates before the ordinary clipper, separated by
  polygon size and bounded state?
- Does exact across-constraint classification duplicate more distance work than it removes?
- Can the existing dot-bearing `u64` key remain in caller scratch without making shell and packed
  frontier code materially more complex?
- Is one fixed window sufficient, or should long-stream workloads use a different width only after
  measured activation?
- Does the additional fused hot code worsen instruction-cache behavior enough to erase arithmetic
  savings?

The working thesis is that a small selected-constraint stage can recover batch information without
making geometric correctness the neighbor index's responsibility. The first implementation should
test that thesis with a narrow rolling window and exact semantics, then earn deeper fusion through
measurement.
