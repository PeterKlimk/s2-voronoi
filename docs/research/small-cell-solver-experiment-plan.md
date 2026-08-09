# Small-Cell Solver Experiment Plan

**Status:** closed negative (2026-08-09); historical experiment record, with no retained harness

This note records alternatives evaluated against the incremental gnomonic polygon clipper for the
ordinary small-cell regime. A temporary polar-dual microprobe initially made complete six-to-eight
constraint solves look materially faster than repeated polygon mutation. Production-shaped and live
controls later showed that buffering and the polar kernel were independently adverse. Fixed K=6,
K=7, and K=8 forms all lost end to end; K=16, homogeneous predicates, scalar pair masks, online hull,
and sorting-network variants also failed their gates.

This family is distinct from the closed selected-constraint batch experiments in
[`constraint-batch-pipeline-idea.md`](constraint-batch-pipeline-idea.md). Those experiments prepared or
classified several constraints and then retained the existing sequential polygon mutation. The
experiments below replace the finite-set half-plane solve itself.

[`work-log.md`](../work-log.md) remains the authoritative implementation queue. Nothing here is an active production commitment. The temporary recorder, solver harness, and live
probe code were removed after the investigation; only this record remains.

## Starting evidence

The temporary microbench used actual nearest-neighbor prefixes from 4,096-site Fibonacci and
deterministic uniform spheres. It compared repeated production `clip_convex` calls, starting from
the production `1e6` bounding triangle, with a one-shot hull of the dual constraints followed by
direct primal-vertex reconstruction. Each accepted result was differentially checked as the same
unordered vertex set, allowing the small coordinate difference caused by incremental clipping from
the artificial bounding triangle.

Representative native-release results:

| Prefix / workload | Candidate / incremental time |
|---|---:|
| Fibonacci, 6 constraints | 0.49x |
| Fibonacci, 7 constraints | 0.61x |
| Fibonacci, 8 constraints | 0.66x initially; 0.77--0.80x after larger harness code-layout changes |
| Uniform, bounded 8-constraint prefixes | 0.63--0.68x |
| Fibonacci, 16 constraints | 2.56x |

A fixed ten-million-cell, seven-constraint Fibonacci counter comparison measured about 0.59x cycles,
1.25x instructions, 2.00x branches, and 0.29x branch misses. Cache counters were directionally lower
but noisy. This is evidence for a different dependency/control shape, not an instruction-count win.

Eight nearest constraints bounded all 1,024 sampled Fibonacci cells and 951/1,024 sampled uniform
cells. Against the exact ninth-neighbor bound, all sampled Fibonacci prefixes but only 224/1,024
uniform prefixes passed the current security certificate. A production design must therefore
continue from an uncertified initial solve; it cannot assume one batch completes every ordinary
cell.

Two adjacent results constrain the design:

- A fixed eight-slot packed first batch was previously rejected on its own: it saved 0.315% of
  Fibonacci instructions but added 2.777% on uniform and raised branches on both. Reopen that choice
  only when combined with a cell solver that removes enough clipping work.
- An arrival-order online dual hull was approximately neutral on Fibonacci at eight constraints,
  about 15% faster on bounded uniform prefixes, and 31--49% slower at sixteen constraints. Do not
  promote that straightforward online form without a different update mechanism.

Temporary source, raw counters, and exact probe notes remain under `/tmp/s2-core-probes/` for the
lifetime of the development environment; they are not repository inputs.

## Completed experiment status

A temporary feature-gated smoke harness regenerated 1,024 real nearest-neighbor prefixes from
4,096-site Fibonacci or deterministic uniform spheres using production bisector coefficients. It
supported fixed `K` values 6, 7, 8, and 16 and compared:

- the production incremental clipper;
- the original affine-coordinate polar hull;
- a division-free homogeneous polar hull;
- the topology-only scalar homogeneous feasible-pair mask; and
- the complete scalar pair-mask solver with owner-cycle recovery and materialization.

Before timing, every accepted result is checked against the incremental authority by owner-conditioned
vertices, independently checked for constraint satisfaction, owner-line incidence, CCW orientation,
and `max_r2`, and then continued through candidate 16 and compared with a fresh full incremental
solve. Ambiguity is allowed only by returning false; an accepted mismatch is loud.

Representative native-release K=8 measurements from the first harness revision were:

| Distribution | Affine polar | Homogeneous polar | Pair mask only | Complete pair solver |
|---|---:|---:|---:|---:|
| Fibonacci | 0.78x | 1.05x | 0.95x | 1.56x |
| Uniform, bounded prefixes | 0.65x | 0.88x | 1.86x | 2.27x |

All ratios are relative to incremental clipping on the common accepted corpus. Code layout remains
visible in these sub-150-nanosecond kernels, so these numbers are directional rather than an
integration claim. The same harness confirms that all three candidate solvers accept and match all
bounded ordinary prefixes in this corpus; K=16 remains decisively adverse.

At this intermediate stage, the results narrowed the next work:

- The scalar pair direction fails its pre-metadata gate on uniform. Do not build the complete SIMD
  adapter unless a mask-only SIMD probe can plausibly erase that roughly 2x deficit.
- Making every polar predicate homogeneous consumed most or all of the Fibonacci win. This made a
  guarded affine fast path the only plausible polar candidate for the later live controls.
- This corpus deliberately has no incoming edge checks and uses global nearest order. It is a smoke
  harness, not yet the production-shaped shared corpus described below.

### Production-shaped follow-up

A later temporary harness revision added a `microbench`-only recorder at the real cell-build seams.
It recorded unique incoming edge checks and directed packed/shell attempts, retained the first
production termination certificate, and consumed up to 32 attempts for continuation experiments.
Every clip after the retained certificate was required to be `Unchanged`; a changed clip, fallback
request, or failure was loud. While present, the inactive recorder used an atomic fast gate and all
hooks were cfg-elided from feature-disabled builds. The recorder and hooks have since been removed.

On 4,096-site, one-worker recordings, the ordinary production-shaped corpus had:

| Distribution | Usable cells | Mean natural stop | Natural stop <= 8 | Hybrid mean stop | Bounded affine K=8 |
|---|---:|---:|---:|---:|---:|
| Fibonacci | 1,024/1,024 | 6.86 | 1,024 | 8.00 | 1,024 |
| Uniform | 1,020/1,024 | 11.33 | 307 | 11.65 | 987 |

The four omitted uniform cells did not provide a complete certified prefix within the current
32-event recorder window and are excluded loudly from the timed pool. For accepted cells, the
harness recomputes the polar security decision from the recorded exact post-event bound and the
polar/continued `max_r2`; every final owner-conditioned cycle matches a fresh incremental replay.

Representative pinned native-release measurements, including a separately measured coefficient
preparation estimate, were:

| Distribution | Affine-8 plus continuation / incremental | With coefficient preparation estimate |
|---|---:|---:|
| Fibonacci | 0.85x | 0.89x |
| Uniform | 0.82x | 0.85x |

These rows write the polar result directly into the continuation buffer seam. An earlier harness
revision copied the complete fixed-capacity `PolyBuffer` before continuation and understated the
ceiling, especially for immediately certified Fibonacci cells.

This is a narrower Fibonacci ceiling than the isolated kernel suggested. It remains enough to
justify one feature-gated end-to-end prototype, but not enough to justify a production design yet.
The recorder still does not serialize every frontier observation, fallback/failure disposition, or
final extraction oracle; extend those records before treating it as a durable corpus rather than an
in-process experiment.

### End-to-end builder probe

A temporary feature-gated builder probe then deferred the first eight real incoming/stream
constraints, solved them with the affine polar kernel, installed the result directly into the live
`PolyBuffer`, retained all eight owner records, and replayed the prefix incrementally when the polar
solve was unbounded. It completed ordinary 100k Fibonacci construction and produced the expected
cell/vertex counts, but the full pipeline result was decisively negative.

Three rotated, pinned, single-worker 1M Fibonacci `perf stat` pairs measured the probe against the
same `tools,microbench` binary with the probe disabled:

| Counter | Probe / baseline |
|---|---:|
| cycles | 1.131x |
| instructions | 1.149x |
| branches | 1.263x |
| branch misses | 1.328x |
| cache references | 1.284x |
| cache misses | 1.061x |

The production-shaped precomputed corpus therefore overstated the transferable win. In the live
builder, coefficients and metadata must be buffered across stream events and reread by a branchy
hull kernel instead of being consumed immediately by the clipper; that interleaved storage/control
shape reverses the microkernel result. Reusing the hull scratch and writing directly into the live
polygon did not recover the loss.

A same-binary attribution control then compared immediate clipping, buffer-eight followed by the
same incremental replay, and buffer-eight followed by polar solving. Three rotated pinned 1M
Fibonacci groups measured:

| Variant / reference | Cycles | Instructions | Branches | Branch misses | Cache references |
|---|---:|---:|---:|---:|---:|
| Buffered incremental / immediate incremental | 1.078x | 1.099x | 1.098x | 1.087x | 1.152x |
| Buffered polar / buffered incremental | 1.047x | 1.041x | 1.125x | 1.232x | 0.975x |

Thus the earlier comparison did conflate two effects, but both are negative: buffering itself is
expensive, and the affine insertion-sort hull loses another 4.7% cycles after both candidates pay
that same buffering cost. A fixed eight-item compare/select sorting network was also visibly slower
than insertion sorting in 100k full builds, so branch removal through that network is not a rescue.
A batch-native gather could remove much of the first row, but it cannot make the measured polar
kernel beat an equivalently gathered incremental replay without a materially different hull
algorithm.

The final live prefix sweep repeated the same controls at K=6 and K=7, where isolated polar timing
had looked strongest. Three rotated pinned 1M Fibonacci groups measured:

| K | Buffered replay / immediate cycles | Polar / buffered replay cycles | Polar / immediate cycles |
|---:|---:|---:|---:|
| 6 | 1.035x | 1.017x | 1.053x |
| 7 | 1.042x | 1.035x | 1.079x |

At K=6, the least adverse case, polar still added 1.7% cycles after the common buffering cost and
5.3% against immediate clipping. Pinned 1M uniform cell-construction medians were likewise adverse:
K=6 measured about 745 ms immediate, 771 ms buffered replay, and 809 ms polar; K=7 measured about
746, 789, and 833 ms respectively. The smaller prefixes therefore reduce but do not reverse either
cost. The K=6/K=7 live controls close the last favorable-prefix interpretation of the isolated
microbench.

**Disposition:** remove all solver, recorder, and attribution code from the worktree and do not
productionize affine polar construction in this form. Exact exploratory patches and raw results are
retained under `/tmp/s2-core-probes/`; this document is the durable repository record.

## Shared mathematical model

The production gnomonic constraints have the form

```text
a_i * u + b_i * v + c_i >= 0.
```

For a distinct generator and neighbor in the generator-centered chart, `c_i > 0`. Define the polar
point

```text
p_i = (-a_i / c_i, -b_i / c_i).
```

Then the constraint is `p_i dot (u, v) <= 1`, and the finite candidate cell is the polar of
`convex_hull({p_i})`. In exact generic-position arithmetic:

- interior dual points are redundant primal constraints;
- dual-hull vertices are active primal edges;
- adjacent dual-hull vertices identify a primal vertex; and
- the primal cell is bounded exactly when the origin is strictly inside the dual hull.

Production candidates should avoid eagerly forming `a/c` and `b/c`. Use homogeneous dual points
`(-a, -b, c)`: rational coordinate comparisons use cross multiplication, and a dual orientation is
the sign of a 3-by-3 determinant because all denominators are positive. Divide only when emitting a
final primal vertex.

This is exact convex-set equivalence, not floating-point or policy equivalence. Any solver must deal
explicitly with collinear exposed dual points, parallel constraints, cocircular/high-degree
vertices, exact-zero edges, cycle orientation, and the current fallback and reporting semantics.

## Common harness before another production edit

Build one reusable, feature-gated solver harness rather than separate synthetic benchmarks for each
candidate. It should capture or deterministically regenerate real cell inputs with:

- exact production `(a, b, c)` coefficients and arrival order;
- incoming edge-check constraints distinguished from packed candidates;
- the first packed batch, subsequent candidates, and exact unseen bounds;
- Fibonacci, uniform, clustered, mega, great-circle, cubed, and adversarial prefixes;
- prefix lengths from 3 through 16, including bounded-but-uncertified cases; and
- current incremental output, active owner cycle, `max_r2`, and fallback disposition as the oracle.

Every solver reports:

- bounded, ambiguous, or fallback-required;
- active constraints in deterministic cyclic order;
- primal vertices and `max_r2`;
- accepted/redundant constraint provenance; and
- operation counts useful for estimating eager preparation and continuation work.

Measure fixed-work instructions, cycles, branches/misses, cache events, code size, and I-cache
behavior. Keep coefficient preparation and final vertex materialization either inside every timed
candidate or outside every candidate; report both if the intended integration changes that seam.

## Direction A — eight-constraint polar solve plus incremental continuation

**Status:** measured and rejected for the buffered affine-hull implementation; reopen only if a
materially different fixed-control solver removes the live buffering/control regression

1. Combine incoming edge-check seeds and enough packed candidates to form an initial set of at most
   eight constraints.
2. Solve that set with a homogeneous polar hull.
3. If it is unbounded or numerically ambiguous, replay the set through the current builder.
4. If bounded, construct the ordinary `PolyBuffer`, owner metadata, and `max_r2`.
5. Apply the existing exact unseen-bound security test.
6. If uncertified, continue with the existing incremental clipper from the produced polygon rather
   than solving a 16-constraint hull.

This shape keeps the observed small-prefix win while avoiding the measured 16-constraint loss. The
first timing-only integration may retain the current 16-slot selection but consume only eight and
charge any required remainder reconstruction explicitly. Only then test whether requesting eight
from packed selection is favorable in combination.

**Key risks**

- Incoming checks carry endpoint expectations, not only a geometric plane.
- The current builder retains constraints that changed the polygon in arrival order; a full-set
  solver sees final redundancy instead. Fallback replay must remain complete even if provenance is
  represented differently.
- Direct plane-plane intersections will not be bitwise equal to vertices produced through repeated
  interpolation from the bounding triangle.
- A new `max_r2` rounding path can move a security decision; termination must remain conservative.

**Gate:** require a clear whole-cell-construction cycle win on both Fibonacci and uniform before
changing packed selection. Reject if code/I-cache growth erases the isolated kernel result.

## Direction B — homogeneous pair-feasibility masks

**Status:** scalar mask and complete solver measured negative; SIMD remains low priority

For `K = 8`, enumerate the 28 constraint pairs. Represent each pair intersection homogeneously,
test it against all eight constraints with fixed-width masks, and materialize coordinates only for
feasible pairs. The feasible pair set is the primal vertex set; owner pairs already supply most
output metadata.

This intentionally spends arithmetic to remove polygon-size transitions, repeated buffer writes,
and loop-carried clipping dependencies. Pair and constraint tests are independent and may map well
to `wide` lanes. It also avoids sorting the dual points.

**Risks:** 28-by-8 classification may simply be too much work; cocircular vertices produce duplicate
feasible pairs; deterministic deduplication and cyclic ordering can reintroduce control. Use
homogeneous determinant signs so the first probe does not pay 28 pair divisions.

**Gate:** compare against both incremental clipping and Direction A on the same eight-constraint
corpus. Stop if it is not competitive before metadata deduplication.

## Direction C — fixed sorting-network polar hull

**Priority:** secondary

Replace data-dependent insertion sorting with a fixed eight-item network, then run a compact
monotone-chain/deque hull. Compare rational dual coordinates without division. This makes sorting
control predictable and may keep the complete solve in registers.

The repository already owns generated small sorting-network infrastructure, but do not modify the
generated body directly. A polar solver may need a comparator over homogeneous rational coordinates
rather than the existing scalar key comparator.

**Risk:** duplicated compare/exchange code can increase executable text and I-cache pressure—the
same failure mode seen in earlier specialization experiments.

**Gate:** require lower cycles and no material I-cache/code-size regression relative to the simpler
polar implementation, not merely fewer branch misses.

## Direction D — batch dual gift wrapping

**Priority:** secondary

Find one extreme dual constraint, then repeatedly select the next hull constraint by scanning the
small candidate set. With roughly eight candidates and six active edges, the expected work is about
`K * h` orientation comparisons. The state is tiny and no complete sort is required; each selection
can potentially use a fixed-width reduction.

This is distinct from the provisionally negative arrival-order online hull: it solves a known batch
and never performs dynamic insertion/splicing.

**Risk:** hull-edge discovery is serial across `h`, and deterministic handling of collinear exposed
points may require additional scans.

**Disposition:** left unmeasured because every surrounding small-cell family failed its live gate.
Rebuild a temporary differential harness only if a materially new result justifies reopening it; do
not integrate from the standalone arithmetic hypothesis.

## Direction E — orientation-matrix / bitset topology

**Priority:** later, only if pair masks expose a useful topology representation

Compute the relevant pair/triple determinant signs into compact bitsets and derive the exposed
constraint cycle through bit operations or a small fixed state machine. Coordinates are produced
only after topology is known.

This is the most control-regular design and the most likely to over-specialize or expand code. Treat
it as a representation extracted from Direction B evidence, not an independent speculative
implementation.

**Gate:** first demonstrate that the orientation matrix is materially smaller or cheaper than pair
feasibility masks on recorded prefixes. Include executable-text and I-cache gates from the start.

## Direction F — primal angle-sorted half-plane intersection

**Priority:** control experiment

Sort primal constraint directions and apply a standard deque half-plane intersection directly,
without dual coordinate construction. This is mathematically close to the polar hull and may offer
better finite-range behavior when `c` is small.

Its primary value is attribution: if it matches polar performance, the win comes from one-shot
ordered construction rather than dual representation specifically. It is not automatically a
production candidate.

## Directions currently disfavored

- **Straight 16-constraint polar solve:** measured at roughly 2.56x incremental time on Fibonacci.
- **Straight arrival-order online dual hull:** neutral on Fibonacci K=8 and adverse at K=16.
- **Prepare/classify a constraint window and then clip survivors sequentially:** already measured
  and closed in the selected-constraint batch plan.
- **Globally request eight packed candidates without a solver win:** previously rejected on uniform
  instructions and branches.
- **Exact/robust predicates on every ordinary dual turn:** begin with f64 homogeneous predicates and
  a conservative ambiguity fallback. Pay adaptive robustness only if evidence shows that the
  fallback rate or differential mismatch rate is material.

## Correctness and acceptance contract

A fast solver may differ bitwise from incremental clipping, but it must not weaken construction
certification. Before production use, establish all of the following:

1. **Coefficient fidelity:** consume the existing off-unit-corrected f64 `(a, b, c)` values without
   replacing them with a unit-vector approximation.
2. **Boundedness:** accept only when the origin is strictly inside the dual hull, or the equivalent
   primal condition is certified.
3. **Constraint satisfaction:** every emitted vertex satisfies every accepted constraint under the
   selected conservative policy.
4. **Degeneracy handling:** parallel, collinear, cocircular, near-coincident, and exact-zero cases
   either preserve the documented owner information or fall back before committing output.
5. **Metadata:** incoming edge checks, endpoint identities, neighbor slots, plane owners, cycle
   orientation, and reverse-side expectations remain complete.
6. **Continuation:** an uncertified small solve can continue incrementally without losing a
   constraint or changing the unseen-bound contract.
7. **Fallback replay:** projection-limit, polygon-capacity, or ambiguous-predicate handoff replays a
   complete deterministic constraint sequence through the current authority.
8. **Termination:** use the existing chart metric correction, radius padding, and exact frontier
   bound; checked builds differentially compare every accepted early termination with exhaustive
   continuation.
9. **Stitching:** backend fingerprints may change only after explicit policy agreement, while edge
   multiplicity/orientation, Euler, owner equality, and strict validation must remain accepted.
10. **Performance:** measure the complete pipeline across ordinary and non-uniform controls. A
    microkernel win alone is insufficient.

## Experiment status and remaining order

Completed and rejected:

1. Shared nearest-prefix and production-event recorder/harness.
2. Affine and homogeneous polar K=6/7/8/16 kernel comparisons.
3. Scalar pair mask and complete pair solver.
4. Fixed K=8 compare/select sorting-network polar solver.
5. Production-shaped affine K=8 continuation and coefficient-preparation estimate.
6. Same-binary immediate, buffered-replay, and buffered-polar attribution controls.
7. Live K=6/K=7/K=8 prefix sweep on Fibonacci and uniform.

No measured polar/pair variant survives the whole-build gate. Do not reopen packed prefix sizing from
this line of work. The only listed solver direction still materially different from a measured
implementation is Direction D's batch gift wrapping. Test it only if another small-cell solver probe
is explicitly desired; it must beat buffered incremental replay by enough to pay the independently
measured buffering cost before receiving any live integration work.
