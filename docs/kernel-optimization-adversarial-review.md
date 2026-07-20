# Kernel optimization adversarial review

Three fresh blind reviewers were each assigned one candidate from the multi-model synthesis and
asked to try to reject it before implementation. All three returned **measure-only**. This file
records the material corrections, proof obligations, and cheapest falsification gates.

## Demand-sized packed prefixes

### Corrections and risks

- `emit_run(n_target)` accepts runtime sizes locally, but the frontier protocol does not yet:
  `PackedQuery::frontier` chooses fixed 16/8 requests and both cache layers are request-unaware.
- Smaller requests do not automatically remove partition work. Both sizes can partition the full
  remainder, and under-requesting can repeat that scan. A small ask can also replace chunk0's
  whole-sort-small path with branch-heavy partitioning.
- Emitted-minus-visited counts therefore measure abandoned sorted/scattered slots, not all saved
  selection work. The whole-sort path may sort beyond the emitted prefix.
- Changing batch boundaries may change which constraints are consumed because post-batch
  certification runs even after a changed clip, while mid-batch checks run only after unchanged
  clips. The required invariant is identical output/topology with sound coverage, not identical
  internal consumption.
- A useful cheap demand signal is not established. The builder knows boundedness and vertex count,
  but the next useful bound is learned by performing the selection the proposal aims to avoid.

### Proof and test obligations

- Random varying-request sequences must preserve full `(descending dot, ascending slot)` order,
  monotone conservative bounds, stage coverage, and tie behavior; zero-length asks are forbidden.
- Re-probing a cached exact frontier must retain the original slots and length or explicitly reject
  an incompatible demand hint.
- Prove segmentation-independent output/topology/fallback behavior across scalar/SIMD, bin counts,
  and thread counts.
- A demand policy must be pure, deterministic, and based only on stable per-cell state.

### Falsification gate

Measure emitted/visited/abandoned slots together with remainder length, partition versus whole-sort
path, actual sort length, scatter count, cached re-probes, and cheap builder state. Strong
abandonment justifies only a fixed 8-versus-16 probe first, not an adaptive 1--4 request policy.

## Compact deferred high-key overflow

### Corrections and risks

- Retention must use the full ascending `make_desc_key(dot, slot)` order, not dot alone. Boundary
  filtering by dot mishandles equal-dot slot ties.
- Current extraction partitions the remainder and usually sorts only the emitted prefix; it does
  not sort every unused high key. The removable work is storage and participation in partitioning.
- Existing key totals combine requested tail materialization with chunk0. Only the chunk0 share is
  directly reclaimable, and peak capacity persists across groups.
- A retained kth dot is a looser bound than the model threshold. Deferred overflow remains part of
  chunk0 and must be exhausted or certified before tail.
- Dense-band queries are a separate regime: reconstructing from raw cell ranges can throw away the
  band's targeted-gather benefit.
- Online top-K maintenance adds comparisons, branches, and irregular writes to every high key. If
  block masks are dense, reconstruction approaches the rejected full-rescan experiment.

### Proof and test obligations

- Maintain exact deterministic top-K by full key and represent each omitted high candidate exactly
  once under directed center/ring eligibility.
- Overflow emission must globally merge all relevant marked blocks before returning a candidate;
  its bounds must cover the retained suffix, deferred overflow, tail, and outside coverage.
- Any-high block marking is conservative but may be much denser than exact deferred membership;
  record both. Choose pressure mode before vectors grow or peak capacity is not recovered.
- Keep dense-band support explicitly excluded until its query-specific gather identity and bound
  are handled.

### Falsification gate

Using existing exact chunk0 keys, simulate caps and block sizes without changing behavior. Record
high count, emitted depth, cap exceedance, exact versus any-high blocks, eligible slots covered by
demanded rescans, and normal versus dense-band mode. A cap is unattractive when it is exceeded
widely and demanded block masks cover roughly half or more of eligible neighborhoods, unless an
isolated streaming-selection cost model shows a large compensating win.

## Builder-aware shell-cell rejection

### Corrections and risks

- Redundancy under a gnomonic polygon is monotone only while that representation remains active. A
  later candidate may trigger spherical fallback; a point rejected earlier might then have been
  processed by the baseline fallback stream.
- Shell materialization currently scans a complete BFS layer before returning candidates, so the
  builder snapshot naturally changes between layers, not between cells within one layer.
- Existing cube-cell caps are conservative radial enclosures. Reusing their lower dot bound with
  the existing strict termination comparison is much easier to prove than a new directional
  all-polygon-vertices certificate.
- A directional certificate must model the raw-f32 chord bisector, including point norm, promoted
  arithmetic, strict inside comparison, outward rounding, chart conditioning, and face/cell ties.
- Rejected cells must still participate in BFS visitation, neighbor discovery, and next-ring
  bounds; only resident-point scanning may disappear.

### Proof and test obligations

- Prove the cell enclosure covers all raw-f32 residents at face dominance ties, cube edges,
  vertices, and UV grid lines.
- A hit must imply the production all-vertices decision under equivalent arithmetic and conservative
  tolerance direction.
- Disable the certificate for unbounded, failed, and fallback builders. If fallback can follow an
  earlier rejection, restart/replay the shell or prove redundancy across representations.
- Preserve retained-candidate order, traversal topology, deterministic cached-frontier behavior,
  and supported backend/thread/bin equivalence.
- Keep certificate state cold and shell-only; do not enlarge ordinary builder/frontier hot state.

### Falsification gate

First run an ideal exact resident-cell oracle at the frozen layer-start builder state. Count
nonempty tested cells, exact all-resident-unchanged hits, saved resident slots, predicate/vertex
evaluations, actual consumed slots, and any later fallback after a hit. If the ideal oracle covers
little shell work—or fallback-after-hit is nonzero and replay is too costly—reject before designing
new cap mathematics. Only a strong ideal upper bound justifies a conservative cap shadow with zero
false positives.
