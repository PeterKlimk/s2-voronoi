# Positive-threshold edge simplification plan

**Status:** proposed implementation plan for RES-002

**Date:** 2026-07-22

This plan extends the exact stored-zero output-resolution work with an explicit, consumer-selected
positive edge threshold. The result is a simplified `SphericalCellMesh`, not a Voronoi diagram of
the original effective generators. The governing policy remains
[`output-resolution-policy.md`](output-resolution-policy.md); this document defines the intended
implementation sequence and acceptance gates.

The central implementation fact is that post-assembly reconciliation already contains most of the
safe-collapse classifier. On a defect-bearing run it proposes nearby vertex unions, rejects
threshold-connected chains whose full component diameter exceeds the configured epsilon, evaluates
all interacting merges over a complete affected-cell cover, and declines merges that would kill or
make a face non-simple. RES-002 should extract and reuse those rules. It should not grow a second,
slightly different positive-distance merge policy in output resolution.

## Goal and non-goals

The goal is an opt-in consuming conversion which:

- finds every final represented edge at or below a positive chord threshold;
- forms deterministic, diameter-bounded collapse components;
- contracts components which pass the topology and selected cell-outcome policy;
- optionally elides cells when the caller explicitly permits it;
- returns a dense, strictly valid `SphericalCellMesh` with original-input provenance; and
- reports the requested threshold, accepted and declined work, removed cells, and geometric
  displacement bounds.

This is not:

- an input-site welding control;
- another reconciliation tolerance or a replacement for mismatch repair;
- an implicit stage in `compute`, `compute_with`, or `compute_with_report`;
- a promise that the result remains a Voronoi diagram for either the original or surviving sites;
- a reason to run Hull3d after simplification; or
- an exact or global minimum-feature-size optimizer.

The existing exact-zero canonicalization and `into_elided_cell_mesh` behavior remain valid and do
not acquire positive-threshold overhead.

## Existing machinery to reuse

### Reconciliation

`knn_clipping/edge_reconcile.rs` already supplies the essential positive-distance safety rules:

- candidate pairs use squared stored-coordinate distance;
- union-find gives deterministic connected candidate components;
- the persistent merge ledger prevents separate rounds from extending an accepted component past
  its diameter bound;
- every pair of original members is checked in promoted f64 against the configured diameter;
- provenance supplies a complete local face cover, with a global fallback when it is incomplete;
- interacting proposals are simulated together; and
- a proposal is rejected if a rewritten face has fewer than three vertices or repeats a vertex.

The simplifier is a one-shot transaction and therefore does not need reconciliation's persistent
multi-round ledger. It does need the same component-diameter and joint face-safety semantics.

### Exact-zero output resolution

`knn_clipping/output_resolution.rs` already supplies:

- deterministic component representatives;
- grouping of components which interact through a cell;
- cyclic face rewriting and consecutive-duplicate removal;
- localized quotient checks for non-cell-killing contractions;
- rollback on a rejected transaction;
- global cell-elision, degree-two vertex suppression, vertex compaction, and strict validation; and
- effective-generator to final-cell mappings.

The positive path should operate on the distinct cell-mesh conversion. It must not broaden the
authority of the ordinary exact-zero stage over `SphericalVoronoi`.

### Public cell-mesh conversion

`cell_mesh.rs` already owns:

- source-output validation;
- conversion from the preferred effective diagram;
- original-input/preprocessing-weld provenance composition;
- recoverable conversion errors which retain the source `ComputeOutput`; and
- embedded unit-sphere parity.

The new conversion should share this preparation and finalization rather than cloning it.

## Proposed policy decisions

These decisions make the first implementation finite and deterministic. Revisit them only with a
specific consumer requirement or contradictory campaign evidence.

### Placement and default cost

Positive simplification is a cold explicit conversion over a successful `ComputeOutput`. Candidate
discovery performs one global scan of final live edges. Do not thread the threshold through the
construction hot loop initially.

The construction hint is only a necessary-condition optimization: widening its x-coordinate bound
would still require exact final chord confirmation, and a threshold selected after computation is
not available to that hot loop. The explicit conversion is already `O(V + E + F)` for rewriting,
compaction, provenance, and validation, so the edge scan is the appropriate first implementation.
Add a sparse construction hint only if conversion profiling later demonstrates a worthwhile
crossover and the `t + 2r` discovery certificate is retained.

### Threshold units and boundary

- The public canonical unit is Euclidean chord length between stored unit-sphere directions.
- The accepted public value is finite, strictly positive, and at most `2.0`.
- The implementation promotes stored f32 coordinates to f64, computes squared chord distance, and
  compares `distance_squared <= threshold_squared`.
- The report records the requested chord threshold and the exact squared f64 threshold used.
- An angular convenience constructor may convert radians to chord length, but it is not required
  for the first implementation and must not use `acos(dot)` for edge classification.

Chord length is easier for callers to interpret than a public squared-distance field while keeping
the hot comparison stable. Exact stored-zero behavior remains a separate named operation rather
than being encoded as a nominal positive threshold.

### Component construction

Candidate edges are sorted and deduplicated before union. The lowest vertex id is the deterministic
representative, matching exact-zero canonicalization and reconciliation.

A threshold-connected component is accepted only if every pair of its original member positions is
within the configured chord threshold. If a short-edge chain exceeds that diameter, decline the
whole connected component. Do not greedily split it: greedy splitting makes the result dependent on
edge order and turns one consumer threshold into an undocumented clustering algorithm.

All components touching the same face are evaluated as one interaction group. A group is committed
or declined transactionally so that several individually safe contractions cannot jointly kill or
make a face non-simple.

### Cell outcomes

The positive conversion exposes three explicit outcomes for a requested component:

- `Preserve`: contract the component only if every effective cell survives; otherwise decline and
  report it.
- `Error`: perform all independently safe work, but return a recoverable simplification error when
  a requested component requires cell removal.
- `Elide`: permit the existing global cell quotient to remove affected effective cells, suppress
  valid degree-two subdivisions, and report all original inputs mapped to no final cell.

This policy belongs to the cell-mesh conversion and should use a conversion-specific enum. Do not
add `Elide` to `CellKillingPolicy`: that enum configures construction returning
`SphericalVoronoi`, which cannot represent removed effective cells honestly.

### Geometry contract

Selecting an existing component representative moves every retired member by no more than the
certified component diameter. The report records at least:

- maximum accepted component diameter;
- maximum vertex displacement to the chosen representative;
- maximum cross-track deviation introduced by degree-two suppression; and
- accepted and declined component counts, split by diameter, face safety, and cell outcome.

The first version promises the contract implemented by `SphericalCellMesh::validate`: a connected,
oriented, closed abstract S2 cell complex with valid spherical vertex storage, coherent provenance,
no zero/antipodal surviving edges, and single-cycle vertex links. It does not claim global
Hausdorff error, preserved bisectors, nearest-site ownership, or a certified absence of crossings
between nonincident shorter great-circle arcs unless a separate geometric-embedding validator is
implemented. User-facing wording must state this limit rather than deriving a stronger geometric
claim from the threshold alone.

## Proposed API shape

Names are provisional until the API stage, but the separation should look like:

```rust,ignore
#[non_exhaustive]
pub struct CellSimplificationOptions {
    // Private fields with validated constructors/builders.
}

#[non_exhaustive]
pub enum SimplificationCellPolicy {
    Preserve,
    Error,
    Elide,
}

impl CellSimplificationOptions {
    pub fn from_chord_length(chord: f32) -> Result<Self, ThresholdError>;
    pub fn with_cell_policy(self, policy: SimplificationCellPolicy) -> Self;
}

impl ComputeOutput {
    pub fn into_simplified_cell_mesh(
        self,
        options: CellSimplificationOptions,
    ) -> Result<SimplifiedCellMeshOutput, CellSimplificationError>;
}
```

`SimplifiedCellMeshOutput` contains the mesh, original `ComputeReport`, and a
`CellSimplificationReport`. A conversion error owns the original successful `ComputeOutput`, as
`CellElisionError` does today.

Keep `into_elided_cell_mesh()` as the exact-zero convenience operation. Internally, exact elision
and positive simplification should share source preparation, provenance composition, quotient
finalization, and embedded wrapping. Do not redefine the exact method as a floating-point
threshold of zero if doing so obscures its exact stored-equality contract.

The first public report should include:

- requested chord threshold and squared comparison threshold;
- short edges and threshold-connected components detected;
- components and edges contracted;
- components declined for excess diameter;
- components declined for non-simple/topology reasons;
- components declined because `Preserve` retained a cell;
- effective cells and original inputs elided;
- degree-two vertices suppressed and unused vertices removed;
- maximum accepted component diameter;
- maximum representative displacement;
- maximum suppression cross-track residual; and
- final `CellMeshValidationReport`.

Counts must say whether they are unique edges, raw cell-edge uses, components, interaction groups,
effective cells, or original inputs.

## Implementation stages

### Stage 0 — Pin behavior and baseline

Before refactoring, add or identify direct tests for reconciliation's existing rules:

- inclusive distance boundary;
- rejection of a transitive over-diameter chain;
- joint rejection of components which interact through one face;
- deterministic representative selection;
- preservation of cell cycles on rejection; and
- localized affected-cell classification matching a global oracle.

Record a release build/counter baseline for the ordinary clean path. The eventual feature must add
no scans, allocations, environment reads, or threshold branches to ordinary computation.

### Stage 1 — Extract the shared collapse classifier

Move the pure one-round machinery needed by both consumers behind an internal module, tentatively
`knn_clipping/collapse_components.rs`:

1. accept sorted candidate vertex pairs, stored positions, and an affected-cell cover;
2. build deterministic threshold-connected components;
3. calculate promoted-f64 component diameter and representative displacement;
4. reject over-diameter components transactionally;
5. group surviving components by face interaction; and
6. classify rewritten faces as safe, cell-killing, or non-simple.

Keep reconciliation-specific record discovery, duplicate-key rules, persistent round ledger,
Hull3d seeds, and telemetry in `edge_reconcile.rs`. Keep output-resolution mutation, rollback,
elision, and public reporting out of the shared classifier.

Acceptance for this stage is no semantic change to reconciliation or exact-zero output, focused
unit parity, `cargo test --profile checked` for affected suites, and no measurable clean-path
regression. If extraction perturbs codegen materially, keep a small duplicated pure predicate
rather than imposing a throughput regression merely to share source.

### Stage 2 — Add cold positive-edge discovery and Preserve

Add a final-edge collector parameterized by a validated squared chord threshold. Deduplicate the
two cell uses of every edge, then feed candidates to the shared classifier.

Implement `Preserve` first:

- contract every safe interaction group;
- decline cell-killing, over-diameter, or non-simple groups;
- retain positions of deterministic representatives;
- compact unused vertices;
- run generic cell-mesh validation; and
- populate the complete positive-simplification report.

This stage establishes the API, units, deterministic candidate behavior, and no-cell-loss contract
without mixing in quotient face removal.

### Stage 3 — Add Error and Elide outcomes

`Error` shares `Preserve` classification but returns a recoverable error if any requested
interaction group is cell-killing. The error reports the affected original input indices and owns
the source output. Specify whether independently safe groups are only simulated or applied to a
temporary result before returning; externally, failure always returns the untouched source.

`Elide` reuses the existing global quotient:

- apply all diameter- and topology-admissible representatives together;
- remove faces that fall below three vertices;
- suppress degree-two subdivision vertices only when owner rotations agree;
- reject pinched links, duplicate faces, invalid incidence, whole-mesh collapse, and invalid arcs;
- compact geometry and provenance; and
- validate the complete final mesh.

The exact-zero candidates still present in the source and the positive candidates must participate
in one deterministic transaction. Do not let processing order make a zero component safe or unsafe
relative to a positive component.

### Stage 4 — Embedded parity and documentation

Add the corresponding consuming conversion for `EmbeddedComputeOutput` as a projection wrapper
over the unit-mesh implementation. Retain the same restriction as exact elision: no locator,
Delaunay, Lloyd, or conditioned Voronoi-measure claims for the simplified mesh.

Update README, correctness, architecture, output-resolution policy, and work-log status together.
Document threshold units, inclusive comparison, approximation semantics, recoverable errors,
provenance behavior, and complexity.

### Stage 5 — Optimize only with conversion evidence

Benchmark the explicit conversion separately on uniform, clustered, mega, great-circle, and a
purpose-built many-short-edge input. Record candidate counts, component sizes, rejected chains,
affected cells, and time by discovery/classification/rewrite/validation phase.

Only if the global edge scan is material should construction accept a threshold-aware sparse hint.
That optimization must widen the existing necessary x-coordinate hint according to the documented
`t + 2r` argument, confirm full final chord distance after assembly, include every post-assembly
mutation footprint, and fall back globally when representative drift or provenance invalidates the
certificate.

## Complexity and resource bounds

The cold conversion has expected linear work in the mesh size for discovery, rewriting,
compaction, provenance, and validation. Sorting candidate edges costs `O(K log K)` for `K`
short-edge candidates.

Exact component diameter checking is `O(m^2)` for a component with `m` original members. This is
appropriate for the expected small local components and matches reconciliation's defensible
diameter rule. The implementation must nevertheless guard pathological caller thresholds:

- track and report maximum component size;
- avoid allocating a dense `V`-sized union-find for a sparse candidate set;
- use checked index/capacity conversions; and
- define a recoverable resource-limit error if a component or retained candidate set exceeds a
  documented safe bound rather than silently weakening the diameter certificate.

Do not replace the all-pairs diameter check with distance from the representative alone: that
permits a component diameter up to twice the requested threshold.

## Test and campaign matrix

### Unit and synthetic topology tests

- distances immediately below, equal to, and above the squared threshold;
- invalid thresholds: zero, negative, NaN, infinity, and greater than the sphere diameter;
- the smallest positive f32 threshold is accepted and deterministically finds no positive edge on
  ordinary stored geometry;
- one safe polygon-edge contraction;
- disjoint safe components;
- several components interacting through one face;
- an over-diameter transitive chain;
- a component producing a repeated nonconsecutive vertex;
- triangular and multi-edge cell killing under Preserve, Error, and Elide;
- degree-two suppression with agreeing and disagreeing owner rotations;
- pinched vertex link, duplicate face, disconnected quotient, and whole-mesh collapse rejection;
- exact-zero and positive components in the same transaction;
- vertex-id permutation, face-order permutation, and cycle-rotation determinism; and
- recoverable failure returning the original computation unchanged.

### End-to-end tests

- extend the existing 18-site exact-zero fixture with thresholds bracketing its positive edges;
- construct clean diagrams with short nonzero edges but no reconciliation trigger;
- verify preprocessing-weld classes all map to `None` when their effective cell is elided;
- confirm source/effective/final mappings under every policy;
- require strict `SphericalCellMesh` validation on every success; and
- compare semantic topology across thread counts, bin counts, default/scalar SIMD, and FMA.

### Performance gates

- ordinary computation with no conversion: no new runtime work and no material counter regression;
- exact `into_elided_cell_mesh`: no material regression from shared preparation/finalization;
- positive conversion: report wall time and peak memory separately from construction;
- candidate-free conversion remains linear with low allocation churn; and
- candidate-heavy behavior is bounded and fails explicitly rather than exhausting memory.

## Completion criteria

RES-002 is complete when:

1. a caller can request a validated positive chord threshold only through an explicit cell-mesh
   conversion;
2. every final edge is considered, independently of whether reconciliation ran;
3. accepted components are deterministic and have certified diameter at most the threshold;
4. threshold chains cannot merge a larger feature;
5. interacting contractions are classified and committed transactionally;
6. Preserve, Error, and Elide have tested cell-killing behavior;
7. every successful result passes strict generic cell-mesh validation and retains coherent
   original-input provenance;
8. the report exposes thresholds, accepted/declined work, removed cells, and displacement bounds;
9. ordinary construction and exact-zero behavior retain their performance and correctness
   contracts; and
10. documentation states the approximate non-Voronoi contract and does not overclaim a global
    geometric-embedding or Hausdorff certificate.

## Deferred extensions

- angular threshold convenience beyond a simple validated conversion;
- pre-storage f64 collision telemetry;
- adaptive or per-region thresholds;
- greedy/optimal partitioning of rejected over-diameter chains;
- a global nonincident spherical-arc intersection certificate;
- global Hausdorff or area-error bounds;
- locator, Delaunay, Lloyd, area, or centroid semantics for simplified cells; and
- a threshold-aware construction hint before conversion profiling justifies it.
