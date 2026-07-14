# Output Resolution and Edge Collapse Policy

**Status:** design record; not yet implemented

**Date:** 2026-07-14

This note records the intended policy for Voronoi features at or below the resolution of the
returned vertex representation. It separates two decisions that are easy to conflate:

1. what happens when resolving an edge would remove a generator cell; and
2. which edges are candidates for collapse.

The first decision is about generator identity and cardinality. The second is about geometric
resolution. They should be independently visible in the API and report, even if a convenience
mode selects both together.

This design was prompted by a downstream mesh containing a Voronoi edge whose distinct vertex IDs
had bit-identical f32 positions. The incident edge is not load-bearing for either owner cell:
contracting it reduces the two cell degrees without removing a cell. The broader generator policy
is therefore not needed to fix that incident, but it is needed to define behavior for genuinely
collapsed cells.

## The representational tradeoff

At fixed output precision, a library cannot always provide all three of the following:

1. one returned cell for every effective generator;
2. an injective geometric embedding with no zero-length edges or zero-area cells; and
3. the combinatorics selected by a higher-precision Voronoi construction.

A positive but sub-f32 cell can have distinct f64 construction vertices which become equal when
stored as f32. An analogous ambiguity can occur within f64 working precision. A deterministic
tie-breaking policy can select a coherent combinatorial answer, but it cannot manufacture
geometric separation in the chosen output representation.

Using f64 output would move this resolution boundary, not remove it. Exact cocircularity can also
produce an unnecessary zero-length diagonal when a generalized Voronoi vertex is represented by a
triangulation. A zero-edge canonicalization step remains useful independently of output precision.

For this policy, an **exact stored-zero edge** means that the two endpoint records represent the
same positive direction. Detection must use equality/canonical equality or a zero squared chord in
the stored representation. It must not use `acos(clamp(dot)) == 0`: dot-product rounding can make a
short positive edge appear to have zero angle.

Equality in the stored representation is not proof that the ideal f64 construction vertices were
equal. Collapsing such an edge is an output-resolution decision.

## Dimension 1: generator outcome

The generator outcome applies only when an otherwise requested contraction would make a cell
unrepresentable, normally by leaving it with fewer than three distinct boundary vertices.

### Preserve (default)

- Preserve one effective cell per effective generator.
- Perform contractions that leave every cell representable.
- Transactionally decline a contraction component that would eliminate a cell.
- Return the coherent combinatorial topology, even if some remaining edges or cells have zero
  geometry in the stored representation.
- Report the unresolved zero-geometry feature so geometry-sensitive consumers can reject it or
  select a different policy.

This is the default because silently changing generator cardinality can invalidate consumer
assumptions. It is also the no-failure choice when topology remains valid but output precision
cannot embed it injectively.

### Error

- Preserve one effective cell per effective generator.
- Perform safe contractions first.
- Return a defined representation error if a requested zero-edge requirement cannot be satisfied
  without eliminating a cell.

This is appropriate for consumers that require both stable generator cardinality and an
injectively embedded output mesh.

### Elide

- Permit contraction to remove a cell whose boundary collapses below representation capacity.
- Return a valid cell complex over the remaining effective generators.
- Report every elided generator and compose that information with preprocessing weld results.
- Do not imply that a removed generator is canonically equivalent to either owner of the collapsed
  edge.

Elision differs from input welding. Welding deliberately constructs the diagram of a quotient site
set and has a representative generator. A generator whose cell disappears at output resolution
need not have a geometrically canonical surviving representative. The eventual public report may
therefore need to distinguish `WeldedInto(effective_id)` from `ResolutionElided`, or otherwise map
an original generator to no returned cell rather than inventing an alias.

Eliding a cell that is zero only after f32 storage returns a diagram chosen for the output
resolution, not the exact Voronoi diagram of every original effective generator.

## Dimension 2: edge-collapse scope

### Exact stored-zero (baseline)

The baseline stage considers exact stored-zero edges. A contraction that does not eliminate a cell
is a representation canonicalization, not generator elision. Subject to local link checks and
whole-result validation, it should occur under all three generator outcomes.

This commonly removes an arbitrary triangulation diagonal and restores a degree-four-or-higher
generalized Voronoi vertex. It also fixes zero-length mesh edges without changing the generator
set.

The stage should run to a fixed point: one contraction can expose another exact stored-zero edge.
Connected zero-edge components must be handled transactionally rather than by an order-dependent
sequence of pairwise edits.

### Within epsilon (optional “extra-elide”)

An optional positive threshold extends the candidate set to short, nonzero edges. Its purpose is
consumer-controlled mesh conditioning: graphical and physics applications may prefer to remove a
tiny sliver rather than retain the exact represented Voronoi feature.

Positive-edge collapse is explicitly approximate:

- the returned boundaries need not remain bisectors of the original generators;
- the graph need not remain a Voronoi diagram for the exact construction sites;
- topology and representation invariants must still hold; and
- the report must record the threshold, accepted components, declined components, removed cells,
  and a bound on each collapsed component's geometric diameter.

The threshold should be expressed using a numerically stable measure such as squared chord length.
Components must be diameter-bounded; pairwise epsilon chains must not merge an arbitrarily large
feature. Candidate processing must be deterministic and transactional.

The generator outcome remains orthogonal. For example, `Preserve` plus a positive threshold may
collapse short edges while declining components that would remove a cell. The common
“extra-elide” use case is `Elide` plus a positive threshold.

## Policy matrix

| Generator outcome | Non-cell-killing stored-zero edge | Cell-killing stored-zero edge | Positive edge within configured threshold |
|---|---|---|---|
| Preserve (default) | Contract | Preserve and report | Contract only if no cell is lost |
| Error | Contract | Defined error | Defined error if the requested contraction requires cell loss |
| Elide | Contract | Remove cell and report | Remove cells as needed and report |

With no positive threshold, the final column is not considered.

The matrix describes requested behavior, not permission to perform an invalid local edit. Every
accepted component must satisfy the topology and representation checks below.

## Pipeline placement and repair consistency

Resolution policy belongs after construction and repair:

```text
construct -> reconcile -> Local3d if required -> output-resolution policy -> validation
```

Local3d reconstructs a Voronoi neighborhood. It must not run after output-resolution collapse and
recreate an edge the policy intentionally removed. A failure of a collapse transaction is handled
by the selected generator outcome, not by asking Local3d to restore the pre-simplified geometry.

The final policy must apply globally, whether or not repair ran. Repair-local tolerance welding
cannot be the only way a tiny edge is collapsed; otherwise an unrelated topology defect can change
the resolution of the returned mesh. The existing fast-repair weld path must be audited against
this rule:

- repair may establish the endpoint equivalence needed for edge agreement;
- it must not silently eliminate a generator under `Preserve`;
- any approximate simplification it performs must obey the same threshold and generator outcome
  as the final global pass; and
- the global pass remains authoritative for repaired and unrepaired diagrams alike.

If later work introduces another repair after this stage, the resolution stage must run again
after the final repair. The simpler design is to keep it terminal.

## Transaction requirements

An accepted contraction component must, at minimum:

- preserve cyclic order while rewriting affected cell boundaries;
- remove consecutive duplicate vertex IDs and reject nonconsecutive repetition that cannot be
  normalized coherently;
- obey the selected policy when a boundary falls below three distinct vertices;
- leave no self-loop, duplicate face, unintended parallel edge, low-incidence edge, or overused
  edge;
- preserve opposite orientation of the two owners of every surviving edge;
- preserve connectivity and spherical Euler characteristic;
- handle multiple coincident edges as one component, including explicit rejection or elision of a
  component that collapses an entire face; and
- pass the selected whole-diagram validation before the transaction is committed.

Deleting a triangular cell after two of its vertices merge is locally plausible: its other two
edges become the candidate shared edge between the two neighboring cells. That operation is valid
only when the link and owner rotations agree. “The other owner of the zero edge” is not a
geometrically justified generator representative and must not be selected merely because it owned
the collapsed diagonal.

The first implementation should favor a simple cold transactional rebuild over intricate live
mutation. Exact-zero incidents are expected to be rare. Common-path scanning cost should be
measured with retired-instruction and branch counters; live hints may optimize candidate discovery
later, but the post-repair result remains authoritative.

## Reporting and consumer contract

The computation report should distinguish at least:

- non-destructive exact-zero contractions;
- unresolved zero-geometry edges or cells under `Preserve`;
- resolution errors under `Error`;
- resolution-elided generators under `Elide`;
- positive-threshold contractions and the configured threshold; and
- declined unsafe components.

Strict topology validity and geometric embedding are separate facts. Validation should expose a
zero-edge/zero-geometry count even when the abstract topology is valid. A consumer can then choose
among stable generator identity, fail-loud geometry, or generator elision without interpreting a
generic topology failure.

The user-facing guarantee for positive-threshold output must say “valid spherical cell complex
after explicit simplification,” not “Voronoi diagram of the original generators.”

## Initial implementation sequence

1. Add detection and reporting for exact stored-zero edges after the final repair.
2. Implement transactional non-cell-killing zero-edge contraction and run it to a fixed point.
3. Reproduce the downstream zero-edge incident and confirm it succeeds under default `Preserve`
   without changing the generator count.
4. Add `Preserve`, `Error`, and `Elide` behavior for components that would eliminate cells,
   including explicit generator-remapping/report semantics.
5. Audit repair-local welding for path independence.
6. Add optional positive-threshold collapse only after the exact-zero policy is stable.
7. Benchmark the no-candidate scan and keep all component reconstruction on the cold path.

Tests must include non-cell-killing zero diagonals, triangle-to-digon cases, both-owner triangles,
zero-edge trees and cycles, simultaneous contractions, permutation independence, repaired output,
near-antipodal edges, and epsilon chains whose pairwise distances pass while their total diameter
does not.

## Decisions still open

- Public type and variant names.
- Whether resolution-elided generators map to `None`, a richer outcome enum, or an explicitly
  noncanonical deterministic representative.
- The exact local link predicate used before transactional whole-diagram validation.
- Whether exact-zero detection uses stored f32 positions exclusively or also retains pre-storage
  f64 collision telemetry.
- Threshold units and whether a convenience angular API converts to the canonical squared-chord
  representation.
- Whether positive-threshold output is stored in the same diagram type with report metadata or a
  distinct simplified-mesh wrapper.
