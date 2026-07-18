# Validation fact inventory

**Status:** QUAL-001C pre-extraction inventory, 2026-07-19

This document maps the three strict sphere-validation consumers before any shared-fact
refactoring. The purpose is to preserve their different cost, input, and diagnostic policies while
identifying facts that are genuinely identical. It is not a proposal to route every consumer
through `ValidationReport` or one universal traversal.

## Consumers and input policy

| Consumer | Input | Role | Failure policy | Weld policy |
|---|---|---|---|---|
| `verify_sphere_fast` | `SphericalVoronoi` | Opt-in `VORONOI_MESH_VERIFY` success gate | Sequential, fail fast with a static reason; caller reruns the report on failure | Validate aliases, then exclude welded twins from topology |
| `verify_sphere_effective_strict` | Generators plus raw effective vertices/cells/indices | Local-rebuild/effective-array acceptance gate | Parallel cell scan with deterministic sequential-equivalent first reason | No weld map; every effective cell is a face |
| `validate_impl` | `SphericalVoronoi` | Public/reporting validation and `compute_with_report` diagnostics | Accumulate counters and representation notes across the full diagram | Count and validate aliases, then exclude welded twins from topology |

The diagram consumers rely on `SphericalVoronoi`'s structural construction contract for cell-array
cardinality and span bounds. The effective-array gate runs before those arrays become a diagram, so
it alone checks generator/cell cardinality and every raw live span. Those are policy differences,
not missing shared checks.

## Fact matrix

| Fact | Fast diagram gate | Effective-array gate | Accumulating report |
|---|---|---|---|
| Generator/cell cardinality | Assumed by diagram storage | Strict first check | Assumed by diagram storage |
| Vertex finite/on sphere | First failing vertex rejects | First failing vertex rejects | Count every off-sphere vertex |
| Weld canonicality and aliasing | Reject first bad twin; skip twins | Not applicable | Count bad aliases; skip twins |
| Live cell-span bounds | Assumed by diagram storage | Reject first bad span | Assumed by diagram storage |
| Vertex-id bounds | Reject first invalid reference | Reject first invalid reference | Count invalid references and affected cells; omit them from later facts |
| Duplicate ids within a cell | Reject first duplicate | Reject first duplicate | Count affected cells; incidence counts each distinct valid id once |
| Fewer than three distinct valid ids | Reject cell | Reject cell | Count degenerate cells |
| Canonical cell signature | Reject first repeated signature | Sort signatures and reject the sequential-equivalent second cell | Count repeated signatures |
| Fewer than three exact stored positions | Not a strict defect | Not a strict defect | Representation note only |
| Referenced vertex incidence 1/2 | Reject first low-incidence vertex | Reject first low-incidence vertex | Histogram plus low-incidence count |
| Orphan stored vertex | Allowed | Allowed | Representation note only |
| Edge use count/orientation | One combined strict reason | Same combined strict reason | Separate boundary, overuse, and same-direction counters |
| Distinct ids with equal stored positions | Allowed | Allowed | Representation note only |
| Owner-conditioned antipodal arc | Reject first invalid grouped edge | Reject first invalid grouped edge | Count invalid exactly-two-use edges |
| Cell adjacency connectivity | Require one canonical component | Require one effective component | Count canonical components |
| Euler characteristic | Require referenced `V - E + F == 2` | Same effective-space equation | Report exact value |

All consumers intentionally ignore orphan vertices in Euler's `V`. All topology facts operate on
canonical/effective faces, not welded aliases. Exact equal-position edges and cells with fewer than
three stored directions remain representation telemetry rather than abstract-topology failures.

## Already-shared facts

The following are already centralized in `validation.rs`; QUAL-001C must not create parallel
replacements for them:

- `vertex_is_on_sphere`: finite and unit-length tolerance classification;
- `cell_signature`: sorted small-inline/heap boundary identity;
- `edge_key` and `edge_vertices`: undirected endpoint identity;
- `owner_arc_class`: owner-conditioned near-antipodal classification;
- `DisjointSet`: connectivity primitive; and
- `EdgeUse` plus `sort_edge_uses`: strict-verifier grouping and deterministic verdict semantics.

The largest remaining duplication is the per-cell scan and grouped-edge loop. Their facts overlap,
but their execution policy does not: sequential early return, deterministic parallel early return,
and full accumulation require different storage and stopping behavior.

## Ordering and classification differences

The two fail-fast gates have a deliberate reason order.

1. Input-only checks: generator/cardinality where applicable, then stored vertex geometry and weld
   aliases where applicable.
2. Per-cell checks: span, vertex reference/duplicate id, degeneracy, duplicate signature, then edge
   emission.
3. Global checks: low incidence, grouped-edge use/orientation and antipodal geometry, connectivity,
   then Euler.

The effective parallel scan ranks per-cell failures by `(cell, check_rank)` and performs duplicate
signature resolution after chunk collection so its result matches the sequential diagram gate for
the shared no-weld input domain. Any extraction must preserve this ordering, not merely the final
valid/invalid boolean.

The report is intentionally different. It continues after local defects, omits invalid references
from downstream facts, and separates grouped-edge failure classes. Reusing a fail-fast result as a
report fact would lose counts; building report state for a success gate would add allocations and
work.

One currently dead fail-fast classification is worth retaining in the inventory: after duplicate-id
and degeneracy checks have passed, a cycle cannot contain an adjacent `(v, v)` self-loop. Therefore
the strict gates' later `"self-loop edge"` return is dominated by earlier checks. The accumulating
report's `self_loop_edges` counter is still reachable and useful because that traversal records
multiple simultaneous defects. Removal or reordering of the strict branch requires a dedicated
behavior test rather than assumption.

## Negative-control coverage before extraction

| Category | Exact fast/effective differential | Other current coverage | Gap |
|---|---|---|---|
| Valid no-weld diagram | Yes | Public validation integration tests | None |
| Boundary edges / low incidence | Yes, through the one-cell fixture | Plain-gate fault injection | Literal expected reason is not separately asserted |
| Duplicate vertex id | Yes | Plain-gate fault injection | Literal expected reason is not separately asserted |
| Duplicate cell signature | Yes | Plain-gate fault injection | Literal expected reason is not separately asserted |
| Off-sphere finite/non-finite vertex | Effective-only exact reason | Input and checked-storage tests | Diagram/effective differential is intentionally unavailable without violating `SpherePoint` construction invariants |
| Same-direction edge pair | No | Effective gate rejects fault injection | Exact reason/differential missing |
| Overused edge | No | Effective gate rejects fault injection | Exact reason/differential missing |
| Antipodal edge | No | Effective gate rejects fault injection | Exact reason/differential missing |
| Disconnected subdivision / Euler | No | Effective gate rejects fault injection | Need fixtures isolating each ordering outcome |
| Invalid vertex id | No | Checked deserialization rejects malformed storage | Fast/effective exact differential missing |
| Degenerate distinct-id count | No isolated case | General correctness tests require real cells | Exact reason/differential missing |
| Generator/cell mismatch | Not applicable to diagram input | Effective code path only | Effective exact negative control missing |
| Invalid live span | Not applicable to a valid diagram | Checked deserialization has separate span tests | Effective exact negative control missing |
| Weld-map inconsistency | Not applicable to effective space | Public report rejects a corrupt-alias fixture | Fast/report semantic comparison missing |
| Self-loop reason | Structurally dominated in fail-fast order | Report counter and earlier duplicate-id rejection | Prove dominance before deleting strict branch |

The current `effective_strict_matches_fast` test is valuable but not yet a complete extraction
oracle. Several fault-injection tests assert only `is_err()`, so they protect the validity contract
without pinning classification or first-error ordering.

## First safe gate

Before sharing another production fact, expand the no-weld differential fixture matrix for every
category constructible in both representations: invalid vertex id, isolated degeneracy,
same-direction/overused edge groups, owner-conditioned antipodal edges, connectivity, and Euler.
Each fixture should assert the exact static reason from both fail-fast consumers. Add separate
effective-only controls for generator/cardinality and invalid spans, and a focused proof test for
self-loop dominance.

After that matrix is independent, the narrowest production candidate is a typed edge-use
classification (`paired`, `boundary`, `overused`, `same direction`) mapped to the existing consumer
outputs. It is allocation-free and semantically shared by all three consumers, while allowing the
strict gates to retain one combined message and the report to retain separate counters. It still
requires release codegen and counter gates; validation cleanup is not exempt from the established
Pareto rule.
