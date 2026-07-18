# Validation fact inventory

**Status:** QUAL-001C inventory and first negative-control expansion, 2026-07-19

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

The first post-inventory extraction is also now shared: `EdgeUseClass` and `classify_edge_uses`
classify paired, boundary, overused, and same-direction groups. Both strict gates still map every
non-paired class to their existing combined reason, while the report maps the same classes to its
three separate counters.

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
| Boundary edges / low incidence | Yes, with literal reason | Report pins the boundary-edge subclass | None for the shared strict reason |
| Duplicate vertex id | Yes, with literal reason | Plain-gate fault injection | None |
| Duplicate cell signature | Yes, with literal reason | Plain-gate fault injection | None |
| Off-sphere finite/non-finite vertex | Effective-only exact reason | Input and checked-storage tests | Diagram/effective differential is intentionally unavailable without violating `SpherePoint` construction invariants |
| Same-direction edge pair | Yes, with combined strict reason | Report pins the same-direction subclass | None for edge-use extraction |
| Overused edge | Yes, with combined strict reason | Report pins the overused subclass | None for edge-use extraction |
| Antipodal edge | Yes, with literal reason | Effective gate rejects fault injection | None |
| Disconnected subdivision / Euler | Yes; isolated fixtures pin both ordering outcomes | Effective gate rejects fault injection | None |
| Invalid vertex id | Yes, with literal reason | Checked deserialization rejects malformed storage | None |
| Degenerate distinct-id count | Yes, with literal reason | General correctness tests require real cells | None |
| Generator/cell mismatch | Not applicable to diagram input | Effective-only literal reason pinned | None |
| Invalid live span | Not applicable to a valid diagram | Effective-only literal reason pinned; deserialization has separate tests | None |
| Weld-map inconsistency | Not applicable to effective space | Public report rejects a corrupt-alias fixture | Fast/report semantic comparison missing |
| Self-loop reason | Structurally dominated in fail-fast order | Exhaustive small-cycle proof plus representative literal reasons | Report telemetry remains independently covered |

The expanded test-only matrix now pins every safely constructible fail-fast reason shared by the
no-weld diagram/effective domain, isolates connectivity from Euler with a connected 3x3 toroidal
quadrangulation, and pins the effective-only structural-input reasons. Separate report fixtures pin
boundary, overused, and same-direction edge counters. The weld-map comparison remains deliberately
outside the effective domain and should be completed before sharing weld-specific facts.

## First accepted fact and next gate

The typed edge-use classification was accepted after the no-weld differential matrix and report
subclass controls were independent. It is allocation-free and preserves all consumer outputs. The
release artifact added 12 text bytes, removed 16 BSS bytes, and remained neutral across seven
instruction/branch counter pairs.

The next narrow candidate is a typed internal strict-failure reason replacing duplicated static
strings in the two fail-fast consumers. It must preserve the effective parallel scan's
`(cell, check_rank)` ordering, the exact strings pinned by the oracle, and the report's independent
counter taxonomy. Introduce it without routing the accumulating report through fail-fast state.
