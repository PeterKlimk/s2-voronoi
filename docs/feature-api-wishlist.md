# Feature and API Wishlist

**Status:** uncommitted design catalogue

This document collects useful caller-visible capabilities that are not release commitments. The
promoted roadmap lives in [`ROADMAP.md`](../ROADMAP.md), and [`work-log.md`](work-log.md) remains the
authoritative source for active and decision-gated work. Entries here intentionally describe the
shape of a feature and its safety boundary without prematurely fixing an API.

Specialized outputs should have explicit types and contracts. Hints may accelerate a certified
computation but must never become correctness authorities, and an incremental result must not imply
that spherical Voronoi influence is local without a certificate.

## Positive-threshold edge collapse

Exact stored-zero edges are already detected and safely contracted when that does not erase an
effective generator cell. The wishlist extension is an explicit positive epsilon that lets graphics
or physics consumers remove represented nonzero slivers.

This is existing decision-gated work, not a new proposal: `RES-002` in
[`work-log.md`](work-log.md#res-002--optional-positive-threshold-edge-simplification) is canonical.
It needs threshold units, option and report fields, component-diameter and geometric-deviation
certificates, and a distinct cell-mesh result whose contract is a valid simplified spherical cell
complex rather than the original exact Voronoi subdivision. The default remains `Preserve`.

## Fused cell-local consumers

Some callers want areas, centroids, Lloyd displacements, or an objective reduction and immediately
discard the shared diagram. A specialized entry point could consume each certified cell while it is
hot and avoid materializing the complete shared-vertex graph. An adjacency- or Delaunay-only result
is a related output-shaped pipeline.

This should be a separate API, not a hidden mode of `compute`: it needs a result and report contract,
deterministic reduction rules, and a clear answer for defects that currently require post-assembly
reconciliation or Local3d. The ordinary full-diagram path and its guarantees remain unchanged.

## Temporal topology hints

A repeated computation could accept the previous diagram or adjacency as a hint. For each site, try
the previous neighbors first, then let the current spatial query and termination certificate prove
that no required constraint was omitted. Invalid, stale, or low-value hints simply fall back to the
normal build.

The API needs stable input identity across frames, explicit behavior for inserted, deleted, welded,
or reordered sites, and telemetry showing hint acceptance and fallback work. Its first useful target
is slowly moving sites and iterative relaxation; it should not promise that topology is unchanged.

## Certified partial rebuild

A regional update API would accept a previous result plus changed generators and return a new full
result, or a patch with explicit application semantics. The difficult part is discovering the dirty
region soundly: moving one site is both a deletion from its old neighborhood and an insertion into
its new one, and its influence can propagate beyond an assumed radius.

A plausible design seeds from old and new neighborhoods, rebuilds those cells, and expands across
the boundary until current query certificates prove that unchanged exterior cells cannot gain or
lose constraints. It may also need a delta overlay or partial rebuild for the cube-grid index.
Temporal hints can reduce the work inside the dirty region, but they do not certify its boundary.

The API must define provenance and index stability, patch atomicity, fallback to a full rebuild, and
how preprocessing weld-class changes invalidate old mappings. A whole-result fingerprint and
differential tests against a fresh full computation are required before exposing incremental
success.

## Promotion rule

A wishlist item moves into [`ROADMAP.md`](../ROADMAP.md) when its caller and compatibility value are
clear. It moves into [`work-log.md`](work-log.md) when the next policy decision or implementation
slice is concrete. Until then, names and signatures in this document are illustrative only.
