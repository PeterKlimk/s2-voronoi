# Escalation Build State (Updated 2026-06-22)

Resume anchor for local repair / exact-reference work. This document supersedes
the earlier projected-vs-raw diagnosis.

## Current Status

- Default repair is **normalized local 3D repair**:
  `repair_local_hull` builds one normalized local 3D hull over the repaired
  closure's gather, splices the repaired closure, grows on residuals, and accepts
  only if whole-diagram validation is strictly valid.
- The repair is default-on via `VoronoiConfig::repair_mode =
  RepairMode::Local3d`, with a valid-or-revert gate. `RepairMode::LocalProjected`
  remains available as an A/B diagnostic path.
- Known mega 100k defects (seeds 1, 2, 15) are repaired to strict validity.
- Broad sweep: 25 defective inputs repaired, clean controls remain valid.

## Major Correction: Normalize Before Exact 3D

The earlier conclusion that local 3D hull repair was the wrong metric was based
on a bad reference: exact 3D hulls were computed on raw f32 coordinates. Those
coordinates have tiny radius drift, and exact predicates preserve that drift.
The result is a Euclidean off-sphere hull, not the S2 Delaunay graph.

For this crate's contract, exact 3D construction must use normalized directions.
Once the 3D oracle f64-renormalizes inputs:

- fast gnomonic output matches normalized CGAL on tested uniform and mega inputs;
- normalized `LocalHull` matches CGAL;
- normalized local 3D repair matches projected repair on known mega defects, and
  is now the default final backstop to avoid projected-chart failure modes.

So: the issue was **not projection drift**. It was failure to normalize before
exact 3D construction.

## Probe Results

CGAL external exact hull probe:

```bash
g++ -O3 -std=c++17 scripts/cgal_hull3.cpp -lgmp -lmpfr -o /tmp/cgal_hull3
```

Fast vs normalized CGAL:

| distribution | n | seed | changed |
|---|---:|---:|---:|
| uniform | 100k | 3 | 0 |
| uniform | 500k | 3 | 0 |
| uniform | 1m | 3 | 0 |
| mega | 12k | 3 | 0 |

Normalized `LocalHull` vs CGAL:

| distribution | n | seed | changed |
|---|---:|---:|---:|
| uniform | 100k | 3 | 0 |

Known mega repair comparison:

| seed | projected repair | normalized local 3D repair |
|---:|---:|---:|
| 1 | valid, 7 spliced, 2 rounds | valid, 7 spliced, 2 rounds |
| 2 | valid, 15 spliced, 2 rounds | valid, 15 spliced, 2 rounds |
| 15 | valid, 9 spliced, 2 rounds | valid, 9 spliced, 2 rounds |

## Active Probe Switches

- `S2_CGAL_HULL3_BIN=/tmp/cgal_hull3`: enables ignored CGAL reference probes in
  `tests/escalate.rs`.
- `S2_ESCALATE_DELAUNATOR=1`: under `escalate_probe`, routes repair through the
  older global/projected delaunator oracle.

## Recommended Next Work

1. Run the broad repair sweep with the `RepairMode::Local3d` default.
2. Compare closure sizes, rounds, strict-valid acceptance, and runtime against
   explicit `RepairMode::LocalProjected`.
3. Keep CGAL retained as an external audit probe for normalized 3D truth.
4. For exact-by-construction mode, use normalized 3D hull / local hull as the
   graph reference. Do not use raw f32 3D hulls as a spherical oracle.

## Code Map

- `src/knn_clipping/compute.rs`: repair-mode wiring and valid-or-revert gate.
- `src/knn_clipping/escalate.rs`: normalized local 3D repair, projected A/B
  repair, and grow/splice machinery.
- `src/knn_clipping/local_hull.rs`: local 3D hull; now f64-renormalizes points
  before exact predicates.
- `scripts/cgal_hull3.cpp`: external CGAL exact hull reference utility.
- `tests/escalate.rs`: CGAL/reference/local-repair probes.
- `tests/escalate_local.rs`: default repair broad sweep.
