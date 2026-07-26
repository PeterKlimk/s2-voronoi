# Local-Rebuild Trigger Hunt

**Status:** stable rejected-transaction fixture selected; no natural accepted fixture found,
2026-07-20

This search supports the candidate transaction selected in
[`code-quality closeout`](code-quality-closeout.md#local-rebuild-transaction). Its purpose is
measurement and regression coverage, not to redefine which adversarial inputs must succeed.

## Signal and protocol

The search used the release `bench_voronoi` artifact at the `95ab99b` source boundary and the
existing `VORONOI_MESH_LOCAL_REBUILD_DEBUG` hook. That environment lookup and its first output occur
only after the real production trigger, so absence of the line means the computation did not enter
the local-rebuild attempt.

Discovery cases used `RAYON_NUM_THREADS=1`. Wide seed sweeps ran four independent single-threaded
processes concurrently because wall time and counters were irrelevant during discovery. Every hit
was then rerun serially before classification.

## Search matrix

No default-preprocessed case triggered:

| Regime | Cases | Triggers |
|---|---:|---:|
| 100k, fraction 0.8, pole/edge/corner, seeds 1–500 | 1,500 | 0 |
| 100k, edge/corner, fractions 0.50/0.65/0.80/0.90/0.95, seeds 1–10 | 100 | 0 |
| 100k, bin counts 16/32/48/64/96, all three placements, fractions 0.65/0.80/0.95, seeds 1–6 | 270 | 0 |
| 300k, all placements, fractions 0.65/0.80/0.95, seeds 1–5 | 45 | 0 |
| 500k, all placements, fractions 0.65/0.80/0.95, seeds 1–4 | 36 | 0 |
| 1M, all placements, fraction 0.8, seeds 1–3 | 9 | 0 |

Disabling preprocessing exposed rare corner-cap triggers:

| Regime | Cases | Trigger seeds | Accepted |
|---|---:|---|---:|
| 100k, pole, fraction 0.8, seeds 1–1000 | 1,000 | none | 0 |
| 100k, edge, fraction 0.8, seeds 1–250 | 250 | none | 0 |
| 100k, corner, fraction 0.8, seeds 1–1000 | 1,000 | 224, 485, 540 | 0 |
| 100k, corner, fraction 0.95, seeds 1–1000 | 1,000 | 407, 899 | 0 |
| 100k, corner, fraction 0.65, seeds 1–1000 | 1,000 | 224 | 0 |

The overlapping rows are intentional parameter searches rather than a claim of unique-case count.
All six triggered runs reached productive overlay growth and the candidate transaction, then the
whole-diagram gate rejected a remaining low-incidence vertex.

## Selected fixture

Use this production Hull3d command:

```bash
VORONOI_MESH_BENCH_CAP_CENTER=corner \
VORONOI_MESH_LOCAL_REBUILD_DEBUG=1 \
RAYON_NUM_THREADS=1 \
target/release/bench_voronoi 100k --no-preprocess --dist mega --dist-param 0.8 --seed 224
```

Three serial confirmations produced the same structural fingerprint:

- 8 normalized defect pairs from 11 unpaired records;
- a low-incidence trigger;
- 2 Hull3d growth rounds;
- 7 spliced generators and 1 stuck component;
- 5 final implicated generators;
- a fully materialized 100,000-cell candidate with 200,002 vertices; and
- strict-gate rejection for a low-incidence vertex, followed by the expected plain-compute error.

The benchmark executable panics because it expects plain `compute` to succeed. That occurs after
the rejected transaction has restored the original geometry. A counter harness for this fixture
must accept the expected nonzero exit and verify the structural debug fingerprint; it must not
silently treat an early failure as a complete sample.

The trigger is not monotonic in `n`. With the same seed, placement, fraction, and policy, 20k
through 99k were clean in coarse/thousand-point probes, while 100k triggered, 110k was clean, and
120k/150k/200k triggered. Finer probes also found a mixture of clean and triggered sizes between
99,100 and 100,000. Keep 100k as the convenient stable fixture; do not describe it as a minimized
threshold.

The feature-gated global projected-Delaunay oracle also rejected the 100k/seed-224 candidate with a
remaining low-incidence vertex. No natural accepted transaction was found. Accepted commit/swap
semantics therefore still require a direct deterministic transaction test rather than a mislabeled
mega benchmark.

## Gate consequence

The transaction refactor can now use three complementary checks:

1. 500k Fibonacci counters protect the ordinary no-trigger layout.
2. The selected seed-224 fixture measures the productive materialize/validate/rollback path.
3. A direct accepted-candidate test pins retained minted vertices, paired cell/index replacement,
   and returned resolution-scan cells.

The rejected fixture is not a reason to weaken the valid-or-error contract or tune Hull3d. Oracle
quality and acceptance-rate work remain separate from the structural cleanup.

The accepted transaction extraction used this fixture for seven interleaved candidate/parent
counter pairs. Every sample reproduced the complete fingerprint above. Candidate/parent ratios
were `0.999991525` instructions and `1.000000202` branches, with zero context switches and CPU
migrations in every sample. The productive materialize/validate/rollback path is therefore
counter-neutral after the ownership extraction.
