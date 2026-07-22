# Experiment plan

This is the initial plan. Freeze exact commands, revisions, compiler options, and datasets in each
campaign under `results/` before collecting publication measurements.

## Comparison tiers

Results from different tiers must not be presented as direct speed ratios.

### G: independent geometry

Nearest-neighbor discovery and per-cell clipping, returning cell/facet geometry without requiring
a globally conforming indexed mesh. This is the closest tier to Caplan et al.'s published scaling
experiment.

### M: explicit shared mesh

Construction of indexed vertices, cells, and shared-edge topology. Compare against CGAL and, if the
path is usable, Vortex with explicit mesh storage and merging enabled.

### C: checked result

Full production outcome including required reconciliation, local recovery, and the chosen topology
checks. State precisely whether the timer includes every check used to make the success claim.

## Principal comparisons

1. **Vortex, same machine, tier G:** common input coordinates, zero weights, matching thread counts,
   warmup policy, compiler optimization, and as closely matched numerical precision as practical.
2. **CGAL, same machine, tier M:** common input and explicit output; report semantic differences.
3. **Production engine, tier C:** end-to-end throughput and memory with the full stated contract.
4. **Production versus independent reference:** quantify the cost of topology and the work recovered
   by ownership filtering/forwarding inside one codebase.

Published numbers from another machine may be shown for context, never as a direct speedup ratio.

## Mechanism ablations

Every timed configuration must remain correct under its stated output contract.

| ID | Query ownership | Incoming edge seeds | Assembly | Purpose |
|---|---|---|---|---|
| A0 | unrestricted | none | post-construction | independent reference |
| A1 | unrestricted | enabled | same as production | isolate seed-first effects without omitting candidates |
| A2 | ownership-filtered | enabled | production | complete coupled mechanism |

Ownership filtering without forwarded constraints is not a valid construction and must not be
reported as an ablation result. Additional internal controls should disable one optimization at a
time only where the replacement path preserves the same semantics.

Useful secondary ablations:

- packed/SIMD versus scalar neighbor-query backend;
- streamed/resumable candidates versus a fully materialized neighbor list;
- sharded live dedup versus independent polygon output plus post-hoc assembly;
- reconciliation disabled on workloads proven not to require it, for cost attribution only;
- validation excluded/included, with both numbers reported rather than selectively chosen;
- one shard/thread versus the production shard policy.

## Workloads

Minimum publication matrix:

- random points on the sphere;
- Fibonacci or subdivided-icosahedron uniform points;
- Lloyd-relaxed/centroidal points;
- clustered and strongly nonuniform points;
- near-coincident/adversarial points as a correctness and recovery campaign;
- sizes spanning cache-resident through memory-pressure regimes;
- one thread and a thread-scaling sweep through all physical cores.

Use common serialized datasets for external comparators. Record generation algorithm, seed,
normalization, ordering, and any preprocessing separately.

## Metrics

### Primary

- wall-clock time and generators/second;
- peak resident memory and bytes/generator;
- success/outcome counts under the stated topology contract;
- scaling versus input size and thread count.

### Work attribution

- spatial-index construction time;
- neighbor-query time and candidate dot products;
- candidates emitted and consumed per generator;
- bisector clip attempts and clips that change a cell;
- same-shard pairs forwarded;
- independently reconstructed cross-shard pairs;
- cell-construction, deduplication, reconciliation, assembly, and validation times;
- local rebuild count and affected generators;
- raw and final vertex/edge/cell counts.

### Statistical protocol

- warm up before measured runs;
- interleave comparator order to reduce thermal and temporal bias;
- use several seeds for randomized distributions;
- retain every measured sample;
- report median and spread/interval, not only the best run;
- use CPU pinning for single-thread comparisons where available;
- treat effects below the measured machine/code-layout noise floor as unresolved.

The existing `scripts/bench_build.sh` and `scripts/bench_run.sh` implement an interleaved internal
comparison protocol and structured CSV output. Publication campaigns must copy their generated
artifacts from `/tmp` into the appropriate immutable results directory.
