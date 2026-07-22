# Paper research notebook

This directory holds evidence, experiment plans, raw measurements, and working notes for a paper
about `voronoi-mesh`. It is intentionally not a manuscript: the paper's author writes and approves
every line of published prose.

The notebook has four jobs:

1. keep proposed contributions separate from established facts;
2. attach prior-art evidence to every novelty claim;
3. make performance results reproducible from a commit, command, machine, and output contract; and
4. preserve negative and inconclusive results instead of selecting only favorable measurements.

## Contents

- [claims-and-scope.md](claims-and-scope.md): candidate thesis, terminology, and claims to avoid.
- [prior-work.md](prior-work.md): literature and implementation comparisons, with source evidence.
- [experiments.md](experiments.md): comparison tiers, ablations, workloads, and required metrics.
- [results/](results/): machine descriptions and immutable benchmark campaign artifacts.

## Working rules

- Notes may suggest claims, but only a completed literature search and supporting experiments can
  promote them into paper claims.
- Record unsuccessful experiments and correctness failures alongside successful ones.
- Never compare timings from different machines as a direct speed ratio.
- Every benchmark must identify whether it measured independent cell geometry, an explicit shared
  mesh, or a checked/certified result.
- Preserve raw command output. Derived tables must name the raw inputs and analysis procedure.
- Record dirty working trees explicitly; prefer committed revisions for publication results.
- Keep citations as DOI, archival URL, or commit-pinned source links rather than recollection.

The initial notebook was created on 2026-07-23 on branch `agent/paper-research`.
