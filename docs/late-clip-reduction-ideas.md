# Late clip reduction ideas

Status: planning note, 2026-06-19.

This note captures the current "do less work" exploration after the passed-edge
geometry audit. It is meant as a branch map for shadow/counter work, not as a
claim that any behavior change is ready.

## Current Read

The remaining work prize is the late reject tail: after the cell is mostly
formed, the builder keeps clipping candidates to prove that no more constraints
matter. The cost profile points at fewer late clips / less candidate production,
not at optimizing seed replay.

The passed-edge geometry idea remains useful as negative evidence:

- Incoming same-bin edge checks average about three edges per cell.
- A timing audit on `agent/passed-edge-geometry-audit` found seed clipping is
  only about 8-11% of total clipping in 100k single-thread samples.
- Final passed edges are active in clean runs, but the connected-chain
  assumption is not generally reliable: fib-like inputs were connected about
  90% of incoming cells, while uniform/splittable were about 74-76%.

So "do not clip passed edges; initialize from their geometry" probably has a
low ceiling and nontrivial topology. The better target is avoiding the last few
clip attempts, where candidates are usually unchanged.

## Candidate Direction Aperture

Candidate direction should be treated as a phase signal. Early in a bin/cell,
the remaining candidate directions can span the full circle. As constraints
accumulate and the cell closes, the effective angular aperture shrinks toward
one or a few narrow gaps. A fixed global circle/radius loses that information.

Working model:

```text
early:   wide 360-degree aperture; clipping is useful
middle:  several open gaps; active edges are mostly found
late:    one/few narrow wedges; most candidates are rejects
```

This favors lazy, wedge-aware certificates over global per-cell certificates.
Pay for directional geometry only when the current exact batch has a narrow
candidate aperture and enough remaining candidates to amortize the setup.

## Ideas To Shadow

### 1. Final-edge rank and phase audit

Measure where final active constraints appear in the stream.

Counters per processed candidate:

- batch source (`PackedChunk0`, `PackedTail`, `ShellExpand`)
- stream rank within the cell
- clip result (`Changed` / `Unchanged`)
- whether the candidate survives as a final edge
- polygon vertex count before the clip
- whether the cell was bounded before the clip
- consecutive unchanged count

Questions:

- Are final edges almost always in the first 6-8 processed candidates?
- How much clipping happens after the last final edge was discovered?
- Does the late reject tail correlate with shell/tail sources or dense regimes?

If final edges are early, certificates are the main path. If final edges are
often late, clip ordering/candidate production deserve more attention.

### 2. Candidate aperture audit

For each exact batch, and for suffixes after each processed candidate, measure
the angular spread of remaining candidate half-plane normals in local `(u,v)`
coordinates.

Counters:

- remaining candidate count
- occupied sector count for K = 8, 16, 32
- minimum covering arc length / aperture
- whether all remaining candidates were actually unchanged
- bucket by bounded state, vertex count, batch source, and consecutive unchanged

Questions:

- Do most skip opportunities occur after aperture becomes narrow?
- Which K is enough to capture the signal?
- What gates avoid paying aperture work on the easy early phase?

### 3. Lazy wedge support certificate

A conservative batch-level certificate: compute support only for sectors that
are occupied by the remaining candidates in the current exact batch.

For a candidate half-plane:

```text
a*u + b*v + c >= -eps
```

Quantize `(a,b)` into a sector. For each occupied sector, compute:

```text
support_lb = min projection of current polygon on sector center
             - angular_error * polygon_radius
```

Then a sector can skip candidates whose `support_lb + c >= -eps` under the
appropriate normalization/bound. This is the old support-envelope idea, but
batch/wedge-gated instead of maintained globally or checked before every clip.

Shadow questions:

- How many exact unchanged candidates would this prove?
- How often does the sector table need to be built?
- What is the best activation gate: bounded, `poly.len >= 5`, suffix count,
  aperture threshold, consecutive unchanged, shell/tail source?

### 4. Different clip order, only after safety gates

Nearest-first is valuable because it shrinks the polygon early and keeps simple
frontier bounds. A reorder should not cross an unseen-bound frontier without a
sound certificate.

Potential safe experiments:

- Keep nearest-first until bounded.
- Reorder only within an exact batch.
- Within similar dot-distance bands, prefer angular diversity.
- In late phase, prioritize candidates facing the largest current support gap.

Shadow questions:

- Would an angular-diverse order discover final edges earlier?
- Does reordering reduce total `Changed`/`Unchanged` attempts in replay?
- Does it hurt termination by delaying near constraints?

### 5. Coarse outer geometry

A coarse geometry test is useful only if it proves the true polygon is inside
the candidate half-plane. For "candidate does not cut", an inner approximation
is not sufficient. A conservative outer approximation can be safe but
pessimistic.

Shapes worth shadowing:

- scalar circle/radius: cheap, already close to current termination proof
- AABB / fixed slabs in local coordinates: cheap but orientation-dependent
- sector slab fan: stronger and naturally aligns with candidate aperture
- batch-aligned wedge/slabs: promising if aperture closes late

The best near-term version is probably the lazy sector fan from idea 3, because
it computes only directions that appear in the current candidate suffix.

## Guardrails

- Prefer shadow counters before behavior changes.
- Keep production correctness unchanged until a certificate is conservative.
- Do not retry late per-candidate support probing unless the trigger is much
  cheaper than the skipped clip.
- Do not optimize only fib/uniform; include splittable/mega and at least one
  dense rebuilt case.

## Suggested Branch Order

1. `agent/final-edge-rank-audit`
2. `agent/candidate-aperture-audit`
3. `agent/lazy-wedge-support-shadow`
4. Only then behavior branches for wedge certificate or intra-batch ordering.

