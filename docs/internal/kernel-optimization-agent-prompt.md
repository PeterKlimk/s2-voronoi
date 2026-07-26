# Read-Only Kernel Optimization Fishing Prompt

You are one of several independent reviewers analyzing the same frozen revision of this repository.
Your task is to find high-leverage optimization hypotheses for the spherical Voronoi construction
kernels. This is a read-only fishing pass, not an implementation pass.

## Hard constraints

- Do not edit, create, delete, or format files.
- Do not produce a patch, pseudo-diff, commit, or implementation plan at statement-by-statement
  granularity.
- Do not read reports produced by other reviewers. Your analysis must be independent.
- Do not run a new benchmark campaign. Use the existing source, telemetry, tests, and recorded
  performance evidence. You may use read-only code-navigation and Git-inspection commands.
- Favor algorithmic and structural reductions in work over local instruction substitutions.
- Treat correctness, determinism, code layout, and cache behavior as first-class constraints.
- If a fact cannot be established from the repository, label it as an unknown rather than assuming
  it.

## Goal

Analyze nearest-neighbor traversal, clipping, and termination as one coupled kernel. Find proposals
that plausibly reduce one or more of:

- candidate dot evaluations;
- retained or materialized candidate keys;
- partitioning and sorting work;
- emitted candidates;
- changed or unchanged clipping operations;
- shell layers or repeated traversal;
- fallback incidence; or
- hot live-state size and consequential memory traffic.

A proposal that merely moves instructions between these categories is not a work-reduction
hypothesis unless it has a clear cache, vectorization, or scheduling mechanism.

## Reading order

Use this order to reduce anchoring:

1. Read the repository `AGENTS.md`.
2. Read the four-file minimal packet below and write private preliminary notes about the data flow
   and possible opportunities before consulting the optimization history:
   - `src/knn_clipping/cell_build/run.rs`
   - `src/cube_grid/packed_knn/scratch/prepare.rs`
   - `src/cube_grid/packed_knn/scratch/emit.rs`
   - `src/knn_clipping/topo2d/clippers/small.rs`
3. Read `docs/internal/kernel-optimization-brief.md`, especially its contracts and focused-context
   routing.
4. Read only the supporting files required to verify your hypotheses.
5. Read `docs/performance.md`, especially “Source-pinned performance decisions,” after forming your
   preliminary hypotheses. Use it to identify collisions with prior work, not to erase an idea whose
   broader mechanism genuinely changes the old cost model.

Do not expand into assembly, reconciliation, or local rebuilding unless you find evidence that a
kernel proposal moves material work across that boundary.

## Analysis requirements

- Explain the current NN–clip–termination data flow accurately enough that another reviewer can spot
  a mistaken premise.
- Return three to five ranked hypotheses. Include fewer rather than padding the report with weak
  ideas.
- Include at least one hypothesis that crosses an existing module or phase boundary.
- For each hypothesis, name the specific source symbols involved, but stay above patch-level detail.
- State the causal mechanism: what work disappears, why it is currently performed, and why the new
  organization can avoid it.
- Identify workload regimes expected to benefit or regress: at minimum ordinary Fibonacci/uniform
  versus clustered/mega or shell-heavy inputs.
- Check the prior-performance ledger and classify overlap as `new`, `related`, or `previously tried`.
  A related or retried idea must explain what changes the previous cost model.
- Give the smallest experiment that could disprove the mechanism before a large implementation.
- Separate proof obligations from empirical risks.
- Name existing telemetry that can validate the mechanism and the minimum new counter, if any.

## Required response format

Use the following headings exactly so a later synthesis pass can compare reports mechanically.

### 0. Review metadata

Record the Git revision, reviewer/model label if known, and any supporting files read beyond the
minimal packet. State explicitly that the worktree was not modified.

### 1. Kernel model

In at most 500 words, describe:

- group preparation;
- per-query frontier production;
- candidate consumption and clipping;
- termination; and
- packed-to-shell takeover.

Call out any premise you could not verify.

### 2. Ranked hypotheses

Provide one summary table:

| Rank | ID | Proposal | Primary reduced work | Expected regimes | Impact | Evidence | Risk | Experiment cost | Prior-work overlap |
|---|---|---|---|---|---|---|---|---|---|

Use `1–5` qualitative scores for impact, evidence, and risk; use `S`, `M`, or `L` for experiment cost.
Do not combine them into a synthetic total score.

### 3. Hypothesis details

For each hypothesis, use this exact template:

#### Hn — Short name

- **Mechanism:**
- **Current work removed:**
- **Affected symbols:**
- **Correctness proof obligations:**
- **Empirical risks:**
- **Expected workload behavior:**
- **Prior-work relationship:**
- **Smallest falsifying experiment:**
- **Telemetry:**
- **Why this is not merely a micro-optimization:**

### 4. Considered but rejected

List plausible ideas you rejected during analysis and the concrete reason for rejecting each. Include
ledger collisions that do not have a changed cost model.

### 5. Cross-cutting unknowns

List missing measurements or unclear invariants that materially affect more than one hypothesis.
Do not request telemetry merely because it would be interesting.

### 6. Recommended first experiment

Choose exactly one hypothesis. Explain why it has the best information value, not merely the largest
imagined upside. Define:

- the reduced-work prediction;
- the smallest implementation boundary;
- semantic validation required;
- workloads and counters;
- success, neutral, and rejection criteria; and
- what result would change your ranking of the remaining hypotheses.

## Review standard

Be inventive but skeptical. A useful report may conclude that the current kernel is already near a
local optimum and recommend only one modest experiment. Novelty without a causal cost model is not
valuable, and consensus with other models will be evaluated later rather than assumed here.
