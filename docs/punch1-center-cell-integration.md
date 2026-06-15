# punch-1 center-cell band-prune — integration design (the "hard part")

Status (2026-06-15): **IMPLEMENTED** on `agent/punch1-axissort` (axis-sort
variant). The dense band-prune center pass is wired into `prepare_group_directed`,
gated on `center_len > DENSE_CELL_THRESHOLD` + a dense sub-index; the completeness
floor flows through `center_bound`/`band_mode` and the shell takeover backstops
everything below the band. NN-contract suite (incl. a new dense-single-cell
scenario) + full `S2_VORONOI_VERIFY=1` topological validation green.

**Validation status (2026-06-15): appears regression-free** across the
distribution matrix at both 500k and 2m (converge A/B vs `main`): a clean ~17×
win on the `cap` pathology, neutral everywhere else (the `grid_rebuilt` gate
keeps the band dormant on the regimes where it would otherwise regress — see
the regression-sweep section below). **NOT yet merged** — held on the branch
for a human correctness review of the kNN-completeness changes before merge.
Pushed to remote for that review.

## Measured outcome (2026-06-15, WSL2/Ryzen 3600, single run)

- `cap` 25k: **106s → ~6.2s (~17×)**. 50k (was: didn't finish) → 23s; 100k → 7s
  (finer rebuild grid splits the cap). `S2_VORONOI_VERIFY=1` passes on all.
- Phase split (cap 25k): `packed_knn` **5722ms → 313ms (~18×)** — the O(occ²)
  `select_partition` is gone. The residual `cell_construction` cost is the shell
  takeover, which fires for ~100% of cap cells: cap cells genuinely need ~3339
  neighbors to terminate (near-equidistant points → slow certificate), and the
  takeover sorts the cell *once* (O(occ log occ)) vs the old per-chunk
  `select_nth` (O(occ²)). So the remaining cap ceiling is the **termination
  certificate depth**, a separate backlog item — NOT the kNN gather.
- `DENSE_BAND_TARGET_COUNT` swept {32,64,128,512,1024}: cap is insensitive
  (takeover dominates regardless); mega 500k best at 64–128. Settled on **128**
  (placeholder; differences within single-run noise, proper quiet-box
  calibration still owed). Larger T is pure overhead on cap.
- uniform 500k: **neutral** (no dense cell → gate never engages).

## Regression sweep + the rebuild gate (2026-06-15)

A converge A/B vs `main` across the distribution matrix found the band path is
NOT universally safe at `DENSE_CELL_THRESHOLD=512`:

- `grid_max_occ` probe (500k): the occupancy rebuild caps mega/splittable/
  bimodal at ~220 (< 512) so the band is **dormant** there; uniform/gradient are
  naturally low. Only `clustered` (occ 1655, `rebuilt=0`) and `outlier` (occ 534,
  `rebuilt=0`) actually engage the band.
- Those two **regressed**: clustered 500k −13.5%, outlier −7.3% (HEAD slower).
- cap-size crossover (isolated dense cell, occ ≈ n): neutral at occ ~2000,
  HEAD 3× faster at 5k, 7.5× at 10k. The band only pays off when cells need a
  **deep certificate** (near-equidistant points → many neighbors to close, e.g.
  cap); on fast-closing moderate clusters main's direct scan never drains the
  cell and the band + takeover is pure overhead.

**Fix — gate the band on `grid_rebuilt`** (`build_query_grid` clears the dense
index when no rebuild fired). A deep-certificate pathology is always highly
concentrated → always trips the occupancy rebuild (Σocc²/n > 500) and survives
it (a cell still over the dense threshold). This is **scale-invariant** (a fixed
occupancy threshold fails because clustered's dense-cell occ grows with n).
After the gate: clustered −13.5% → −2.6% (CI touches 0 = neutral, band dormant
→ identical hot code, residual is cold-code layout noise), outlier −7.3% →
+0.5% (neutral), cap 5k win preserved (2.7×). The `nn_contract` dense gate still
exercises the band directly (it builds the grid via `CubeMapGrid::new`, which
keeps the index regardless of the production rebuild gate).

## Latent bug found + fixed during integration

`CubeMapGrid::compact_welded` (near-coincident weld) rewrites `cell_offsets` /
`point_indices` / `cell_points_*` in place but had left `dense_index` stale,
yielding out-of-range band slots on welded clustered inputs. `compact_welded`
now rebuilds `dense_index` from the compacted arrays.

## Original design (kept for reference)

## Target (measured)

The `cap` dist (everything in one tight cell) is O(occ²): at 25k, ~106s;
50k+ doesn't finish; `grid_max_occ=24500` (the occupancy rebuild *cannot* split
a cap tighter than a max-res cell). Phase split (cap 10k): `packed_knn` is 5309
of 5722 ms of `cell_construction`, of which **`packed_select_partition` = 4480ms
(78%)** + `center_pass` = 767ms. The clip is a non-issue (278ms).

Root: `prepare_group_directed` (scratch/prepare.rs) scans the **whole center
cell for every query in the group** (center_len × num_queries dots, ~line 262),
building per-query candidate lists ~occ long, then `select_nth_unstable` over
~occ (emit.rs:185) — O(occ²) regionally.

## Why a naive band-prune doesn't work

The candidate gather must be **complete to the query's security threshold**
(everything with `dot > security` found, else the kNN/Voronoi is wrong). But
`security_thresholds[qi]` is the dot to the **3×3 boundary** (prepare.rs:150) —
for a dense cell the *entire cell* is inside it, so any band that covers the
security radius = the whole cell. No prune.

## The safe + winning design

Exploit `band_slots`' **superset property** (validated): `band_slots(cell, q, r)`
returns a superset of every point within Euclidean `r` of `q`. So if we process
that band and keep points with `dot > cos_angle(r)`, we have **all** points
within `r` — and `cos_angle(r)` is then a *sound* completeness bound.

For a **dense** center cell (`center_len > DENSE_CELL_THRESHOLD` &&
`dense_index.has(cell)`), per query `qi`:
1. Pick `r` sized for ~a few × `chunk0_size` nearest, from local density:
   `ρ ≈ β·sqrt(T / center_len)` (β = cell cap half-angle from `cell_cap_cos`,
   T = target count), `r = 2·sin(ρ/2)`.
2. `band_slots(cell, q_qi, r, &mut band)` → candidate slots (≈ occ^(2/3), not occ).
3. For each band slot: dot; **apply the directed intra-bin filter** (the
   `qi >= i` self/ordering rule, prepare.rs:282 — required for the stitching
   contract); split into chunk0 (`dot > hi`) / tail (`security < dot ≤ hi`).
4. Set this query's effective completeness bound to `max(security_qi,
   cos_angle(r))` — we have certified completeness only to `r`. The chunk0
   `unseen_bound` / tail handling already carry this downstream.
5. **Grow / fallback**: if the cell builder consumes the whole band without
   closing (needs beyond `r`), it must get the rest. Two options:
   (a) lazy re-band at `2r` (fits the existing chunk0→tail two-stage as a third
   "grow" stage), or (b) fall through to the shell-takeover cursor, which scans
   the full cell once (O(occ), correct, slow — acceptable if rare). Start with
   (b) for simplicity + safety; measure how often it triggers; add (a) only if
   the fallback is hot.

The fast path (non-dense cells) is the existing SIMD batch, untouched →
**uniform pays nothing** (gated on `center_len > DENSE_CELL_THRESHOLD`).

## Soundness

- Completeness rests on the `band_slots` superset property (no point within `r`
  is missed) + the `cos_angle(r)` bound (downstream knows coverage stops at r) +
  the existing cursor/tail for beyond-r. The **NN-contract suite** is the gate;
  add a debug differential (dense path vs the full-cell scan: same neighbor set
  within `r`) if practical.
- The directed filter must be replicated exactly, or the stitching/dedup
  contract breaks (one-sided edges). This is the subtlest part.

## Risk

This is the most correctness-critical path in the engine (kNN completeness). A
subtle error silently corrupts diagrams. Implement carefully, NN-contract-gated,
with the dense-vs-full differential — not as a rushed change.

## Validation plan

- NN-contract suite (all variants) green.
- `cap` dist: O(occ²) → ~O(occ^(5/3)); 25k 106s → target a few s; 50k/100k
  finish; via `bench_voronoi --dist cap`.
- uniform 500k/2M `--converge`: NEUTRAL (gate never engages).
- splittable/mega: neutral-to-better (their cells are rebuild-capped, mostly
  below the gate; no regression).
