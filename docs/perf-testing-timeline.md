# Perf-testing timeline — 2026-06 strict-plane + occupancy arc

When the quiet-box perf pass happens, this says **which commits to A/B and
which to skip**, so paired runs aren't wasted on commits that can't move
release runtime. Built from the git log of the 2026-06-14 session.

## How to use

Paired, interleaved, single-thread (the established protocol):
```
./scripts/bench_build.sh --chain <n>     # build the commit chain
./scripts/bench_run.sh -s 500k -r 20 -m total
./scripts/bench_run.sh -s 2m   -r 12 -m total
```
Box is noisy — use order-rotated paired medians; treat sub-1% deltas as noise
(per the micro-opt matrix finding, per-binary code-layout offsets alone are
±1-2% at 500k ST). For the occupancy anchor, also bench a **clustered** input,
not just uniform.

## Already characterized (do NOT re-test)

- **Sphere strict clip rule** (`CLIP_EPS_INSIDE = 0.0`, pre-session): benched
  perf-neutral (500k +0.6%, 2M −1.7%, within noise). Done.

## Anchor commits (real release-perf candidates — A/B these)

| commit | change | hot path? | what to measure | expectation |
|--------|--------|-----------|-----------------|-------------|
| `ff3528b` | plane strict clip rule (`PLANE_CLIP_EPS_INSIDE = 0.0`); tolerances.rs + types.rs | **plane only** — changes plane clip decisions, output, and repair frequency | plane `compute_plane` build, uniform 500k/2M | neutral (the sphere equivalent was; confirm plane too — repair-rate could shift it) |
| `0def19e` | skip per-clip `eps` sqrt (dead when `base_eps==0`); types.rs `new_unnormalized_base_eps` | **yes, both backends** (one sqrt per neighbor per cell removed) | sphere + plane build, uniform 500k/2M | tiny win (<1%); static estimate caps the sqrt at <1% of total. The batched micro-opt. |
| `33b4962` | occupancy rebuild re-trigger on `Σocc²/n`; policy.rs + compute.rs | **all builds** add an O(cells) `Σocc²` pass; **clustered builds** change drastically | (a) uniform 500k/2M — regression check on the extra pass; (b) a **clustered** input — should be much faster (modest clusters no longer re-grid) | (a) neutral; (b) faster on clustered, unchanged on uniform |

Notes:
- `0def19e`'s sqrt skip is gated on `base_eps==0`, which holds for the plane
  only *because of* `ff3528b`. So on the plane the two are coupled; on the
  sphere `0def19e` stands alone (`CLIP_EPS_INSIDE` was already 0).
- `33b4962` is the one with a genuine *clustered-input* story — that's where
  the win is (the old trigger was 1.5–9× slower on clusters). Don't judge it
  on uniform alone.

## Verify-only (likely neutral; low priority)

- `833477f` — plane report-channel refactor (split `compute_plane` into
  `_built` cores). Touches the plane fast path structurally but should be
  neutral; a quick plane no-regression check suffices.
- `0b081f1`, `336bb01`, `3f877c1`, `404d6f1` — repair-path hardening
  (fixpoint, output-invariant scan, collinear drop, slot-conflict). All
  **defect-gated**: on a clean input `reconcile_unresolved_edges` early-returns,
  so clean-path uniform/plane perf is unaffected. They only change cost on
  *defect-bearing* inputs (rare). Skip for the main pass; bench only if
  profiling a known defect-carrier.

## Skip — zero release-perf impact (don't spend paired runs here)

- **Docs only**: `93c2090`, `50d412a`, `67f3d14`, `841c0e8`, `84fb3c3`,
  `eefb295`, `6056d1e`, `3420d30`.
- **Tests/scripts only**: `c7e1479` (plane campaign harness), `a3a79bc`
  (sweep matrix), `2ae683b` (campaign harness), `19d124d` (sweep + doc).
- `41f5bef` — early-return scan is `#[cfg(debug_assertions)]`, **compiled out
  of release**. Release-neutral by construction.
- `aff8cc3` — removed stale `debug_assert!`s (compiled out of release anyway);
  affects debug-build cost only.
- `2147345` — clippy `collapsible_if`; identical codegen.
- `2c1156e` — `S2_VORONOI_VERIFY` gate, **off by default**; zero cost unless
  the env var is set.

## Suggested minimal pass

Three A/B comparisons cover the whole arc:
1. `0def19e` vs its parent — sphere + plane uniform (the micro-opt).
2. `ff3528b` vs its parent — plane uniform (strict-plane perf).
3. `33b4962` vs its parent — uniform (no-regression) **and** a clustered input
   (the actual win).

Everything else is doc/test/compiled-out/opt-in and needs no run.
