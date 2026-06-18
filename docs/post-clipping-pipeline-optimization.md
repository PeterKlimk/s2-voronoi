# Post-Clipping Pipeline Optimization

Status: open, 2026-06-18. Two branch-worthy ideas after a full timing
breakdown of `cell_construction` (580.8ms at 500k, uniform input).

## Timing Breakdown (500k uniform, single-threaded)

```
cell_construction:   580.8ms (85.1%)
    knn_query:           0.2ms ( 0.0%)
    packed_knn:        158.6ms (27.3%)
      select_sort        43.0ms    ← sorting networks already; microbenched
      ring_pass          38.3ms    ← cap-prune tried, measured loss
      center_pass        25.3ms
      select_scatter     19.7ms
      select_prep        18.0ms
      ring_thresholds     2.6ms
      ring_fallback       2.1ms    ← lazy; rarely fires
    clipping:          218.7ms (37.7%) ← per-N SIMD clippers; near-optimal
    certification:      55.2ms ( 9.5%)
    key_dedup:          50.6ms ( 8.7%) ← O(1) per vertex; near-optimal
    edge_collect:       27.7ms ( 4.8%)
    edge_resolve:       27.7ms ( 4.8%) ← edge-check matching; parked
    edge_emit:          42.1ms ( 7.3%)
    neighbors: mean=8.9 max=141
```

## Already Near-Optimal (Do Not Touch)

| slice | ms | why |
|---|---|---|
| clipping | 218.7 | Per-N monomorphized (N=3..8), SIMD distance eval via `wide::f64x4`, branchless bitmask, early-unchanged check (`clippers.rs:217-238`). Escalation is dead code in production (`CLIP_ESCALATION_FACTOR=0.0`). ~49ns/clip. |
| packed_knn.select_sort | 43.0 | Hand-tuned sorting networks for N≤35 (`sort.rs:345`, `generated/sort_nets.rs`); `select_nth_unstable` for large remainders. Microbenched (`bin/microbench_packed_select.rs`). |
| packed_knn.ring_pass | 38.3 | SIMD-8 dots over 8 ring cells. Per-(ring cell, query) cap-prune was tried and measured as a net loss (`prepare.rs:472-476`). |
| key_dedup | 50.6 | O(1) per vertex: one `Vec` lookup (`generator_bin[key[0]]`), one branch, one push (`emit.rs:170-206`). No hashmap, no sort. ~8ns/vertex. |
| edge_resolve | 27.7 | Linear scan over ~3 incoming edge checks. Already near-optimal for sane regime; see `docs/edgecheck-matching-optimization.md` (parked). |

## Branch-Worthy Ideas

### 1. Precompute edge thirds at certification

**Branch:** `agent/cert-edge-thirds`

**Problem:** `edge_emit` recomputes the "third" generator for each edge
endpoint by loading the vertex key and XOR-ing
(`src/live_dedup/emit.rs:68-71`):

```rust
let (a, b) = unpack_edge_key(entry.key);
let thirds = [
    third_for_edge_endpoint(cell_vertices[locals[0] as usize].0, a, b),
    third_for_edge_endpoint(cell_vertices[locals[1] as usize].0, a, b),
];
```

Each `third_for_edge_endpoint` (`edge_checks.rs:24-37`) loads a `[u32; 3]`
vertex key from `cell_vertices` and does `key[0] ^ key[1] ^ key[2] ^ a ^ b`.
That's 2 vertex-key cache loads per edge (~6 edges → ~12 loads per cell).

At certification time (`to_vertex_data_full`, `extract.rs:44-102`), the
third is available **in registers**: for vertex `i` with plane neighbors
`n1, n2` (`extract.rs:73-78`) and edge neighbor `E` (`extract.rs:90,99`),
the third is the one of `{n1, n2}` that is not `E` — i.e.
`n1 ^ n2 ^ edge_neighbor`, all already loaded.

**Change:** Store a per-vertex `edge_thirds: [[u32; 2]]` in `CellOutputBuffer`
(`src/live_dedup/cell_output.rs:43-48`) at `extract.rs:82`, computed inline
during certification. `edge_emit` reads from this array instead of
recomputing via vertex-key loads.

**Layout:** Two `u32` per vertex = 8 bytes/vertex alongside the existing
`edge_neighbor_{globals,slots,eps}` arrays (4+4+4 = 12 bytes/vertex). Fits
naturally; one more `Vec<u32>` (flattened) or `Vec<[u32;2]>`.

**Expected saving:** ~10-15ms (eliminates ~12 cache loads per cell across
500k cells). Low risk — pure precomputation, no algorithmic change, no
correctness sensitivity (the thirds are the same values, just computed
earlier).

**Measurement:** perf counters should show reduced `L1-dcache-load-misses`
and `cache-misses` in `edge_emit`; instruction count roughly flat (XORs
moved, not eliminated). Use paired/interleaved wall-clock runs to confirm.

### 2. SIMD certification back-projection

**Branch:** `agent/cert-simd-project`

**Problem:** `to_vertex_data_full` (`extract.rs:22-106`) does the 2D→3D
gnomonic back-projection **scalar per vertex**, despite the polygon already
being SoA (`us: [f64; 64]`, `vs: [f64; 64]`, `types.rs:100-101`).

The projection per vertex (`extract.rs:49-65`):
```rust
let dir = glam::DVec3::new(
    fp::fma_f64(u, basis.t1.x, basis.g.x),
    fp::fma_f64(u, basis.t1.y, basis.g.y),
    fp::fma_f64(u, basis.t1.z, basis.g.z),
);
let dir = glam::DVec3::new(
    fp::fma_f64(v, basis.t2.x, dir.x),
    fp::fma_f64(v, basis.t2.y, dir.y),
    fp::fma_f64(v, basis.t2.z, dir.z),
);
```

6 `fma_f64` ops per vertex, scalar. The basis vectors `(t1, t2, g)` are
shared across all vertices in a cell (cell-constant).

**Change:** Process 4 vertices at once with `wide::f64x4` (already a
dependency, `Cargo.toml:50`, used in `src/fp.rs`):

- Load 4 contiguous `u` values from `poly.us[i..i+4]` (SoA → contiguous).
- Load 4 contiguous `v` values from `poly.vs[i..i+4]`.
- Broadcast basis components (cell-constant, in registers).
- 6 SIMD `fma` ops (instead of 6×4 = 24 scalar `fma_f64`).
- SIMD `length_squared` + degeneracy check + `sqrt`/`recip`/normalize.

For `poly.len = 6` (typical): 2 batches (4+2) vs 6 scalar iterations.
The scalar tail (sort3_u32 key, 4 buffer pushes, edge metadata) stays
scalar — those are gather/scatter, not worth vectorizing.

**Expected saving:** ~15-20ms (the projection + normalization is ~30-40%
of per-vertex certification cost). Clean diff — SoA layout is already
there, `wide::f64x4` is already used, basis vectors are cell-constant.

**Risk:** The degeneracy guard (`len2 < EXTRACT_DEGENERATE_LEN2`,
`extract.rs:68-70`) must still produce the exact same error on the same
inputs. SIMD comparison + horizontal-any preserves this, but the error
*path* must identify which vertex failed — need a scalar re-check on the
failing batch (rare, cold path).

**Measurement:** perf counters should show reduced `instructions` and
`cycles` in certification; `L1-dcache-load-misses` flat (same data, fewer
loads). Confirm bit-identical output via `tests/correctness.rs` and
`tests/validation.rs`.

### 3. Structural: fuse post-clipping passes (not yet branch-worthy)

The post-clipping pipeline runs **four separate passes** over the same ~6
vertices + ~6 edges:

| pass | time | iterates |
|---|---|---|
| certification | 55.2ms | all vertices (projection, key, edge metadata) |
| collect+resolve | 55.4ms | all edges (classify, match incoming checks) |
| key_dedup | 50.6ms | all vertices (owner lookup, index assignment) |
| edge_emit | 42.1ms | ~3 later edges (push edge checks) |

The data is L1-resident across passes (~18 cache lines for a 6-vertex cell),
so fusing saves **loop overhead, branch mispredictions, and Vec iteration
setup** across 500k cells — not cache misses.

**Blocker:** `edge_emit` needs `vertex_indices` from `key_dedup`, and
`collect+resolve` runs before `key_dedup`. Fusing requires reordering the
data flow: `collect_and_resolve` would leave a per-vertex "pending emit"
tag instead of a separate `edges_to_later` vec; `key_dedup` would emit
each edge once both endpoint indices are assigned.

**Estimated saving:** ~10-15ms (one fewer pass over ~6 elements × 500k
cells). Higher risk — the path propagates vertex indices between cells
(correctness-sensitive), and the diff touches `emit.rs`, `edge_checks.rs`,
and `cell_output.rs`. Not branch-worthy until (1) and (2) are measured.

### 4. The real big lever: fewer neighbors (mean=8.9)

Reducing mean neighbors from 8.9 to ~7 would save ~20% of **both** clipping
(218.7ms) and packed_knn (158.6ms) — ~75ms. This is the existing
"directional certificate" thread (`docs/optimization-ideas.md:35`, parked
promising, branch `agent/directional-certificates`). Not a new idea; not
in scope here.

## Suggested Branch Order

1. **`agent/cert-edge-thirds`** — lowest risk, ~10-15ms, pure
   precomputation. Can be done in isolation.
2. **`agent/cert-simd-project`** — clean, ~15-20ms, SoA already there.
   Independent of (1).
3. Together: ~25-35ms out of 580ms (~4-6%), two clean independent branches.
4. (3) only if (1)+(2) land and the fused-pass idea still looks worth the
   complexity.

## Measurement Protocol

Wall-clock time is unreliable on this machine for sub-percent changes.
Use hardware counters as primary evidence.

**Per-branch baseline + measurement:**
```bash
# Build both branches + main for paired comparison
./scripts/bench_build.sh main agent/cert-edge-thirds agent/cert-simd-project

# Hardware counters (primary evidence)
RAYON_NUM_THREADS=1 perf stat -e cycles,instructions,\
cache-references,cache-misses,\
L1-dcache-load-misses,L1-dcache-loads,\
LLC-load-misses,LLC-loads,\
branch-misses,branches \
  cargo run --release --features tools --bin bench_voronoi -- 500k --no-preprocess

# Timing breakdown (secondary, for slice-level attribution)
S2_VORONOI_TIMING_KV=1 RAYON_NUM_THREADS=1 \
  cargo run --release --features tools,timing --bin bench_voronoi -- 500k --no-preprocess

# Interleaved wall-clock runs (tertiary, noisy)
./scripts/bench_run.sh -s 500k -r 20 -m total
```

**Correctness guards (every branch, before measuring):**
```bash
cargo test --release --test api --test correctness --test validation
cargo test --release --test adversarial
cargo clippy
cargo fmt
```

**What to look for in counters:**
- (1) cert-edge-thirds: reduced `L1-dcache-load-misses` and `cache-misses`
  in `edge_emit`; `instructions` roughly flat (XORs moved, not eliminated).
- (2) cert-simd-project: reduced `instructions` and `cycles` in
  certification; `L1-dcache-load-misses` flat (same data, fewer loads).
  Output must be bit-identical (guard with correctness tests).
