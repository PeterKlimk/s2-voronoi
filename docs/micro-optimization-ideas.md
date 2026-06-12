# Micro-Optimization Ideas — Agent Sweep (2026-06-12)

Raw output of a 43-agent parallel scan of the hot paths, one agent per file (or
small file group) crossed with a micro-optimization lens (SIMD lane use,
arithmetic strength reduction, branch shape, memory layout, allocation,
bounds checks, inlining/codegen, hashing, sorting, parallelism). Read-only
sweep: **nothing here has been benchmarked or prototyped**. Impact / risk /
confidence are the scanning agent's honest estimate from reading the code, not
measurements. Treat this as a candidate pool; promote ideas into
`docs/optimization-ideas.md` (the evidence-backed ledger) only after measuring
with `scripts/bench_build.sh` / `scripts/bench_run.sh`.

Line numbers are as of commit `0601165`.

**Inventory:** 185 ideas — 1 high / 58 medium / 126 low estimated impact.
By category: branch 40, bounds-checks 34, memory-layout 25, arithmetic 21,
allocation 21, inlining-codegen 16, simd 10, sorting 9, build-flags 4,
parallelism 3, hashing 2.

## Top picks

Curated for (estimated impact) × (risk) × (independent rediscovery by multiple
scanners). Verify before landing — none are measured.

| # | Idea | Where | Found by |
|---|------|-------|----------|
| 1 | Kill per-batch `out.clone()` in the cached-frontier path — one malloc+memcpy+free per kNN batch per cell, and on the sphere it happens at **two** layers (`PackedQuery` and `DirectedNeighborStream`), with the inner clone likely never read. Replace with a persistent reused buffer. | `cube_grid/query/stream.rs:150`, `cube_grid/packed_knn/mod.rs:281`, `plane_grid/packed/mod.rs:342` | 4 scanners |
| 2 | Move `ShellFrontier`'s `current`/`next`/`pending` Vecs into `CubeMapGridScratch` — the frontier is built eagerly per cell, so `vec![start_cell]` is a malloc per cell even when takeover never runs. The plane side (`PlaneGridScratch::pending`) already does exactly this. | `cube_grid/query/shells.rs:72` | 3 scanners |
| 3 | `to_vertex_data_full` calls `debug_extraction_failure()` unconditionally, projecting **every vertex twice per cell** (9× `fma_f64` each). Keep the O(1) pre-checks, fold the per-vertex checks into the emission loop, keep the debug pass for the cold diagnostics path. The sweep's only high-impact finding. | `knn_clipping/topo2d/builder/extract.rs:27` | 1 scanner (high) |
| 4 | Store the `total_cmp` sign-fold inside `OrdF32` at construction so sort comparisons are raw integer compares (the fold is an involution, so `get()` recovers the exact f32; ordering bit-identical). Bonus: fixes the latent Eq/Ord disagreement on -0.0/NaN. The plane variant: pack `(dist_sq.to_bits() << 32) | slot` and sort raw u64s. | `fp.rs:21`, `plane_grid/query.rs:110` | 4 scanners |
| 5 | `assign_bins`: `bin_for_cell` runs 6 integer divisions and is called twice per cell — cache it in a u8 scratch on the counting pass; and fuse the `slot_gen_map` build (pass 3, two random gathers per slot) into the placement pass where the values are already in registers. | `live_dedup/binning.rs:101,160` | 1 scanner, 2 ideas |
| 6 | Slice `packed_chunk` to `batch.n` once before the clip loop — the opaque clip call between iterations forces a bounds check plus Vec ptr/len reload per neighbor in the innermost loop. | `knn_clipping/cell_build/run.rs:341` | 3 scanners |
| 7 | Replace the `min_cos` sqrt+div threshold test with a precomputed `max_r2` compare — removes an f64 sqrt + fdiv + `is_finite` per committed clip (monotone transform of the same test). | `knn_clipping/topo2d/builder/clip.rs:50` | 1 scanner |
| 8 | Periodic-plane division purge: `rem_euclid` per cell in `for_each_box_cell` and per push in `collect_ring` (hardware idiv each), and two `fmodf` libcalls per point in periodic normalization — all replaceable by conditional add/subtract given already-guaranteed bounds. | `plane_grid/periodic.rs:333,470`, `plane_clipping/compute.rs:326` | 1 scanner, 3 ideas |
| 9 | `point_to_face_uv`: one reciprocal + `copysign` instead of two divides per point (divss ≈ 10c on Zen 2); `cell_to_face_ij`: do the div/mod in u32, not 64-bit usize (~2× faster idiv on Zen 2). | `cube_grid/projection.rs:30,89` | 1 scanner |
| 10 | Build flags: add `[profile.release]` with `codegen-units = 1`, `lto = "thin"` (currently 16 CGUs, no LTO — blocks intra-crate inlining); check in `.cargo/config.toml` with `target-cpu` so default builds get 256-bit AVX2 codegen for `f32x8` (docs/performance.md already measures native at ~6%). Caveat: shifts baselines for `bench_build.sh` cross-commit chains. | `Cargo.toml`, `.cargo/config.toml` | 3 scanners |
| 11 | `merge_result_from_pairs` does n SipHash `HashMap` lookups that almost all return "absent ⇒ self" — run `find` only on indices appearing in `pairs` and lockstep-merge a sorted exception list. On the main weld path. | `knn_clipping/preprocess.rs:240` | 1 scanner |
| 12 | `ensure_tail_directed_for`: query coords and thresholds re-loaded (bounds-checked) inside the per-8-lane chunk loop because the `push` call defeats hoisting — copy to scalar locals before the loop. | `cube_grid/packed_knn/scratch/emit.rs:45` | 1 scanner |

## Convergent findings

Ideas independently rediscovered by scanners with different lenses (a weak
proxy for "really there" given nothing was benchmarked):

- **`out.clone()` frontier caching** — `pknn:mod`, `query:stream`, `plane:packed`, `x:allocs` (top pick 1).
- **OrdF32 / total_cmp precompute** — `fp:simd`, `fp:arith`, `fp:codegen`, plus the `plane:grid` packed-u64 variant (top pick 4).
- **ShellFrontier scratch buffers** — `query:shells`, `query:scratch`, `x:allocs` (top pick 2).
- **`packed_chunk` slice-to-`batch.n`** — `cell:run-branch`, `cell:run-mem`, `cell:mod` (top pick 6).
- **`signed_dists_mask8` fixed-array refs** (`&[f64; 8]` signature to elide the four per-call subslice checks) — `fp:simd`, `fp:arith`.
- **`from_slices` → `&[f32; 8]` constructors** (drop per-chunk len checks at all packed-kNN load sites, via `as_chunks::<8>()`, stable on MSRV 1.88) — `fp:simd`, `fp:arith`.
- **Checked-in `target-cpu` config** — `fp:simd`, `x:buildflags` (top pick 10).
- **`PolyBuffer` plane-index narrowing usize → u32** — `topo2d:builder`, `topo2d:types`.
- **SIMD-izing the packed-kNN center-pass scalar remainder** (reuse the ring-pass zero-pad + `valid_bits` masking pattern) — `pknn:prepare`, `pknn:scratch`.

## Full catalog

Grouped by module area; within each area sorted by estimated impact. Entry
format: title, location, `category · impact · risk · confidence · scanner`,
then the scanner's description. Duplicates across scanners were deliberately
kept (see convergent findings above) — the slightly different framings are
useful when implementing.

## fp.rs — the 8-lane NN kernel

### Store the total_cmp-canonicalized i32 key inside OrdF32 instead of re-deriving it per comparison

`src/fp.rs` — lines 21-50, struct OrdF32 and impl Ord (lines 33-38)

*sorting · impact **medium** · risk low · confidence medium · scanner `fp:simd`*

OrdF32 stores a raw f32 and calls `total_cmp` on every comparison; total_cmp internally applies the sign-flip bit transform (`b ^= ((b >> 31) as u32 >> 1) as i32`) to BOTH operands on EVERY compare. Instead, apply that transform once in `OrdF32::new` and store the resulting i32; derive `PartialOrd`/`Ord` so comparisons become a single integer cmp, and have `get` re-apply the same xor (the transform is an involution: the sign bit is unchanged by the xor, so applying it twice round-trips exactly) before `f32::from_bits`. The ordering is bit-identical to total_cmp by construction. OrdF32 keys the `(dist, slot)` pending vectors that get sorted in the shell frontiers (cube_grid/query/shells.rs:105, plane_grid/query.rs:110, plane_grid/periodic.rs:463), where each sort comparison currently pays the transform twice; this also makes the (i32, u32) tuple compare fully branch-predictable integer code.

### Check in a .cargo/config.toml so default builds get 256-bit codegen for f32x8/f64x4

`.cargo/config.toml` — new file; affects backend module at src/fp.rs lines 210-327

*build-flags · impact **medium** · risk low · confidence high · scanner `fp:simd`*

Without target features, wide's f32x8 lowers to two 128-bit SSE2 registers and `move_mask` (lines 244, 282, 318-319) becomes two `movmskps` plus shift/or per call; f64x4 in signed_dists_mask8 likewise splits in half. The bench scripts pass `-C target-cpu=native`, but the documented bench command (`cargo run --release --bin bench_voronoi`) and `cargo test --release` do not, and docs/performance.md already measures native as worth ~6%. Adding `[target.x86_64-unknown-linux-gnu] rustflags = ["-C", "target-cpu=x86-64-v3"]` (or native) to a checked-in .cargo/config.toml makes every local build use full-width AVX2 vectors and single-instruction mask extraction. With the `fma` feature off there is no contraction, so results stay bit-identical to the SSE2 lowering.

### Precompute total_cmp key in OrdF32 so sort comparisons are raw integer compares

`src/fp.rs` — lines 18-50, struct OrdF32 / Ord::cmp / new / get

*sorting · impact **medium** · risk low · confidence high · scanner `fp:codegen`*

OrdF32 stores a raw f32 and every Ord::cmp call runs f32::total_cmp, which applies the sign-magnitude bit transform (2 shifts + xor) to BOTH operands per comparison. These keys are sorted O(n log n) times per query batch (plane_grid/query.rs:174 sort_unstable_by_key, and the descending-dot sort in cube_grid/query/shells.rs), while construction/extraction is only O(n). Change OrdF32 to store the transformed bits once in new() (`let b = v.to_bits() as i32; OrdF32(b ^ ((((b >> 31) as u32) >> 1) as i32))`) and make cmp a plain i32 compare; the transform is an involution, so get() reapplies it and from_bits recovers the exact f32. Bonus: derived PartialEq on the i32 becomes consistent with Ord (currently derived f32 PartialEq disagrees with total_cmp for -0.0/+0.0 and NaN, a latent Eq/Ord contract violation).

### Collapse four slice bounds checks to one per slice in signed_dists_mask8 (wide backend)

`src/fp.rs` — lines 311-314, backend::signed_dists_mask8

*bounds-checks · impact **low** · risk low · confidence high · scanner `fp:simd`*

The wide backend does four checked slice ops per call: `&us[..4]`, `&us[4..8]`, `&vs[..4]`, `&vs[4..8]`, each a separate panicking length check that LLVM cannot merge because they are independent. Insert `let us: &[f64; 8] = us[..8].try_into().unwrap(); let vs: &[f64; 8] = vs[..8].try_into().unwrap();` at the top so the subsequent 4-element splits index fixed-size arrays and all inner checks fold away — 4 cmp/branch pairs become 2. This function is called per half-plane per 8-vertex chunk in the topo2d clipping hot loop (clippers/bitmask.rs:26, clippers/small.rs:200,336), so the branches are paid on every clip step. Lane math is untouched, so results stay bit-identical.

### Eliminate per-chunk subslice + length checks in from_slices via as_chunks / fixed-array refs

`src/fp.rs` — lines 98-105 (PointChunk8::from_slices), 149-155 (PlaneChunk8::from_slices), backend::load_slice line 216-219

*bounds-checks · impact **low** · risk low · confidence medium · scanner `fp:simd`*

Every packed-kNN chunk load does `from_slices(&xs[i..], &ys[i..], &zs[i..])` (e.g. cube_grid/packed_knn/scratch/prepare.rs:261, emit.rs:66; plane_grid/packed/scratch.rs:386,504,594): each argument pays an `i <= len` check for the subslice plus a `len >= 8` check inside load_slice — 6 checked branches per PointChunk8 (4 per PlaneChunk8). Add a `from_chunk_refs(x: &[f32; 8], y: &[f32; 8], z: &[f32; 8])` constructor and switch the chunk loops to `slice::as_chunks::<8>()` (stable since 1.88, the crate MSRV), which hoists all checking out of the loop; or, cheaper, replace the three independent checks with one non-short-circuit `assert!((xs.len() >= 8) & (ys.len() >= 8) & (zs.len() >= 8))` so LLVM elides the per-slice checks behind a single branch. The loads are amortized over the inner per-query loop, so the win is real but modest.

### Precompute total_cmp's sign-fold in OrdF32 so comparisons are a single u32 cmp

`src/fp.rs` — lines 21-50, OrdF32 (cmp at lines 33-38)

*sorting · impact **low** · risk low · confidence high · scanner `fp:arith`*

OrdF32::cmp calls f32::total_cmp, which recomputes the monotone bit-fold (bits ^ (((bits as i32) >> 31 as u32) >> 1) on BOTH operands at every comparison; the sort sites (cube_grid/query/shells.rs:148, plane_grid/query.rs:175, plane_grid/periodic.rs:533) run this O(n log n) times per frontier batch. Change OrdF32 to store the folded u32 key once in new() and make cmp a plain u32 compare; the fold never flips the sign bit so it is an involution, letting get() recover the exact f32 by re-applying it. Ordering is bit-identical to total_cmp, saving ~6 integer ops per comparison inside sort_unstable_by_key. Note derived PartialEq changes from f32== to bits equality (only affects -0.0 vs +0.0 / NaN, and only Ord matters to the sorts).

### Take &[f64; 8] in signed_dists_mask8 to elide four slice length checks per call

`src/fp.rs` — lines 198-207 and 303-314 (signed_dists_mask8; try_from slices at 311-314)

*bounds-checks · impact **low** · risk low · confidence medium · scanner `fp:arith`*

The wide backend does four checked conversions (&us[..4], &us[4..8], &vs[..4], &vs[4..8]) per call. The small.rs callers (knn_clipping/topo2d/clippers/small.rs:200,336) pass fixed arrays so checks fold after inlining, but bitmask.rs:26 passes &poly.us[i..] with a runtime loop offset, so LLVM keeps the len>=8 checks in the per-8-vertex clipping inner loop. Change the signature to us: &[f64; 8], vs: &[f64; 8]; small.rs uses poly.us.first_chunk::<8>().unwrap() (compile-time, array length known) and bitmask.rs iterates poly.us.as_chunks::<8>().0 (stable since 1.88), eliminating all per-chunk checks. Lane math unchanged, so numerics are bit-identical.

### Add &[f32; 8] constructors for PointChunk8/PlaneChunk8 to drop per-chunk len checks in packed kNN

`src/fp.rs` — lines 98-105 (PointChunk8::from_slices), 149-155 (PlaneChunk8::from_slices), 216-219 (backend::load_slice)

*bounds-checks · impact **low** · risk low · confidence medium · scanner `fp:arith`*

load_slice does s[..8].try_into().unwrap(), a len>=8 cmp+branch per lane array — 3 per PointChunk8 and 2 per PlaneChunk8 load — at the packed kNN hot sites (cube_grid/packed_knn/scratch/emit.rs:66, prepare.rs:261/359, plane_grid/packed/scratch.rs:386/504/594) where slices are tails like &xs[i..] whose length LLVM can't bound. Add from_array_refs(&[f32; 8], ...) and have callers get array refs via xs.as_chunks::<8>() (stable on MSRV 1.88) or get(i..i+8) restructuring. The branches are perfectly predicted so the win is small (shorter loop bodies, better unrolling), but the change is mechanical and zero-risk to numerics.

### Defer the [f64; 8] materialization in signed_dists_mask8 behind a Dots8-style opaque struct

`src/fp.rs` — lines 302-326 (wide signed_dists_mask8; to_array + rebuild at 320-325)

*inlining-codegen · impact **low** · risk low · confidence low · scanner `fp:arith`*

signed_dists_mask8 always spills both f64x4 registers through to_array and rebuilds a 64-byte [f64; 8] on the stack, but the dominant clip outcomes in small.rs (mask==0 -> Changed, mask==full -> Unchanged at small.rs:206-215) only consume the mask (the dists go to maybe_escalate, which is default-off P5 plumbing). Return (SignedDists8([f64x4; 2]), u32) with a deferred #[inline(always)] to_array(), mirroring the documented Dots8 design where extraction is deferred until the mask is known non-trivial. Saves two 32-byte stores plus the stack array on the all-inside/all-outside fast paths; LLVM may already sink some of this given inline(always), so confidence is moderate.

### Take &[f32; 8] in PointChunk8/PlaneChunk8 loaders and use as_chunks at call sites to elide per-chunk bounds checks

`src/fp.rs` — load_slice (lines 216-219), PointChunk8::from_slices (line 99), PlaneChunk8::from_slices (line 150)

*bounds-checks · impact **low** · risk low · confidence medium · scanner `fp:codegen`*

load_slice does `s[..8].try_into().unwrap()` — a runtime length check per lane-array, i.e. 3 checks per PointChunk8 (2 per PlaneChunk8), on top of the `&xs[i..]` re-slice checks at the call sites (cube_grid/packed_knn/scratch/prepare.rs:261,359, emit.rs:66, plane_grid/packed/scratch.rs:386,504,594). That is up to 6 compare+branch pairs guarding each 8-lane SIMD chunk in the kNN hot loops, and LLVM cannot always prove `len - i >= 8` through the step-by-8 loop structure. Change from_slices to take `&[f32; 8]` per axis and have call sites iterate `xs.as_chunks::<8>().0` (slice::as_chunks is stable as of Rust 1.88, exactly the MSRV), which proves length once per loop instead of per chunk.

### Hoist one &[f64; 8] conversion at top of signed_dists_mask8 to halve its four slice range checks

`src/fp.rs` — backend::signed_dists_mask8, lines 311-314

*bounds-checks · impact **low** · risk low · confidence medium · scanner `fp:codegen`*

The wide backend builds four f64x4 loads via `try_from(&us[..4])`, `&us[4..8]`, `&vs[..4]`, `&vs[4..8]` — four independent runtime length checks (the `[..4]` check is not subsumed by the later `[4..8]` check because it executes first). For the small.rs callers passing whole fixed arrays the checks fold after inlining, but the bitmask.rs caller passes `&poly.us[i..]` with runtime length, so all four checks survive in that clipper's inner loop. Add `let us: &[f64; 8] = us[..8].try_into().unwrap();` (same for vs) at the top; the subsequent const-range subslices of a fixed-size array compile to check-free loads, leaving 2 checks instead of 4.

### Drop redundant & 0xff / & 0xf masking after move_mask in mask_gt, mask_lt, and signed_dists_mask8

`src/fp.rs` — mask_gt (line 244), mask_lt (line 282), signed_dists_mask8 mask combine (lines 318-319)

*inlining-codegen · impact **low** · risk low · confidence medium · scanner `fp:codegen`*

`move_mask()` on f32x8 yields a value in [0,255] and on f64x4 in [0,15] across all of wide's lowering paths, but when it lowers to the opaque AVX movmsk intrinsic LLVM has no range information, so the `& 0xff` (and both `& 0xf` in signed_dists_mask8 — the low-half one is fully redundant) each emit a real AND instruction in the hottest mask-extraction path. Delete the redundant masks (keep only the `<< 4` combine). One ALU op saved per 8-lane chunk; tiny but free, and the value range is guaranteed by the lane count, not the masking.

## Packed kNN (cube_grid/packed_knn)

### Replace per-batch `out.clone()` in cached frontier with a reused slot buffer

`src/cube_grid/packed_knn/mod.rs` — lines 281-284 in PackedQuery::frontier (also dropped at line 302 in advance_frontier)

*allocation · impact **medium** · risk low · confidence high · scanner `pknn:mod`*

Every ExactBatch produced by frontier() heap-allocates a fresh Vec<u32> via `slots: out.clone()` to populate CachedFrontier::ExactBatch, and advance_frontier() immediately drops it (`Some(CachedFrontier::ExactBatch { .. }) => {}`), so it's one malloc+memcpy+free per kNN batch per query. Worse, the only consumer (DirectedNeighborStream::frontier in src/cube_grid/query/stream.rs ~line 150) caches its own `out.clone()` at the stream level, so the PackedQuery-level slot copy is likely never even read on a cache hit. Change CachedFrontier::ExactBatch to store only the batch metadata plus a length, and keep the slots in a persistent `Vec<u32>` field stashed in PreparedPackedGroup/PackedKnnCellScratch (which outlives per-query PackedQuery instances), refilled with clear()+extend_from_slice — zero allocations in steady state.

### SIMDize the scalar center-pass remainder with an overlapped directed-masked chunk

`src/cube_grid/packed_knn/scratch/prepare.rs` — lines 309-330, prepare_group_directed (center cell remainder loop)

*simd · impact **medium** · risk medium · confidence medium · scanner `pknn:scratch`*

The center-pass remainder is a scalar double loop: for each of rem (~3.5 avg at density 24) trailing positions, it does up to num_queries (~24) scalar dot3_f32 + 2 compares, ~77 scalar iterations per group, which exceeds the ~48 inner iterations of the entire SIMD chunk loop above it. When center_len >= 8 (full_chunks >= 1, the common case), replace it with one overlapped window load `PointChunk8::from_slices(&xs[center_len-8..], ...)`, valid_bits masking off lanes already covered by full chunks (lanes < 8-rem), and the same directed mask logic as the chunk loop (lane position must be > qi), reusing the existing hi/band split. Values loaded are identical real points, and fp.rs deliberately matches scalar/SIMD dot ordering (left-associative, fma off by default), so results should be bit-identical; verify backend::dot3 lane order matches dot3_f32 before landing.

### SIMD-ize the center-pass scalar remainder with the ring-pass zero-pad + valid_bits pattern

`src/cube_grid/packed_knn/scratch/prepare.rs` — lines 309-330, prepare_group_directed (center tail loop)

*simd · impact **medium** · risk low · confidence high · scanner `pknn:prepare`*

The center pass processes full 8-lane chunks with PointChunk8, but the remainder (center_len % 8, avg ~3.5 at density 24) falls back to a doubly-nested scalar loop costing rem * num_queries (~24) scalar dot3_f32 calls plus two compares each. The ring pass already solves this (lines 383-418) by copying the remainder into [0.0f32; 8] stack buffers and masking with valid_bits; the same padded chunk can reuse the existing directed-mask logic from the full-chunk body (lines 266-305) verbatim, just AND-ing valid_bits into mask_bits (padding lanes produce dot 0.0 which can pass negative security thresholds, so the mask is load-bearing). This turns ~84 scalar dots per group into ~num_queries SIMD iterations.

### Hoist query coords and thresholds out of the range/chunk loops in ensure_tail_directed_for

`src/cube_grid/packed_knn/scratch/emit.rs` — ensure_tail_directed_for, lines 45-104 (loads at 57-61, 69-70, 101)

*inlining-codegen · impact **medium** · risk low · confidence high · scanner `pknn:emit`*

The query slot lookup (group_queries[qi]) and the three grid.cell_points_{x,y,z}[query_slot] loads are re-executed on every iteration of the `for r in &self.cell_ranges[1..]` loop, and self.security_thresholds[qi] / self.thresholds[qi] are re-indexed (with bounds checks) inside the innermost per-8-lane chunk loop and the scalar remainder loop. Because the loop body calls self.tail_keys[qi].push(...) (a heap write LLVM cannot prove non-aliasing with the threshold Vec buffers), these loads are not hoisted automatically. Copy all five values to scalar locals before line 45: `let query_slot = group_queries[qi]; let (qx_s, qy_s, qz_s) = ...; let sec = self.security_thresholds[qi]; let hi = self.thresholds[qi];` and use the locals in mask_gt and the scalar test. Removes 2 bounds-checked loads per 8-lane chunk and ~5 per range from the ring-fallback hot loop.

### Drop the never-read u32::MAX sentinel fill before next_chunk

`src/cube_grid/packed_knn/mod.rs` — line 264 (`out.resize(k, u32::MAX)`) in PackedQuery::frontier

*allocation · impact **low** · risk low · confidence high · scanner `pknn:mod`*

frontier() does `out.clear()` (line 225) then `out.resize(k, u32::MAX)`, filling all k slots, but the next_chunk implementation in scratch/emit.rs only writes `out[..n]` and the caller immediately does `out.truncate(chunk.n)` — the sentinel values are never read. Move the `out.clear()` into the cached-hit branches and replace line 264 with `if out.len() < k { out.resize(k, 0) } else { out.truncate(k) }` so stale contents are reused as scratch and the memset only covers growth beyond the previous batch length. Saves a small per-batch fill (k is chunk-size tens of u32s); cheap but real on a path executed once per frontier batch.

### Cache the query point coordinates in PackedQuery::new for slot_dot

`src/cube_grid/packed_knn/mod.rs` — lines 207-218 (slot_dot), called at line 271; constructor at lines 187-204

*bounds-checks · impact **low** · risk low · confidence medium · scanner `pknn:mod`*

slot_dot re-derives `query_slot` via `self.prepared.group().queries()[self.query_index]` and loads the query's x/y/z from three separate SoA arrays (4 bounds-checked indexed loads) on every call, even though the query point is fixed for the lifetime of the PackedQuery. Load (qx, qy, qz) once in PackedQuery::new (the group and query_index are available there) and store them as three f32 fields; slot_dot then only indexes the neighbor slot arrays. It is only called once per batch (first_dot of out[0]), so the win is small but the change is trivial and strictly removes work.

### Split self borrows to hoist invariant loads out of the tail-build ring loop

`src/cube_grid/packed_knn/scratch/emit.rs` — lines 45-105, ensure_tail_directed_for

*bounds-checks · impact **low** · risk low · confidence high · scanner `pknn:scratch`*

query_slot and the three bounds-checked grid loads qx_s/qy_s/qz_s (lines 57-61) are recomputed inside `for r in &self.cell_ranges[1..]` though they do not depend on r, and `self.security_thresholds[qi]` / `self.thresholds[qi]` (lines 69-70) are re-indexed (with bounds checks) every 8-lane chunk; `self.tail_keys[qi]` is also re-indexed on every push. LLVM cannot hoist potentially-panicking indexed loads out of a possibly-zero-trip loop. Destructure `let Self { cell_ranges, tail_keys, security_thresholds, thresholds, .. } = self;`, copy the query xyz and both thresholds into locals before the loop, and bind `let tk = &mut tail_keys[qi]` once. Pure invariant-hoisting, no behavior change.

### Replace ring-pass remainder stack copies with an overlapped 8-wide load

`src/cube_grid/packed_knn/scratch/prepare.rs` — lines 383-418, prepare_group_directed (ring pass remainder)

*simd · impact **low** · risk low · confidence medium · scanner `pknn:scratch`*

When rem != 0 (87.5% of ring cells at density 24), the code zero-initializes three [f32; 8] stack buffers, does three copy_from_slice calls, and loads via from_arrays. When range_len >= 8, instead load the last 8 points directly with `from_slices(&xs[range_len-8..], ...)` and set `valid_bits = (0xFF << (8 - rem)) & 0xFF` (masking off lanes already handled by full chunks), with slots based at soa_start + range_len - 8; keep the pad path only for range_len < 8. Identical values are loaded and the existing valid_bits mask already guards stray lanes, so output is bit-identical. The same trick applies to the scalar tail remainder in emit.rs lines 89-104.

### Use chunks_exact zips to elide per-chunk length checks feeding PointChunk8::from_slices

`src/cube_grid/packed_knn/scratch/prepare.rs` — lines 258-261 and 356-359 (also emit.rs line 64-66)

*bounds-checks · impact **low** · risk low · confidence medium · scanner `pknn:scratch`*

Each chunk iteration builds `&xs[i..]` open-ended subslices, and backend::load_slice then does `s[..8].try_into().unwrap()` (fp.rs:217), leaving a length check plus panic branch per load whose elision depends on LLVM proving i+8 <= range_len through the full_chunks division. Rewriting the chunk loops as `xs.chunks_exact(8).zip(ys.chunks_exact(8)).zip(zs.chunks_exact(8)).enumerate()` (or slicing `&xs[i..i+8]` with a from_array path) gives the compiler exact fixed-length slices so the checks vanish statically. Three hot loops are affected; zero numerical change.

### Make f32_to_ordered_u32 branchless in make_desc_key

`src/cube_grid/packed_knn/scratch/helpers.rs` — lines 13-20, f32_to_ordered_u32 (called from make_desc_key)

*branch · impact **low** · risk low · confidence low · scanner `pknn:scratch`*

make_desc_key runs once per accepted candidate across the center, ring, and tail emit loops, and f32_to_ordered_u32 branches on the sign bit. Replace with the standard branchless form `b ^ ((((b as i32) >> 31) as u32) | 0x8000_0000)`, which is identical for all bit patterns and removes the branch (LLVM may already emit cmov, hence low confidence). Dots are usually positive so the branch predicts well; this is a small win mainly when security thresholds go negative on coarse grids.

### Replace resize+fill with clear+resize for chunk0_pos/tail_pos resets

`src/cube_grid/packed_knn/scratch/prepare.rs` — lines 180-181 and 190-191, prepare_group_directed

*allocation · impact **low** · risk low · confidence high · scanner `pknn:scratch`*

`self.chunk0_pos.resize(num_queries, 0); self.chunk0_pos.fill(0);` writes any newly grown region twice (resize memsets the tail, fill rewrites everything); same for tail_pos. Using `clear()` followed by `resize(num_queries, 0)` produces a single memset of exactly num_queries elements. With num_queries ~24 this is a tiny per-group saving, but it is free and clarifies intent.

### Re-slice query coordinate slices to [..num_queries] to elide ring-pass bounds checks

`src/cube_grid/packed_knn/scratch/prepare.rs` — lines 250-252 (query_x/query_y/query_z bindings), consumed at lines 362, 399

*bounds-checks · impact **low** · risk low · confidence medium · scanner `pknn:prepare`*

query_x/y/z alias qx_src..qz_src whose length is center_len (q_end - q_start), while the ring-pass inner loops index them with qi bounded by queries.len() == num_queries; equality is only debug_asserted (line 246), so LLVM cannot prove the indexes in-bounds and emits 3 bounds checks per (chunk, query) iteration in the hottest loop. Change the bindings to `let query_x = &qx_src[..num_queries];` (likewise y, z), matching what is already done for security_thresholds/hi_thresholds at lines 253-254. The checks then hoist to one slice op per group; behavior on a violated invariant changes only from index-panic to slice-panic.

### Vectorize the 4-plane security-threshold min across 8 query lanes

`src/cube_grid/packed_knn/scratch/prepare.rs` — lines 137-157, interior_planes Some-branch of security threshold loop

*simd · impact **low** · risk low · confidence medium · scanner `pknn:prepare`*

For the common interior-cell case, the loop computes 4 scalar dot3_f32 products and a running min per query, fully serially. Since queries are already SoA (qx_src/qy_src/qz_src), chunk by 8: build PointChunk8::from_slices over the query coords, call .dots(n.x, n.y, n.z) for each of the 4 planes, take a lane-wise min, then run the scalar sqrt/clamp epilogue (and the rare s_min<=0 fallback) per extracted lane. This converts 4*num_queries scalar dots into num_queries/8 * 4 SIMD dots in the add_security_thresholds lap. The GRID_PLANE_PAD slack absorbs any ulp-level reassociation differences, and the default backend is non-FMA like dot3_f32.

### Use overlapping last-8 load instead of zero-padded stack copy in ring-pass remainder

`src/cube_grid/packed_knn/scratch/prepare.rs` — lines 383-418, ring-pass remainder block

*memory-layout · impact **low** · risk low · confidence medium · scanner `pknn:prepare`*

When a ring range has a remainder, the code zeroes three [f32; 8] stack buffers and copies rem elements of each coordinate before loading. When full_chunks > 0 (almost always at density 24), instead load the last 8 elements directly with from_slices(&xs[range_len-8..], ...) and set valid_bits = 0xFFu32 << (8 - rem) to mask out the 8-rem lanes already handled by the full-chunk loop (slot base becomes soa_start + range_len - 8 + lane). This removes ~96 bytes of stack writes plus 3 small memcpys per ring cell (up to 8 cells per group) and avoids the store-to-load forwarding hop into the SIMD load. Keep the padded path only for range_len < 8. Masking is load-bearing (unmasked lanes would emit duplicate keys), but the shift expression is mechanical.

### Collapse directed intra-bin mask to a single shift in center pass

`src/cube_grid/packed_knn/scratch/prepare.rs` — lines 276-285, full-chunk directed filter

*branch · impact **low** · risk low · confidence high · scanner `pknn:prepare`*

The directed filter clears bits below rel with a branch on rel > 0 plus `!((1u32 << rel) - 1)`, then clears the self bit with a second AND. Both reduce to clearing bits 0..=rel, i.e. `mask_bits &= u32::MAX << (rel + 1)` (rel <= 7 so the shift is in range), removing one data-dependent branch and two ALU ops per affected (chunk, query) pair. The outer `qi >= i` guard stays (it is well-predicted: true only for the diagonal chunk).

### Software-prefetch the next ring range's SoA slices

`src/cube_grid/packed_knn/scratch/prepare.rs` — lines 344-354, ring-pass loop over cell_ranges[1..]

*memory-layout · impact **low** · risk low · confidence low · scanner `pknn:prepare`*

Ring neighbor cells are scattered ranges of grid.cell_points_x/y/z (cells on other faces or distant rows), ~96 bytes per coordinate array at density 24, so the hardware prefetcher gets no run-up across range boundaries while ~24 query iterations execute per chunk. Since cell_ranges is fully materialized before the loop, issue `core::arch::x86_64::_mm_prefetch::<_MM_HINT_T0>` (stable) for the start of the next range's xs/ys/zs at the top of each iteration, gated on target_arch = "x86_64". Purely a hint: zero correctness risk, but the win depends on whether these lines actually miss, which only a profile can confirm.

### Bind &mut self.tail_keys[qi] once before the range loop to drop per-push outer-Vec indexing

`src/cube_grid/packed_knn/scratch/emit.rs` — ensure_tail_directed_for, push sites at lines 83 and 102

*bounds-checks · impact **low** · risk low · confidence high · scanner `pknn:emit`*

Each accepted candidate does self.tail_keys[qi].push(...), which re-indexes the outer Vec<Vec<u64>> (bounds check) and reloads the inner Vec's ptr/len/cap from memory on every push. Hoist `let tk = &mut self.tail_keys[qi];` before line 45 and push via `tk.push(...)`; field-level borrow splitting keeps the immutable reads of self.cell_ranges/thresholds legal (especially once those are locals per the previous idea). This keeps the inner Vec header hot in registers across the bit-decode loop and removes one bounds check per emitted key.

### Use chunks_exact(8) instead of manual chunk*8 slicing to guarantee elision of load_slice bounds checks

`src/cube_grid/packed_knn/scratch/emit.rs` — ensure_tail_directed_for, lines 63-66 (with fp.rs load_slice at src/fp.rs:216-219)

*bounds-checks · impact **low** · risk low · confidence medium · scanner `pknn:emit`*

The SIMD loop does `&xs[i..]` three times per chunk and fp::PointChunk8::from_slices then does `s[..8].try_into().unwrap()` per slice — six length checks per chunk whose elision depends on LLVM proving `chunk*8 + 8 <= range_len` from `full_chunks = range_len / 8`, which is not guaranteed. Rewrite as `for (chunk_idx, ((cx, cy), cz)) in xs.chunks_exact(8).zip(ys.chunks_exact(8)).zip(zs.chunks_exact(8)).enumerate()` and load via from_slices (chunks_exact slices have a compiler-known length of 8, so the `[..8]` check folds away). Compute `i = chunk_idx * 8` for the slot. Also lets the leftover remainder come from `.remainder()` instead of recomputing tail_start.

### Reuse the new position local for has_more instead of re-indexing pos vectors in next_chunk

`src/cube_grid/packed_knn/scratch/emit.rs` — next_chunk, lines 158-159, 186-187, 225-226, 263-264

*bounds-checks · impact **low** · risk low · confidence high · scanner `pknn:emit`*

All three stage arms write `self.*_pos[qi] = start + n;` and then immediately re-read `self.*_pos[qi] < keys.len()` — a second bounds-checked indexed load of the position Vec that LLVM may not fold because `keys` is a live &mut borrow of a sibling field. Change to `let new_pos = start + n; self.chunk0_pos[qi] = new_pos; let has_more = new_pos < keys.len();` (and likewise for tail_pos/expand2_pos). Saves one bounds check plus a memory round-trip per emitted chunk in the hottest emission function.

### Mask the decoded lane index to statically bound dots_arr access

`src/cube_grid/packed_knn/scratch/emit.rs` — ensure_tail_directed_for, lines 78-86 (dots_arr[lane] at line 82)

*bounds-checks · impact **low** · risk low · confidence medium · scanner `pknn:emit`*

In the mask-to-index decode loop, `lane = tail_bits.trailing_zeros() as usize` indexes the stack array dots_arr[lane]; elision of the bounds check requires LLVM to propagate the `& 0xff` range from mask_gt through two function layers and the `safe & !hi` combination, which is fragile across compiler versions. Writing `dots_arr[lane & 7]` makes the in-bounds proof local and free (the AND folds into the existing trailing_zeros result), guaranteeing no check or panic path inside the per-candidate loop. Zero behavioral change since tail_bits is always <= 0xff.

## Cube-grid query stack (cube_grid/query)

### Reorder cell_mode: test cell > start_cell before the 3-load cell_bin chain

`src/cube_grid/query/directed.rs` — lines 45-66, fn cell_mode

*branch · impact **medium** · risk low · confidence medium · scanner `query:directed`*

cell_mode currently begins with layout.cell_bin(), a dependent chain of three loads (cell_offsets[cell], cell_offsets[cell+1], slot_gen_map[start]), then compares bins, and only then does the free register compare against start_cell. Hoist `let cell_u32 = cell as u32; if cell_u32 > start_cell { return DirectedCellMode::EmitAll; }` to the top: every later-indexed cell (about half of visited cells, and most ring cells) returns from a single register compare with no memory traffic. The only semantic delta is empty cells with index > start_cell returning EmitAll instead of TransitOnly, which is behaviorally identical at all three call sites (shells.rs:86-106, plane_grid/query.rs:91-111, plane_grid/periodic.rs:448-464 all slice an empty range and do zero iterations). Falls through to the existing cell_bin/bin/start checks for cell_u32 <= start_cell, preserving all other outcomes.

### Reuse a persistent slots buffer instead of out.clone() when caching frontier batches

`src/cube_grid/query/stream.rs` — lines 39-49 (CachedFrontier), 113-115, 150-153, 177-180, 197 (frontier/advance_frontier)

*allocation · impact **medium** · risk low · confidence high · scanner `query:stream`*

Every ExactBatch frontier caches its slot list via `slots: out.clone()` (lines 150-153 and 177-180), which heap-allocates a fresh Vec<u32>; `advance_frontier()` then `take()`s and drops it (line 197). In the hot loop (consume_stream in cell_build/run.rs) this is one malloc + memcpy + free per frontier batch per cell, across millions of cells. Change `CachedFrontier::ExactBatch` to hold only the `DirectedNeighborBatch` metadata and add a persistent `cached_slots: Vec<u32>` field on `DirectedNeighborStream`, populated with `self.cached_slots.clear(); self.cached_slots.extend_from_slice(out);` and read back on the cache-hit path at lines 113-115. The buffer's capacity is reused for the lifetime of the stream, eliminating the per-batch allocator round-trip while preserving the idempotent-frontier contract.

### Replace per-point query_idx compare with precomputed query slot

`src/cube_grid/query/shells.rs` — scan_cell lines 91-97; ShellFrontier::new lines 51-55

*branch · impact **medium** · risk low · confidence high · scanner `query:shells`*

scan_cell loads indices[i] for every scanned point solely to skip the query point (line 95), but the query point can only live in start_cell (point_cells[query_idx]), and on locate/quality paths query_idx >= point_cells.len() so the compare never fires at all. In new(), scan point_indices[start..end] of start_cell once to find query_slot: u32 (u32::MAX if absent), then change the loop test to `slot != query_slot` (slot = (start+i) as u32, already computed). This removes the point_indices load stream and the usize-widening compare from the hot per-point scan loop; the indices slice binding at line 91 can be deleted, also dropping one bounds-checked slice per cell.

### Move current/next/pending buffers into CubeMapGridScratch

`src/cube_grid/query/shells.rs` — ShellFrontier::new lines 72-74 (and struct fields lines 33-37)

*allocation · impact **medium** · risk low · confidence high · scanner `query:shells`*

Every ShellFrontier::new allocates `vec![start_cell]` plus two fresh Vecs that regrow during the BFS, and the frontier is constructed once per query (stream.rs:87, locate.rs:137, quality.rs:463). Add shell_current/shell_next/shell_pending Vecs to CubeMapGridScratch (scratch is already passed in by &mut), clear them in new(), and borrow them through self.scratch. Eliminates 3 allocations plus growth reallocs per takeover/locate query; capacity persists across queries.

### Move ShellFrontier's current/next/pending Vecs into CubeMapGridScratch

`src/cube_grid/query/shells.rs` — ShellFrontier::new lines 72-74 (struct fields lines 33-37); CubeMapGridScratch in src/cube_grid/mod.rs:248-251; scratch.rs CubeMapGridScratch::new

*allocation · impact **medium** · risk low · confidence high · scanner `query:scratch`*

ShellFrontier::new allocates `current: vec![start_cell]`, `next: Vec::new()`, `pending: Vec::new()` per construction, and stream.rs:86-87 builds the ShellFrontier eagerly in DirectedNeighborStream::new for every directed kNN query — so the `vec![start_cell]` is one malloc+free per cell build even when takeover never runs, and `next`/`pending` re-grow from capacity 0 on every takeover that does run. The frontier already holds `&mut CubeMapGridScratch`, which persists per bin/thread (cell_build/run.rs:74), so move all three Vecs into CubeMapGridScratch (initialized in scratch.rs new), and in ShellFrontier::new do clear()+push(start_cell) instead. Eliminates one allocator round-trip per query and amortizes pending/next capacity across the whole bin.

### Collapse allows_center_slot to one precomputed packed compare

`src/cube_grid/query/directed.rs` — lines 69-72, fn allows_center_slot (plus new field in struct at lines 8-13, init at lines 31-42)

*branch · impact **low** · risk medium · confidence medium · scanner `query:directed`*

allows_center_slot unpacks bin and local (load, shift, mask) then does a short-circuit `bin_b != query_bin || local_b >= query_local`. But it is only ever called under EmitCenterDirected, which cell_mode emits only when the cell's bin == query_bin, and cells are bin-contiguous (documented invariant on PackedSlotLayout::cell_bin), so the bin disjunct is always false in practice. Precompute `query_packed = ((query_bin as u32) << local_shift) | query_local` in from_layout and reduce the body to `self.layout.slot_gen_map()[slot as usize] >= self.query_packed` (equal high bin bits make the packed compare exactly equal to local_b >= query_local), with a debug_assert that the slot's bin equals query_bin to guard the invariant. Removes the shift, mask, and short-circuit branch per slot in the per-query start-cell loop. If the invariant feels too coupled, the strictly-safe fallback is replacing `||` with `|` to make the existing test branchless.

### Return (mode, start, end) from cell_mode to avoid re-loading cell_offsets in callers

`src/cube_grid/query/directed.rs` — lines 45-66, fn cell_mode; callers shells.rs:86-87, plane_grid/query.rs:91-92, plane_grid/periodic.rs:448-449

*bounds-checks · impact **low** · risk low · confidence high · scanner `query:directed`*

cell_mode already loads cell_offsets[cell] and cell_offsets[cell+1] inside cell_bin (with bounds checks), and all three scan_cell callers immediately reload the exact same two entries (with two more bounds checks) to build their slot range. Change cell_mode to return the start/end it computed alongside the mode (or a small struct), and have callers use those. Saves two bounds-checked loads per non-transit cell per query; the loads are L1-hot so the win is mostly the elided bounds checks and shorter dependency chain. Combine with the reorder idea so start/end are only computed when actually needed.

### Outline the packed_mut invariant panic into a #[cold] #[inline(never)] helper

`src/cube_grid/query/stream.rs` — lines 64-76, fn packed_mut (call sites at lines 130 and 203)

*inlining-codegen · impact **low** · risk low · confidence medium · scanner `query:stream`*

`packed_mut` is `#[inline(always)]` and its None arm expands a 4-argument `panic!` format inline, so the `fmt::Arguments` construction (stage debug, context str, two bools) is duplicated into both hot callers `frontier()` and `advance_frontier()`, bloating their code size. Extract the panic into a separate `#[cold] #[inline(never)] fn packed_invariant_panic(stage: StreamStage, context: &str, cached: bool, did_packed: bool) -> !` and call it from the None arm, leaving the hot inlined body as a bare null-check plus a call to a cold function. This shrinks the hot functions and improves icache density; behavior on the panic path is unchanged.

### Keep ring certificate in dot space, drop dist_sq round trip

`src/cube_grid/query/shells.rs` — build_pending lines 117, 133-134, 139-143 (and cell_min_dist_sq in src/cube_grid/query/mod.rs:120-137)

*arithmetic · impact **low** · risk low · confidence high · scanner `query:shells`*

cell_min_dist_sq already computes the bound in dot space (max_dot_upper, mod.rs:135) and converts it to squared distance via `2.0 - 2.0*max_dot_upper`, then build_pending converts back with `(1.0 - 0.5 * next_min_dist_sq)` at line 142 — a pure round trip with two extra roundings. Add a `cell_max_dot_upper` variant returning max_dot_upper directly (1.0 in the inside-cap case at mod.rs:129), track `next_max_dot = next_max_dot.max(bound)` per neighbor cell, and set pending_bound = next_max_dot (or -1.0 when ring empty). Saves a mul+sub per discovered neighbor cell (9-neighborhood per ring cell) and gives a marginally tighter, still-conservative certificate; clamp already exists in cell_min_dist_sq so the line-142 clamp folds away.

### Sort packed u64 keys instead of (OrdF32, u32) tuples

`src/cube_grid/query/shells.rs` — scan_cell line 105; build_pending lines 147-148; frontier lines 167-170

*sorting · impact **low** · risk low · confidence medium · scanner `query:shells`*

pending is Vec<(OrdF32, u32)> sorted by Reverse(OrdF32), and OrdF32::cmp is f32::total_cmp (src/fp.rs:36), which performs the sign-magnitude bit transform on both operands at every one of the O(n log n) comparisons. Push a single u64 instead: `(((bits ^ (((bits as i32) >> 31) as u32 >> 1) ^ 0x8000_0000)) as u64) << 32 | slot` (the standard total-order key, inverted for descending), then plain sort_unstable() — one transform per element, single-word branchless compares, and slot in the low bits gives a deterministic tie-break. Recover first_dot by undoing the transform on element 0; out.extend takes the low 32 bits. Tie ordering for exactly-equal dots changes (currently unspecified by sort_unstable), but the consumer dedups and within-layer order is only a nearest-first heuristic.

### mem::take the current layer to iterate without indexing through &mut self

`src/cube_grid/query/shells.rs` — build_pending lines 119-138

*bounds-checks · impact **low** · risk low · confidence medium · scanner `query:shells`*

The layer loop indexes `self.current[layer_idx]` because scan_cell takes &mut self; this costs a bounds check per cell plus forces the compiler to reload current's ptr/len after every scan_cell/mark_visited call since it cannot prove they leave self.current untouched. Use `let cur = std::mem::take(&mut self.current); for &cell in &cur { ... }` then rotate buffers (`self.current = std::mem::take(&mut self.next); self.next = cur;`) in place of the swap at line 138, preserving buffer reuse. Removes the per-cell bounds check and the redundant reloads.

### Make mark_visited branchless with a single bounds-checked access

`src/cube_grid/query/scratch.rs` — mark_visited, lines 13-20

*branch · impact **low** · risk low · confidence medium · scanner `query:scratch`*

mark_visited indexes visited_stamp twice (read at line 15, conditional write at line 18), giving two bounds checks (the second is usually but not guaranteeably elided) plus a data-dependent branch and conditional store in the BFS ring-expansion inner loop. Replace with `let slot = &mut self.visited_stamp[cell as usize]; let fresh = *slot != self.stamp; *slot = self.stamp; fresh` — one bounds check, one address computation, and an unconditional store to a line already in cache, removing the in-function branch. Semantics are identical (re-storing the same stamp is a no-op).

### Shrink visited_stamp to u16 generations to halve scratch footprint

`src/cube_grid/query/scratch.rs` — CubeMapGridScratch::new line 6 (struct fields src/cube_grid/mod.rs:250-251; wrap logic src/cube_grid/query/shells.rs:58-62)

*memory-layout · impact **low** · risk low · confidence medium · scanner `query:scratch`*

visited_stamp is Vec<u32> sized 6*res*res — about 1 MB at 1M points — but each entry only needs to distinguish recent queries, and wrap-refill logic already exists at shells.rs:58-62. Change visited_stamp to Vec<u16> and stamp to u16: the fill(0) then fires every 65534 queries instead of ~4 billion (a ~0.5 MB memset per 65k queries, negligible), the stamp array halves in size for better L2 residency across consecutive spatially-coherent queries in a bin, and twice as many adjacent cells share a cache line during ring scans. No tolerance or numeric behavior is involved.

## Cube-grid build & projection

### Fuse point_slots inverse-map construction into the scatter loop

`src/cube_grid/build.rs` — lines 312-325 (point_slots pass); scatter loops at lines 167-181 (parallel) and 245-256 (sequential)

*memory-layout · impact **medium** · risk low · confidence high · scanner `grid:build`*

After scatter, a separate sequential pass allocates vec![u32::MAX; n] and does n random writes: point_slots[point_indices[slot]] = slot. The scatter loop already knows both (original_idx, pos), and original_idx = global_offset + i is sequential within each chunk, so writing point_slots[original_idx] = pos inside the scatter is a streaming (cache-friendly) write. Share point_slots via the same as_mut_ptr-as-usize trick used for the other four arrays, drop the u32::MAX fill (use set_len; every slot is written exactly once), and delete the whole extra n-element pass with its random writes.

### Scatter one interleaved 16-byte record instead of 4 random writes to 4 arrays

`src/cube_grid/build.rs` — lines 131-184 (parallel scatter), 228-256 (sequential scatter)

*memory-layout · impact **medium** · risk low · confidence medium · scanner `grid:build`*

The scatter writes the same random position `pos` into four separate arrays (point_indices, cell_points_x/y/z), so each point costs 4 random cache lines (and up to 4 TLB pages). Instead scatter into one scratch Vec of a #[repr(C)] {idx: u32, x: f32, y: f32, z: f32} (exactly 16 bytes, 4 records per cache line), then do a cheap sequential deinterleave pass into the four SoA arrays the queries need. Cuts random write traffic ~4x at the price of one streaming pass; classic counting-sort layout trick, bitwise-identical output.

### Make face selection branch-free (cmov axis pick + sign-bit face index)

`src/cube_grid/projection.rs` — point_to_face_uv, lines 34-55 (called per point from build.rs:62 and query paths)

*branch · impact **medium** · risk medium · confidence medium · scanner `grid:build`*

For uniformly distributed sphere points the 3-way if-chain plus nested sign test is mispredicted most of the time (face is ~uniform over 6 outcomes), costing ~15+ cycles/point in the classify pass. Rewrite as: pick axis with two cmov-friendly compares (`let mut a = 0; if ay > ax { a = 1 } if az > comp { a = 2 }`), compute `face = 2*a + (p[a] < 0.0) as usize`, and derive (u, v) from indexed components with sign flips taken from a small const [[f32;2];6] table, multiplied by 1/|p[a]|. Must preserve the exact `>=` tie-break order so boundary points land in the same cells as today.

### Batch the per-point cell classification with f32x8 lanes

`src/cube_grid/build.rs` — lines 60-65 (point_cells map), helpers point_to_face_uv / uv_to_st / face_uv_to_cell in projection.rs

*simd · impact **medium** · risk medium · confidence low · scanner `grid:build`*

The classify pass does per point: abs/max face pick, 2 divides, 2 sqrts (uv_to_st), 2 float-to-int conversions — all scalar. With the crate's existing wide f32x8 seam, process 8 points at a time: mask-blend the axis/sign selection, one vdivps for the reciprocal, vsqrtps for both uv_to_st calls (uv_to_st is already a select: s = 0.5*sqrt(1+3|u|), then blend s vs 1-s on sign), and lane-wise iu/iv clamp+index. Removes the face-selection mispredicts entirely and amortizes div/sqrt 8x. Risk is 1-ulp lane-vs-scalar divergence: queries (locate) use the scalar path, so a boundary point's stored cell could disagree with its query-time cell — confirm consumers only need self-consistency of cell_offsets/point_cells, or reuse the SIMD path at query time too.

### Replace two divides with one reciprocal + copysign in point_to_face_uv

`src/cube_grid/projection.rs` — lines 30-56, point_to_face_uv

*arithmetic · impact **medium** · risk low · confidence high · scanner `grid:proj`*

Each face branch does two f32 divides by the same denominator (e.g. `-z/ax, y/ax`) plus an unpredictable sign branch. Rust's strict FP means LLVM will not merge the two divides. Change each axis arm to `let inv = 1.0/ax;` then `u = -z * inv.copysign(x); v = y * inv; face = (x.to_bits() >> 31) as usize` (analogous for Y: `u = x*inv, v = -z*inv.copysign(y)`; Z: `u = x*inv.copysign(z), v = y*inv`). This is 1 divss + 2 mulss + sign-bit ops instead of 2 divss + a 50/50 branch on Zen 2 where divss is ~10c. If bit-exactness is required, the branch-only variant `u = -z/x` (signed denominator) is bit-identical to the original since x == ±ax exactly; the reciprocal variant changes u,v by <=1 ulp, which only matters for points sitting exactly on grid-cell boundaries of an acceleration structure.

### Branchless bit-exact uv_to_st via abs + copysign around 0.5

`src/cube_grid/projection.rs` — lines 10-16, uv_to_st

*branch · impact **medium** · risk low · confidence high · scanner `grid:proj`*

uv_to_st branches on `u >= 0.0`, a 50/50 data-dependent branch executed twice per point in face_uv_to_cell (grid build and every query seed). Replace with `let t = 0.5 * (1.0 + 3.0 * u.abs()).sqrt(); 0.5 + (t - 0.5).copysign(u)`. This is bit-exact: 3.0*u.abs() is an exact sign flip of 3.0*u, t is in [0.5, 1] for u in [-1, 1] so `t - 0.5` is exact by Sterbenz, the positive case reconstructs t exactly, and the negative case rounds the same real value as `1.0 - t`. One sqrt, zero branches, two cheap sign-bit ops.

### Do cell_to_face_ij divisions in u32 instead of usize

`src/cube_grid/projection.rs` — lines 89-95, cell_to_face_ij

*arithmetic · impact **medium** · risk low · confidence high · scanner `grid:proj`*

The function does div+mod by `res*res` and by `res` as 64-bit usize ops. It is called per generator in live_dedup/binning.rs:102 and in weld.rs:85 and packed_knn helpers. On Zen 2 (the reference Ryzen 3600), 64-bit DIV is roughly 2x the latency of 32-bit DIV. Cast once: `let c = cell as u32; let r = res as u32; let r2 = r * r; let face = c / r2; let rem = c % r2; let iv = rem / r; let iu = rem - iv * r;` and widen back at return. Valid because 6*res^2 fits comfortably in u32 for any realistic grid (res is a few thousand at most); add a debug_assert on res. Results are integer-identical.

### Add an 8-lane batched point->cell kernel for the grid build loop

`src/cube_grid/projection.rs` — lines 30-69 (point_to_face_uv + face_uv_to_cell); consumer at src/cube_grid/build.rs:62-63

*simd · impact **medium** · risk medium · confidence medium · scanner `grid:proj`*

The grid-build pass maps every input point to a cell index with the scalar branchy pair point_to_face_uv + face_uv_to_cell. The face selection, reciprocal, uv_to_st sqrt, scale, clamp, and index math are all expressible branchlessly (cmp masks + blends + one sqrt lane op + one reciprocal-style lane op), so a `points_to_cells(&[Vec3]) -> cells` batch function using the existing f32x8 backend in src/fp.rs would get ~8x lane utilization on this per-point pass. Requires the branchless scalar reformulations above as the lane recipe; only worth it if build-phase timing shows this loop is material.

### Reuse chunk_counts in place as chunk_cursors; drop per-chunk zeroed Vec allocs

`src/cube_grid/build.rs` — lines 100-121 (prefix sum), chunk_cursors alloc at line 105

*allocation · impact **low** · risk low · confidence high · scanner `grid:build`*

Prefix sum allocates `vec![vec![0u32; num_cells]; num_chunks]` for cursors and only ever copies running positions into it while reading chunk_counts. Rewrite the inner loop in place: `let c = chunk_counts[chunk][cell]; chunk_counts[chunk][cell] = current_pos; current_pos += c;` and feed chunk_counts into the scatter zip. Eliminates num_threads heap allocations plus zero-filling num_threads*num_cells u32 (~1 MB at 1M points / 12 threads). Optionally flatten counts into a single contiguous Vec<u32> of len num_chunks*num_cells to remove pointer-chasing across separate allocations in the cell-major prefix loop.

### Replace two divides with one reciprocal multiply in point_to_face_uv

`src/cube_grid/projection.rs` — point_to_face_uv, lines 30-56 (hot call site: build.rs:60-65 per-point classify map)

*arithmetic · impact **low** · risk low · confidence medium · scanner `grid:build`*

Every branch computes two f32 divisions by the same denominator (e.g. `(-z / ax, y / ax)`). Hoist `let r = 1.0 / ax;` (computed once per point, before the sign branch can pick the axis value) and emit `(-z * r, y * r)` etc., trading 2 dependent divides for 1 divide + 2 independent muls per point. Result can differ by 1 ulp, which only shifts boundary points between adjacent bins; since the same function is used at build and query time the grid stays self-consistent.

### Drop redundant .max(0.0) and hoist res casts in face_uv_to_cell

`src/cube_grid/projection.rs` — lines 60-69, face_uv_to_cell

*arithmetic · impact **low** · risk low · confidence high · scanner `grid:proj`*

`(fu as usize)` in Rust is a saturating cast: negative floats (and NaN) already map to 0, so `.max(0.0)` on lines 64-65 is dead and costs two maxss per call. Remove both and write `let iu = ((su * resf) as usize).min(res - 1);` with `let resf = res as f32;` computed once (currently `res as f32` is converted twice). Integer-identical output, two fewer FP ops and one fewer int-to-float conversion on a path executed once per point and per query.

### Simplify the Y-axis predicate to a single compare in point_to_face_uv

`src/cube_grid/projection.rs` — line 41, point_to_face_uv

*branch · impact **low** · risk low · confidence high · scanner `grid:proj`*

The else-if tests `ay >= ax && ay >= az`, but given the first branch failed, `ay >= az` alone is equivalent: if ay >= az held while ay < ax, then first-branch failure forces ax < az <= ay < ax, a contradiction, so the extra `ay >= ax` test never changes the outcome. Replace with `else if ay >= az`, saving one compare+branch on a path taken for ~2/3 of uniformly distributed points. Bit-exact; subsumed by the full branchless rewrite if that lands.

## Cell-build loop (knn_clipping/cell_build)

### Slice packed_chunk to batch.n once in clip_batch to elide per-iteration bounds checks and Vec reloads

`src/knn_clipping/cell_build/run.rs` — clip_batch, lines 341-342 and 401-402; StreamPhase.packed_chunk decl line 272

*bounds-checks · impact **medium** · risk low · confidence high · scanner `cell:run-mem`*

The inner loop indexes `phase.packed_chunk[pos]` (line 342) and `phase.packed_chunk[pos + 1]` (line 401) through a `&'x mut Vec<u32>` double indirection, paying a bounds check plus a ptr/len reload through the Vec header on every neighbor (the opaque `clip_with_slot_result_policy` call between iterations prevents LLVM from caching them). Field-split the phase struct at the top of clip_batch (`let StreamPhase { builder, packed_chunk, attempted_neighbors, .. } = phase;`) and take `let chunk: &[u32] = &packed_chunk[..batch.n];` once, then use `chunk[pos]`/`chunk[pos + 1]`. One up-front length check replaces N per-iteration checks, since `pos + 1 < batch.n == chunk.len()` is provable; the immutable slice coexists with the mutable builder borrow because they are disjoint fields.

### Narrow AttemptedNeighbors stamp table from u32 to u16

`src/knn_clipping/cell_build/run.rs` — struct AttemptedNeighbors, lines 20-58 (seen_stamp: Vec<u32>, clear() at 35-41)

*memory-layout · impact **medium** · risk low · confidence medium · scanner `cell:mod`*

seen_stamp is one u32 per input point (4 MB at 1M points) and is probed once per candidate neighbor in clip_batch/clip_seed_neighbors with scattered point-id indices, so probes are cache misses at large n. Change seen_stamp to Vec<u16> and stamp to u16; clear() already handles wrap by filling, which would now fire every ~65534 cells (one memset of n*2 bytes amortized over 65k cells, negligible). Halving the table doubles the effective cache hit rate for the dedup probe with zero semantic change.

### Index the seen-stamp table by grid slot instead of point id for cache locality

`src/knn_clipping/cell_build/run.rs` — AttemptedNeighbors::insert/mark call sites: clip_batch lines 342-358, clip_seed_neighbors line 297

*memory-layout · impact **medium** · risk medium · confidence low · scanner `cell:mod`*

Both call sites already hold neighbor_slot (clip_batch reads it at line 342; SeedNeighbor carries neighbor_slot). Slots are grid-ordered, so a cell's candidate neighbors cluster into a few cache lines, whereas point ids are input-order and scatter. Switch insert/mark to take the slot, sizing the table by point_indices().len(). Requires point_indices to be a permutation (slot<->id 1:1) so the ShellExpand dedup-on-insert semantics are preserved; a duplicate mapping would only cause a redundant idempotent re-clip, but verify the invariant before changing.

### Slice packed_chunk to batch.n before the clip loop to elide per-iteration bounds checks

`src/knn_clipping/cell_build/run.rs` — clip_batch, lines 341-343 and 401-403

*bounds-checks · impact **low** · risk low · confidence high · scanner `cell:run-branch`*

The loop indexes `phase.packed_chunk[pos]` for `pos in 0..batch.n` and a lookahead `phase.packed_chunk[pos + 1]`, each carrying a bounds check because LLVM cannot prove `batch.n <= packed_chunk.len()` across the opaque clip call. Add `let chunk = &phase.packed_chunk[..batch.n];` before the loop and index `chunk[pos]` / `chunk[pos + 1]`; the single slice-prefix check up front lets both per-iteration checks fold away (the lookahead is already guarded by `pos + 1 < batch.n`). One panic-branch fewer per neighbor in the innermost loop, and it unties `packed_chunk` from the mutable borrow set the optimizer must reason about.

### Make AttemptedNeighbors::insert branchless with a single slice access

`src/knn_clipping/cell_build/run.rs` — AttemptedNeighbors::insert, lines 44-51

*branch · impact **low** · risk low · confidence high · scanner `cell:run-branch`*

insert reads `self.seen_stamp[id]`, branches, then conditionally writes `self.seen_stamp[id]` — two indexing ops (two bounds checks) plus a data-dependent branch on a random-access cache line; in ShellExpand takeover batches re-covered vs fresh points are interleaved so the branch predicts poorly. Replace with `let slot = &mut self.seen_stamp[id]; let fresh = *slot != self.stamp; *slot = self.stamp; fresh` — one bounds check, an unconditional store, and a branchless compare returned to the caller (the line is dirty in cache from the read anyway). This runs once per neighbor in both clip_seed_neighbors and clip_batch.

### Hoist loop-invariant batch.source branching out of clip_batch and make the shell bound branchless

`src/knn_clipping/cell_build/run.rs` — clip_batch, lines 348-359 and 409-413

*branch · impact **low** · risk low · confidence medium · scanner `cell:run-branch`*

Both the dedup-policy match on `batch.source` and the ShellExpand-only `next_dot.max(batch.unseen_bound)` widening branch test the same loop-invariant enum per iteration. Hoist `let is_shell = batch.source == DirectedNeighborBatchSource::ShellExpand;` and `let shell_floor = if is_shell { batch.unseen_bound } else { f32::NEG_INFINITY };` before the loop, then use `if is_shell { insert } else { mark; true }` and `let bound = next_dot.max(shell_floor);` — `max(x, -inf) == x` for the non-NaN dots of unit vectors, so the termination bound is bit-identical while becoming a branchless maxss. Removes one enum compare per iteration and one branch per termination probe.

### Coalesce per-neighbor trace stores into locals flushed at loop exit

`src/knn_clipping/cell_build/run.rs` — clip_batch, lines 364-367

*inlining-codegen · impact **low** · risk low · confidence medium · scanner `cell:run-branch`*

Every clipped neighbor performs four stores into the `&mut BuildTrace` (`last_neighbor_idx`, `last_neighbor_slot`, `last_batch_source`, `last_clip_phase`) — two of them Option-wrapped two-word writes — purely for failure diagnostics that are only read after the build terminates. Write `last_batch_source`/`last_clip_phase` once before the loop (they are batch-invariant), keep neighbor idx/slot in plain `usize`/`u32` locals (registers), and flush them into trace after the loop and immediately before each `break`. Identical diagnostics on every exit path, ~6 memory stores per neighbor removed from the hot loop.

### Hoist generator point load out of the mid-batch bound computation

`src/knn_clipping/cell_build/run.rs` — clip_batch, line 403

*memory-layout · impact **low** · risk low · confidence medium · scanner `cell:run-branch`*

The termination-bound path recomputes `points[generator_idx]` (bounds check + 12-byte load) on every Unchanged-clip iteration; hoisting across the loop is hard for LLVM because `points` is also passed into the cold `enter_fallback` call in the same loop body. Add `let generator = points[generator_idx];` at the top of clip_batch and use `generator.dot(points[next])`. One bounds check and load per batch instead of per termination probe; numerically identical.

### Reorder termination guard to test the register-resident bool before is_bounded()

`src/knn_clipping/cell_build/run.rs` — clip_batch, line 399

*branch · impact **low** · risk low · confidence high · scanner `cell:run-branch`*

The guard reads `phase.builder.is_bounded() && should_check_termination`. `is_bounded()` is an enum-dispatch method (discriminant load + match through BuilderImpl) while `should_check_termination` is a freshly computed bool already in a register, and Changed clips (the common early case) make it false. Swap to `should_check_termination && phase.builder.is_bounded()` so the cheap operand short-circuits the discriminant load on every Changed iteration. Pure reorder of side-effect-free reads.

### Hoist loop-invariant batch.source branches and points[generator_idx] out of the clip_batch loop

`src/knn_clipping/cell_build/run.rs` — clip_batch, lines 348-359, 403, 409-413

*branch · impact **low** · risk low · confidence high · scanner `cell:run-mem`*

Three loop-invariant computations sit inside the per-neighbor loop: the `match batch.source` dedup-mode dispatch (lines 348-359), the bounds-checked 12-byte load `points[generator_idx]` in the termination probe (line 403), and the `batch.source == ShellExpand` branch choosing the unseen bound (lines 409-413). Before the loop, compute `let dedup_insert = batch.source == DirectedNeighborBatchSource::ShellExpand;`, `let gen_point = points[generator_idx];`, and `let unseen_floor = if dedup_insert { batch.unseen_bound } else { f32::NEG_INFINITY };` so the bound is the branchless `next_dot.max(unseen_floor)`. Identical float results (max with NEG_INFINITY is the identity for the non-shell case, and unseen_bound is a real dot bound, not NaN), so numerical behavior is unchanged; LLVM may not unswitch this large loop body on its own.

### Defer per-neighbor BuildTrace stores to loop-exit paths in clip_batch

`src/knn_clipping/cell_build/run.rs` — clip_batch lines 364-367 (and clip_seed_neighbors lines 292-295)

*memory-layout · impact **low** · risk low · confidence medium · scanner `cell:run-mem`*

Every clipped neighbor performs ~6 stores into the caller-owned BuildTrace (two Option discriminant+payload pairs, an Option<enum>, and a 16-byte &'static str) purely for failure diagnostics; because trace escapes via &mut, the compiler must emit these stores on every iteration even though they are only read after a failure. Instead track a local `last_pos: usize` (register-resident) and materialize `trace.last_neighbor_idx/slot/batch_source/clip_phase` only on the `Err(_) => break` path, the NeedsFallback arm, and once after the loop. Semantics ('last attempted neighbor') are preserved since the failing/last pos is known at every exit. Pure store-traffic reduction in the hottest loop; diagnostics-only risk.

### Elide grid-invariant bounds checks on point_indices and seen_stamp via get_unchecked

`src/knn_clipping/cell_build/run.rs` — AttemptedNeighbors::insert/mark lines 44-57; clip_batch lines 343, 351-356, 369, 402

*bounds-checks · impact **low** · risk medium · confidence medium · scanner `cell:run-mem`*

Each neighbor pays three more bounds checks: `point_indices[neighbor_slot as usize]` (line 343, again at 402), `self.seen_stamp[id]` load+store in insert/mark (lines 46-49, 56), and `points[neighbor_idx]` (line 369). All indices are grid invariants: slots come from the grid's own chunk fill so slot < point_indices.len(), and entries of point_indices are point ids < num_points == seen_stamp.len() == points.len(). The debug_asserts at lines 45/55 already document this; back them with `unsafe { get_unchecked / get_unchecked_mut }` in insert/mark and at the two point_indices lookups. Removes 3-4 compare+branch pairs per neighbor; risk is the usual unsafe-on-invariant exposure if a future grid change emits an out-of-range slot.

### Hoist loop-invariant BuildTrace stores out of the per-neighbor clip loops

`src/knn_clipping/cell_build/run.rs` — clip_batch lines 364-367; clip_seed_neighbors lines 292-295

*inlining-codegen · impact **low** · risk low · confidence high · scanner `cell:mod`*

Every clipped neighbor writes 4 trace fields, but two of them are invariant across the loop: in clip_batch, last_batch_source = Some(batch.source) and last_clip_phase = "stream" are constant per batch; in clip_seed_neighbors, last_batch_source = None and last_clip_phase = "edgecheck_seed" are constant for the whole phase. Move those two stores above each loop, keeping only the per-iteration last_neighbor_idx/slot writes. Removes 2 stores (incl. an Option discriminant write) per neighbor in the hottest loop; trace is diagnostic-only so the slight failure-report skew (batch source set even if zero neighbors clipped) is harmless.

### Slice packed_chunk to batch.n to elide per-iteration bounds checks in clip_batch

`src/knn_clipping/cell_build/run.rs` — clip_batch, lines 341-342 and lookahead at 400-401

*bounds-checks · impact **low** · risk low · confidence high · scanner `cell:mod`*

The loop `for pos in 0..batch.n` indexes `phase.packed_chunk[pos]` with a bounds check per iteration because the compiler cannot relate batch.n to packed_chunk.len(). Add `let chunk = &phase.packed_chunk[..batch.n];` before the loop and index chunk instead (or iterate chunk.iter().enumerate()); the existing `pos + 1 < batch.n` guard then also lets LLVM elide the check on the `chunk[pos + 1]` lookahead. One check up front replaces one per neighbor.

### Pass the generator point into clip_batch instead of reloading points[generator_idx] in the termination probe

`src/knn_clipping/cell_build/run.rs` — clip_batch line 403 (next_dot computation); source value available at build_cell_into line 547

*bounds-checks · impact **low** · risk low · confidence medium · scanner `cell:mod`*

The mid-batch termination bound computes `points[generator_idx].dot(points[next])`; points[generator_idx] is invariant for the whole cell (it is already loaded once at builder.reset, line 547) but here it is a bounds-checked slice load inside a conditional, which LLVM cannot speculatively hoist. Add a `generator: Vec3` field to StreamPhase (or a clip_batch parameter) set once per cell. Saves a checked load per Unchanged-clip termination probe.

### Use unchecked indexing in AttemptedNeighbors::insert/mark

`src/knn_clipping/cell_build/run.rs` — AttemptedNeighbors::insert lines 44-51, mark lines 53-57

*bounds-checks · impact **low** · risk medium · confidence medium · scanner `cell:mod`*

insert does a checked read then a checked write of seen_stamp[id], and mark a checked write, once per candidate neighbor in the hot loop; the debug_assert at lines 45/55 already documents the invariant (id comes from point_indices, all entries < num_points = table size). Replace with get_unchecked/get_unchecked_mut under the existing debug_assert, removing 1-2 bounds checks per neighbor. Requires unsafe and trusting the point_indices invariant, hence medium risk; consider only after confirming entries are validated at grid build.

## Topo2d clipping (knn_clipping/topo2d)

### Merge debug_extraction_failure's per-vertex pass into the emission loop to kill duplicate projection

`src/knn_clipping/topo2d/builder/extract.rs` — GnomonicBuilder::to_vertex_data_full line 27, loop lines 38-88; debug_extraction_failure lines 145-203

*arithmetic · impact **high** · risk medium · confidence high · scanner `topo2d:extract`*

to_vertex_data_full (called once per cell from cell_build/run.rs:520, the hot path) unconditionally calls debug_extraction_failure(), whose per-vertex loop (lines 145-203) computes the exact same 9x fma_f64 gnomonic-to-3D projection, f32 truncation, and length_squared that the emission loop (lines 43-65) then recomputes — every vertex is projected twice per cell. Keep the cheap O(1) pre-checks (is_bounded, poly.len < 3, metadata length equality) and move the per-vertex checks into the emission loop: replace `len2 < EXTRACT_DEGENERATE_LEN2` at line 62 with `!len2.is_finite() || len2 < EXTRACT_DEGENERATE_LEN2` (covers the NonFiniteProjectedVertex case, since non-finite u/v yields non-finite len2), and use `.get(plane_a)`/`.get(plane_b)`/`.get(edge_plane)` on neighbor_indices/neighbor_slots/half_planes returning Err(CellFailure::NoValidSeed) on None (this also replaces 5 panicking bounds checks per vertex at lines 67-68 and 80-86 with the error path the caller already handles). All failure cases map to the same CellFailure::NoValidSeed today, so caller behavior is unchanged; keep debug_extraction_failure intact for the cold diagnostics call in cell_build/run/failure.rs:33.

### Replace min_cos sqrt+div threshold test with precomputed max_r2 compare

`src/knn_clipping/topo2d/builder/clip.rs` — commit_clip, lines 50-55 (calls Poly::min_cos at src/knn_clipping/topo2d/types.rs:132)

*arithmetic · impact **medium** · risk low · confidence high · scanner `topo2d:clip`*

Every committed clip without a bounding ref evaluates `poly.min_cos()` = `1.0/(1.0+max_r2).sqrt()` (f64 sqrt + fdiv, ~25-40 cycles) and then tests `!min_cos.is_finite() || min_cos <= MIN_PROJECTION_COS`. The test is monotone in max_r2, so precompute `const MAX_R2_LIMIT: f64 = 1.0/(MIN_PROJECTION_COS*MIN_PROJECTION_COS) - 1.0` and test `!(poly.max_r2() < MAX_R2_LIMIT)` instead (NaN/inf max_r2 correctly fails via the negated compare; add a trivial max_r2 accessor). This removes the sqrt, the divide, and the is_finite bit test, leaving one compare per clip. Only boundary cases within ~1 ulp of the threshold could flip, and this is a coarse projection-validity tolerance, not an exact predicate.

### Fuse bisector_coefficients with plane_to_line to skip normal_unnorm and shorten the dependency chain

`src/knn_clipping/topo2d/builder/projection.rs` — lines 145-173 (GnomonicBuilder::bisector_coefficients) + lines 53-55 (TangentBasis::plane_to_line)

*arithmetic · impact **medium** · risk medium · confidence medium · scanner `topo2d:proj`*

Currently each neighbor computes len_sq = n.n, scale, then builds normal_unnorm = g*scale - n (3 mul-adds), then takes 3 dot products against t1/t2/g. The code's own comment (lines 156-162) states t1 and t2 are orthogonal to g so a and b mathematically equal -t.h. Exploit that: return a = -(n_raw.dot(t1)), b = -(n_raw.dot(t2)), c = fp::fma_f64(scale, gg, -n_raw.dot(g)), where gg = g.length_squared() is stored once in new()/reset() next to inv_two_gg. This deletes the 3-component normal construction and, more importantly, breaks the serial chain (n.n -> scale -> normal -> dots): a and b become independent dots computable immediately, so the downstream clip loop starts sooner. Bit-level results change by the dropped scale*(g.t1) ~1e-16 term, so margins shift by ulps; needs the contract/P5 suites to confirm.

### Narrow PolyBuffer plane metadata from usize to u32

`src/knn_clipping/topo2d/types.rs` — PolyBuffer fields lines 73-77 (vertex_planes, edge_planes); embedded twice in GnomonicBuilder at src/knn_clipping/topo2d/builder.rs:75-76

*memory-layout · impact **medium** · risk medium · confidence medium · scanner `topo2d:builder`*

PolyBuffer stores vertex_planes: [(usize, usize); 64] and edge_planes: [usize; 64], i.e. 1536 of 2560 bytes per buffer are plane indices that are bounded by the per-cell half-plane count (Vec capacity 32, realistically < a few hundred; sentinel is usize::MAX at extract.rs:73/190/377 and clippers.rs:54). Switching to u32 with u32::MAX sentinel shrinks each PolyBuffer from ~2560 to ~1792 bytes, so the poly_a/poly_b ping-pong copy that clip_convex performs on every Changed clip writes 768 fewer bytes per pass and both buffers plus hot scalars fit comfortably in L1. Pure index plumbing, no effect on f64 tolerances; churn is mechanical casts in clippers.rs and extract.rs.

### Hoist exit-side distance reads and lerp_t division above the copy loop in clip_small_ptr

`src/knn_clipping/topo2d/clippers/small.rs` — clip_small_ptr, lines 263-304 (specifically move lines 289-291 up next to lines 264-266)

*inlining-codegen · impact **medium** · risk low · confidence high · scanner `topo2d:small`*

clip_small_ptr (the N=4,8 specialization) computes t_entry, runs the vertex copy loop, and only then reads d_exit/d_exit_next and issues the second division. The sibling clip_small_ptr_d was already restructured (lines 399-407, with the comment 'Read all four distances first, then issue both divisions for ILP') so both divsd latencies overlap the copy loop. Apply the identical reorder to clip_small_ptr: read all four dists and compute both lerp_t values before the entry push. Pure reordering of independent FP computation, bit-identical results.

### Fold the maybe_escalate near-margin pre-screen into signed_dists_mask8 as a second SIMD mask

`src/knn_clipping/topo2d/clippers/small.rs` — lines 199-204 and 335-340 (call sites); co-change in fp.rs signed_dists_mask8 (fp.rs:303-326) and clippers.rs maybe_escalate (clippers.rs:100-103)

*simd · impact **medium** · risk low · confidence medium · scanner `topo2d:small`*

Every clip call currently runs a scalar loop of up to 8 f64 abs+compare+or in maybe_escalate's pre-screen, over dists that signed_dists_mask8 just had in f64x4 registers (d_lo/d_hi). Extend signed_dists_mask8 to also return near_bits = move_mask(d.abs() < filter_eps) for the two halves (pass filter_eps = hp.eps * CLIP_ESCALATION_FACTOR in alongside neg_eps), so the hot-path pre-screen becomes a single '(near_bits & full) != 0' test before calling the #[cold] escalate_mask. NaN lanes compare false in both forms, so semantics are identical; this removes ~8 scalar FP ops from every clip invocation including the dominant Unchanged path.

### Const-gate the disabled escalation pre-screen out of maybe_escalate

`src/knn_clipping/topo2d/clippers.rs` — maybe_escalate, lines 86-109 (loop at 100-103)

*branch · impact **medium** · risk low · confidence high · scanner `topo2d:clippers`*

CLIP_ESCALATION_FACTOR is 0.0 in production (src/tolerances.rs:85), so filter_eps = hp.eps * 0.0 is +/-0.0 (or NaN) and `d.abs() < filter_eps` is always false — but LLVM can't fold this through the runtime hp.eps, so every clip call (called from small.rs:204,340 and bitmask.rs:38, i.e. the innermost hot loop) pays an n-lane abs+compare+OR scan plus the `&dists[..n]` slice bounds check for nothing. Add `#[cfg(not(feature = "p5_shadow"))] if crate::tolerances::CLIP_ESCALATION_FACTOR == 0.0 { return mask; }` at the top; the const comparison folds at compile time and the whole pre-screen disappears in default builds, while p5_shadow builds keep the override path. Behavior is bit-identical for any finite hp.eps.

### Shrink PolyBuffer plane-index arrays from usize to u32

`src/knn_clipping/topo2d/types.rs` — lines 75-76 (PolyBuffer.vertex_planes / edge_planes), push_raw line 119, init_bounding lines 100-105

*memory-layout · impact **medium** · risk medium · confidence medium · scanner `topo2d:types`*

vertex_planes is [(usize, usize); 64] (1024 B) and edge_planes is [usize; 64] (512 B), so each PolyBuffer is ~2.6 KB and the builder keeps two (poly_a/poly_b at builder.rs:75-76) ping-ponging through L1 every clip round. Plane indices index the per-cell half_planes vec (capacity ~32, bounded by kNN candidate count), so [(u32, u32); 64] and [u32; 64] are lossless: buffer shrinks to ~1.8 KB (saving ~1.5 KB across the pair), and the per-surviving-vertex metadata copy in the clip output loops (clippers/small.rs:250-251, 386; output.rs:49-50) drops from 16+8 bytes to a single 8-byte pair move plus 4 bytes. Sentinel usize::MAX becomes u32::MAX; mechanical type change across clippers.rs, clippers/small.rs, clippers/bitmask.rs, builder/extract.rs, p5_shadow.rs.

### Fold sync_neighbor_positions push into commit_clip's Changed arm

`src/knn_clipping/topo2d/builder/clip.rs` — commit_clip lines 35-41, sync_neighbor_positions lines 120-128, call sites lines 90-91 and 163-164

*branch · impact **low** · risk low · confidence high · scanner `topo2d:clip`*

sync_neighbor_positions runs after every clip (including Unchanged, the common case) and compares two Vec lengths to decide whether to push — but neighbor_indices only grows inside commit_clip's ClipResult::Changed arm, including the case that subsequently errs with ClippedAway. Pass `neighbor: Vec3` into commit_clip and push to neighbor_positions_raw unconditionally next to the other three pushes (line 38), deleting sync_neighbor_positions and its two call sites. This removes a branch plus two len loads per clip and keeps the four parallel vec pushes adjacent for the allocator/branch predictor.

### Route commit_clip failure sites through a #[cold] helper

`src/knn_clipping/topo2d/builder/clip.rs` — commit_clip lines 31-34, 46-49, 52-55; early-out checks at lines 68-70 and 141-143

*inlining-codegen · impact **low** · risk low · confidence medium · scanner `topo2d:clip`*

commit_clip has three inline failure paths (TooManyVertices, ClippedAway, ProjectionInvalid) that each write self.failed and construct an Err; these are essentially never taken on the hot path but their code is interleaved with the hot fall-through. Add `#[cold] #[inline(never)] fn fail(&mut self, f: CellFailure) -> Result<ClipResult, CellFailure> { self.failed = Some(f); Err(f) }` and call it from all three sites. This shrinks the inlined hot body of clip_with_slot_result and tells LLVM to lay out the error blocks out-of-line, improving icache density in the per-neighbor loop.

### Mark FallbackBuilder clip methods #[cold] to favor the Gnomonic dispatch arm

`src/knn_clipping/topo2d/builder/clip.rs` — FallbackBuilder::clip_with_slot_result line 201, clip_with_slot_edgecheck line 212; dispatch matches at lines 297-304 and 316-323

*inlining-codegen · impact **low** · risk low · confidence medium · scanner `topo2d:clip`*

clip_with_slot_result_policy and clip_with_slot_edgecheck_policy match on BuilderImpl per clip; the Fallback variant is the rare escalation path but its methods carry no inlining hints (inline(never) only under the profiling feature), so LLVM may inline FallbackConstraint::from_neighbor construction into the hot dispatch. Mark both FallbackBuilder methods `#[cold] #[inline(never)]` unconditionally so the Gnomonic arm becomes the predicted straight-line path and the fallback body is laid out out-of-line. Zero behavioral change; only block layout.

### Replace modulo with wrap-compare for next-vertex index in fallback emission

`src/knn_clipping/topo2d/builder/extract.rs` — FallbackBuilder::to_vertex_data_full line 329; debug_extraction_failure line 373

*arithmetic · impact **low** · risk low · confidence high · scanner `topo2d:extract`*

`vertices[(i + 1) % vertices.len()]` performs a hardware integer division per vertex because the length is not a compile-time constant. Replace with `let j = i + 1; let next = vertices[if j == vertices.len() { 0 } else { j }];` (a predictable compare/select). Trivially equivalent; only matters on the rare fallback path, so the win is small.

### Hoist pos_a/pos_b f64 conversions out of the find_map closure in shared_edge_constraint

`src/knn_clipping/topo2d/builder/extract.rs` — FallbackBuilder::shared_edge_constraint lines 227-249

*inlining-codegen · impact **low** · risk low · confidence medium · scanner `topo2d:extract`*

Inside the `find_map` closure, `pos_a` and `pos_b` (six f32-to-f64 conversions plus DVec3 construction, lines 232-241) are loop-invariant — they depend only on `a` and `b`, not on the constraint being tested — yet are written inside the per-constraint closure. Hoist them above the `.iter().enumerate().find_map(...)` chain. LLVM may already hoist them, but the closure is cold-path fallback code where the slow scan runs over all constraints with two dot products each, so making it explicit is free and guarantees the elision.

### Fuse the three edge_neighbor_* parallel Vec pushes into one AoS push

`src/knn_clipping/topo2d/builder/extract.rs` — lines 74-87 and 334-336 (push sites); struct in src/live_dedup/cell_output.rs lines 43-48

*memory-layout · impact **low** · risk medium · confidence medium · scanner `topo2d:extract`*

Each vertex does three separate pushes into edge_neighbor_globals/slots/eps (plus three reserve calls per cell), i.e. three capacity branches and three length updates per vertex where one would do. Merging them into a single `Vec<(u32, u32, f32)>` field (12-byte AoS records) halves the per-vertex bookkeeping and improves locality for the dedup/emit consumers that read all three fields together. Risk is churn rather than correctness: consumers in live_dedup/emit.rs, live_dedup/edge_checks.rs (line 127), and the plane builders (plane_clipping/builder.rs:318-350, periodic_builder.rs:285-389) all touch these fields, so it is a wider edit than this file alone.

### Hoist loop-invariant generator.normalize() out of FallbackConstraint::from_neighbor

`src/knn_clipping/topo2d/builder/projection.rs` — lines 62-81 (FallbackConstraint::from_neighbor), callers builder.rs:281 and clip.rs:179

*arithmetic · impact **low** · risk low · confidence high · scanner `topo2d:proj`*

from_neighbor calls generator.normalize() (length_squared + sqrt + div + 3 mul) on every invocation, but FallbackBuilder::from_gnomonic (builder.rs:274-289) calls it in a map over all existing constraints with the identical generator, and clip.rs:179 reuses the same builder generator per cell. Change the signature to take the pre-normalized generator (normalize once before the loop / cache it on FallbackBuilder). Bit-identical output since normalize is deterministic; pure redundant-work removal on the fallback escalation path, which is rare, so overall impact is small.

### Replace Box<dyn Iterator> in Topo2DBuilder::neighbor_indices_iter with an enum iterator

`src/knn_clipping/topo2d/builder/projection.rs` — lines 317-323 (Topo2DBuilder::neighbor_indices_iter)

*allocation · impact **low** · risk low · confidence high · scanner `topo2d:proj`*

The dispatch wrapper heap-allocates a Box<dyn Iterator> and pays virtual dispatch per next() just to unify two concrete iterators; the sole caller (cell_build/run/failure.rs:35) immediately collects into a Vec. Replace with a two-variant Either-style enum implementing Iterator (matching the pattern already used elsewhere), or expose a collect_neighbor_indices(&mut Vec<usize>) that matches on inner and extends. Removes one allocation plus indirect calls per failed cell; only fires on the failure path, so the win is small but the change is mechanical.

### Branchless arbitrary-axis pick in TangentBasis::new

`src/knn_clipping/topo2d/builder/projection.rs` — lines 38-49 (TangentBasis::new), called per cell from new()/reset() (lines 97, 130)

*branch · impact **low** · risk low · confidence medium · scanner `topo2d:proj`*

The smallest-|component| axis selection uses a two-branch cascade whose outcome is data-dependent and effectively unpredictable across random generators (~2 mispredicts per cell reset). Compute the axis index branchlessly, e.g. let ax=g.x.abs(); ay=...; az=...; let idx = (!(ax<=ay && ax<=az)) as usize * (1 + (!(ay<=az)) as usize); then index a const [DVec3::X, DVec3::Y, DVec3::Z] table — keeping the exact same tie-break conditions so the chosen axis (and thus t1/t2 bits) is unchanged for every input. Saves only ~10-30 cycles per cell against a microsecond-scale cell build, so strictly a polish item.

### mem::take constraints in fallback re-entry and mark enter_fallback #[cold]

`src/knn_clipping/topo2d/builder.rs` — enter_fallback lines 198-212; FallbackBuilder::from_fallback lines 299-306

*allocation · impact **low** · risk low · confidence high · scanner `topo2d:builder`*

from_fallback clones builder.constraints (heap alloc + memcpy) even though the source FallbackBuilder is overwritten and dropped on the next line of enter_fallback (self.inner = ...). Match on &mut self.inner instead and use std::mem::take(&mut builder.constraints) to move the Vec for free. Additionally, enter_fallback is the rare escalation path called from three hot-loop sites (cell_build/run.rs:110, 311, 378); annotating it #[cold] #[inline(never)] keeps its code and the FallbackBuilder construction out of the hot clip loop's layout and tells the compiler those branches are unlikely.

### Retain GnomonicBuilder scratch across fallback episodes instead of drop/realloc

`src/knn_clipping/topo2d/builder.rs` — BuilderImpl enum lines 18-26; enter_fallback line 211; Topo2DBuilder::reset at builder/projection.rs:275-281

*allocation · impact **low** · risk medium · confidence high · scanner `topo2d:builder`*

enter_fallback replaces self.inner, dropping the GnomonicBuilder and its four heap Vecs (half_planes, neighbor_indices, neighbor_slots, neighbor_positions_raw, each with_capacity 32) plus ~5KB of poly buffers; the next reset() then rebuilds GnomonicBuilder::new from scratch, re-allocating all four Vecs and re-running angle_pad.sin_cos(). Restructure Topo2DBuilder to always own the GnomonicBuilder and add an Option<Box<FallbackBuilder>> (or fallback_active flag), so a fallback cell costs only the FallbackBuilder and the gnomonic scratch survives for the following cells. Impact is low because fallback fires only on ProjectionLimit, but fallbacks cluster geometrically (polar/degenerate regions), making each occurrence a double allocation burst today.

### Replace poly_a/poly_b + use_a branch with [PolyBuffer; 2] indexed select

`src/knn_clipping/topo2d/builder.rs` — GnomonicBuilder fields lines 75-77; consumers at builder/clip.rs:84-87, 100-103, 157-160 and builder/projection.rs:176-182 (current_poly)

*branch · impact **low** · risk low · confidence medium · scanner `topo2d:builder`*

Every clip and every current_poly() call selects the active buffer via if self.use_a { &self.poly_a } else { &self.poly_b }, a data-dependent branch on a flag that flips on each Changed clip. Storing polys: [PolyBuffer; 2] with cur: u8 lets selection become &self.polys[usize::from(self.cur)] and the target buffer self.polys[usize::from(self.cur ^ 1)] — bounds-check-free (index provably < 2) address arithmetic instead of a branch, and it collapses the duplicated then/else clip_convex call pairs in clip.rs into single calls. Win is modest since an alternating flag predicts well, but it also shrinks code size in the hottest function.

### Replace the i == exit_idx loop exit with a precomputed run-length counted loop

`src/knn_clipping/topo2d/clippers/small.rs` — copy loops at lines 281-287 (clip_small_ptr) and 423-432 (clip_small_ptr_d)

*branch · impact **low** · risk low · confidence medium · scanner `topo2d:small`*

Both copy loops carry two per-iteration conditional branches: the 'i == exit_idx' exit test and the index wrap (modulo or compare-reset). The kept run length is known up front: keep = ((exit_idx + N - entry_next) % N) + 1, computed once with a constant-N modulo (this matches the first-arc semantics exactly, unlike popcnt of the mask). Rewrite as 'for k in 0..keep' with index wrap only, giving LLVM a bounded trip count (<= N <= 8) it can unroll and removing the loop-carried compare against exit_idx. Output bytes are identical since the visited indices are unchanged.

### Use masked add for prev-index in the power-of-two clipper

`src/knn_clipping/topo2d/clippers/small.rs` — lines 226-231 in clip_small_ptr (entry_idx / exit_idx computation)

*arithmetic · impact **low** · risk low · confidence high · scanner `topo2d:small`*

clip_small_ptr is only dispatched with N=4 and N=8 (clippers.rs match arms), so the 'if entry_next == 0 { N - 1 } else { entry_next - 1 }' selects can be '(entry_next + N - 1) & (N - 1)': one lea+and instead of cmp+cmov, and it shortens the dependency chain feeding the entry/exit pointer loads. Leave the _d variant (non-power-of-two N) as-is since constant modulo there costs more than the cmov. Exact same indices produced; zero numerical effect.

### Const-fold the eps*factor term in clip_convex's early-unchanged clearance

`src/knn_clipping/topo2d/clippers.rs` — clip_convex, line 213

*arithmetic · impact **low** · risk low · confidence high · scanner `topo2d:clippers`*

`let t = hp.c - hp.eps * crate::tolerances::CLIP_ESCALATION_FACTOR;` with the factor const 0.0 still emits a mulsd+subsd per call because LLVM can't fold x*0.0. Replace with `let t = if crate::tolerances::CLIP_ESCALATION_FACTOR == 0.0 { hp.c } else { hp.c - hp.eps * crate::tolerances::CLIP_ESCALATION_FACTOR };` — the const branch folds to just `hp.c`. Identical result for finite hp.eps (c - (+0.0) == c exactly). Tiny per-call win but it sits on the early-unchanged fast path taken for every n>=5 unbounded clip.

### Fold has_bounding_ref into the dispatch jump-table index

`src/knn_clipping/topo2d/clippers.rs` — dispatch_clip, lines 135-188

*branch · impact **low** · risk low · confidence medium · scanner `topo2d:clippers`*

The match compiles to a jump table on n followed by a separate conditional branch on poly.has_bounding_ref inside each arm — two control-flow decisions per clip. Match on a combined dense index instead: `match (n << 1) | has_bounding_ref as usize { 7 => clip_small_ptr_d::<3,true>(..), 6 => clip_small_ptr_d::<3,false>(..), ... }` so the 12 monomorphizations sit behind a single indexed jump. has_bounding_ref is mostly predictable (true early in a cell's build, then false), so the win is one fewer branch slot plus slightly tighter code; n itself varies per clip so the indirect jump cost is unchanged.

### Stop fully inlining the 12-way dispatch body into the rare edgecheck entry

`src/knn_clipping/topo2d/clippers.rs` — dispatch_clip #[inline(always)] at line 134; clip_convex_edgecheck at lines 228-241

*inlining-codegen · impact **low** · risk low · confidence low · scanner `topo2d:clippers`*

dispatch_clip is #[inline(always)] and every target (clip_small_ptr/_d at small.rs:177/313, plus clip_bitmask) is also #[inline(always)], so the entire ~13-body clipper blob is duplicated into both clip_convex and clip_convex_edgecheck. The edgecheck variant is explicitly for rare seed constraints, so its copy is pure I-cache pressure next to the hot copy. Have clip_convex_edgecheck call a non-inlined shim (`#[inline(never)] fn dispatch_clip_outline(...)` that forwards to dispatch_clip) while clip_convex keeps the inlined version — one direct call on the rare path in exchange for roughly halving the duplicated dispatch code. Needs a perf check since the authors chose inline(always) deliberately.

### Make lerp_t branchless with a max/min clamp chain

`src/knn_clipping/topo2d/clippers.rs` — lerp_t, lines 117-124

*branch · impact **low** · risk medium · confidence medium · scanner `topo2d:clippers`*

Current code computes t = d0/(d0-d1), then takes a data-dependent `t.is_finite()` branch before clamp; lerp_t runs 2-4 times per clip (small.rs:266,291,406,407; bitmask.rs:91,104). Replace the body with `(d0 / (d0 - d1)).max(0.0).min(1.0)` — Rust's f64::max/min return the non-NaN operand, so the degenerate 0/0 NaN still maps to 0.0, and the abs+cmp+branch disappears in favor of two branchless min/max ops. Behavior change only when t is exactly +/-inf (d0 == d1 with d0 != 0): current returns 0.0, new returns the nearer clamp endpoint (1.0 for +inf), which is arguably the better clamp but is a numerical-behavior change in a degenerate tie case — verify against the flipped-decision tests before landing.

### Cache-line align PolyBuffer with #[repr(C, align(64))] and arrays-first field order

`src/knn_clipping/topo2d/types.rs` — lines 68-77 (struct PolyBuffer)

*memory-layout · impact **low** · risk low · confidence medium · scanner `topo2d:types`*

The hot loads are f64x4 pairs over us/vs in fp::signed_dists_mask8, taken at offsets i*8 bytes with i a multiple of 8, i.e. exact 64-byte strides from the start of each array (bitmask.rs:24-30). PolyBuffer is repr(Rust) embedded in the builder with only 8-byte alignment, so these 32-byte loads can straddle cache lines on every chunk. Change to #[repr(C, align(64))] with field order us, vs, vertex_planes, edge_planes, len, max_r2, has_bounding_ref: each array then starts on a 64-byte boundary (512 B each keeps successors aligned), making every 8-lane chunk load split-free. One-line layout attribute, zero numerical change.

### Shrink HalfPlane 48->40 bytes: plane_idx as u32, eps as f32

`src/knn_clipping/topo2d/types.rs` — lines 16-23 (struct HalfPlane), new_unnormalized lines 26-39

*memory-layout · impact **low** · risk medium · confidence medium · scanner `topo2d:types`*

HalfPlane is 5 f64 + usize = 48 bytes and the per-cell Vec<HalfPlane> (builder.rs:71) is re-walked every clip round reading a, b, c, eps, ab2 (early-reject at clippers.rs:214). plane_idx fits u32 (bounded by candidate count), and eps already carries only f32 precision in its sqrt ((ab2 as f32).sqrt() as f64, line 28); storing eps: f32 and widening at use (neg_eps = -(hp.eps as f64)) plus plane_idx: u32 gives a 40-byte struct, ~17% better cache density on the constraint scan. Caveat: the final multiply by CLIP_EPS_INSIDE is currently done in f64, so storing f32 adds one rounding to the tolerance, which can flip inside/outside decisions in a ~1e-7-relative band of eps; this crate's P5 work shows margin bits matter, so verify with the parity/audit suites.

## Live dedup engine (live_dedup)

### Fuse slot_gen_map build into the placement pass, eliminating the two-gather third pass

`src/live_dedup/binning.rs` — assign_bins_with, lines 160-185 (placement loop) and 205-218 (slot_gen_map loop)

*memory-layout · impact **medium** · risk low · confidence high · scanner `dedup:binning`*

Pass 3 walks all cells and for each slot does two random gathers (generator_bin[g] from a u8 array, global_to_local[g] from a u32 array — typically two cache misses per element at large n) to rebuild a packed value that pass 2 already had in registers. Because pass 2 iterates points in exactly cell-major slot order, the slot index equals the existing `visited` counter: write `slot_gen_map[visited] = (b_usize as u32) << local_shift | local_usize as u32` inside the pass-2 inner loop (right where the dangling 'Pack:' comment at line 175 sits) and delete pass 3. This also lets slot_gen_map become a sequential push/with_capacity build, dropping the n*4-byte u32::MAX prefill memset at line 157.

### Compute bin_for_cell once per cell into a u8 scratch instead of twice with 6 divisions each

`src/live_dedup/binning.rs` — assign_bins bin_for_cell closure lines 101-106; calls at lines 147 and 161

*arithmetic · impact **medium** · risk low · confidence high · scanner `dedup:binning`*

bin_for_cell does cell/(res*res), cell%(res*res), rem/res, rem%res, iu/bin_stride, iv/bin_stride — six integer divisions per call — and is evaluated 2x per cell (counting pass and placement pass). During the counting pass, store the result into a `cell_bin: Vec<u8>` (num_bins <= 96 fits u8; size 6*res^2 ~ n/4 bytes) and read it sequentially in the placement pass, halving the division work. Optionally also replace the iu/iv divides with `bu_of[iu]`/`bv_of[iv]` LUTs of length res built once, since bin_stride is a runtime divisor LLVM cannot strength-reduce.

### Narrow bin_generators from Vec<Vec<usize>> to Vec<Vec<u32>>

`src/live_dedup/binning.rs` — BinAssignment::bin_generators line 18; built at lines 151-153, pushed at line 170

*memory-layout · impact **medium** · risk low · confidence medium · scanner `dedup:binning`*

Generator indices originate as u32 from point_indices (g_u32 is widened to usize at line 164 just to be stored), and n provably fits u32 since global_to_local is u32. Storing them as usize doubles the write traffic of the placement pass and the read footprint of the per-bin generator lists that the hot drivers iterate (knn_clipping/driver.rs:64, plane_clipping/driver.rs:160, plane_clipping/periodic_driver.rs:141). Change the element type to u32 and add `as usize` at the three driver use sites; halves the cache footprint of this n*8-byte structure.

### Hoist per-vertex checked_u32 overflow checks out of the dedup loop

`src/live_dedup/emit.rs` — emit_cell_output, lines 185 and 195-196 (loop at 172-205)

*branch · impact **medium** · risk low · confidence medium · scanner `dedup:emit`*

Inside the per-vertex loop, `checked_u32(shard.output.vertices.len(), ...)?` and `checked_u32(shard.output.cell_indices.len(), ...)?` each execute a Result branch per vertex, and each call site inlines a `format!`-based error-construction closure into the loop body. Since the cell pushes at most `count` entries and `count <= 255` is already proven by `checked_u8` at line 167, do two checks once before the loop (`checked_u32(shard.output.vertices.len() + count, ...)` and same for `cell_indices`), then use plain `as u32` casts inside the loop. This removes two compare+branch pairs per vertex and, more importantly, evicts the format!/String landing-pad codegen from the hot loop, shrinking it for icache and register allocation.

### Parallelize into_final() so per-generator Vec frees in ShardDedup drop run across threads

`src/live_dedup/assemble.rs` — lines 113-116, assemble_sharded_live_dedup

*parallelism · impact **medium** · risk low · confidence medium · scanner `dedup:assemble`*

The shards are converted serially: `std::mem::take(&mut data.shards).into_iter().map(|s| s.into_final()).collect()`. Each `into_final` drops `ShardDedup`, whose `edge_checks: Vec<Vec<EdgeCheck>>` holds one inner Vec per local generator, so this is O(num_generators) allocator frees done on one thread between two timed phases. Under `#[cfg(feature = "parallel")]`, change to `data.shards.into_par_iter().map(|s| s.into_final()).collect()` (IndexedParallelIterator preserves order), spreading the frees across the pool.

### Drop side from the overflow sort key and sort by raw u64

`src/live_dedup/edge_checks.rs` — resolve_edge_check_overflow, line 305

*sorting · impact **medium** · risk medium · confidence medium · scanner `dedup:edges`*

The cross-bin overflow records (40 bytes each) are sorted by the tuple (EdgeKey, u8 side), which compiles to a two-field, two-compare comparator. The pairing logic that follows is side-symmetric: reconcile_edge_endpoints' fast path and fallback both patch a/b crosswise regardless of which record comes first, and a.side==b.side is still checked explicitly for the duplicate-side diagnostic. Change to `sort_unstable_by_key(|entry| entry.key.as_u64())` for a single u64 compare per element, which ipnsort handles with cheaper branchless compares. This phase has its own timer (OverflowResolveTiming::sort), so the win is directly measurable.

### Flatten edge_checks Vec<Vec<EdgeCheck>> into inline fixed-capacity slots with rare overflow

`src/live_dedup/shard.rs` — lines 10-24 (ShardDedup struct/new); hot push/take in src/live_dedup/edge_checks.rs:71-99 (push_edge_check, take_edge_checks, recycle_edge_checks)

*memory-layout · impact **medium** · risk medium · confidence medium · scanner `dedup:shard`*

ShardDedup stores per-local edge checks as Vec<Vec<EdgeCheck>> plus a Vec pool: every push does a bounds-checked outer index, a capacity()==0 probe, a possible pool pop+clear, then a pointer-chased inner push; every cell does a mem::take and a later pool recycle. Average incoming checks per cell is small (~3, one per edge to a later local). Replace with a flat arena: `checks: Vec<[EdgeCheck; 4]>` + `counts: Vec<u8>` indexed by LocalId, with a rare overflow Vec<(u32, EdgeCheck)> drained at take time. EdgeCheck is 32 bytes so 4 inline slots are exactly 2 cache lines per local; this removes one heap indirection per push and per take, deletes the pool and all per-local allocations, and replaces take/recycle Vec shuffling with a count reset.

### Narrow cell_indices from u64 packed refs to u32 (8-bit bin + 24-bit vertex index)

`src/live_dedup/shard.rs` — line 34 (cell_indices: Vec<u64>); writers src/live_dedup/emit.rs:192,197 (pack_ref/DEFERRED), reader src/live_dedup/assemble.rs:~285

*memory-layout · impact **medium** · risk medium · confidence medium · scanner `dedup:shard`*

cell_indices is the largest per-vertex output stream (~6 entries per generator) and stores a (BinId u8, v_idx u32) pair packed into a u64 plus a u64 DEFERRED sentinel. BinId already fits in 8 bits and per-shard vertex counts are ~2*(N/num_bins), so a 24-bit index suffices for all benched configurations; pack into u32 with a checked_u24-style guard (the code already errors via checked_u32/checked_u8 for analogous limits). Halves write bandwidth in the emit loop and read bandwidth in assembly concat. Risk is the 16M-vertices-per-shard ceiling (e.g. forced tiny bin counts at 10M+ points), which must become a checked error or fallback rather than silent truncation.

### Hoist per-element validate_local_capacity to one per-bin check using counts

`src/live_dedup/binning.rs` — assign_bins_with line 168 (call inside per-point inner loop); counts available from lines 145-149

*branch · impact **low** · risk low · confidence high · scanner `dedup:binning`*

validate_local_capacity is called once per point inside the hot placement loop, adding a compare-and-branch plus Result plumbing per element, even though the final population of every bin is already known from the counting pass. After the counting loop, run a single 0..num_bins loop checking `counts[b] <= local_mask as usize + 1` (reporting counts[b] as local_population on failure) and drop the per-element call. Same error condition, removed from n iterations down to num_bins (<=96).

### Hoist &mut bin_generators[b_usize] out of the per-point inner loop

`src/live_dedup/binning.rs` — assign_bins_with lines 163-184; indexed at lines 167 and 170

*bounds-checks · impact **low** · risk low · confidence medium · scanner `dedup:binning`*

Each point does `bin_generators[b_usize].len()` then `bin_generators[b_usize].push(g)`: two bounds-checked outer-Vec indexings plus reloads of the inner Vec's ptr/len/cap per element, which LLVM cannot fully cache because push writes those fields back through memory. Since b_usize is constant for the whole cell, bind `let bg = &mut bin_generators[b_usize];` once per cell and use `bg.len()`/`bg.push(g)` in the inner loop, keeping len in a register and removing the per-element bounds checks.

### Outline the off-shard deferred-slot branch into a #[cold] helper

`src/live_dedup/emit.rs` — emit_cell_output, else branch lines 193-204

*inlining-codegen · impact **low** · risk low · confidence medium · scanner `dedup:emit`*

The `owner_bin != bin` branch (build a DeferredSlot, push to two Vecs) only fires for vertices whose min-generator lives in another bin, i.e. bin-boundary cells; interior cells never take it. Extract that block into a `#[cold] #[inline(never)]` function taking (`&mut shard.output`, key, pos, bin). The hot on-shard path then becomes a short straight-line loop body with the rare path as a call, improving branch layout and icache density. Pairs naturally with hoisting the `checked_u32` at line 196 into the cold helper or per-cell precheck.

### Elide bounds check on generator_bin owner lookup with get_unchecked

`src/live_dedup/emit.rs` — emit_cell_output, line 182 (`assignment.generator_bin[key[0] as usize]`)

*bounds-checks · impact **low** · risk medium · confidence medium · scanner `dedup:emit`*

Every vertex does a bounds-checked index into `assignment.generator_bin` keyed by `key[0]`, which is always a valid generator index by construction (binning.rs builds `generator_bin` with len == n and keys are triplets of generator indices). Replace with `unsafe { *assignment.generator_bin.get_unchecked(key[0] as usize) }` plus a `debug_assert!(key[0] as usize) < len`. The crate already uses unsafe in hot paths (topo2d/clippers/small.rs, sort.rs), so this fits existing norms; saves one compare+branch per vertex in the hottest emission loop.

### Remove duplicated bounds-checked indexing in EdgeScratch::emit edge loops

`src/live_dedup/emit.rs` — EdgeScratch::emit, lines 65-104 (both drain loops)

*bounds-checks · impact **low** · risk medium · confidence medium · scanner `dedup:emit`*

Per drained edge entry, the code performs four bounds-checked indexings: `cell_vertices[locals[i] as usize]` twice (lines 69-70, 90-91) and `self.vertex_indices[locals[i] as usize]` twice (lines 79-81, 98-101); `locals` values are local vertex ids already known < cell vertex count when the entry was created, and `vertex_indices` was resized to that same count at line 42. Hoist `let (la, lb) = (locals[0] as usize, locals[1] as usize);` and use `get_unchecked` with debug_asserts (or fetch both `cell_vertices` entries and both `vertex_indices` entries once). With roughly one edge entry per vertex per cell, this trims ~4 cmp+branch pairs per entry; minor but free.

### Use get_unchecked for vertex_offsets gather in the per-element cell-index rewrite loop

`src/live_dedup/assemble.rs` — line 320, inner loop of the maybe_par_into_iter closure (lines 305-321)

*bounds-checks · impact **low** · risk low · confidence high · scanner `dedup:assemble`*

The hottest loop in assembly rewrites every packed cell index (~6 per cell): `dst.add(i).write(vertex_offsets[vbin.as_usize()] + local)`. The `vertex_offsets[...]` index is bounds-checked per element even though the loop body is already inside an `unsafe` block and line 314 debug_asserts `vbin.as_usize() < num_bins`. Replace with `*vertex_offsets.get_unchecked(vbin.as_usize())` to drop a data-dependent branch from a loop that runs total_cell_indices times.

### Accumulate prefix sum in u64 with one final overflow check instead of per-cell checked_add

`src/live_dedup/assemble.rs` — lines 209-221, prefix-sum loop in assemble_sharded_live_dedup

*branch · impact **low** · risk low · confidence medium · scanner `dedup:assemble`*

The serial prefix-sum loop does `total_cell_indices.checked_add(count).ok_or_else(...)` per generator, putting a branch plus an error-closure on every iteration, and indexes `generator_bin[gen_idx]` and `global_to_local[gen_idx]` with bounds checks. Since `cell_count` returns u8, accumulate in `u64` (cannot overflow: num_cells * 255 << 2^64), write `total as u32` each iteration, and check `total <= u32::MAX as u64` once after the loop (on failure the buffer is discarded, so truncated intermediates are harmless). Iterate via `generator_bin.iter().zip(&global_to_local).enumerate()` to elide both bounds checks; optionally build cell_starts_global with `Vec::with_capacity(num_cells+1)` + push to skip the `vec![0; ...]` memset pass (line 209).

### Reuse already-loaded count for count_u16 instead of a second cell_count() lookup

`src/live_dedup/assemble.rs` — lines 283 and 323, maybe_par_into_iter closure

*arithmetic · impact **low** · risk low · confidence high · scanner `dedup:assemble`*

Line 283 loads `let count = shard.output.cell_count(local) as usize;`, then line 323 re-reads the same bounds-checked `cell_counts[local]` slot via `u16::from(shard.output.cell_count(local))`. Replace line 323 with `let count_u16 = count as u16;` (count came from a u8, so the cast is lossless). Removes a redundant bounds-checked load per cell.

### Slice edge_neighbor_* arrays to [..n] to elide per-edge bounds checks

`src/live_dedup/edge_checks.rs` — collect_and_resolve_cell_edges, lines 125-128 and loop at 178-194

*bounds-checks · impact **low** · risk low · confidence high · scanner `dedup:edges`*

The per-edge loop indexes edge_neighbor_slots[i], edge_neighbor_globals[i], and edge_neighbor_eps[i] where only cell_vertices.len()==n is known to the optimizer (the length equalities are debug_asserts, gone in release), so each iteration carries three bounds checks. Rebind the three slices as `&edge_neighbor_slots[..n]` etc. before the loop so the `for i in 0..n` indexing is provably in bounds. This also lets LLVM sink the currently-unconditional `edge_neighbor_eps[i]` load (line 194) into the edges_to_later branch where hp_eps is actually used, since the load is no longer pinned by a potential panic. This loop runs once per edge across every cell, i.e. tens of millions of iterations at 1m-point scale.

### Replace position()+re-index with a single enumerate().find() in the check scan

`src/live_dedup/edge_checks.rs` — collect_and_resolve_cell_edges, lines 219-222

*branch · impact **low** · risk low · confidence high · scanner `dedup:edges`*

The earlier-neighbor resolve does `incoming_checks.iter().position(|c| c.key == edge_key).map(|idx| (idx, incoming_checks[idx]))`, which re-indexes the Vec (a second bounds-checked load) and copies the full 32-byte EdgeCheck by value even though only thirds/indices are read. Replace with `incoming_checks.iter().enumerate().find(|(_, c)| c.key == edge_key)` and use the borrowed `&EdgeCheck` directly. Removes one bounds check and a 32-byte stack copy per resolved in-bin edge (roughly half of all edges).

### Move unresolved_edges pushes into a #[cold] #[inline(never)] helper

`src/live_dedup/edge_checks.rs` — collect_and_resolve_cell_edges, lines 247-251, 254-257, 273-277

*inlining-codegen · impact **low** · risk low · confidence medium · scanner `dedup:edges`*

Per project findings, unresolved-edge mismatches occur 1-20 times per multi-million-cell run, yet the three `shard.output.unresolved_edges.push(UnresolvedEdgeMismatch{..})` arms (including the Vec grow path) are inlined into the per-edge hot loop body, bloating it and consuming i-cache. Add `#[cold] #[inline(never)] fn push_unresolved(out: &mut Vec<UnresolvedEdgeMismatch>, key: EdgeKey, origin: UnresolvedEdgeOrigin)` and call it from all three arms (and optionally the two arms in resolve_edge_check_overflow). LLVM then lays the calls out-of-line and biases branch layout toward the fall-through hot path.

### Pre-size fresh edge-check vecs in push_edge_check when the pool is empty

`src/live_dedup/edge_checks.rs` — ShardDedup::push_edge_check, lines 78-85

*allocation · impact **low** · risk low · confidence high · scanner `dedup:edges`*

When `slot.capacity() == 0` and edge_check_pool.pop() returns None, the subsequent `slot.push(check)` grows the Vec through the 1->2->4 doubling sequence (up to three reallocs per cell, since cells average ~3 incoming checks at 24-32 bytes each). Add an else branch: `*slot = Vec::with_capacity(4)` when the pool pop fails, so a fresh vec allocates once. Only matters during per-shard pool warm-up before recycled vecs dominate, so the win is bounded but the change is one line.

### Merge the three parallel edge_neighbor vecs into one Vec of a packed 12-byte struct

`src/live_dedup/cell_output.rs` — lines 45-47 (CellOutputBuffer fields) and clear() lines 53-55

*memory-layout · impact **low** · risk low · confidence medium · scanner `dedup:cellout`*

CellOutputBuffer keeps edge data as three parallel Vecs (edge_neighbor_globals: Vec<u32>, edge_neighbor_slots: Vec<u32>, edge_neighbor_eps: Vec<f32>). Every extraction site does three reserves and three pushes per edge (extract.rs:33-35/74-86/314-336, plane_clipping/builder.rs:318-356, periodic_builder.rs:285-321), i.e. three len/cap checks and three distinct allocation tails (cache lines) touched per edge in the per-cell hot loop; the consumer (live_dedup/edge_checks.rs:126-127) reads globals and slots together anyway. Replace with a single `edge_neighbors: Vec<EdgeNeighbor>` where `#[repr(C)] struct EdgeNeighbor { global: u32, slot: u32, eps: f32 }` (12 bytes, zero padding): one reserve, one push, one cache line per edge, and clear() drops to two calls. Only periodic_builder.rs:389's globals-only scan gets a 12-byte stride instead of 4, which is negligible.

### Split vertices Vec<(VertexKey, P)> into SoA keys/positions so edge emit touches only keys

`src/live_dedup/cell_output.rs` — line 44 (CellOutputBuffer::vertices), type VertexData line 12

*memory-layout · impact **low** · risk low · confidence medium · scanner `dedup:cellout`*

vertices is AoS `Vec<([u32;3], P)>` (24 bytes/elem for P=Vec3). The edge-emit passes in live_dedup/emit.rs:69-70 and 89-90 do indexed lookups `cell_vertices[locals[i] as usize].0` — they only ever need the 12-byte key, but each random access drags the full 24-byte tuple stride; collect_and_resolve also only consumes keys via vertex_indices. Splitting into `keys: Vec<VertexKey>` and `positions: Vec<P>` halves the footprint of those indexed lookups and keeps the dedup loop at emit.rs:172-199 sequential over both arrays; the cost is one extra push at extract.rs:70/327 and builder push sites. Win is modest because cells are small (~6 verts, usually one cache line either way), but it also lets emit pass `&keys` instead of the full VertexData slice.

### Shrink CellBuildError by storing detail as Option<Box<str>> (or box the whole error at return sites)

`src/live_dedup/cell_output.rs` — lines 34-39 (CellBuildError), field detail line 38

*memory-layout · impact **low** · risk low · confidence medium · scanner `dedup:cellout`*

CellBuildError is { usize, enum, Option<String> } = 40 bytes and rides in `Result<CellBuildStats, CellBuildError>` returned per cell from the hot build path (knn_clipping/cell_build/run.rs:495, 538), so even the common Ok path pays the larger Result move/spill. Changing detail to Option<Box<str>> cuts the error to 32 bytes for free (errors are cold, construction cost irrelevant); boxing the whole error (`Box<CellBuildError>`) at the run.rs signatures would shrink the Result payload to 8 bytes plus stats and is the bigger win, at the cost of touching ~10 construction sites in compute.rs/failure.rs.

### Pack (VertexKey, side) into one u128 sort key for the edge_check_overflow sort

`src/live_dedup/cell_output.rs` — VertexKey type alias line 9; sort site live_dedup/edge_checks.rs:305

*sorting · impact **low** · risk low · confidence medium · scanner `dedup:cellout`*

VertexKey is [u32;3], so `sort_unstable_by_key(|e| (e.key, e.side))` at edge_checks.rs:305 compares a ([u32;3], u8) tuple lexicographically — up to 4 branchy element compares per comparison. Add an #[inline] helper next to the alias, e.g. `fn key_with_side(k: VertexKey, side: u8) -> u128 { ((k[0] as u128) << 72) | ((k[1] as u128) << 40) | ((k[2] as u128) << 8) | side as u128 }`, and sort by that single u128 (two-instruction compare, branch-poor). Order is identical to the tuple order. Impact limited because the overflow list is normally small.

### Replace compare-and-swap pair in pack_edge with pack-then-rotate select

`src/live_dedup/packed.rs` — lines 27-30, fn pack_edge (feeds hot per-edge loop at src/live_dedup/edge_checks.rs:193)

*branch · impact **low** · risk low · confidence medium · scanner `dedup:packed`*

pack_edge currently selects (min,max) via `if a <= b { (a, b) } else { (b, a) }`, which lowers to two 32-bit cmovs (or a min/max pair) before two zero-extends, a shift, and an or. Because swapping the two 32-bit halves of the packed u64 is exactly a 32-bit rotate, you can pack unconditionally first and then conditionally rotate: `let k = (a as u64) | ((b as u64) << 32); let k = if a > b { k.rotate_left(32) } else { k };` This replaces two dependent cmovs with one cmp + one cmov on a single 64-bit value, shaving roughly one uop and shortening the dependency chain in the per-cell-edge loop that builds EdgeKeys for the dedup hash map. The result is bit-identical (min in low 32, max in high 32), so hashing/unpack behavior is unchanged.

### Fuse cell_starts (u32) and cell_counts (u8) into one per-local record

`src/live_dedup/shard.rs` — lines 35-36 (fields), 48-49 (init), 53-71 (set_cell_start/cell_start/set_cell_count/cell_count)

*memory-layout · impact **low** · risk low · confidence high · scanner `dedup:shard`*

Each cell performs two separately bounds-checked stores into two parallel arrays (set_cell_start in the driver, set_cell_count in emit_cell_output at emit.rs:168), and assembly reads both per local in the concat loop. Replace with a single `Vec<CellSlot { start: u32, count: u8 }>` (8 bytes after padding vs 5 split): one bounds check and one cache line touched per cell on both write and read sides. Writes are sequential so the prefetcher already mitigates much of this, hence low expected impact, but it is a strict reduction in checks and touched lines with no numerical behavior change.

### Elide the release-mode bounds check on the per-push edge_checks index

`src/live_dedup/edge_checks.rs` — push_edge_check, lines 71-86 (indexing at line 78); invariant already debug_asserted at lines 73-76

*bounds-checks · impact **low** · risk low · confidence high · scanner `dedup:shard`*

push_edge_check is called once per edge-to-earlier-local (millions of times at 1M+ points) and does `&mut self.edge_checks[local_idx]` with a release-mode bounds check immediately after a debug_assert stating the invariant (local ids are constructed < num_local_generators). Replace with `get_unchecked_mut` under a SAFETY comment mirroring the debug_assert (the crate already uses unsafe in assemble.rs). Saves one cmp+branch per push; small but free given the invariant is already documented and debug-checked.

## Drivers & orchestration (knn_clipping)

### Schedule bins largest-first (LPT) to cut parallel tail latency

`src/knn_clipping/driver.rs` — lines 58-60 (`maybe_par_into_iter!(0..num_bins)`) and merge loop lines 231-240, fn build_cells_sharded_live_dedup

*parallelism · impact **medium** · risk low · confidence medium · scanner `drv:sphere`*

Bins are the rayon task unit and there are only ~2x threads of them (S2_BIN_COUNT default), so one oversized bin scheduled late bounds the whole phase. Build `order: Vec<usize>` of bin indices sorted descending by `assignment.bin_generators[b].len()`, parallel-iterate that instead of `0..num_bins`, and have each task return `(bin_usize, shard, sub_accum)`; after collect, place shards into a `Vec<Option<ShardState>>` (or sort by bin) so `ShardedCellsData::from_parts` still sees shards in BinId order. With work-stealing over per-item tasks, starting the biggest bins first measurably shrinks the tail on clustered inputs; the sort itself is O(num_bins log num_bins) over ~tens of items.

### Eliminate n SipHash DSU lookups in merge_result_from_pairs

`src/knn_clipping/preprocess.rs` — lines 240-250, merge_result_from_pairs

*hashing · impact **medium** · risk medium · confidence high · scanner `drv:preproc`*

The loop calls dsu.find(i) for every original index 0..n, where dsu is SparseUnionFind backed by std HashMap<u32,(u32,u8)> with the default SipHash hasher; with typical sparse welds (a handful of pairs out of 100k-1M points) that is n SipHash hashes and probes that almost all return 'absent => self'. Instead, run find only on the indices that appear in `pairs` (the only ones that can have rep != self), collect the non-root (idx, rep) entries into a sorted Vec, and do the 0..n pass as a lockstep merge against that sorted list with zero hashing. union_keep_min guarantees the rep is the class minimum, so first-occurrence rep order is unchanged. This is on the main weld path (compute.rs:408), not just the fallback detector.

### Hoist PackedSlotLayout into GridContext instead of rebuilding per cell

`src/knn_clipping/driver.rs` — lines 275-283 in build_and_emit_cell; per-bin copy already exists at lines 107-111; GridContext at lines 23-27

*inlining-codegen · impact **low** · risk low · confidence high · scanner `drv:sphere`*

build_and_emit_cell reconstructs `PackedSlotLayout::new(&grid_ctx.assignment.slot_gen_map, ...)` for every cell to feed `DirectedEligibility::from_layout`, re-deref'ing the Vec into ptr+len and reloading `local_shift`/`local_mask` from `assignment` per cell, even though the bin loop already built an identical `packed_layout` at line 107. Add a `packed_layout: PackedSlotLayout<'a>` field to `GridContext` (it is Copy), set it once per bin, and use `grid_ctx.packed_layout` in `from_layout`. Saves a handful of dependent loads per cell that the compiler cannot hoist across the non-inlined `build_cell_into` call; purely mechanical, zero behavior change.

### Branchless XOR endpoint select + extend() in seed-neighbor loop

`src/knn_clipping/driver.rs` — lines 262-273 in build_and_emit_cell, especially line 266

*branch · impact **low** · risk low · confidence high · scanner `drv:sphere`*

The incoming-check loop does `let neighbor_idx = if a == cell_idx { b } else { a }` (data-dependent branch, ~50/50 by edge-key packing order, so it mispredicts) followed by per-element `push`. Since one endpoint of every edge key is the cell itself, replace with `let neighbor_idx = (a ^ b ^ cell_idx) as usize;` (keep a debug_assert that a==cell_idx || b==cell_idx), and replace the push loop with `live_ctx.seed_neighbors.extend(incoming_checks.iter().map(...))` so the exact size_hint elides the per-push capacity branch. Runs once per incoming edge check per cell (~3/cell average), so the win is small but free.

### Right-size per-shard vertices/vertex_keys reservation (6x generators overshoots owned-vertex count ~3x)

`src/knn_clipping/driver.rs` — lines 70-78, fn build_cells_sharded_live_dedup

*allocation · impact **low** · risk low · confidence low · scanner `drv:sphere`*

Each bin reserves `len*6` for `output.vertices` and `output.vertex_keys`, but with live dedup each Voronoi vertex (degree 3) is owned by exactly one of its three incident cells, so the expected owned-vertex count is ~2 per generator; only `cell_indices` genuinely needs ~6 per generator. Reserving `len.saturating_mul(5)/2` for vertices/vertex_keys cuts the two largest per-shard up-front allocations ~2.4x, reducing allocator pressure and peak VM with the same zero-realloc behavior in the common case. Verify the ~2/cell ratio against emit's dedup-by-owner-bin semantics before changing; reserve is only a hint so correctness risk is nil, the only downside is a possible mid-run realloc if undersized.

### Fuse finiteness validation into the canonicalization pass

`src/knn_clipping/compute.rs` — validate_generator_finiteness (lines 322-341), canonicalize_unit_points (lines 360-381), call sites at lines 23-26 and 106-109

*parallelism · impact **low** · risk low · confidence high · scanner `drv:compute`*

Both entry cores do two separate full O(n) parallel sweeps over the input: a rayon position_first finiteness scan, then a par_chunks_mut canonicalization pass. Fuse them: inside canonicalize_chunk, track the first non-finite index per chunk and fold it into a shared AtomicUsize via fetch_min, returning the InvalidInput error after the single pass. Non-finite points already fail the (0.25..=4.0).contains gate so they stay byte-identical for the error message; on error the owned points are dropped, so mutating good points before detecting a bad one is observationally equivalent. Saves one full memory sweep (~24MB read at 2M points) plus one rayon dispatch per compute call.

### Vectorize canonicalize_unit_points with wide::f64x4 (bit-identical div/sqrt)

`src/knn_clipping/compute.rs` — canonicalize_chunk inner loop, lines 361-370

*simd · impact **low** · risk medium · confidence medium · scanner `drv:compute`*

The scalar loop does per-point f64 length_squared, sqrt, and a 3-component divide (~10ns/point, ~20ms ST at 2M per the in-file comment; it dominates P5 stage 0's +0.5-0.8% ST cost). Process 4 points per iteration with wide::f64x4: transpose to x/y/z lanes, len_sq = x*x+y*y+z*z, f64x4::sqrt, then component vectors divided by the length vector, blending out-of-range/NaN lanes back to the originals via a cmp mask. vdivpd/vsqrtpd are IEEE correctly rounded, so keeping the same div-by-sqrt formulation yields bit-identical canonical points to the scalar path — the P5 'identical bits per generator' contract is preserved. Win is single-threaded only (parallel build already hides this pass); AoS->SoA transpose of 12-byte Vec3s is the main overhead.

### Strength-reduce 3 f64 divides to recip-multiply in canonicalization

`src/knn_clipping/compute.rs` — canonicalize_chunk, line 366 (`let n = v / len_sq.sqrt();`)

*arithmetic · impact **low** · risk medium · confidence high · scanner `drv:compute`*

glam's DVec3 / f64 lowers to three independent f64 divides; replacing with `let inv = len_sq.sqrt().recip(); let n = v * inv;` turns that into one divide plus three multiplies, roughly halving the per-point arithmetic cost of this pass. Caveat: this changes the canonical bits by up to 1 ulp per component. The pipeline stays self-consistent (canonicalization is the single source of truth at entry), but outputs shift versus prior runs, which matters in this tolerance-sensitive crate — treat as the cheap fallback if the bit-identical f64x4 idea above is not taken (the two are mutually exclusive).

### Drop double indexing and bounds checks in weld remap loop

`src/knn_clipping/compute.rs` — remap_cells_to_original_indices, lines 575-582

*bounds-checks · impact **low** · risk low · confidence high · scanner `drv:compute`*

The weld-path loop indexes merge_result.original_to_effective[orig_idx] (bounds check) and then reads eff_to_canonical[eff_idx] twice — once in the if, once for the push. Rewrite as `for (orig_idx, &eff_idx) in merge_result.original_to_effective.iter().enumerate()` and bind `let canon = &mut eff_to_canonical[eff_idx];`, reusing the local for both the conditional store and the weld_map.push. Eliminates one bounds check and one redundant load per original point in this O(n) loop. Only runs when welds occurred, so the win is confined to that path.

### Precompute per-point boundary flag during keying; early-skip interior points in scan_boundary

`src/knn_clipping/preprocess.rs` — lines 51-92 (axis/boundary_neighbor_keys), 126-134 (keyed map), 174-176 (scan_boundary)

*branch · impact **low** · risk low · confidence medium · scanner `drv:preproc`*

scan_boundary calls boundary_neighbor_keys for every point, recomputing all three axis() quantizations (f64 convert, mul, cast, two pad compares each) and building three [Option<u64>;3] arrays plus a triple iter().flatten() loop, even though ~90%+ of points are interior (pad/cell ~ 1/64 per wall) and produce zero neighbor keys. The keyed map pass at lines 126-134 already computes axis() per coordinate but key() discards the near_low/near_high flags. Change key() to also return a 1-bit 'any wall within pad' flag, store it in a per-point bitmap during the keyed pass, and have scan_boundary skip points with the bit clear before touching boundary_neighbor_keys. Minimal fallback: add an early `if !(near_low|near_high on any axis) { return; }` at the top of boundary_neighbor_keys before constructing the options arrays.

### Return Option/empty instead of full identity copy when no welds found

`src/knn_clipping/preprocess.rs` — lines 95-101 (identity_result), 119-121, 210-212

*allocation · impact **low** · risk low · confidence high · scanner `drv:preproc`*

On the no-weld path, merge_close_points returns identity_result, which allocates and fills points.to_vec() (12 B/point) plus (0..n).collect() (8 B/point). The only production caller (compute.rs:421-427) checks num_merged > 0 and drops the entire MergeResult otherwise, so ~20 B/point of allocation and writes are pure waste on clean inputs. Change merge_close_points to return Option<MergeResult> (None = no merges), or leave both Vecs empty in the identity case; the function is not exported from lib.rs, so only this call site and the unit tests need touching. Applies to the large-MergeWithin fallback path only.

### Shrink rep_to_effective from Vec<Option<usize>> to Vec<u32> with MAX sentinel

`src/knn_clipping/preprocess.rs` — lines 235, 245-249, merge_result_from_pairs

*memory-layout · impact **low** · risk low · confidence high · scanner `drv:preproc`*

rep_to_effective is Vec<Option<usize>> (16 bytes/entry) sized n, probed twice per point (is_none then unwrap). Replace with Vec<u32> initialized to u32::MAX as the 'unassigned' sentinel: 4x less scratch memory traffic on an n-sized array, and the Option discriminant test plus unwrap panic path become a plain integer compare. Point counts are already bounded by u32 elsewhere (pair indices are u32). Subsumed if the lockstep-merge restructure above is taken instead.

### Bound cross-wall binary search to the suffix after the current element

`src/knn_clipping/preprocess.rs` — line 181, scan_boundary closure

*sorting · impact **low** · risk low · confidence medium · scanner `drv:preproc`*

Each boundary point does keyed.partition_point over the entire sorted array per neighbor key, but the nkey > key gate (line 178) guarantees every match lies strictly after the current element's position. Track the element's global index (enumerate the chunk range) and search only keyed[pos+1..], halving the average search range and improving locality; for the common z-wall neighbor (nkey = key + 1, immediately after the current run) a short linear/galloping probe from pos before the binary search makes it near O(1). ~9% of points are boundary, so this trims roughly 0.09*n*log(n) cache-missing comparisons.

### Return segments via fixed 2-slot inline buffer instead of Vec in edge_segments_for_neighbor

`src/knn_clipping/edge_reconcile.rs` — edge_segments_for_neighbor, lines 67-103; consumers at lines 214-216

*allocation · impact **low** · risk low · confidence high · scanner `drv:reconcile`*

Each call heap-allocates a Vec<(u32,u32)> (line 80), and collect_merges calls it twice per edge record (lines 214-215). The only production consumer distinguishes counts 0 / 1 / >=2 and reads only segment [0], so any count >=2 routes to the same `continue`. Change the return type to e.g. ([(u32,u32); 2], usize) (or take a &mut Vec scratch from collect_merges), capping collection at 2 matches with an early break. Eliminates two mallocs plus a possible grow per defective edge; the two test call sites in src/live_dedup/assemble.rs (lines 605-607) need a matching signature update.

### Carry vertex key forward and drop the modulo in edge_segments_for_neighbor loop

`src/knn_clipping/edge_reconcile.rs` — edge_segments_for_neighbor, lines 81-101

*arithmetic · impact **low** · risk low · confidence high · scanner `drv:reconcile`*

The loop fetches every vertex key twice — kj of iteration i is ki of iteration i+1 — and computes the wrap index with `(i + 1) % n` (line 83), a runtime integer modulo since n is dynamic. Fetch slice[0]'s key once before the loop, carry `ki = kj` across iterations, and compute the successor index as `let j = i + 1; let j = if j == n { 0 } else { j };` (branchless cmov in practice). Halves the bounds-checked vertex_keys.get + format!-closure setup per edge and removes one div/mod per iteration.

### Pre-filter edges with key_contains(ki, neighbor) before shared_neighbor

`src/knn_clipping/edge_reconcile.rs` — edge_segments_for_neighbor line 98; shared_neighbor lines 30-37

*branch · impact **low** · risk low · confidence high · scanner `drv:reconcile`*

For each cell edge, shared_neighbor first verifies cell_idx is in both keys (invariantly true for boundary vertices of that cell) and then runs an iter().find over the triplet — yet most edges of the cell are not against `neighbor`. Since shared_neighbor(...) == Some(neighbor) requires neighbor to appear in both keys, add a cheap necessary-condition reject first: `if !key_contains(ki, neighbor) || !key_contains(kj, neighbor) { continue; }`. One 3-way compare rejects the common case before the two redundant cell_idx checks and the find loop; semantics are unchanged because the guard is strictly weaker than the equality test.

### Hoist the four endpoint positions in the d00/d01 pairing test

`src/knn_clipping/edge_reconcile.rs` — collect_merges, lines 294-297

*memory-layout · impact **low** · risk low · confidence high · scanner `drv:reconcile`*

The cross-pairing distance test fetches each of a0, a1, b0, b1 via Result-wrapped vertex_pos twice (8 lookups, each with a bounds check and lazy-format error closure) to compute d00 and d01. Bind `let pa0 = vertex_pos(vertices, a0)?;` etc. once (4 lookups) and reuse for both sums. Pure redundancy removal with identical float results; also shrinks the error-path codegen in this block.

### Reuse the `seen` scratch Vec across cells in apply_merges_rebuild

`src/knn_clipping/edge_reconcile.rs` — apply_merges_rebuild, lines 328-337 (line 330)

*allocation · impact **low** · risk low · confidence high · scanner `drv:reconcile`*

The rebuild backend allocates a fresh `seen: Vec<u32>` per cell — O(cells) allocations over the whole diagram (millions when the S2_EDGE_REPAIR_REBUILD differential oracle runs at 2M points). Hoist `let mut seen = Vec::with_capacity(32);` above the loop and `seen.clear()` per iteration. Only matters in the diagnostic/differential-test path (InPlace is the production default), but it is the largest allocation churn in the file and the fix is two lines.

## Planar backend (plane_grid, plane_clipping)

### Specialize scan_cell inner loop for non-start cells (drop per-point query-idx and mode checks)

`src/plane_grid/query.rs` — PlaneShellFrontier::scan_cell, lines 84-112 (per-point checks at 98-107)

*branch · impact **medium** · risk low · confidence high · scanner `plane:grid`*

The hot per-point gather loop does three things per candidate: loads indices[i] to compare against query_idx, re-tests mode == EmitCenterDirected, and conditionally calls allows_center_slot. But the query point can only live in start_cell (point_cells[query_idx] == start_cell), and cell_mode (cube_grid/query/directed.rs:61-62) only returns EmitCenterDirected when cell == start_cell. Branch once on cell == self.start_cell as usize: the start-cell path keeps the current loop; the path for every other cell (the vast majority of points scanned across rings 1..k) becomes a tight loop over xs/ys only — no indices[] load, no compare, no mode test — just dist_sq + push. Also lets the indices slice go entirely untouched in that path.

### Pack (dist_sq bits, slot) into one u64 and sort raw u64s instead of (OrdF32, u32) tuples via total_cmp

`src/plane_grid/query.rs` — scan_cell push line 110, build_pending sort lines 172-175, frontier unpack lines 201-202 (and PlaneGridScratch::pending in src/plane_grid/mod.rs:61)

*sorting · impact **medium** · risk low · confidence high · scanner `plane:grid`*

pending is Vec<(OrdF32, u32)> sorted with sort_unstable_by_key, where OrdF32 compares via f32::total_cmp — that does a sign-flip bit transform of both operands at every comparison. dist_sq = dx*dx+dy*dy is always nonnegative, so its raw f32 bit pattern already orders correctly as an unsigned integer. Change pending to Vec<u64> with key = (dist_sq.to_bits() as u64) << 32 | slot, sort with plain sort_unstable (single branchless u64 compare per element pair, and ipnsort's integer fast path), and reconstruct exactly via f32::from_bits in frontier — which also collapses the current two passes over pending (lines 201-202) into one unpack loop. Tie order becomes deterministic by slot, which is a behavior improvement, not a hazard.

### Replace per-batch Vec clones in frontier cache with a (stage, start, n) range into scratch keys

`src/plane_grid/packed/mod.rs` — lines 247-258 (CachedFrontier) and 342-357 (PlanePackedQuery::frontier)

*allocation · impact **medium** · risk medium · confidence high · scanner `plane:packed`*

Every exact batch does `slots: out.clone(), dists: out_dists.clone()` — two fresh heap allocations plus copies per (query, batch), i.e. millions of small allocs at scale, even though the common case calls frontier() once before advance_frontier(). next_chunk emits from `keys[start..start+n]` and only ever mutates `keys[*pos..]` afterward, so the emitted prefix stays intact and sorted in scratch. Change CachedFrontier::ExactBatch to store `{ batch, stage: PlanePackedStage, start: usize, n: usize }` and have the cached-replay branch re-extract via key_to_idx/key_to_dist_sq from the stage's key vec, eliminating both allocs and copies per batch.

### Use raw dist_sqs instead of dist_sqs_wrapped for wrap-free periodic groups

`src/plane_grid/packed/mod.rs` — trait PackedGeometry lines 26-46 (chunk_dist_sqs seam); kernel call sites scratch.rs lines 392, 506, 530, 595

*simd · impact **medium** · risk medium · confidence medium · scanner `plane:packed`*

PeriodicGrid::chunk_dist_sqs always calls fp::dist_sqs_wrapped (2x abs + 2x splat-sub + 2x min extra SIMD ops per 8-lane chunk vs raw), but when the radius-r box doesn't cross the boundary (cx>=r, cx+r<res, same for y) and res >= 2(r+1), every lane has |d| <= p/2, so minimum-image distance is bit-identical to raw (|d|^2 == d^2 in IEEE). Add a `box_is_wrap_free(&self, cx, cy, radius) -> bool` trait method (default true), compute it once per group in prepare_group, and route the kernel to `chunk.dist_sqs(qx, qy)` for wrap-free groups (e.g. via a hint flag in the chunk_dist_sqs call). Interior cells dominate ((res-2r)^2/res^2), so most periodic groups drop ~6 ops from the innermost distance kernel with zero numerical change.

### Vectorize the ExpandR2 band scan and hoist the per-slot center-range test

`src/plane_grid/packed/scratch.rs` — ensure_expand_r2_band_for, lines 728-755

*simd · impact **medium** · risk low · confidence high · scanner `plane:packed`*

The 5x5-box band scan is fully scalar: per slot it does two indexed loads from points_x()/points_y() (bounds-checked), a scalar dist_sq, a `slot == query_slot` compare, and a 3-way center-range compare (`range.soa_start == center_range.soa_start && range.soa_end == ...`) re-evaluated per slot. Mirror ensure_tail_for's 8-wide pattern: per range, slice rxs/rys once, compute the band mask as `mask_lt(security2) & !mask_lt(security1)` via PlaneChunk8, and hoist the center-range test out of the slot loop — for the center range simply start iteration at `query_slot+1` (the directed filter is exactly slot > query_slot since slots are contiguous), removing both per-slot compares entirely. ~100-point scans (25 cells x density 4) become ~13 chunk evaluations.

### Replace per-cell rem_euclid with incremental wrap in for_each_box_cell

`src/plane_grid/periodic.rs` — lines 333-344, PeriodicGrid::for_each_box_cell (PackedGeometry impl)

*arithmetic · impact **medium** · risk low · confidence high · scanner `plane:periodic`*

The wrapped box enumeration computes `y.rem_euclid(res)` per row and `x.rem_euclid(res)` per cell — `res` is a runtime usize, so each is a hardware idiv (20+ cycles) inside the (2r+1)^2 loop that feeds the packed SIMD stage (called from packed/scratch.rs:303 and :647). Since x,y range over [c-r, c+r] with 0<=c<res and 2r<res (guaranteed by box_radius_distinct gating), wrap once at loop start (`let mut xw = if cx>=r {cx-r} else {cx+res-r}`) and advance with `xw += 1; if xw == res { xw = 0; }` per step; same for yw per row. Also hoist `yw * res` as a row base. Removes all divisions from the enumeration.

### Drop the two rem_euclid divisions per push in collect_ring

`src/plane_grid/periodic.rs` — lines 470-493, PeriodicShellFrontier::collect_ring (push closure, lines 476-477)

*arithmetic · impact **medium** · risk low · confidence high · scanner `plane:periodic`*

Every ring-cell push does `x.rem_euclid(res)` and `y.rem_euclid(res)` — two idivs per cell, and this runs for every query because the shell takeover re-covers all neighbors after the packed stage. With cx,cy in [0,res) and k <= res-1, coordinates are bounded in (-res, 2res), so replace with two conditional corrections: `let mut xw = x; if xw < 0 { xw += res; } else if xw >= res { xw -= res; }` (branchless via cmov in practice). Note the top edge `push(.., x, cy - k)` and bottom edge `push(.., x, cy + k)` also recompute the same xw twice per x — compute xw once per x iteration and reuse for both pushes, and hoist the two row bases `yw_top * res` / `yw_bot * res` out of the x loop.

### Replace 4 modulo ops with conditional subtract in outside_wrapped_box_dist_sq

`src/plane_grid/periodic.rs` — lines 196-199, PeriodicGrid::outside_wrapped_box_dist_sq

*arithmetic · impact **medium** · risk low · confidence high · scanner `plane:periodic`*

The four wall-index computations `(cx + res - k) % res` / `(cx + k + 1) % res` (and the y twins) are four idivs per call; this certificate runs once per query in the packed stage (packed/scratch.rs:348, :679) and once per ring in the takeover (unseen_bound_after, line 499-502). After the early `2*k + 1 >= res` return, k < res/2, so `cx + res - k < 2*res` and `cx + k + 1 < 2*res` — a single `if v >= res { v -= res; }` per index is exact. Pure integer strength reduction, no float behavior change.

### Slice subranges and hoist eligibility-mode branch in scan_cell

`src/plane_grid/periodic.rs` — lines 441-465, PeriodicShellFrontier::scan_cell inner loop (lines 451-464)

*bounds-checks · impact **medium** · risk low · confidence high · scanner `plane:periodic`*

The per-slot loop indexes three parallel arrays (`point_indices[slot]`, `cell_points_x[slot]`, `cell_points_y[slot]`) with a bounds check each, re-tests the loop-invariant `mode == DirectedCellMode::EmitCenterDirected` every iteration, and compares `point_indices[slot] as usize == self.query_idx` with a widening cast per slot. Pre-slice `&point_indices[start..end]`, `&cell_points_x[start..end]`, `&cell_points_y[start..end]` and iterate them zipped (one bounds check total), hoist `let center_directed = mode == EmitCenterDirected;` and `let q = self.query_idx as u32;` before the loop. This is the takeover scan that executes for every query; eliding three checks plus a branch per candidate also gives LLVM a chance to unroll the min_image_dist_sq body.

### Elide per-iteration bounds checks in to_vertex_data via hoisted slices

`src/plane_clipping/builder.rs` — to_vertex_data, lines 324-358

*bounds-checks · impact **medium** · risk low · confidence high · scanner `plane:builder`*

The loop `for i in 0..poly.len` indexes four fixed [_; 64] arrays (poly.us, poly.vs, poly.vertex_planes, poly.edge_planes); since poly.len is a plain pub usize, LLVM cannot prove len <= 64, so each access keeps a bounds check (4/iter). Hoist `let len = poly.len; let us = &poly.us[..len];` etc. before the loop (or zip the slices) to pay one check per array total. Additionally, the guard at line 335 compares plane indices against `self.half_planes.len()` but then indexes `self.neighbor_indices[plane_a/plane_b/edge_plane]` (a different Vec, same length by construction), so those three checks also survive; bind `let nbr = self.neighbor_indices.as_slice();` and guard against `nbr.len()` instead so LLVM elides them. This runs once per cell over every output vertex, in the hot extraction path (driver.rs:461).

### Replace rem_euclid with one conditional subtract in periodic normalization

`src/plane_clipping/compute.rs` — lines 326-330 in compute_plane_periodic_impl (normalization loop)

*arithmetic · impact **medium** · risk low · confidence high · scanner `plane:driver`*

The periodic per-point loop calls n.x.rem_euclid(px) and n.y.rem_euclid(py), each of which lowers to an fmodf libcall plus a sign fixup — two libcalls per input point. The inputs are provably in [0, px]: x >= min.x makes (x - min.x) exactly nonnegative in IEEE, and rounding monotonicity of (x - min.x) * scale against the identical expression rect.width() * scale (line 37) bounds it by px. So `let mut vx = n.x; if vx >= px { vx -= px; }` is bit-exact equivalent to rem_euclid here (px wraps to 0, everything else unchanged), removing both fmodf calls per point while keeping the existing .min(next_below(px)) clamp.

### Slice cell range in pairs_against_cell to elide three bounds checks per point

`src/plane_grid/mod.rs` — PlaneGrid::pairs_against_cell, lines 300-318 (inner loop 311-317)

*bounds-checks · impact **low** · risk low · confidence high · scanner `plane:grid`*

The loop indexes self.cell_points_x[slot], self.cell_points_y[slot], and self.point_indices[slot] for slot in start..end against the full Vecs; LLVM cannot prove end <= len from the cell_offsets load, so all three accesses keep bounds checks. Slice once before the loop (let xs = &self.cell_points_x[start..end]; etc., iterate with zip as collect_pairs_for_cell already does at lines 211-213) to hoist the checks out. Only the near-coincident weld preprocess hits this, so the win is confined to that pass.

### Drop the redundant infinity early-return in outside_box_dist_sq

`src/plane_grid/mod.rs` — outside_box_dist_sq, lines 369-371

*branch · impact **low** · risk low · confidence high · scanner `plane:grid`*

After the four min-chains the function branches on d == f32::INFINITY and returns INFINITY, then otherwise computes d.max(0.0) squared. INFINITY.max(0.0) * itself is already INFINITY, so the branch is pure overhead: deleting it makes the tail branchless (maxss + mulss). Called once per ring advance per query, so the win is small but the change is mechanical and bit-identical.

### Fuse the diagonal-chunk directed-filter mask clears into one AND

`src/plane_grid/packed/scratch.rs` — prepare_group center pass, lines 401-410

*branch · impact **low** · risk low · confidence high · scanner `plane:packed`*

When qi >= i in the center pass, the code clears bits below rel with a branch (`if rel > 0 { mask_bits &= !((1u32 << rel) - 1); }`) and then separately clears the self bit (`mask_bits &= !(1u32 << rel)`). Clearing bits 0..=rel is a single expression: `mask_bits &= !((2u32 << rel) - 1)` (rel < 8 so no overflow). Removes one branch and one AND from the diagonal chunk of every center-cell pass; tiny but it sits in the hottest 8-wide loop and the change is one line.

### Drop the dead slot != query_slot check in ensure_tail_for ring loops

`src/plane_grid/packed/scratch.rs` — ensure_tail_for, lines 604-610 (lane pop loop) and 613-616 (scalar remainder)

*branch · impact **low** · risk low · confidence high · scanner `plane:packed`*

The tail builder only iterates `cell_ranges[1..]` (ring cells; the closure in prepare_group skips ncell == cell), and CSR SoA ranges of distinct cells are disjoint from the center cell that contains query_slot, so `slot != query_slot` is always true and `slot == query_slot` in the remainder loop is always false. Remove both checks (keep a debug_assert), eliminating a compare+branch per emitted tail candidate inside the bit-pop loop. Same reasoning likely applies to the sphere twin.

### Hoist &mut chunk0_keys[qi] out of the lane bit-pop push loops

`src/plane_grid/packed/scratch.rs` — lines 412-418 (center pass), 511-517 and 533-541 (ring pass)

*bounds-checks · impact **low** · risk low · confidence medium · scanner `plane:packed`*

Inside the trailing_zeros pop loops, every push indexes the outer slice again (`chunk0_keys[qi].push(...)`), forcing a per-push re-borrow that can reload the Vec's ptr/len/cap and re-check the slice bound instead of keeping them in registers across the loop. Bind `let kv = &mut chunk0_keys[qi];` right after the `mask_bits == 0` early-out (where to_array is materialized) and push to `kv`. Three sites, mechanical change, in the single hottest selection loop of the planar packed stage.

### Drop redundant len<3 re-checks using the non-failed => len>=3 invariant

`src/plane_clipping/builder.rs` — commit_clip line 198; can_terminate line 287

*branch · impact **low** · risk medium · confidence medium · scanner `plane:builder`*

commit_clip checks `self.current_poly().len < 3` even on the ClipResult::Unchanged path, where len cannot have changed and was >= 3 on entry (seed_rect starts at 4, every prior Changed commit re-checked, and `failed` short-circuits at function entry). Move the check inside the Changed arm only. Likewise can_terminate (called once per clipped neighbor in the driver hot loop, driver.rs:417/427) re-evaluates `self.vertex_count() < 3`, which re-derives current_poly through the use_a branch; under the same invariant it is dead and can be dropped, leaving only the `failed.is_some()` test. Saves a branch plus a dependent load per neighbor in the termination check.

### Select (src, dst) buffer pair once instead of duplicating the clip call in if/else

`src/plane_clipping/builder.rs` — clip_with_slot_result lines 157-171; clip_with_slot_edgecheck lines 237-251

*branch · impact **low** · risk low · confidence high · scanner `plane:builder`*

Both clip entry points duplicate the entire clip_convex / clip_convex_edgecheck call in the two arms of `if self.use_a`. clip_convex is a large non-#[inline] function (all dispatch_clip specializations are #[inline(always)] into it), so each arm is a separate call with duplicated argument setup. Replace with `let (src, dst) = if self.use_a { (&self.poly_a, &mut self.poly_b) } else { (&self.poly_b, &mut self.poly_a) };` followed by a single call; the borrow split on disjoint fields compiles, the branch becomes two pointer cmovs, and code size in this hot per-neighbor function shrinks. use_a flips on every Changed clip so the branch alternates; cmov-ing it removes any mispredict exposure.

### Derive HalfPlane ab2 from c (ab2 = c + c) to skip the duplicated dot product

`src/plane_clipping/builder.rs` — bisector_coefficients lines 135-141 and clip_with_slot_result line 155 (constructor in src/knn_clipping/topo2d/types.rs:26-39)

*arithmetic · impact **low** · risk low · confidence low · scanner `plane:builder`*

bisector_coefficients computes `c = 0.5 * fma_f64(qu, qu, qv*qv)`, then HalfPlane::new_unnormalized recomputes the identical quantity as `ab2 = fma_f64(a, a, b*b)` with a = -qu, b = -qv. Since multiplying by 0.5 is exact in binary FP, `c + c` reproduces ab2 bit-exactly; add a `new_unnormalized_with_ab2(a, b, c, ab2, plane_idx)` variant (eps = CLIP_EPS_INSIDE * (ab2 as f32).sqrt() as f64 unchanged) and call it from clip_with_slot_result, saving one fma+mul per neighbor clip. Confidence is low because if new_unnormalized inlines, InstCombine's fneg-mul canonicalization may already CSE the two dot products.

### Store neighbor_indices as Vec<u32> instead of Vec<usize>

`src/plane_clipping/builder.rs` — field at line 29; writes at lines 104, 190; reads at lines 340-344

*memory-layout · impact **low** · risk low · confidence medium · scanner `plane:builder`*

Every value pushed into neighbor_indices fits u32: seed_rect pushes `(wall_base + side) as usize` (wall_base is already u32) and clip paths push generator indices that are cast back to u32 at every read site in to_vertex_data (lines 340, 341, 344). Switching to Vec<u32> halves the bytes touched by the per-vertex gathers `nbr[plane_a]/nbr[plane_b]/nbr[edge_plane]` in to_vertex_data and removes the widening/narrowing casts; clip_with_slot's signature can keep usize with a single cast at push. Pure layout change, no numeric behavior involved.

### Fold per-vertex is_finite early-returns into one post-loop check in to_vertex_data

`src/plane_clipping/builder.rs` — to_vertex_data line 327

*branch · impact **low** · risk low · confidence medium · scanner `plane:builder`*

The extraction loop branches on `!u.is_finite() || !v.is_finite()` and early-returns per vertex, even though the caller (driver.rs:461) discards the output buffer on Err and the next cell clears it. Accumulate branchlessly instead (`all_finite &= u.is_finite() & v.is_finite();`) and return Err once after the loop, removing a two-condition branch chain from every iteration and unblocking tighter scheduling of the slice-based loop from the bounds-check idea. NaN positions transiently pushed into the buffer are never observed because Err aborts the cell.

### Slice batch/batch_dists to ..batch.n to elide bounds checks in the clip loop

`src/plane_clipping/driver.rs` — lines 391-421 in build_and_emit_cell_plane (ExactBatch inner loop)

*bounds-checks · impact **low** · risk low · confidence high · scanner `plane:driver`*

The hottest planar loop indexes `worker.batch[pos]` and `worker.batch_dists[pos + 1]` with `pos in 0..batch.n`; the compiler cannot prove batch.n <= worker.batch.len(), so every neighbor pays two bounds checks. Right after the ExactBatch arm, bind `let slots = &worker.batch[..batch.n]; let dists = &worker.batch_dists[..batch.n];` (disjoint field borrows alongside the existing `&mut worker.builder` are fine) and iterate `slots.iter().enumerate()`, reading `dists[pos + 1]` via `dists.get(pos + 1)` folded into the existing pos+1 < batch.n branch. One panic-on-slice check per batch replaces two checks per neighbor.

### Drop redundant is_finite checks from per-point validation loops

`src/plane_clipping/compute.rs` — lines 178-190 (compute_plane_impl) and 310-321 (compute_plane_periodic_impl)

*branch · impact **low** · risk low · confidence high · scanner `plane:driver`*

validate_rect already guarantees a finite rect, and PlaneRect::contains (plane_diagram.rs:137) is `x >= min && x <= max && ...`, which is false for NaN and both infinities — so the four is_finite tests per point are redundant in the hot path. Move the finiteness classification inside the `!rect.contains(x, y)` branch (pick non-finite vs outside-domain error there, ideally via a #[cold] helper) to cut ~4 compares+branches per point from both normalization loops.

### Move weld_map out of WeldResult instead of cloning it

`src/plane_clipping/compute.rs` — lines 209-224 (weld match, clone at 220) and 346-361 (clone at 357)

*allocation · impact **low** · risk low · confidence high · scanner `plane:driver`*

On the weld path the code does `Some(weld.weld_map.clone())`, an O(n) Vec<u32> alloc+copy at full point count, only because the match borrows `&weld` to keep `&weld.original_to_effective` alive. Destructure the Some arm into its three fields (or mem::take weld_map from a `Some(mut weld)` binding) so weld_map is moved, not cloned. Rare path (welds are uncommon), but free when it fires.

### Use linear contains on tiny seed_ids and drop the per-cell sort

`src/plane_clipping/driver.rs` — lines 342-357 (seed loop + sort_unstable) and 394-398 (binary_search per neighbor)

*branch · impact **low** · risk low · confidence medium · scanner `plane:driver`*

seed_ids holds the incoming edge-check neighbors, which is empty for almost every cell and a handful at most otherwise, yet every emitted neighbor runs slice::binary_search and every cell pays a sort_unstable. Replace with `!worker.seed_ids.is_empty() && worker.seed_ids.contains(&(neighbor_idx as u32))` and delete the sort at line 357: a short-circuiting linear scan beats binary search at n<=8 and the empty check makes the common case one load+branch. Decision outcome is identical (membership test only), so no tolerance interaction.

## Sorting (sort.rs, generated/sort_nets.rs)

### Make merge_sorted_suffix_back branchless like merge_down_u64

`src/sort.rs` — merge_sorted_suffix_back, lines 503-526

*branch · impact **medium** · risk medium · confidence medium · scanner `sort:utils`*

The back-merge loop re-derives the suffix value each iteration via a 3-arm `match right_idx` and then takes a data-dependent `if lv > rv` branch, while every other merge in this file uses `select_unpredictable`. Replace the match with a stack array `let rs = [r0, r1, r2]` indexed by `right_idx` (raw-ptr load, no bounds check), and make the step branchless: `let take_left = lv > rv; *p.add(out as usize) = select_unpredictable(take_left, lv, rv); left_idx -= take_left as isize; right_idx -= !take_left as isize;`, keeping the `left_idx >= 0` exhaustion case as a cold epilogue loop (left exhaustion requires the whole suffix to be smaller than the whole >=8-element prefix, which is rare). For unsorted suffix values this loop can run up to base+rem iterations with a mispredict per step today; this path fires for every n with n%8 in {1,2} plus n=35.

### Use register-lean sort16_tail_out_12_4 for the exact n==32 case

`src/sort.rs` — sort32_maybe_padded, lines 562-563 (dispatched from sort_small lines 375/412)

*sorting · impact **medium** · risk low · confidence medium · scanner `sort:utils`*

The comment at line 366 says the `_12_4` hybrids exist specifically to reduce live registers/spills, and they are used for fixed 16/24 — but the n==32 path (rem==0) calls `sort32_maybe_padded`, which sorts both halves with the generic `sort16_tail_out(.., tail_len=8)`, the maximal-spill variant. Add a fast branch in sort32_maybe_padded: when `n == 32`, call `sort16_tail_out_12_4(base, base.add(12), 4)` and `sort16_tail_out_12_4(base.add(16), base.add(28), 4)` for the two halves before the merge. n==32 is the largest network size and exact multiples of 8 are common kNN batch shapes, so the spill savings apply where the sort is most expensive.

### Monomorphize tail_len via const generic so sentinel comparators and dispatch matches fold away

`src/generated/sort_nets.rs` — fn signatures at lines 91 (sort16_tail_out), 292 (sort24_tail_out), 554 (sort16_tail_out_12_4), 686 (sort24_tail_out_20_4); load matches 104-160/305-361/563-586/695-718; store matches 223-276/484-537/649-670/841-862

*inlining-codegen · impact **medium** · risk low · confidence medium · scanner `sort:nets`*

tail_len is a runtime parameter, but 6 of 8 call sites in src/sort.rs pass compile-time constants (lines 373-374, 389, 394, 410-411 pass 3 or 4), and the functions carry no #[inline], so the constants never propagate into these large outlined bodies. Change the generator (scripts/gen_sort_nets.py) to emit `unsafe fn sort16_tail_out<const TAIL_LEN: usize>(base, out)` and dispatch the two variable-tail_len sites (sort.rs:422-423, 562-563) through a small match. With TAIL_LEN const, both jump-table matches vanish, and every register initialized to the u64::MAX sentinel makes `cond = val <= r` provably true, so both selects in cswap_reg_ptr/cswap_reg fold to identity and the comparator is dead-code-eliminated; sentinels propagate, so e.g. the tail_len=3 instantiation of the 8-tail 16-net (sort.rs:389) sheds a large fraction of its ~64 comparators — effectively free pruned 11/12/19-element networks, addressing the size-mismatch between padded networks and actual input sizes.

### Replace 17-arm copy-back jump table with two overlapping constant 16-element copies

`src/sort.rs` — copy_back_u64_16_to_32, lines 575-599

*inlining-codegen · impact **low** · risk low · confidence high · scanner `sort:utils`*

copy_back_u64_16_to_32 dispatches through a 17-arm match to get constant-size memcpys, costing a jump table plus ~17 inlined copy bodies of code bloat per call site. Since n >= 16, the same effect is two fixed 128-byte copies: `ptr::copy_nonoverlapping(src, dst, 16); ptr::copy_nonoverlapping(src.add(n - 16), dst.add(n - 16), 16);`. The overlap region in dst is written twice with identical values (each call's src/dst still don't overlap, so copy_nonoverlapping's contract holds), and both reads stay within the n initialized elements of the MaybeUninit scratch. Removes the indirect branch and shrinks sort32_maybe_padded's footprint.

### Fuse the two fallback range checks in sort_small into one branch

`src/sort.rs` — sort_small, lines 346-355

*branch · impact **low** · risk low · confidence high · scanner `sort:utils`*

sort_small currently does `if n < 8 { sort_unstable; return }` followed by `if n > 35 { sort_unstable; return }` — two compare+branch pairs and two duplicated sort_unstable call sites on every invocation of this hot per-query function. Replace with a single check `if n.wrapping_sub(8) > 27 { v.sort_unstable(); return; }` (covers n<8 and n>35 in one unsigned compare). Saves one always-taken-not branch and one cold call site; trivially behavior-preserving for usize n.

### Collapse tail load/store dispatch to one exact jump table with unreachable_unchecked

`src/generated/sort_nets.rs` — lines 104-160 and 223-276 (sort16_tail_out); same pattern at 305-361/484-537 (sort24_tail_out), 563-586/649-670 (sort16_tail_out_12_4), 695-718/841-862 (sort24_tail_out_20_4)

*branch · impact **low** · risk medium · confidence high · scanner `sort:nets`*

Each function does `if tail_len == 8 { ...full load... } else { match tail_len { 0..=7, _ => unreachable!() } }`, then a second `match` with `_ => unreachable!()` for stores. That costs a compare-and-branch before the jump table (despite the doc comment saying tail_len < 8 is the common case), plus a panic-guarded range check on each jump table. Replace with a single `match tail_len { 0..=8 ..., _ => unsafe { core::hint::unreachable_unchecked() } }` for both load and store: one exact jump table, no range guard, no extra ==8 branch. The tail_len <= 8 contract is already a documented safety precondition with a debug_assert at line 92, so the unchecked hint adds no new caller obligation. Subsumed by the const-generic idea if that lands, but trivially applicable on its own.

## Cross-cutting (build flags, allocation sweep)

### Add [profile.release] with codegen-units = 1 and lto = "thin"

`Cargo.toml` — profile sections, lines 101-117 (no [profile.release] exists; add one above [profile.checked])

*build-flags · impact **medium** · risk low · confidence medium · scanner `x:buildflags`*

Cargo.toml defines [profile.checked] and [profile.profiled] but no [profile.release], so every release build (tests, bench bins) uses the defaults: 16 codegen units and no LTO. That splits the crate's own hot modules (knn_clipping, live_dedup, cube_grid) across CGUs, blocking intra-crate inlining of non-#[inline] functions, and skips cross-crate inlining into wide/glam/rayon beyond #[inline] hints. Adding `[profile.release]\ncodegen-units = 1\nlto = "thin"` is a standard low-single-digit-percent win for SIMD-heavy code and also makes release match [profile.profiled] (which already pins codegen-units = 1 at line 116), so profiles reflect shipping codegen. Note: this perturbs scripts/bench_build.sh cross-commit chains for commits before the change.

### Check in .cargo/config.toml with -C target-cpu=native for local dev

`Cargo.toml` — crate root (.cargo/config.toml is absent; wide = "0.7" at line 50, fma feature comment at line 30)

*build-flags · impact **medium** · risk medium · confidence high · scanner `x:buildflags`*

There is no .cargo/config.toml, so the documented commands (`cargo run --release --features tools --bin bench_voronoi`, `cargo test --release`) compile at baseline x86-64: wide's f32x8 lowers to paired 128-bit SSE2 halves instead of single AVX2 256-bit ops, and the fma feature's mul_add becomes a slow libm call (docs/optimization-ideas.md:177 documents a 25-35% LOSS) instead of vfmadd. docs/performance.md:15 measures target-cpu=native alone at ~6% on the reference Ryzen 3600; scripts/bench_build.sh:113 already injects it but ad-hoc runs do not. Add `.cargo/config.toml` with `[build]\nrustflags = ["-C", "target-cpu=native"]` (or at minimum copy the flag into the CLAUDE.md/README bench commands). Risk is portability/reproducibility of locally built binaries and a one-time baseline shift, not correctness — AVX lowering does not change f32 results unless fma is enabled.

### Recycle cached-frontier slots buffer instead of out.clone() per kNN batch (sphere)

`src/cube_grid/query/stream.rs` — lines 150-153 and 177-180 (DirectedNeighborStream::frontier), plus the matching site at src/cube_grid/packed_knn/mod.rs:281-284 (PackedQuery::frontier)

*allocation · impact **medium** · risk low · confidence high · scanner `x:allocs`*

Every exact batch served to a cell does `slots: out.clone()` to populate `CachedFrontier::ExactBatch` — and it happens at TWO layers (PackedQuery caches one clone, the stream wrapping it caches another), so each chunk0/tail/expandR2 batch costs 1-2 fresh Vec<u32> mallocs per cell. Change: keep a `spare: Vec<u32>` field on the stream/query (or hoist `cached_slots: Vec<u32>` out of the enum into the struct); in `advance_frontier`, when taking the old `ExactBatch`, mem::take its Vec into `spare`; when caching a new batch, `spare.clear(); spare.extend_from_slice(out)` instead of clone. Cuts per-batch heap traffic in the hottest loop to near zero with identical contents.

### Recycle slots/dists buffers in PlanePackedQuery cached frontier (plane)

`src/plane_grid/packed/mod.rs` — lines 352-356, PlanePackedQuery::frontier (CachedFrontier::ExactBatch construction)

*allocation · impact **medium** · risk low · confidence high · scanner `x:allocs`*

Same pattern as the sphere side: each exact batch caches `slots: out.clone(), dists: out_dists.clone()` — two Vec allocations per batch per query in the planar kNN path. Apply the identical recycle fix: reclaim the Vecs from the previous `CachedFrontier::ExactBatch` in `advance_frontier` (mem::take into spare fields), then clear + extend_from_slice when re-caching. Mechanical, behavior-identical.

### Set panic = "abort" in the release profile for bench binaries

`Cargo.toml` — new [profile.release] section (see idea 1); bin targets at lines 71-99

*build-flags · impact **low** · risk low · confidence medium · scanner `x:buildflags`*

Release currently uses the default panic = "unwind", so every call that can panic (bounds checks, asserts, Vec growth) carries unwind landing pads and inhibits some inlining in the bench bins (bench_voronoi, bench_plane, microbench_*). Adding `panic = "abort"` to [profile.release] shrinks code and removes cleanup paths; cargo automatically forces unwind for test/bench harness targets, so `cargo test --release` still builds, and a grep shows the suite has zero #[should_panic] tests. Impact is typically small (~0-2%) since the hot loops are panic-light, but it is free to try alongside idea 1.

### Move ShellFrontier's current/next/pending Vecs into CubeMapGridScratch

`src/cube_grid/query/shells.rs` — lines 72-74 in ShellFrontier::new (fields declared at lines 33-37)

*allocation · impact **low** · risk low · confidence high · scanner `x:allocs`*

`ShellFrontier::new` allocates `current: vec![start_cell]` (a 1-element heap Vec) plus lazily-growing `next`/`pending` Vecs, and `DirectedNeighborStream::new` (src/cube_grid/query/stream.rs:86-87, called per cell from src/knn_clipping/cell_build/run.rs:559) constructs the takeover frontier EAGERLY — so this malloc happens once per cell even when takeover is never used. The plane side already fixed exactly this: `PlaneGridScratch` (src/plane_grid/mod.rs:56-62) holds a reusable `pending` Vec 'to avoid a heap allocation per cell in the million-cell driver loop'. Mirror it: add `current`/`next`/`pending` to `CubeMapGridScratch` (src/cube_grid/mod.rs:248) and clear+push in `new`; use mem::swap for the current/next layer flip to satisfy the borrow checker.

### Switch SparseUnionFind map from SipHash HashMap to FxHashMap

`src/knn_clipping/union_find.rs` — line 80, SparseUnionFind::nodes

*hashing · impact **low** · risk low · confidence high · scanner `x:allocs`*

`SparseUnionFind` uses `std::collections::HashMap<u32, (u32, u8)>` with the default SipHash hasher, while the crate already depends on rustc_hash (`FxHashMap` in src/live_dedup/assemble.rs:5,30). Every find/union does multiple hashed lookups; FxHash on a u32 key is several times cheaper per op. The doc comment at lines 75-77 explicitly states lookups never iterate the map, so hasher/order cannot leak into results (and `roots_snapshot` collects-then-sorts). One-line type swap. Impact is bounded because edge_reconcile maps are small (defect counts are low), hence low impact despite high safety confidence.

### Avoid full O(N) point copy in the no-weld preprocess fast path

`src/knn_clipping/preprocess.rs` — lines 95-101, identity_result (used by merge_close_points early-outs)

*allocation · impact **low** · risk medium · confidence medium · scanner `x:allocs`*

When no near-coincident points exist (the common case), `identity_result` still does `points.to_vec()` (12 B/point) plus `(0..n).collect()` for an identity index map (8 B/point) — roughly 20 MB of allocation+memcpy at 1M points before any real work starts. Change `MergeResult` to carry `effective_points: Cow<'_, [Vec3]>` (or an `is_identity: bool` with empty vecs) and have `original_to_effective` lookups short-circuit to identity when no merges occurred. Saves a one-time O(N) copy; ranked lower because it is setup-phase, not per-cell, and touches the MergeResult consumers (medium risk of ripple).

