# Refactor Plan (Step-by-Step)

Goals: improve readability, reduce boilerplate, and make future changes safer without altering behavior.

Guardrails
- Keep each step behavior-neutral unless explicitly stated.
- Keep diffs small and reviewable.
- Run `cargo test --release` after each phase (or at least at phase boundaries).

Phase 0: Baseline + Prep
1. Capture current state (tests pass, rough clippy status).
2. Decide any required naming conventions for new modules/types.

Phase 1: Split `live_dedup.rs` (highest ROI)
1. Create folder `src/knn_clipping/live_dedup/` with `mod.rs`.
2. Move small, dependency-light items first:
   - Constants + pack/unpack helpers -> `packed.rs`
   - Bin layout + generator assignment -> `binning.rs`
3. Move shared POD structs and small helpers -> `types.rs`.
4. Move shard-local structs/impls -> `shard.rs`.
5. Move edge-check bookkeeping/matching -> `edge_checks.rs`.
6. Move final assembly + output shaping -> `assemble.rs`.
7. Keep `mod.rs` as the public facade, re-exporting only what `knn_clipping/mod.rs` needs.
8. Verify tests.

Phase 2: Simplify Cell Build Flow (still behavior-neutral)
1. Replace the `process_cell` closure with a named function.
2. Introduce a `CellContext` struct for shared scratch buffers.
3. Reduce argument count for hot-path helpers (e.g., edge collection).
4. Verify tests.

Phase 3: Type Safety + Intent (low-risk improvements)
1. Add small newtypes where confusion is easy: `BinId(u32)`, `LocalId(u32)`, `EdgeKey(u64)`.
2. Replace raw `u32` in new code, keep conversions localized.
3. Verify tests.

Phase 4: Split `knn_clipping/mod.rs`
1. Extract preprocessing (`merge_close_points`) -> `preprocess.rs`.
2. Extract diagnostics/debug logging -> `debug.rs`.
3. Extract edge repair -> `edge_repair.rs`.
4. Extract main compute path -> `compute.rs`.
5. Keep `mod.rs` as thin orchestration + re-exports.
6. Verify tests.

Phase 5: Split `cube_grid/mod.rs`
1. Extract grid build logic -> `build.rs`.
2. Extract query logic -> `query.rs`.
3. Keep `packed_knn` under `cube_grid/` as-is (or split later if needed).
4. Verify tests.

Phase 6: Config + Cleanup
1. Replace `S2V_*` env reads with `VoronoiConfig` fields (keep env as CLI-only).
2. Revisit diagnostics checks to reduce duplicated logic.
3. Chase remaining clippy warnings that indicate real cleanup wins.

Definition of Done
- Tests pass in release mode.
- `live_dedup` and other big modules are navigable by concern.
- Minimal or zero dead-code warnings for core library code paths.

