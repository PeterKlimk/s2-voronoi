# RFC 0002: Packed kNN Chunked Emission Until Termination (r=1)

## Summary

Replace the current "packed kNN to reach K (then fallback)" flow with
"packed kNN emits safe candidates in chunks until clipping terminates".
This keeps the dominant fast path as "3x3 packed -> clip -> terminate" while
allowing additional packed chunks to be requested when termination is not
proven. Resumable kNN and full scan remain as fallback paths for now.

This RFC targets the r=1 (3x3) case only. Ring expansion is deferred.

## Goals

- Preserve the fast initial 3x3 path for typical inputs.
- Avoid the "reach K then give up" behavior that fails at high density.
- Keep termination logic sound with correct unseen bounds.
- Keep grouped packed queries (per center cell) and reuse work across queries.

## Non-Goals

- Implement ring expansion (r>1). This is a later phase.
- Redesign the large-candidate path (SlowPath). We keep existing behavior.

## Background (Current)

- Packed kNN runs once per group (3x3) with a fixed `k`, returning a sorted list.
- `process_cell` clips against the packed neighbors, then falls back to resumable
  kNN stages and possibly full scan.
- This fails when termination requires more neighbors than `k`, especially at
  higher point densities.

## Proposed Change (r=1 Only)

### Safety rule

Define `security` as the outside-of-3x3 dot bound (current ring-2 cap).
A candidate is **safe** iff `dot >= security`. Unsafe candidates are never
emitted in r=1 and are deferred for future ring expansion work.

### Candidate sets

For each query:

1. **Center safe**: safe candidates in the center cell.
2. **Ring safe-hi**: ring candidates with `dot >= worst_center_safe`.
3. **Ring safe-lo**: ring candidates with `security <= dot < worst_center_safe`.
4. **Unsafe**: `dot < security` (not emitted).

Notes:
- If there are no center-safe candidates, skip ring safe-hi and start with
  ring safe-lo directly.
- "Worst center safe" is the minimum dot among center-safe candidates.

### Chunked emission

Emit candidates in chunks of size `k0` for the first chunk, then `k1` for
subsequent chunks (`k0 > k1` to avoid very small early chunks).

Chunk order:
1. Chunk0 = center safe + ring safe-hi (if center safe exists).
2. Ring safe-lo (only if still not terminated).

Each chunk is sorted by dot (desc). Chunks are selected by "top-M" selection
using `select_nth_unstable` per query to avoid a full sort.

### Termination bounds

Let `last_dot` be the dot of the last candidate in the current chunk
(smallest dot in that sorted chunk).

- While the chunk is being consumed:
  - Termination checks may use `next_in_chunk` for sorted chunks.
  - It is acceptable to skip the check for the last element.
- After the chunk:
  - If any safe candidates remain, `unseen_bound = last_dot`.
  - If no safe candidates remain, `unseen_bound = security`.

Important: We must never use `security` as the bound while safe candidates
remain un-emitted.

### Fallbacks

If all safe packed candidates are consumed and termination is still not proven:
- fall back to resumable kNN (existing path),
- then full scan (existing path), if needed.

## API Sketch

Introduce a group-level packed interface that can emit chunks per query.
Only the essential shape is shown here.

```
pub struct PackedKnnCellGroup<'a> {
    // inputs, thresholds, candidate slabs
}

pub enum PackedStage { Chunk0, Tail }

impl<'a> PackedKnnCellGroup<'a> {
    pub fn prepare_directed(...) -> PackedKnnCellStatus;
    pub fn ensure_tail(&mut self, ...);
    pub fn next_chunk(
        &mut self,
        qi: usize,
        stage: PackedStage,
        k: usize,
        out: &mut [u32],
    ) -> Option<(usize, bool)>;
    pub fn security(&self, qi: usize) -> f32;
}
```

The `next_chunk` return is:
- `usize`: number of neighbors written to `out`,
- `bool`: whether more safe candidates remain in that stage.

## Cell Builder Loop (Sketch)

```
while let Some((n, has_more)) = group.next_chunk(qi, Chunk0, k_cur, tmp) {
    for pos in 0..n {
        let neighbor_idx = point_indices[tmp[pos] as usize] as usize;
        clip(neighbor_idx);
        if pos + 1 < n {
            let next_dot = dot(next neighbor);
            if can_terminate(next_dot) { return; }
        }
    }
    if can_terminate(if has_more { last_dot } else { security }) { return; }
    k_cur = k1;
}

if !terminated {
    group.ensure_tail(...);
    // repeat with PackedStage::Tail
}
```

## Performance Notes

- The dominant path remains "3x3 -> chunk0 -> terminate".
- Chunked selection avoids full sort for long tails.
- The first chunk is larger (`k0`) to avoid tiny early chunks that rarely
  terminate cells.

## Future Work

- Ring expansion (r>1) with safe/unsafe promotion.
- Better handling of large-candidate (SlowPath) queries.
- Per-group adaptive chunk sizes and fast-path tuning.
