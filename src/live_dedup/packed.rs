//! Packed identifiers and helper functions for live dedup.

use super::types::EdgeKey;

pub(super) const INVALID_INDEX: u32 = u32::MAX;

#[inline]
fn pack_bc(b: u32, c: u32) -> u64 {
    (b as u64) | ((c as u64) << 32)
}

#[inline]
pub(crate) fn pack_edge(a: u32, b: u32) -> EdgeKey {
    let (min, max) = if a <= b { (a, b) } else { (b, a) };
    EdgeKey::from(pack_bc(min, max))
}
