//! Packed identifiers and helper functions for live dedup.

pub(super) const DEFERRED: u64 = u64::MAX;
pub(super) const INVALID_INDEX: u32 = u32::MAX;

#[inline]
pub(super) fn pack_ref(bin: u32, local: u32) -> u64 {
    ((bin as u64) << 32) | (local as u64)
}

#[inline]
pub(super) fn unpack_ref(packed: u64) -> (u32, u32) {
    ((packed >> 32) as u32, (packed & 0xFFFF_FFFF) as u32)
}

#[inline]
fn pack_bc(b: u32, c: u32) -> u64 {
    (b as u64) | ((c as u64) << 32)
}

#[inline]
pub(super) fn pack_edge(a: u32, b: u32) -> u64 {
    let (min, max) = if a <= b { (a, b) } else { (b, a) };
    pack_bc(min, max)
}
