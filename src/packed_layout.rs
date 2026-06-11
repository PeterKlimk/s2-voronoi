#[derive(Debug, Clone, Copy)]
pub(crate) struct PackedSlotLayout<'a> {
    slot_gen_map: &'a [u32],
    local_shift: u32,
    local_mask: u32,
}

impl<'a> PackedSlotLayout<'a> {
    #[inline]
    pub(crate) fn new(slot_gen_map: &'a [u32], local_shift: u32, local_mask: u32) -> Self {
        Self {
            slot_gen_map,
            local_shift,
            local_mask,
        }
    }

    #[inline]
    pub(crate) fn slot_gen_map(self) -> &'a [u32] {
        self.slot_gen_map
    }

    #[inline]
    pub(crate) fn local_shift(self) -> u32 {
        self.local_shift
    }

    #[inline]
    pub(crate) fn local_mask(self) -> u32 {
        self.local_mask
    }

    #[inline]
    pub(crate) fn bin_local(self, slot: u32) -> (u8, u32) {
        let packed = self.slot_gen_map[slot as usize];
        let bin = (packed >> self.local_shift) as u8;
        let local = packed & self.local_mask;
        (bin, local)
    }

    /// Bin of a grid cell via its first slot (`None` for empty cells).
    ///
    /// Takes the CSR `cell_offsets` directly so any grid layout (cube-map or
    /// planar) with bin-contiguous cells can use it.
    #[inline]
    pub(crate) fn cell_bin(self, cell_offsets: &[u32], cell: usize) -> Option<u8> {
        let start = cell_offsets[cell] as usize;
        let end = cell_offsets[cell + 1] as usize;
        if start >= end {
            return None;
        }
        Some(self.bin_local(start as u32).0)
    }
}
