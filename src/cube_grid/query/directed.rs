//! Directed-eligibility rules for the stitching order.
//!
//! Grid-agnostic: operates on the CSR `cell_offsets` and the packed
//! (bin, local) slot layout only, so the cube-map and planar grids share it.

use crate::packed_layout::PackedSlotLayout;

#[derive(Debug, Clone, Copy)]
pub(crate) struct DirectedEligibility<'a> {
    query_bin: u8,
    query_local: u32,
    layout: PackedSlotLayout<'a>,
}

impl<'a> DirectedEligibility<'a> {
    #[inline]
    pub(crate) fn new(
        query_bin: u8,
        query_local: u32,
        slot_gen_map: &'a [u32],
        local_shift: u32,
        local_mask: u32,
    ) -> Self {
        Self::from_layout(
            query_bin,
            query_local,
            PackedSlotLayout::new(slot_gen_map, local_shift, local_mask),
        )
    }

    #[inline]
    pub(crate) fn from_layout(
        query_bin: u8,
        query_local: u32,
        layout: PackedSlotLayout<'a>,
    ) -> Self {
        Self {
            query_bin,
            query_local,
            layout,
        }
    }

    #[inline]
    pub(crate) fn cell_mode(
        self,
        cell_offsets: &[u32],
        start_cell: u32,
        cell: usize,
    ) -> DirectedCellMode {
        let cell_u32 = cell as u32;
        if cell_u32 > start_cell {
            return DirectedCellMode::EmitAll;
        }

        let Some(bin_b) = self.layout.cell_bin(cell_offsets, cell) else {
            return DirectedCellMode::TransitOnly;
        };
        if bin_b != self.query_bin {
            return DirectedCellMode::EmitAll;
        }

        if cell_u32 < start_cell {
            DirectedCellMode::TransitOnly
        } else if cell_u32 == start_cell {
            DirectedCellMode::EmitCenterDirected
        } else {
            DirectedCellMode::EmitAll
        }
    }

    #[inline]
    pub(crate) fn allows_center_slot(self, slot: u32) -> bool {
        let (bin_b, local_b) = self.layout.bin_local(slot);
        bin_b != self.query_bin || local_b >= self.query_local
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum DirectedCellMode {
    TransitOnly,
    EmitAll,
    EmitCenterDirected,
}
