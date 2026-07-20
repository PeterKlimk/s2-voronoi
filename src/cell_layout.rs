//! Internal read-only view of live cell spans and their backing index buffer.
//!
//! Reconciliation may shrink a cell's live span without compacting the shared
//! buffer, so consumers must follow each [`VoronoiCell`] record rather than
//! iterate the buffer directly. This module owns that representation rule.

use crate::diagram::VoronoiCell;

/// Why a checked cell-span lookup could not produce a live slice.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum CellSpanError {
    /// The requested cell is not present in the layout.
    CellOutOfBounds { cell: usize, cell_count: usize },
    /// The cell's declared live-span end cannot be represented as `usize`.
    SpanEndOverflow {
        cell: usize,
        start: usize,
        count: usize,
    },
    /// The cell's declared live span exceeds the backing index buffer.
    SpanOutOfBounds {
        cell: usize,
        start: usize,
        end: usize,
        index_count: usize,
    },
}

/// Borrowed read-only pairing of cell records with their shared index buffer.
#[derive(Clone, Copy)]
pub(crate) struct LiveCellLayout<'cells, 'indices> {
    cells: &'cells [VoronoiCell],
    indices: &'indices [u32],
}

impl<'cells, 'indices> LiveCellLayout<'cells, 'indices> {
    /// Pair cell records with the index buffer that backs their live spans.
    #[inline]
    pub(crate) const fn new(cells: &'cells [VoronoiCell], indices: &'indices [u32]) -> Self {
        Self { cells, indices }
    }

    /// Number of cell records paired with the backing index buffer.
    #[inline]
    pub(crate) const fn cell_count(self) -> usize {
        self.cells.len()
    }

    /// Length of the backing index buffer, including any stale tail slots.
    #[inline]
    pub(crate) const fn index_count(self) -> usize {
        self.indices.len()
    }

    /// Assert the structural invariants required by every live-span reader.
    ///
    /// This is compiled only when debug assertions are enabled so production
    /// traversal and code generation remain unchanged.
    #[cfg(debug_assertions)]
    pub(crate) fn debug_assert_valid(self) {
        assert!(
            u32::try_from(self.cells.len()).is_ok(),
            "live cell layout has {} cells, exceeding u32 cell-id capacity",
            self.cells.len()
        );
        assert!(
            u32::try_from(self.indices.len()).is_ok(),
            "live cell layout has {} indices, exceeding u32 offset capacity",
            self.indices.len()
        );

        for (cell, record) in self.cells.iter().enumerate() {
            let start = record.vertex_start();
            let count = record.vertex_count();
            let end = start.checked_add(count).unwrap_or_else(|| {
                panic!("live cell layout cell {cell} span start {start} + count {count} overflows usize")
            });
            assert!(
                end <= self.indices.len(),
                "live cell layout cell {cell} span [{start}..{end}) exceeds index buffer len {}",
                self.indices.len()
            );
        }
    }

    /// Return a live span when both the cell id and declared buffer range are valid.
    #[inline]
    pub(crate) fn checked_span(self, cell: usize) -> Result<&'indices [u32], CellSpanError> {
        if cell >= self.cells.len() {
            return Err(CellSpanError::CellOutOfBounds {
                cell,
                cell_count: self.cells.len(),
            });
        }
        let record = &self.cells[cell];
        let start = record.vertex_start();
        let count = record.vertex_count();
        let end = start
            .checked_add(count)
            .ok_or(CellSpanError::SpanEndOverflow { cell, start, count })?;
        if end > self.indices.len() {
            return Err(CellSpanError::SpanOutOfBounds {
                cell,
                start,
                end,
                index_count: self.indices.len(),
            });
        }
        Ok(&self.indices[start..end])
    }

    /// Return the live span for a trusted in-bounds cell id.
    #[inline]
    pub(crate) fn span(self, cell: usize) -> &'indices [u32] {
        let record = &self.cells[cell];
        let start = record.vertex_start();
        let end = start + record.vertex_count();
        &self.indices[start..end]
    }

    /// Return the live span for a record already obtained from this layout.
    ///
    /// This skips a second cell-id lookup but retains normal slice bounds
    /// checking, matching direct internal traversal of a valid cell record.
    #[inline]
    pub(crate) fn span_for(self, cell: &VoronoiCell) -> &'indices [u32] {
        let start = cell.vertex_start();
        let end = start + cell.vertex_count();
        &self.indices[start..end]
    }
}

#[cfg(test)]
mod tests {
    use super::{CellSpanError, LiveCellLayout};
    use crate::diagram::VoronoiCell;

    #[test]
    fn follows_live_spans_instead_of_stale_tail_slots() {
        let cells = [VoronoiCell::new(0, 2), VoronoiCell::new(4, 3)];
        let indices = [10, 11, 99, 98, 20, 21, 22];
        let layout = LiveCellLayout::new(&cells, &indices);

        #[cfg(debug_assertions)]
        layout.debug_assert_valid();
        assert_eq!(layout.cell_count(), 2);
        assert_eq!(layout.index_count(), 7);
        assert_eq!(layout.span(0), &[10, 11]);
        assert_eq!(layout.span(1), &[20, 21, 22]);
        assert_eq!(layout.span_for(&cells[0]), &[10, 11]);
        assert_eq!(layout.checked_span(1), Ok(&[20, 21, 22][..]));
    }

    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "cell 0 span [2..4) exceeds index buffer len 3")]
    fn debug_validation_rejects_out_of_bounds_live_span() {
        let cells = [VoronoiCell::new(2, 2)];
        let indices = [10, 11, 12];

        LiveCellLayout::new(&cells, &indices).debug_assert_valid();
    }

    #[test]
    fn checked_lookup_distinguishes_cell_and_span_errors() {
        let cells = [VoronoiCell::new(2, 2)];
        let indices = [10, 11, 12];
        let layout = LiveCellLayout::new(&cells, &indices);

        assert_eq!(
            layout.checked_span(1),
            Err(CellSpanError::CellOutOfBounds {
                cell: 1,
                cell_count: 1,
            })
        );
        assert_eq!(
            layout.checked_span(0),
            Err(CellSpanError::SpanOutOfBounds {
                cell: 0,
                start: 2,
                end: 4,
                index_count: 3,
            })
        );
    }

    #[cfg(target_pointer_width = "32")]
    #[test]
    fn checked_lookup_rejects_span_end_overflow() {
        let cells = [VoronoiCell::new(u32::MAX, 1)];
        let indices = [];
        let layout = LiveCellLayout::new(&cells, &indices);

        assert_eq!(
            layout.checked_span(0),
            Err(CellSpanError::SpanEndOverflow {
                cell: 0,
                start: u32::MAX as usize,
                count: 1,
            })
        );
    }
}
