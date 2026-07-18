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
        let end = start + record.vertex_count();
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

        assert_eq!(layout.span_for(&cells[0]), &[10, 11]);
        assert_eq!(layout.checked_span(1), Ok(&[20, 21, 22][..]));
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
}
