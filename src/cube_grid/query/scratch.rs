use super::super::CubeMapGridScratch;

impl CubeMapGridScratch {
    pub fn new(num_cells: usize) -> Self {
        Self {
            visited_stamp: vec![0; num_cells],
            stamp: 0,
            exhausted: false,
        }
    }

    #[inline]
    pub(super) fn mark_visited(&mut self, cell: u32) -> bool {
        let idx = cell as usize;
        if self.visited_stamp[idx] == self.stamp {
            return false;
        }
        self.visited_stamp[idx] = self.stamp;
        true
    }
}
