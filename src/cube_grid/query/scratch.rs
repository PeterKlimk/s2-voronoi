use super::super::CubeMapGridScratch;

impl CubeMapGridScratch {
    pub fn new(_num_cells: usize) -> Self {
        Self {
            visited_stamp: Vec::new(),
            stamp: 0,
            current: Vec::new(),
            next: Vec::new(),
            pending: Vec::new(),
        }
    }

    #[inline]
    pub(super) fn begin_visit(&mut self, num_cells: usize) {
        if self.visited_stamp.len() != num_cells {
            self.visited_stamp.clear();
            self.visited_stamp.resize(num_cells, 0);
            self.stamp = 1;
            return;
        }
        self.stamp = self.stamp.wrapping_add(1).max(1);
        if self.stamp == u32::MAX {
            self.visited_stamp.fill(0);
            self.stamp = 1;
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
