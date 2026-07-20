use super::super::CubeMapGridScratch;

impl CubeMapGridScratch {
    pub(crate) fn new(num_cells: usize) -> Self {
        Self {
            visited_stamp: vec![0; num_cells],
            stamp: 0,
            shell_schedule_start: u32::MAX,
            shell_schedule_cells: Vec::new(),
            shell_layer_offsets: Vec::new(),
            pending: Vec::new(),
        }
    }

    pub(super) fn reset_shell_schedule(&mut self, start_cell: u32) {
        self.stamp = self.stamp.wrapping_add(1).max(1);
        if self.stamp == u32::MAX {
            self.visited_stamp.fill(0);
            self.stamp = 1;
        }
        self.shell_schedule_start = start_cell;
        self.shell_schedule_cells.clear();
        self.shell_schedule_cells.push(start_cell);
        self.shell_layer_offsets.clear();
        self.shell_layer_offsets.extend([0, 1]);
        self.mark_visited(start_cell);
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
