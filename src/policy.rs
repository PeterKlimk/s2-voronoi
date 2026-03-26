//! Internal policy and heuristic configuration for neighbor search.
//!
//! This module centralizes the crate's behavior-preserving tuning knobs so the
//! query/stream code can depend on named policy decisions instead of scattered
//! constants.

pub(crate) const DEFAULT_PACKED_CHUNK0_SIZE: usize = 16;
pub(crate) const DEFAULT_PACKED_CHUNK_SIZE: usize = 8;
pub(crate) const DEFAULT_TERMINATION_CHECK_START: usize = 8;
pub(crate) const DEFAULT_TERMINATION_CHECK_STEP: usize = 1;

pub(crate) const PACKED_HI_BUDGET: usize = 32;
pub(crate) const PACKED_COUNT_MODEL_IGNORE_DIRECTED_CENTER: bool = false;
pub(crate) const PACKED_COUNT_MODEL_INCLUDE_SAME_BIN_EARLIER: bool = false;
pub(crate) const PACKED_MAX_EXPAND_R2_CANDIDATES_PER_QUERY: usize = 8_192;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct PackedNeighborPolicy {
    chunk0_size: usize,
    chunk_size: usize,
    expand_r2_enabled: bool,
}

impl PackedNeighborPolicy {
    #[inline]
    pub(crate) fn for_point_count(num_points: usize, expand_r2_enabled: bool) -> Self {
        let max_neighbors = num_points.saturating_sub(1);
        Self {
            chunk0_size: DEFAULT_PACKED_CHUNK0_SIZE.min(max_neighbors),
            chunk_size: DEFAULT_PACKED_CHUNK_SIZE.min(max_neighbors),
            expand_r2_enabled,
        }
    }

    #[inline]
    pub(crate) fn enabled(self) -> bool {
        self.chunk0_size > 0
    }

    #[inline]
    pub(crate) fn chunk0_size(self) -> usize {
        self.chunk0_size
    }

    #[inline]
    pub(crate) fn chunk_size(self) -> usize {
        self.chunk_size
    }

    #[inline]
    pub(crate) fn expand_r2_enabled(self) -> bool {
        self.expand_r2_enabled
    }

    #[inline]
    pub(crate) fn scratch_chunk_capacity(self) -> usize {
        self.chunk0_size.max(self.chunk_size)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct TerminationPolicy {
    check_start: usize,
    check_step: usize,
    max_k_cap: Option<usize>,
}

impl Default for TerminationPolicy {
    fn default() -> Self {
        Self {
            check_start: DEFAULT_TERMINATION_CHECK_START,
            check_step: DEFAULT_TERMINATION_CHECK_STEP,
            max_k_cap: None,
        }
    }
}

impl TerminationPolicy {
    #[inline]
    pub(crate) fn new(check_start: usize, check_step: usize, max_k_cap: Option<usize>) -> Self {
        Self {
            check_start,
            check_step,
            max_k_cap,
        }
    }

    #[inline]
    pub(crate) fn should_check(self, neighbors_processed: usize) -> bool {
        self.check_step > 0
            && neighbors_processed >= self.check_start
            && (neighbors_processed - self.check_start).is_multiple_of(self.check_step)
    }

    #[inline]
    pub(crate) fn max_k_cap(self) -> Option<usize> {
        self.max_k_cap
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct KnnPolicy {
    packed: PackedNeighborPolicy,
    termination: TerminationPolicy,
}

impl KnnPolicy {
    #[inline]
    pub(crate) fn new(packed: PackedNeighborPolicy, termination: TerminationPolicy) -> Self {
        Self {
            packed,
            termination,
        }
    }

    #[inline]
    pub(crate) fn for_point_count(
        num_points: usize,
        termination: TerminationPolicy,
        expand_r2_enabled: bool,
    ) -> Self {
        Self::new(
            PackedNeighborPolicy::for_point_count(num_points, expand_r2_enabled),
            termination,
        )
    }

    #[inline]
    pub(crate) fn packed(self) -> PackedNeighborPolicy {
        self.packed
    }

    #[inline]
    pub(crate) fn termination(self) -> TerminationPolicy {
        self.termination
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn packed_policy_clamps_to_available_neighbors() {
        let zero = PackedNeighborPolicy::for_point_count(0, true);
        assert!(!zero.enabled());
        assert_eq!(zero.chunk0_size(), 0);
        assert_eq!(zero.chunk_size(), 0);

        let one = PackedNeighborPolicy::for_point_count(1, true);
        assert!(!one.enabled());
        assert_eq!(one.chunk0_size(), 0);
        assert_eq!(one.chunk_size(), 0);

        let small = PackedNeighborPolicy::for_point_count(5, true);
        assert!(small.enabled());
        assert_eq!(small.chunk0_size(), 4);
        assert_eq!(small.chunk_size(), 4);
        assert_eq!(small.scratch_chunk_capacity(), 4);

        let large = PackedNeighborPolicy::for_point_count(100, false);
        assert_eq!(large.chunk0_size(), 16);
        assert_eq!(large.chunk_size(), 8);
        assert_eq!(large.scratch_chunk_capacity(), 16);
        assert!(!large.expand_r2_enabled());
    }

    #[test]
    fn termination_policy_uses_expected_cadence() {
        let policy = TerminationPolicy::new(8, 3, None);
        for n in 0..8 {
            assert!(!policy.should_check(n));
        }
        assert!(policy.should_check(8));
        assert!(!policy.should_check(9));
        assert!(!policy.should_check(10));
        assert!(policy.should_check(11));
        assert!(policy.should_check(14));
    }

    #[test]
    fn knn_policy_keeps_packed_and_termination_together() {
        let termination = TerminationPolicy::new(12, 2, Some(64));
        let policy = KnnPolicy::for_point_count(20, termination, true);
        assert_eq!(policy.packed().chunk0_size(), 16);
        assert_eq!(policy.packed().chunk_size(), 8);
        assert!(policy.packed().expand_r2_enabled());
        assert_eq!(policy.termination().max_k_cap(), Some(64));
        assert!(policy.termination().should_check(12));
        assert!(policy.termination().should_check(14));
        assert!(!policy.termination().should_check(13));
    }
}
