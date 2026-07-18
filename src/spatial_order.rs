//! Shared policy for choosing between caller/global order and spatial order.

/// Measured relationship between an identity order and a spatial order.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum SpatialOrderCorrelation {
    /// Adjacent entries usually remain near each other in the other order.
    Correlated,
    /// Adjacent entries usually jump across the other order's domain.
    Scrambled,
}

impl SpatialOrderCorrelation {
    #[inline(always)]
    pub(crate) fn is_scrambled(self) -> bool {
        matches!(self, Self::Scrambled)
    }
}

/// Classify a sampled mean absolute delta against the greater of one element
/// and one percent of its domain.
///
/// Callers choose samples from the order they can traverse cheaply. The
/// absolute floor prevents unit-stride order in domains below 100 entries
/// from being mislabeled as scrambled.
#[inline(always)]
pub(crate) fn classify_spatial_correlation(
    abs_delta: u64,
    samples: u64,
    domain_len: usize,
) -> SpatialOrderCorrelation {
    // Representation limits in both callers keep these products well
    // below u64::MAX.
    let scaled_domain = domain_len.max(100) as u64;
    if samples != 0 && abs_delta * 100 > scaled_domain * samples {
        SpatialOrderCorrelation::Scrambled
    } else {
        SpatialOrderCorrelation::Correlated
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn one_percent_threshold_is_strict() {
        assert_eq!(
            classify_spatial_correlation(32 * 100, 32, 10_000),
            SpatialOrderCorrelation::Correlated
        );
        assert_eq!(
            classify_spatial_correlation(32 * 101, 32, 10_000),
            SpatialOrderCorrelation::Scrambled
        );
        assert_eq!(
            classify_spatial_correlation(0, 0, 10_000),
            SpatialOrderCorrelation::Correlated
        );
    }

    #[test]
    fn small_domains_keep_unit_stride_correlated() {
        assert_eq!(
            classify_spatial_correlation(32, 32, 64),
            SpatialOrderCorrelation::Correlated
        );
        assert_eq!(
            classify_spatial_correlation(64, 32, 64),
            SpatialOrderCorrelation::Scrambled
        );
    }
}
