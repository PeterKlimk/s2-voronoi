use crate::live_dedup::{EdgeMismatch, EdgeMismatchOrigin};

#[derive(Debug, Default, PartialEq, Eq)]
struct OriginCounts {
    in_bin_missing: usize,
    in_bin_thirds: usize,
    in_bin_duplicate: usize,
    in_bin_unconsumed: usize,
    cross_bin_thirds: usize,
    cross_bin_single: usize,
    cross_bin_duplicate: usize,
    cross_bin_slot: usize,
    post_reconciliation_unpaired: usize,
    endpoint_key: usize,
}

impl OriginCounts {
    fn collect(edge_mismatches: &[EdgeMismatch]) -> Self {
        let mut counts = Self::default();
        for mismatch in edge_mismatches {
            match mismatch.origin {
                EdgeMismatchOrigin::InBinMissingCheck => counts.in_bin_missing += 1,
                EdgeMismatchOrigin::InBinThirdsMismatch => counts.in_bin_thirds += 1,
                EdgeMismatchOrigin::InBinDuplicateSide => counts.in_bin_duplicate += 1,
                EdgeMismatchOrigin::InBinUnconsumedCheck => counts.in_bin_unconsumed += 1,
                EdgeMismatchOrigin::CrossBinThirdsMismatch => counts.cross_bin_thirds += 1,
                EdgeMismatchOrigin::CrossBinSingleSided => counts.cross_bin_single += 1,
                EdgeMismatchOrigin::CrossBinDuplicateSide => counts.cross_bin_duplicate += 1,
                EdgeMismatchOrigin::CrossBinSlotConflict => counts.cross_bin_slot += 1,
                EdgeMismatchOrigin::PostReconciliationUnpaired => {
                    counts.post_reconciliation_unpaired += 1;
                }
                EdgeMismatchOrigin::EndpointKeyMismatch => counts.endpoint_key += 1,
            }
        }
        counts
    }

    fn render(&self, total: usize) -> String {
        format!(
            "[origins] total={total} | InBin(miss={} thirds={} dup={} unconsumed={}) \
             CrossBin(thirds={} single={} dup={} slot={}) post_reconcile={} endpoint_key={}",
            self.in_bin_missing,
            self.in_bin_thirds,
            self.in_bin_duplicate,
            self.in_bin_unconsumed,
            self.cross_bin_thirds,
            self.cross_bin_single,
            self.cross_bin_duplicate,
            self.cross_bin_slot,
            self.post_reconciliation_unpaired,
            self.endpoint_key,
        )
    }
}

/// Emit the opt-in mismatch-origin diagnostic. The caller keeps the common
/// empty-input path outside this cold function, avoiding an environment lookup
/// for clean builds.
#[cold]
pub(super) fn maybe_emit_edge_mismatch_origins(edge_mismatches: &[EdgeMismatch]) {
    debug_assert!(!edge_mismatches.is_empty());
    if std::env::var_os("VORONOI_MESH_EDGE_MISMATCH_ORIGINS").is_none() {
        return;
    }
    eprintln!(
        "{}",
        OriginCounts::collect(edge_mismatches).render(edge_mismatches.len())
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::live_dedup::EdgeKey;

    fn mismatch(origin: EdgeMismatchOrigin) -> EdgeMismatch {
        EdgeMismatch {
            key: EdgeKey::from(0),
            origin,
        }
    }

    #[test]
    fn origin_line_reports_post_reconciliation_and_endpoint_counts() {
        let mismatches = [
            mismatch(EdgeMismatchOrigin::PostReconciliationUnpaired),
            mismatch(EdgeMismatchOrigin::PostReconciliationUnpaired),
            mismatch(EdgeMismatchOrigin::EndpointKeyMismatch),
        ];
        let line = OriginCounts::collect(&mismatches).render(mismatches.len());
        assert!(line.contains("total=3"));
        assert!(line.contains("post_reconcile=2"));
        assert!(line.contains("endpoint_key=1"));
    }
}
