use super::super::types::PolyBuffer;
use crate::fp;

pub(super) fn build_output(
    poly: &PolyBuffer,
    out: &mut PolyBuffer,
    n: usize,
    entry_pt: (f64, f64),
    entry_edge_plane: usize,
    entry_next: usize,
    exit_pt: (f64, f64),
    exit_edge_plane: usize,
    exit_idx: usize,
    hp_plane_idx: usize,
) {
    out.len = 0;
    let mut max_r2 = 0.0f64;
    let mut has_bounding = false;
    let track_bounding = poly.has_bounding_ref;

    macro_rules! push {
        ($u:expr, $v:expr, $vp:expr, $ep:expr) => {{
            let u = $u;
            let v = $v;
            let vp = $vp;
            out.push_raw(u, v, vp, $ep);
            max_r2 = max_r2.max(fp::fma_f64(u, u, v * v));
            if track_bounding {
                has_bounding |= vp.0 == usize::MAX;
            }
        }};
    }

    push!(
        entry_pt.0,
        entry_pt.1,
        (entry_edge_plane, hp_plane_idx),
        entry_edge_plane
    );

    let mut i = entry_next;
    loop {
        push!(
            poly.us[i],
            poly.vs[i],
            poly.vertex_planes[i],
            poly.edge_planes[i]
        );
        if i == exit_idx {
            break;
        }
        i = (i + 1) % n;
    }

    push!(
        exit_pt.0,
        exit_pt.1,
        (exit_edge_plane, hp_plane_idx),
        hp_plane_idx
    );

    out.max_r2 = max_r2;
    out.has_bounding_ref = if track_bounding { has_bounding } else { false };
}
