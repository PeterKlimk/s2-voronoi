#![allow(clippy::needless_range_loop)] // indices address parallel arrays
use super::super::*;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

#[test]
fn tiny_cell_cap_avoids_cosine_to_sine_cancellation() {
    for (res, iu, iv) in [
        (26_754usize, 13_377usize, 13_377usize),
        (26_754, 6_688, 6_688),
        (1_536, 668, 464),
    ] {
        let inv_res = 1.0 / res as f32;
        let u0 = st_to_uv(iu as f32 * inv_res);
        let u1 = st_to_uv((iu + 1) as f32 * inv_res);
        let v0 = st_to_uv(iv as f32 * inv_res);
        let v1 = st_to_uv((iv + 1) as f32 * inv_res);
        let uc = st_to_uv((iu as f32 + 0.5) * inv_res);
        let vc = st_to_uv((iv as f32 + 0.5) * inv_res);
        let center = face_uv_to_3d(0, uc, vc);
        let corners = [
            face_uv_to_3d(0, u0, v0),
            face_uv_to_3d(0, u0, v1),
            face_uv_to_3d(0, u1, v0),
            face_uv_to_3d(0, u1, v1),
        ];

        let old_cos = corners
            .iter()
            .map(|corner| center.dot(*corner))
            .fold(1.0f32, f32::min);
        let true_radius = corners
            .iter()
            .map(|corner| {
                let c = center.as_dvec3();
                let p = corner.as_dvec3();
                (c.cross(p).length() / (c.length() * p.length())).asin()
            })
            .fold(0.0f64, f64::max);
        let old_sin = (1.0 - old_cos * old_cos).max(0.0).sqrt()
            + old_cos * crate::tolerances::GRID_CAP_ANGULAR_PAD;
        let true_padded_sin = (true_radius + crate::tolerances::GRID_CAP_ANGULAR_PAD as f64).sin();
        assert!(
            (old_sin as f64) < true_padded_sin,
            "fixture must reproduce the old underestimated sine radius"
        );

        let (stored_cos, stored_sin) = super::super::build::conservative_cell_cap(center, &corners);
        let padded_radius = true_radius + crate::tolerances::GRID_CAP_ANGULAR_PAD as f64;
        assert!(
            stored_cos as f64 <= padded_radius.cos(),
            "stored cosine radius is not rounded outward"
        );
        assert!(
            stored_sin as f64 >= padded_radius.sin(),
            "stored sine radius is not rounded outward"
        );
    }
}

#[test]
fn direct_cell_cap_contains_coarse_and_rotated_face_corners() {
    for (res, face, iu, iv) in [(1usize, 5usize, 0usize, 0usize), (17, 4, 0, 16)] {
        let inv_res = 1.0 / res as f32;
        let u0 = st_to_uv(iu as f32 * inv_res);
        let u1 = st_to_uv((iu + 1) as f32 * inv_res);
        let v0 = st_to_uv(iv as f32 * inv_res);
        let v1 = st_to_uv((iv + 1) as f32 * inv_res);
        let center = face_uv_to_3d(
            face,
            st_to_uv((iu as f32 + 0.5) * inv_res),
            st_to_uv((iv as f32 + 0.5) * inv_res),
        );
        let corners = [
            face_uv_to_3d(face, u0, v0),
            face_uv_to_3d(face, u0, v1),
            face_uv_to_3d(face, u1, v0),
            face_uv_to_3d(face, u1, v1),
        ];
        let (cos_radius, sin_radius) = super::super::build::conservative_cell_cap(center, &corners);
        let stored_radius = (sin_radius as f64).atan2(cos_radius as f64);

        for corner in corners {
            let c = center.as_dvec3().normalize();
            let p = corner.as_dvec3().normalize();
            let corner_radius = c.cross(p).length().atan2(c.dot(p));
            assert!(corner_radius < stored_radius);
        }
    }
}

#[test]
fn near_antipodal_shell_cap_bounds_raw_dot() {
    let res = 5usize;
    let grid = CubeMapGrid::new(&[], res);
    let cell = grid.point_to_cell(Vec3::X);
    assert_eq!(grid.cell_centers[cell], Vec3::X);

    let (_, iu, iv) = cell_to_face_ij(cell, res);
    let s = ((iu + 1) as f32 / res as f32).next_down();
    let t = ((iv + 1) as f32 / res as f32).next_down();
    let point = face_uv_to_3d(0, st_to_uv(s), st_to_uv(t));
    assert_eq!(grid.point_to_cell(point), cell);

    let delta = 2.0f64.powi(-13);
    let tangent = glam::DVec3::new(0.0, point.y as f64, point.z as f64).normalize();
    let query64 = (glam::DVec3::NEG_X + delta * tangent).normalize();
    let query = Vec3::new(query64.x as f32, query64.y as f32, query64.z as f32);
    assert_eq!(
        query.x, -1.0,
        "fixture must hit the rounded antipodal endpoint"
    );

    let min_dist_sq = grid.cell_min_dist_sq(query.as_dvec3().normalize(), cell);
    let exported =
        (1.0 - 0.5 * min_dist_sq).clamp(-1.0, 1.0) + crate::tolerances::GRID_DOT_BOUND_PAD;
    let actual = crate::fp::dot3_f32(query.x, query.y, query.z, point.x, point.y, point.z);
    assert!(
        actual <= exported,
        "near-antipodal cell cap underestimates raw dot: actual={actual:?}, \
         exported={exported:?}, gap={:?}",
        actual - exported
    );
}

#[test]
fn test_security_ring2_captures_outside_cap_max() {
    #[inline]
    fn max_dot_to_cap(q: Vec3, center: Vec3, cos_r: f32, sin_r: f32) -> f32 {
        let cos_d = q.dot(center).clamp(-1.0, 1.0);
        if cos_d > cos_r {
            return 1.0;
        }
        let sin_d = (1.0 - cos_d * cos_d).max(0.0).sqrt();
        (cos_d * cos_r + sin_d * sin_r).clamp(-1.0, 1.0)
    }

    fn sample_point_in_cell(cell: usize, res: usize, rng: &mut impl Rng) -> Vec3 {
        let (face, iu, iv) = cell_to_face_ij(cell, res);

        let eps = 1e-4f32;
        let fu = (rng.gen::<f32>() * (1.0 - 2.0 * eps) + eps) / res as f32;
        let fv = (rng.gen::<f32>() * (1.0 - 2.0 * eps) + eps) / res as f32;
        let su = iu as f32 / res as f32 + fu;
        let sv = iv as f32 / res as f32 + fv;

        let u = st_to_uv(su);
        let v = st_to_uv(sv);
        face_uv_to_3d(face, u, v)
    }

    let res = 8usize;
    let grid = CubeMapGrid::new(&[], res);
    let num_cells = 6 * res * res;
    let mut rng = ChaCha8Rng::seed_from_u64(12345);
    let samples_per_cell = 8usize;

    for cell in 0..num_cells {
        let neighbors = grid.cell_neighbors(cell);
        let mut in_neighborhood = vec![false; num_cells];
        for &c in neighbors.iter() {
            if c == u32::MAX {
                continue;
            }
            in_neighborhood[c as usize] = true;
        }

        let ring2 = grid.cell_ring2(cell);
        for &c in ring2 {
            assert!(
                !in_neighborhood[c as usize],
                "ring2 cell is inside neighborhood: cell={}, ring2_cell={}",
                cell, c
            );
        }

        for _ in 0..samples_per_cell {
            let q = sample_point_in_cell(cell, res, &mut rng);
            assert_eq!(
                grid.point_to_cell(q),
                cell,
                "sample not inside cell: cell={}",
                cell
            );

            let mut outside_max = f32::NEG_INFINITY;
            for other in 0..num_cells {
                if in_neighborhood[other] {
                    continue;
                }
                let dot = max_dot_to_cap(
                    q,
                    grid.cell_centers[other],
                    grid.cell_cos_radius[other],
                    grid.cell_sin_radius[other],
                );
                outside_max = outside_max.max(dot);
            }

            let mut ring2_max = f32::NEG_INFINITY;
            for &other in ring2 {
                let idx = other as usize;
                let dot = max_dot_to_cap(
                    q,
                    grid.cell_centers[idx],
                    grid.cell_cos_radius[idx],
                    grid.cell_sin_radius[idx],
                );
                ring2_max = ring2_max.max(dot);
            }

            let diff = outside_max - ring2_max;
            assert!(
                diff <= 1e-5,
                "ring2 missed outside max: cell={}, outside_max={}, ring2_max={}, diff={}",
                cell,
                outside_max,
                ring2_max,
                diff
            );
        }
    }
}

#[test]
fn test_uv_line_planes_match_uv_rect_interior() {
    let res = 16usize;
    let grid = CubeMapGrid::new(&[], res);
    let mut rng = ChaCha8Rng::seed_from_u64(123);
    let samples = [(3usize, 4usize), (6, 6), (10, 9)];

    for face in 0..6usize {
        for &(iu, iv) in &samples {
            let cell = face * res * res + iv * res + iu;
            let center = grid.cell_centers[cell];

            let mut planes = [
                grid.face_u_line_plane(face, iu - 1),
                grid.face_u_line_plane(face, iu + 2),
                grid.face_v_line_plane(face, iv - 1),
                grid.face_v_line_plane(face, iv + 2),
            ];
            for n in &mut planes {
                if n.dot(center) < 0.0 {
                    *n = -*n;
                }
            }

            let umin = st_to_uv((iu - 1) as f32 / res as f32);
            let umax = st_to_uv((iu + 2) as f32 / res as f32);
            let vmin = st_to_uv((iv - 1) as f32 / res as f32);
            let vmax = st_to_uv((iv + 2) as f32 / res as f32);
            let eps = 5e-4f32;

            for _ in 0..256 {
                let u = rng.gen_range((umin + eps)..(umax - eps));
                let v = rng.gen_range((vmin + eps)..(vmax - eps));
                let p = face_uv_to_3d(face, u, v);
                let (f2, u2, v2) = point_to_face_uv(p);
                assert_eq!(f2, face);
                assert!(u2 >= umin - 1e-5 && u2 <= umax + 1e-5);
                assert!(v2 >= vmin - 1e-5 && v2 <= vmax + 1e-5);

                for n in &planes {
                    assert!(
                        n.dot(p) >= -1e-5,
                        "inside point violates plane: face={}, iu={}, iv={}, n·p={}",
                        face,
                        iu,
                        iv,
                        n.dot(p)
                    );
                }
            }

            let delta = 2e-3f32;
            for &(u, v) in &[
                (umin - delta, (vmin + vmax) * 0.5),
                (umax + delta, (vmin + vmax) * 0.5),
                ((umin + umax) * 0.5, vmin - delta),
                ((umin + umax) * 0.5, vmax + delta),
            ] {
                assert!(u > -1.0 && u < 1.0 && v > -1.0 && v < 1.0);
                let p = face_uv_to_3d(face, u, v);
                let (f2, ..) = point_to_face_uv(p);
                assert_eq!(f2, face);

                let ok = planes.iter().all(|n| n.dot(p) >= -1e-6);
                assert!(
                    !ok,
                    "outside point unexpectedly inside: face={}, iu={}, iv={}, u={}, v={}",
                    face, iu, iv, u, v
                );
            }
        }
    }
}
