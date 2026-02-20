use super::super::*;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

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
