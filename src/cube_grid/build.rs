//! Grid build helpers for CubeMapGrid.

use glam::Vec3;

use super::{
    cell_to_face_ij, diagonal_from_edge_neighbors, face_uv_to_3d, face_uv_to_cell,
    point_to_face_uv, st_to_uv, step_one, CubeMapGrid, CubeMapGridBuildTimings, EdgeDir, RING2_MAX,
};

impl CubeMapGrid {
    /// Build a cube-map grid from points on unit sphere.
    ///
    /// `res` controls grid resolution: 6 * res² total cells.
    /// For n points, good choices:
    /// - res ≈ sqrt(n / 300) for ~50 points per cell
    /// - res ≈ sqrt(n / 600) for ~100 points per cell
    pub fn new(points: &[Vec3], res: usize) -> Self {
        #[cfg(feature = "timing")]
        let mut timings = CubeMapGridBuildTimings::default();
        #[cfg(feature = "timing")]
        return Self::new_impl(points, res, Some(&mut timings));
        #[cfg(not(feature = "timing"))]
        return Self::new_impl(points, res, None);
    }

    #[cfg(feature = "timing")]
    pub fn new_with_build_timings(
        points: &[Vec3],
        res: usize,
        timings: &mut CubeMapGridBuildTimings,
    ) -> Self {
        Self::new_impl(points, res, Some(timings))
    }

    fn new_impl(
        points: &[Vec3],
        res: usize,
        #[cfg(feature = "timing")] mut timings: Option<&mut CubeMapGridBuildTimings>,
        #[cfg(not(feature = "timing"))] _timings: Option<&mut CubeMapGridBuildTimings>,
    ) -> Self {
        assert!(res > 0, "CubeMapGrid requires res > 0");
        let num_cells = 6 * res * res;

        // Step 1: Count points per cell
        #[cfg(feature = "timing")]
        let t = std::time::Instant::now();
        let mut cell_counts = vec![0u32; num_cells];
        let mut point_cells = Vec::with_capacity(points.len());
        for p in points {
            let (face, u, v) = point_to_face_uv(*p);
            let cell = face_uv_to_cell(face, u, v, res);
            point_cells.push(cell as u32);
            cell_counts[cell] += 1;
        }
        #[cfg(feature = "timing")]
        if let Some(timings) = timings.as_deref_mut() {
            timings.count_cells += t.elapsed();
        }

        // Step 2: Prefix sum to get offsets
        #[cfg(feature = "timing")]
        let t = std::time::Instant::now();
        let mut cell_offsets = Vec::with_capacity(num_cells + 1);
        cell_offsets.push(0);
        let mut sum = 0u32;
        for &count in &cell_counts {
            sum += count;
            cell_offsets.push(sum);
        }
        #[cfg(feature = "timing")]
        if let Some(timings) = timings.as_deref_mut() {
            timings.prefix_sum += t.elapsed();
        }

        // Step 3: Scatter points into cells and build SoA layout (points stored contiguous by cell).
        //
        // This is a single pass that avoids an additional gather pass over a random permutation.
        #[cfg(feature = "timing")]
        let t = std::time::Instant::now();
        let n = points.len();
        debug_assert_eq!(cell_offsets[num_cells] as usize, n, "prefix sum mismatch");

        let mut point_indices = Vec::<u32>::with_capacity(n);
        let mut cell_points_x = Vec::<f32>::with_capacity(n);
        let mut cell_points_y = Vec::<f32>::with_capacity(n);
        let mut cell_points_z = Vec::<f32>::with_capacity(n);

        let indices_spare = point_indices.spare_capacity_mut();
        let x_spare = cell_points_x.spare_capacity_mut();
        let y_spare = cell_points_y.spare_capacity_mut();
        let z_spare = cell_points_z.spare_capacity_mut();

        let mut cell_cursors = cell_offsets[..num_cells].to_vec();
        for (i, cell_u32) in point_cells.iter().copied().enumerate() {
            let cell = cell_u32 as usize;
            let pos = cell_cursors[cell] as usize;
            let p = points[i];

            indices_spare[pos].write(i as u32);
            x_spare[pos].write(p.x);
            y_spare[pos].write(p.y);
            z_spare[pos].write(p.z);

            cell_cursors[cell] += 1;
        }

        debug_assert!(
            cell_cursors
                .iter()
                .zip(cell_offsets.iter().skip(1))
                .all(|(cursor, offset_end)| cursor == offset_end),
            "scatter cursors did not reach cell end offsets"
        );

        // SAFETY: The prefix-sum scatter visits each `pos` in `0..n` exactly once, so the first
        // `n` elements of each vec's spare capacity have been fully initialized.
        unsafe {
            point_indices.set_len(n);
            cell_points_x.set_len(n);
            cell_points_y.set_len(n);
            cell_points_z.set_len(n);
        }
        #[cfg(feature = "timing")]
        if let Some(timings) = timings.as_deref_mut() {
            timings.scatter_soa += t.elapsed();
        }

        // Step 4: Precompute neighbors and ring-2 cells for each cell
        #[cfg(feature = "timing")]
        let t = std::time::Instant::now();
        let neighbors = Self::compute_all_neighbors(res);
        #[cfg(feature = "timing")]
        if let Some(timings) = timings.as_deref_mut() {
            timings.neighbors += t.elapsed();
        }

        #[cfg(feature = "timing")]
        let t = std::time::Instant::now();
        let (ring2, ring2_lens) = Self::compute_ring2(res, &neighbors);
        #[cfg(feature = "timing")]
        if let Some(timings) = timings.as_deref_mut() {
            timings.ring2 += t.elapsed();
        }

        #[cfg(feature = "timing")]
        let t = std::time::Instant::now();
        let (cell_centers, cell_cos_radius, cell_sin_radius) = Self::compute_cell_bounds(res);
        #[cfg(feature = "timing")]
        if let Some(timings) = timings.as_deref_mut() {
            timings.cell_bounds += t.elapsed();
        }

        // Step 5: Precompute security_3x3 threshold per cell using ring-2 caps
        #[cfg(feature = "timing")]
        let t = std::time::Instant::now();
        let security_3x3 = Self::compute_security_3x3(
            res,
            &cell_centers,
            &cell_cos_radius,
            &cell_sin_radius,
            &ring2,
            &ring2_lens,
        );
        #[cfg(feature = "timing")]
        if let Some(timings) = timings.as_deref_mut() {
            timings.security_3x3 += t.elapsed();
        }

        CubeMapGrid {
            res,
            cell_offsets,
            point_indices,
            point_cells,
            neighbors,
            ring2,
            ring2_lens,
            cell_centers,
            cell_cos_radius,
            cell_sin_radius,
            security_3x3,
            cell_points_x,
            cell_points_y,
            cell_points_z,
        }
    }

    /// Compute 3×3 neighborhood for all cells.
    fn compute_all_neighbors(res: usize) -> Vec<u32> {
        let num_cells = 6 * res * res;
        let mut neighbors = vec![u32::MAX; num_cells * 9];

        for cell in 0..num_cells {
            let (face, iu, iv) = cell_to_face_ij(cell, res);
            let base = cell * 9;

            // Fast path for interior cells: all 3×3 neighbors stay on the same face, so we can
            // compute them with simple index arithmetic and avoid cube-edge stitching logic.
            //
            // Margin is 1 cell (unlike ring-2 which needs margin 2).
            if res >= 3 && iu >= 1 && iv >= 1 && iu + 1 < res && iv + 1 < res {
                let center = cell as u32;
                let left = (cell - 1) as u32;
                let right = (cell + 1) as u32;
                let down = (cell - res) as u32;
                let up = (cell + res) as u32;

                neighbors[base + 0] = (cell - res - 1) as u32; // down_left
                neighbors[base + 1] = down;
                neighbors[base + 2] = (cell - res + 1) as u32; // down_right
                neighbors[base + 3] = left;
                neighbors[base + 4] = center;
                neighbors[base + 5] = right;
                neighbors[base + 6] = (cell + res - 1) as u32; // up_left
                neighbors[base + 7] = up;
                neighbors[base + 8] = (cell + res + 1) as u32; // up_right
                continue;
            }

            let center = cell as u32;
            let (lf, lu, lv) = step_one(face, iu, iv, EdgeDir::Left, res);
            let left = (lf * res * res + lv * res + lu) as u32;
            let (rf, ru, rv) = step_one(face, iu, iv, EdgeDir::Right, res);
            let right = (rf * res * res + rv * res + ru) as u32;
            let (df, duu, dvv) = step_one(face, iu, iv, EdgeDir::Down, res);
            let down = (df * res * res + dvv * res + duu) as u32;
            let (uf, uu, uv) = step_one(face, iu, iv, EdgeDir::Up, res);
            let up = (uf * res * res + uv * res + uu) as u32;

            let down_left = diagonal_from_edge_neighbors(center, down, left, res);
            let down_right = diagonal_from_edge_neighbors(center, down, right, res);
            let up_left = diagonal_from_edge_neighbors(center, up, left, res);
            let up_right = diagonal_from_edge_neighbors(center, up, right, res);

            neighbors[base + 0] = down_left;
            neighbors[base + 1] = down;
            neighbors[base + 2] = down_right;
            neighbors[base + 3] = left;
            neighbors[base + 4] = center;
            neighbors[base + 5] = right;
            neighbors[base + 6] = up_left;
            neighbors[base + 7] = up;
            neighbors[base + 8] = up_right;

            debug_assert!(
                {
                    let ns = [
                        down_left, down, down_right, left, center, right, up_left, up, up_right,
                    ];
                    let mut ok = true;
                    for i in 0..9 {
                        for j in (i + 1)..9 {
                            if ns[i] == u32::MAX || ns[j] == u32::MAX {
                                continue;
                            }
                            ok &= ns[i] != ns[j];
                        }
                    }
                    ok
                },
                "duplicate neighbor cell: cell={}, neighbors={:?}",
                cell,
                &neighbors[base..base + 9]
            );
        }

        neighbors
    }

    fn compute_ring2(res: usize, neighbors: &[u32]) -> (Vec<[u32; RING2_MAX]>, Vec<u8>) {
        let num_cells = 6 * res * res;
        let mut ring2 = vec![[u32::MAX; RING2_MAX]; num_cells];
        let mut ring2_lens = vec![0u8; num_cells];

        for cell in 0..num_cells {
            let base = cell * 9;
            let near = &neighbors[base..base + 9];
            debug_assert_eq!(near.len(), 9);

            // Interior cells (>=2 away from any edge) have a fixed ring-2 pattern on the face.
            let (face, iu, iv) = cell_to_face_ij(cell, res);
            if iu >= 2 && iv >= 2 && iu + 2 < res && iv + 2 < res {
                let mut ring = [u32::MAX; RING2_MAX];
                let mut ring_len = 0usize;
                let face_base = face * res * res;
                let iu = iu as isize;
                let iv = iv as isize;
                for dv in -2isize..=2 {
                    for du in -2isize..=2 {
                        if du == 0 && dv == 0 {
                            continue;
                        }
                        if du.abs().max(dv.abs()) != 2 {
                            continue;
                        }
                        let u = (iu + du) as usize;
                        let v = (iv + dv) as usize;
                        ring[ring_len] = (face_base + v * res + u) as u32;
                        ring_len += 1;
                    }
                }
                debug_assert_eq!(
                    ring_len, RING2_MAX,
                    "interior ring2 size mismatch: {}",
                    ring_len
                );
                ring2[cell] = ring;
                ring2_lens[cell] = ring_len as u8;
                continue;
            }

            // Unroll near-membership checks: this is hot (called for every neighbor-of-neighbor).
            let near0 = near[0];
            let near1 = near[1];
            let near2 = near[2];
            let near3 = near[3];
            let near4 = near[4];
            let near5 = near[5];
            let near6 = near[6];
            let near7 = near[7];
            let near8 = near[8];

            #[inline(always)]
            fn ring_contains(ring: &[u32; RING2_MAX], ring_len: usize, v: u32) -> bool {
                let mut i = 0usize;
                while i < ring_len {
                    if ring[i] == v {
                        return true;
                    }
                    i += 1;
                }
                false
            }

            let mut ring = [u32::MAX; RING2_MAX];
            let mut ring_len = 0usize;
            for &n1_cell in near {
                if n1_cell == u32::MAX {
                    continue;
                }
                let n1_idx = n1_cell as usize;
                let b1 = n1_idx * 9;
                let near_of_n1 = &neighbors[b1..b1 + 9];
                for &cand in near_of_n1 {
                    if cand == u32::MAX {
                        continue;
                    }
                    // Skip the 3×3 neighborhood (including self).
                    if cand == near0
                        || cand == near1
                        || cand == near2
                        || cand == near3
                        || cand == near4
                        || cand == near5
                        || cand == near6
                        || cand == near7
                        || cand == near8
                    {
                        continue;
                    }
                    if ring_contains(&ring, ring_len, cand) {
                        continue;
                    }
                    debug_assert!(ring_len < RING2_MAX, "ring2 exceeded max size");
                    ring[ring_len] = cand;
                    ring_len += 1;
                    if ring_len == RING2_MAX {
                        break;
                    }
                }
                if ring_len == RING2_MAX {
                    break;
                }
            }

            debug_assert!(ring_len > 0, "ring2 must be non-empty");

            ring2[cell][..ring_len].copy_from_slice(&ring[..ring_len]);
            ring2_lens[cell] = ring_len as u8;
        }

        (ring2, ring2_lens)
    }

    /// Compute security_3x3 threshold for all cells using ring-2 caps.
    ///
    /// For each cell, finds the max dot product from any point in the cell to any point
    /// in the ring-2 set (Chebyshev distance 2). Uses spherical caps as a conservative bound.
    fn compute_security_3x3(
        res: usize,
        cell_centers: &[Vec3],
        cell_cos_radius: &[f32],
        cell_sin_radius: &[f32],
        ring2: &[[u32; RING2_MAX]],
        ring2_lens: &[u8],
    ) -> Vec<f32> {
        let num_cells = 6 * res * res;
        let mut security = vec![f32::NEG_INFINITY; num_cells];

        for cell in 0..num_cells {
            let ring = &ring2[cell];
            let ring_len = ring2_lens[cell] as usize;

            let center_a = cell_centers[cell];
            let cos_ra = cell_cos_radius[cell];
            let sin_ra = cell_sin_radius[cell];

            let mut max_dot = f32::NEG_INFINITY;
            for i in 0..ring_len {
                let ring_cell = ring[i] as usize;
                let center_b = cell_centers[ring_cell];
                let cos_rb = cell_cos_radius[ring_cell];
                let sin_rb = cell_sin_radius[ring_cell];

                // Compute max dot between two spherical caps
                let cos_d = center_a.dot(center_b).clamp(-1.0, 1.0);

                // If center cell is inside ring cell's cap (distance < radius), max_dot = 1.0
                if cos_d > cos_rb {
                    max_dot = 1.0;
                    break;
                }

                // Max dot = cos(d - ra - rb) where d is center-to-center angle
                // cos(d - r) = cos_d * cos_r + sin_d * sin_r
                let sin_d = (1.0 - cos_d * cos_d).max(0.0).sqrt();

                // First subtract ra: cos(d - ra)
                let cos_d_minus_ra = cos_d * cos_ra + sin_d * sin_ra;
                let sin_d_minus_ra = sin_d * cos_ra - cos_d * sin_ra;

                // Then subtract rb: cos(d - ra - rb)
                let cos_min_angle = cos_d_minus_ra * cos_rb + sin_d_minus_ra.abs() * sin_rb;
                max_dot = max_dot.max(cos_min_angle);
            }

            security[cell] = max_dot;
        }

        security
    }

    fn compute_cell_bounds(res: usize) -> (Vec<Vec3>, Vec<f32>, Vec<f32>) {
        let num_cells = 6 * res * res;
        let mut centers = Vec::with_capacity(num_cells);
        let mut cos_r = Vec::with_capacity(num_cells);
        let mut sin_r = Vec::with_capacity(num_cells);

        // Precompute ST→UV transform for grid lines and cell centers.
        //
        // `st_to_uv` is sqrt-heavy; doing this once avoids ~6 calls per cell.
        let res_f = res as f32;
        let inv_res = 1.0 / res_f;
        let mut uv_lines: Vec<f32> = Vec::with_capacity(res + 1);
        for i in 0..=res {
            let s = i as f32 * inv_res;
            uv_lines.push(st_to_uv(s));
        }
        let mut uv_centers: Vec<f32> = Vec::with_capacity(res);
        for i in 0..res {
            let s = (i as f32 + 0.5) * inv_res;
            uv_centers.push(st_to_uv(s));
        }

        #[inline]
        fn plane_normal_u(face: usize, u: f32) -> Vec3 {
            let n = match face {
                0 | 1 => Vec3::new(u, 0.0, 1.0),  // z + u x = 0
                2 => Vec3::new(1.0, -u, 0.0),     // x - u y = 0
                3 => Vec3::new(1.0, u, 0.0),      // x + u y = 0
                4 | 5 => Vec3::new(1.0, 0.0, -u), // x - u z = 0
                _ => unreachable!(),
            };
            n.normalize()
        }

        #[inline]
        fn plane_normal_v(face: usize, v: f32) -> Vec3 {
            let n = match face {
                0 => Vec3::new(-v, 1.0, 0.0),    // y - v x = 0
                1 => Vec3::new(v, 1.0, 0.0),     // y + v x = 0
                2 | 3 => Vec3::new(0.0, v, 1.0), // z + v y = 0
                4 => Vec3::new(0.0, 1.0, -v),    // y - v z = 0
                5 => Vec3::new(0.0, 1.0, v),     // y + v z = 0
                _ => unreachable!(),
            };
            n.normalize()
        }

        // Precompute normalized boundary plane normals for each face and each grid line.
        // Planes depend only on (face, u_line) or (face, v_line), not on the points.
        let line_count = res + 1;
        let mut u_planes: Vec<Vec3> = Vec::with_capacity(6 * line_count);
        let mut v_planes: Vec<Vec3> = Vec::with_capacity(6 * line_count);
        for face in 0..6 {
            for i in 0..=res {
                let uv = uv_lines[i];
                u_planes.push(plane_normal_u(face, uv));
                v_planes.push(plane_normal_v(face, uv));
            }
        }

        #[inline]
        fn inside_all(p: Vec3, planes: &[Vec3; 4]) -> bool {
            planes.iter().all(|n| n.dot(p) >= -1e-6)
        }

        for cell in 0..num_cells {
            let (face, iu, iv) = cell_to_face_ij(cell, res);

            let uc = uv_centers[iu];
            let vc = uv_centers[iv];
            let center = face_uv_to_3d(face, uc, vc);

            // Cell boundary UV coordinates (grid lines).
            let u0 = uv_lines[iu];
            let u1 = uv_lines[iu + 1];
            let v0 = uv_lines[iv];
            let v1 = uv_lines[iv + 1];

            let mut planes = [
                u_planes[face * line_count + iu],
                u_planes[face * line_count + (iu + 1)],
                v_planes[face * line_count + iv],
                v_planes[face * line_count + (iv + 1)],
            ];
            for n in planes.iter_mut() {
                if n.dot(center) < 0.0 {
                    *n = -*n;
                }
            }

            #[inline]
            fn update_min_dot(center: Vec3, p: Vec3, min_dot: &mut f32) {
                let dot = center.dot(p).clamp(-1.0, 1.0);
                *min_dot = (*min_dot).min(dot);
            }

            // Track min dot (= cos of max angle) using exact edge/vertex candidates.
            let mut min_dot = 1.0f32;

            // Corners: intersections of adjacent u/v planes.
            let corners = [
                (planes[0], planes[2]),
                (planes[0], planes[3]),
                (planes[1], planes[2]),
                (planes[1], planes[3]),
            ];
            for (a, b) in corners {
                let cross = a.cross(b);
                if cross.length_squared() <= 1e-12 {
                    continue;
                }
                let mut p = cross.normalize();
                if inside_all(p, &planes) {
                    update_min_dot(center, p, &mut min_dot);
                }
                p = -p;
                if inside_all(p, &planes) {
                    update_min_dot(center, p, &mut min_dot);
                }
            }

            // Edge midpoints: intersection of one plane with opposite boundary plane.
            for &(plane, flip) in [
                (planes[0], planes[1]),
                (planes[1], planes[0]),
                (planes[2], planes[3]),
                (planes[3], planes[2]),
            ]
            .iter()
            {
                let cross = plane.cross(flip);
                if cross.length_squared() <= 1e-12 {
                    continue;
                }
                let mut p = cross.normalize();
                if inside_all(p, &planes) {
                    update_min_dot(center, p, &mut min_dot);
                }
                p = -p;
                if inside_all(p, &planes) {
                    update_min_dot(center, p, &mut min_dot);
                }
            }

            // Sample the cell corners directly from projected bounds.
            for &u in &[u0, u1] {
                for &v in &[v0, v1] {
                    update_min_dot(center, face_uv_to_3d(face, u, v), &mut min_dot);
                }
            }

            // Sample additional points for stability on highly curved faces.
            let mid_u = (u0 + u1) * 0.5;
            let mid_v = (v0 + v1) * 0.5;
            for &u in &[u0, mid_u, u1] {
                for &v in &[v0, mid_v, v1] {
                    update_min_dot(center, face_uv_to_3d(face, u, v), &mut min_dot);
                }
            }

            let cos_radius = min_dot.clamp(-1.0, 1.0);
            let sin_a = (1.0 - cos_radius * cos_radius).max(0.0).sqrt();
            const SIN_EPS: f32 = 1e-5;
            let sin_radius = (sin_a + min_dot * SIN_EPS).clamp(0.0, 1.0);

            centers.push(center);
            cos_r.push(cos_radius);
            sin_r.push(sin_radius);
        }

        (centers, cos_r, sin_r)
    }
}
