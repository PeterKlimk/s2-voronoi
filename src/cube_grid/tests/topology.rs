use super::super::*;

#[test]
fn test_step_one_has_unique_inverse_direction() {
    for res in [4usize, 5, 8, 16] {
        let num_cells = 6 * res * res;
        for cell in 0..num_cells {
            let (face, iu, iv) = cell_to_face_ij(cell, res);
            for dir in [EdgeDir::Left, EdgeDir::Right, EdgeDir::Down, EdgeDir::Up] {
                let (f1, u1, v1) = step_one(face, iu, iv, dir, res);
                let mut count = 0usize;
                for back in [EdgeDir::Left, EdgeDir::Right, EdgeDir::Down, EdgeDir::Up] {
                    let (fb, ub, vb) = step_one(f1, u1, v1, back, res);
                    if (fb, ub, vb) == (face, iu, iv) {
                        count += 1;
                    }
                }
                assert_eq!(
                    count, 1,
                    "step_one inverse not unique: res={}, cell={}, dir={:?}, step=({},{},{})",
                    res, cell, dir, f1, u1, v1
                );
            }
        }
    }
}

#[test]
fn test_cell_neighbors_unique() {
    for res in [4usize, 5, 8, 16] {
        let grid = CubeMapGrid::new(&[], res);
        let num_cells = 6 * res * res;
        for cell in 0..num_cells {
            let (face, iu, iv) = cell_to_face_ij(cell, res);
            let neighbors = grid.cell_neighbors(cell);
            let mut seen = std::collections::HashSet::<u32>::with_capacity(9);
            for &ncell in neighbors.iter() {
                if ncell == u32::MAX {
                    continue;
                }
                assert!(
                    (ncell as usize) < num_cells,
                    "invalid neighbor cell: res={}, cell={}, neighbor={}",
                    res,
                    cell,
                    ncell
                );
                assert!(
                    seen.insert(ncell),
                    "duplicate neighbor cell: res={}, cell={}, neighbor={}, neighbors={:?}",
                    res,
                    cell,
                    ncell,
                    neighbors
                );
            }
            assert!(
                seen.contains(&(cell as u32)),
                "center missing: res={}, cell={}",
                res,
                cell
            );

            let last = res - 1;
            let is_face_corner = (iu == 0 || iu == last) && (iv == 0 || iv == last);
            let expected = if is_face_corner { 8 } else { 9 };
            assert_eq!(
                seen.len(),
                expected,
                "unexpected neighborhood size: res={}, cell={}, face={}, iu={}, iv={}, got={}",
                res,
                cell,
                face,
                iu,
                iv,
                seen.len()
            );
        }
    }
}

#[test]
fn test_ring2_unique_non_empty() {
    for res in [4usize, 5, 8, 16] {
        let grid = CubeMapGrid::new(&[], res);
        let num_cells = 6 * res * res;
        for cell in 0..num_cells {
            let ring2 = grid.cell_ring2(cell);
            assert!(
                !ring2.is_empty(),
                "ring2 is empty: res={}, cell={}",
                res,
                cell
            );
            assert!(
                ring2.len() <= RING2_MAX,
                "ring2 too large: res={}, cell={}, len={}",
                res,
                cell,
                ring2.len()
            );
            let mut seen = std::collections::HashSet::<u32>::with_capacity(ring2.len());
            for &ncell in ring2 {
                assert!(
                    (ncell as usize) < num_cells,
                    "invalid ring2 cell: res={}, cell={}, ring_cell={}",
                    res,
                    cell,
                    ncell
                );
                assert!(
                    seen.insert(ncell),
                    "duplicate ring2 cell: res={}, cell={}, ring_cell={}",
                    res,
                    cell,
                    ncell
                );
            }
        }
    }
}
