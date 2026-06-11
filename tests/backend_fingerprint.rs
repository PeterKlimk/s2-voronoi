//! Diagnostic: print a structural fingerprint of a computed diagram so SIMD
//! backends can be compared for bit-identical output.

mod support;

use s2_voronoi::compute;
use support::points::random_sphere_points;

#[test]
#[ignore = "cross-backend determinism check - run manually"]
fn backend_fingerprint() {
    let points = random_sphere_points(100_000, 7);
    let diagram = compute(&points).unwrap();

    let mut h: u64 = 0xcbf29ce484222325;
    let mut mix = |v: u64| {
        h ^= v;
        h = h.wrapping_mul(0x100000001b3);
    };
    for v in diagram.vertices() {
        mix(v.x.to_bits() as u64);
        mix(v.y.to_bits() as u64);
        mix(v.z.to_bits() as u64);
    }
    for cell in diagram.iter_cells() {
        for &i in cell.vertex_indices {
            mix(i as u64);
        }
    }
    eprintln!(
        "fingerprint: {:016x} vertices={} cells={}",
        h,
        diagram.num_vertices(),
        diagram.num_cells()
    );
}
