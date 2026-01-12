use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use s2_voronoi::UnitVec3;

/// Generate random points uniformly distributed on the unit sphere.
pub fn random_sphere_points(n: usize, seed: u64) -> Vec<UnitVec3> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    random_sphere_points_with_rng(n, &mut rng)
}

pub fn random_sphere_points_with_rng<R: Rng + ?Sized>(n: usize, rng: &mut R) -> Vec<UnitVec3> {
    use std::f32::consts::PI;
    (0..n)
        .map(|_| {
            let z: f32 = rng.gen_range(-1.0..1.0);
            let theta: f32 = rng.gen_range(0.0..2.0 * PI);
            let r = (1.0 - z * z).sqrt();
            UnitVec3::new(r * theta.cos(), r * theta.sin(), z)
        })
        .collect()
}

/// Generate Fibonacci sphere points (more uniform than random).
pub fn fibonacci_sphere_points(n: usize, jitter: f32, seed: u64) -> Vec<UnitVec3> {
    use std::f32::consts::PI;
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let golden_angle = PI * (3.0 - 5.0f32.sqrt());

    (0..n)
        .map(|i| {
            let y = 1.0 - (2.0 * i as f32 + 1.0) / n as f32;
            let radius = (1.0 - y * y).sqrt();
            let theta = golden_angle * i as f32;

            let mut x = radius * theta.cos();
            let mut z = radius * theta.sin();

            if jitter > 0.0 {
                x += rng.gen_range(-jitter..jitter);
                z += rng.gen_range(-jitter..jitter);
            }

            let len = (x * x + y * y + z * z).sqrt();
            UnitVec3::new(x / len, y / len, z / len)
        })
        .collect()
}
