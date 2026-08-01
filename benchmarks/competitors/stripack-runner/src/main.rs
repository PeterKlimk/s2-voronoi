use std::fs;
use std::hint::black_box;
use std::time::Instant;
use stripack::DelaunayTriangulation;

fn mix(hash: u64, value: u64) -> u64 {
    hash ^ value
        .wrapping_add(0x9e37_79b9_7f4a_7c15)
        .wrapping_add(hash << 6)
        .wrapping_add(hash >> 2)
}

fn read_points(path: &str) -> Result<(Vec<f64>, Vec<f64>, Vec<f64>), String> {
    let bytes = fs::read(path).map_err(|error| error.to_string())?;
    if bytes.is_empty() || bytes.len() % 12 != 0 {
        return Err("input must contain packed little-endian f32 xyz triples".into());
    }
    let n = bytes.len() / 12;
    let mut x = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);
    let mut z = Vec::with_capacity(n);
    for chunk in bytes.chunks_exact(12) {
        let px = f32::from_le_bytes(chunk[0..4].try_into().unwrap()) as f64;
        let py = f32::from_le_bytes(chunk[4..8].try_into().unwrap()) as f64;
        let pz = f32::from_le_bytes(chunk[8..12].try_into().unwrap()) as f64;
        // stripack 0.1.2 rejects unit vectors whose f64 norm differs by more
        // than 1e-10. Packed f32 inputs cannot generally meet that threshold.
        let inverse_norm = (px * px + py * py + pz * pz).sqrt().recip();
        x.push(px * inverse_norm);
        y.push(py * inverse_norm);
        z.push(pz * inverse_norm);
    }
    Ok((x, y, z))
}

fn main() -> Result<(), String> {
    let mut arguments = std::env::args().skip(1);
    let input = arguments
        .next()
        .ok_or("usage: bench-stripack-sphere INPUT.f32 [--repeat N]")?;
    let mut repeat = 1usize;
    let mut construct_only = false;
    while let Some(flag) = arguments.next() {
        match flag.as_str() {
            "--repeat" => {
                repeat = arguments
                    .next()
                    .ok_or("missing repeat count")?
                    .parse()
                    .map_err(|_| "invalid repeat count")?;
            }
            "--construct-only" => construct_only = true,
            _ => return Err("expected --repeat N or --construct-only".into()),
        }
    }
    if repeat == 0 {
        return Err(
            "usage: bench-stripack-sphere INPUT.f32 [--repeat N] [--construct-only]".into(),
        );
    }

    let (base_x, base_y, base_z) = read_points(&input)?;
    for iteration in 1..=repeat {
        // Clone outside the timed region: it is an ownership-adapter cost, not
        // part of STRIPACK's triangulation algorithm.
        let x = base_x.clone();
        let y = base_y.clone();
        let z = base_z.clone();
        let start = Instant::now();
        let mut triangulation = DelaunayTriangulation::new(x, y, z)
            .map_err(|error| format!("STRIPACK construction failed: {error}"))?;
        let construct_ms = start.elapsed().as_secs_f64() * 1_000.0;

        let materialize_start = Instant::now();
        if construct_only {
            let boundary = triangulation.boundary_nodes();
            let checksum = mix(
                mix(0xa409_3822_299f_31d0_u64, boundary.num_triangles as u64),
                boundary.num_arcs as u64,
            );
            black_box(checksum);
            let materialize_ms = materialize_start.elapsed().as_secs_f64() * 1_000.0;
            println!(
                "RESULT backend=stripack-construct n={} iteration={} construct_ms={:.6} materialize_ms={:.6} total_ms={:.6} vertices={} cells={} incidences={} checksum={:x}",
                base_x.len(),
                iteration,
                construct_ms,
                materialize_ms,
                construct_ms + materialize_ms,
                boundary.num_triangles,
                base_x.len(),
                boundary.num_triangles * 3,
                checksum,
            );
            continue;
        }
        let mesh = triangulation
            .triangle_mesh()
            .map_err(|error| format!("STRIPACK mesh extraction failed: {error}"))?;
        let voronoi = triangulation
            .voronoi_cells()
            .map_err(|error| format!("STRIPACK Voronoi extraction failed: {error}"))?;
        let mut checksum = 0xa409_3822_299f_31d0_u64;
        for &index in &mesh.indices {
            checksum = mix(checksum, index as u64);
        }
        for cell in &voronoi {
            checksum = mix(checksum, cell.position[0].to_bits());
            checksum = mix(checksum, cell.position[1].to_bits());
            checksum = mix(checksum, cell.position[2].to_bits());
        }
        black_box(checksum);
        let materialize_ms = materialize_start.elapsed().as_secs_f64() * 1_000.0;
        println!(
            "RESULT backend=stripack n={} iteration={} construct_ms={:.6} materialize_ms={:.6} total_ms={:.6} vertices={} cells={} incidences={} checksum={:x}",
            base_x.len(),
            iteration,
            construct_ms,
            materialize_ms,
            construct_ms + materialize_ms,
            voronoi.len(),
            base_x.len(),
            mesh.indices.len(),
            checksum,
        );
    }
    Ok(())
}
