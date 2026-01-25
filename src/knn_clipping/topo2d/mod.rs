//! 2D topology builder using gnomonic projection.

pub(crate) mod builder;
pub(crate) mod clippers;
pub mod types;

#[cfg(any(test, feature = "microbench"))]
pub mod microbench;

#[cfg(feature = "microbench")]
pub use microbench::{run_batch_clip_microbench, run_clip_convex_microbench};

// Re-export core types and builder for external use within the crate
pub use builder::Topo2DBuilder;

// Internal re-exports for sibling modules
#[cfg(any(test, feature = "microbench"))]
pub(crate) use clippers::clip_convex_small_bool;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::knn_clipping::topo2d::builder::TangentBasis;
    use glam::{DVec3, Vec3};

    #[test]
    fn test_tangent_basis() {
        let g = DVec3::new(0.0, 0.0, 1.0);
        let basis = TangentBasis::new(g);

        assert!((basis.t1.dot(basis.t2)).abs() < 1e-10);
        assert!((basis.t1.dot(basis.g)).abs() < 1e-10);
        assert!((basis.t2.dot(basis.g)).abs() < 1e-10);
        assert!((basis.t1.length() - 1.0).abs() < 1e-10);
        assert!((basis.t2.length() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_incremental_triangle() {
        let g = Vec3::new(0.0, 0.0, 1.0);
        let mut builder = Topo2DBuilder::new(0, g);

        let h1 = Vec3::new(1.0, 0.0, 0.5).normalize();
        let h2 = Vec3::new(-0.5, 0.866, 0.5).normalize();
        let h3 = Vec3::new(-0.5, -0.866, 0.5).normalize();

        assert!(!builder.is_bounded());

        builder.clip_with_slot(1, u32::MAX, h1).unwrap();
        assert!(!builder.is_bounded());

        builder.clip_with_slot(2, u32::MAX, h2).unwrap();
        assert!(!builder.is_bounded());

        builder.clip_with_slot(3, u32::MAX, h3).unwrap();
        assert!(builder.is_bounded());

        assert!(builder.vertex_count() >= 3);
    }

    #[test]
    fn test_incremental_square() {
        let g = Vec3::new(0.0, 0.0, 1.0);
        let mut builder = Topo2DBuilder::new(0, g);

        let h1 = Vec3::new(1.0, 0.0, 0.5).normalize();
        let h2 = Vec3::new(0.0, 1.0, 0.5).normalize();
        let h3 = Vec3::new(-1.0, 0.0, 0.5).normalize();
        let h4 = Vec3::new(0.0, -1.0, 0.5).normalize();

        builder.clip_with_slot(1, u32::MAX, h1).unwrap();
        builder.clip_with_slot(2, u32::MAX, h2).unwrap();
        builder.clip_with_slot(3, u32::MAX, h3).unwrap();
        builder.clip_with_slot(4, u32::MAX, h4).unwrap();

        assert!(builder.is_bounded());
        assert_eq!(builder.vertex_count(), 4);
    }

    #[test]
    fn test_early_termination_check() {
        let g = Vec3::new(0.0, 0.0, 1.0);
        let mut builder = Topo2DBuilder::new(0, g);

        let h1 = Vec3::new(0.1, 0.0, 0.99).normalize();
        let h2 = Vec3::new(-0.05, 0.087, 0.99).normalize();
        let h3 = Vec3::new(-0.05, -0.087, 0.99).normalize();

        builder.clip_with_slot(1, u32::MAX, h1).unwrap();
        builder.clip_with_slot(2, u32::MAX, h2).unwrap();
        builder.clip_with_slot(3, u32::MAX, h3).unwrap();

        assert!(builder.is_bounded());

        let far_dot = 0.5f32;
        let can_term = builder.can_terminate(far_dot);
        assert!(can_term);
    }

    #[test]
    fn test_to_vertex_data() {
        let g = Vec3::new(0.0, 0.0, 1.0);
        let mut builder = Topo2DBuilder::new(0, g);

        let h1 = Vec3::new(1.0, 0.0, 0.5).normalize();
        let h2 = Vec3::new(-0.5, 0.866, 0.5).normalize();
        let h3 = Vec3::new(-0.5, -0.866, 0.5).normalize();

        builder.clip_with_slot(1, u32::MAX, h1).unwrap();
        builder.clip_with_slot(2, u32::MAX, h2).unwrap();
        builder.clip_with_slot(3, u32::MAX, h3).unwrap();

        let mut vertices = Vec::new();
        let mut edge_neighbors = Vec::new();
        let mut edge_neighbor_slots = Vec::new();
        builder
            .to_vertex_data_full(&mut vertices, &mut edge_neighbors, &mut edge_neighbor_slots)
            .unwrap();

        assert_eq!(vertices.len(), 3);
        for (_key, pos) in &vertices {
            let len = pos.length();
            assert!(
                (len - 1.0).abs() < 1e-5,
                "vertex not on sphere: len={}",
                len
            );
        }
    }
}
