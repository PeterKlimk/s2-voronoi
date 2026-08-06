use std::time::{Duration, Instant};

/// Low-frequency wall-clock timer for whole-pipeline boundaries.
pub(crate) struct Timer(Instant);

impl Timer {
    #[inline]
    pub(crate) fn start() -> Self {
        Self(Instant::now())
    }

    #[inline]
    pub(crate) fn elapsed(&self) -> Duration {
        self.0.elapsed()
    }
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct PhaseTimings {
    total: Duration,
    input_validation: Duration,
    preprocess: Duration,
    grid_build: Duration,
    cell_construction: Duration,
    shard_assembly: Duration,
    edge_reconcile: Duration,
    postprocess: Duration,
    output_remap: Duration,
    output_validation: Duration,
}

impl PhaseTimings {
    pub(crate) fn report(&self, n: usize) {
        let ms = |d: Duration| d.as_secs_f64() * 1000.0;
        let pct = |d: Duration| {
            if self.total.is_zero() {
                0.0
            } else {
                100.0 * d.as_secs_f64() / self.total.as_secs_f64()
            }
        };
        eprintln!("timing n={n}");
        for (name, value) in [
            ("input_validation", self.input_validation),
            ("grid_build", self.grid_build),
            ("preprocess", self.preprocess),
            ("cell_construction", self.cell_construction),
            ("shard_assembly", self.shard_assembly),
            ("edge_reconcile", self.edge_reconcile),
            ("postprocess", self.postprocess),
            ("output_remap", self.output_remap),
            ("output_validation", self.output_validation),
        ] {
            eprintln!("  {name:<18} {:7.1}ms ({:4.1}%)", ms(value), pct(value));
        }
        eprintln!("  total              {:7.1}ms", ms(self.total));
        if std::env::var_os("VORONOI_MESH_TIMING_KV").is_some() {
            eprintln!(
                "TIMING_KV n={n} total_ms={:.3} input_validation_ms={:.3} grid_build_ms={:.3} preprocess_ms={:.3} cell_construction_ms={:.3} shard_assembly_ms={:.3} edge_reconcile_ms={:.3} postprocess_ms={:.3} output_remap_ms={:.3} output_validation_ms={:.3}",
                ms(self.total), ms(self.input_validation), ms(self.grid_build), ms(self.preprocess),
                ms(self.cell_construction), ms(self.shard_assembly), ms(self.edge_reconcile),
                ms(self.postprocess), ms(self.output_remap), ms(self.output_validation),
            );
        }
    }
}

pub(crate) struct TimingBuilder {
    started: Instant,
    input_validation: Duration,
    preprocess: Duration,
    grid_build: Duration,
    cell_construction: Duration,
    shard_assembly: Duration,
    edge_reconcile: Duration,
    postprocess: Duration,
    output_remap: Duration,
    output_validation: Duration,
}

impl TimingBuilder {
    pub(crate) fn new() -> Self {
        Self {
            started: Instant::now(),
            input_validation: Duration::ZERO,
            preprocess: Duration::ZERO,
            grid_build: Duration::ZERO,
            cell_construction: Duration::ZERO,
            shard_assembly: Duration::ZERO,
            edge_reconcile: Duration::ZERO,
            postprocess: Duration::ZERO,
            output_remap: Duration::ZERO,
            output_validation: Duration::ZERO,
        }
    }
    pub(crate) fn set_input_validation(&mut self, d: Duration) {
        self.input_validation = d;
    }
    pub(crate) fn set_preprocess(&mut self, d: Duration) {
        self.preprocess = d;
    }
    pub(crate) fn add_grid_build(&mut self, d: Duration) {
        self.grid_build += d;
    }
    pub(crate) fn set_cell_construction(&mut self, d: Duration) {
        self.cell_construction = d;
    }
    pub(crate) fn set_shard_assembly(&mut self, d: Duration) {
        self.shard_assembly = d;
    }
    pub(crate) fn set_edge_reconcile(&mut self, d: Duration) {
        self.edge_reconcile = d;
    }
    pub(crate) fn set_postprocess(&mut self, d: Duration) {
        self.postprocess = d;
    }
    pub(crate) fn set_output_remap(&mut self, d: Duration) {
        self.output_remap = d;
    }
    pub(crate) fn set_output_validation(&mut self, d: Duration) {
        self.output_validation = d;
    }
    pub(crate) fn finish(self) -> PhaseTimings {
        PhaseTimings {
            total: self.started.elapsed(),
            input_validation: self.input_validation,
            preprocess: self.preprocess,
            grid_build: self.grid_build,
            cell_construction: self.cell_construction,
            shard_assembly: self.shard_assembly,
            edge_reconcile: self.edge_reconcile,
            postprocess: self.postprocess,
            output_remap: self.output_remap,
            output_validation: self.output_validation,
        }
    }
}
