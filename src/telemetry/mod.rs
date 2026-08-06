//! Optional algorithmic work telemetry.
//!
//! These counters describe workload shape and resource use. They are kept
//! separate from wall-clock timing because they intentionally instrument hot
//! loops and are not suitable for performance attribution.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum KnnCellStage {
    PackedChunk0,
    PackedTail,
    ShellExpand,
}

#[cfg(feature = "telemetry")]
mod real;
#[cfg(not(feature = "telemetry"))]
mod stub;
#[cfg(feature = "telemetry")]
pub(crate) use real::*;
#[cfg(not(feature = "telemetry"))]
pub(crate) use stub::*;
