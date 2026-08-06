//! Low-distortion wall-clock timing at whole-pipeline boundaries.
//!
//! Fine hot-path attribution belongs to sampling profilers. Algorithmic work
//! counters live separately in `crate::telemetry`.

#[cfg(feature = "timing")]
mod real;
#[cfg(not(feature = "timing"))]
mod stub;

#[cfg(feature = "timing")]
pub(crate) use real::*;
#[cfg(not(feature = "timing"))]
pub(crate) use stub::*;
