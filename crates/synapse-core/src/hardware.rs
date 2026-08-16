//! Runtime detection of compute backends.
//!
//! The same code works whether the binary was built with
//! `--features cuda`, `--features metal`, or no accelerator at all —
//! the feature gates the binding; detection just looks at runtime.
//!
//! On a CPU-only build the probe always reports `ComputeBackend::Cpu`
//! with `has_accelerator = false`. On a GPU-enabled build it picks
//! CUDA first (Linux/Windows), then Metal (macOS), and falls back
//! to CPU if neither is actually usable at runtime.

use serde::{Deserialize, Serialize};

/// Which compute backend the host will use for inference.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ComputeBackend {
    Cpu,
    Cuda,
    Metal,
}

impl ComputeBackend {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Cpu => "cpu",
            Self::Cuda => "cuda",
            Self::Metal => "metal",
        }
    }
}

/// Result of probing the local host for accelerators.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HardwareProbe {
    pub backend: ComputeBackend,
    /// True iff candle found at least one usable device for `backend`.
    pub has_accelerator: bool,
    /// Free VRAM in bytes. `None` on CPU. For CUDA we read via
    /// the driver; for Metal we don't have a portable API, so we
    /// leave this `None` and rely on the system-RAM estimate.
    pub vram_free_bytes: Option<u64>,
    /// Total VRAM in bytes (if known).
    pub vram_total_bytes: Option<u64>,
