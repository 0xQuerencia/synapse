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
    /// Heuristic FP16 dense matmul TFLOPS. We don't have a portable
    /// micro-benchmark; this is a coarse lookup by GPU model name.
    /// `None` on CPU or when the model is unknown.
    pub estimated_fp16_tflops: Option<f32>,
    /// Number of CPU cores (logical). Always populated; useful for
    /// the planner even on CPU-only boxes.
    pub cpu_threads: u16,
}

impl HardwareProbe {
    /// Detect everything in one call. The cost is one syscall on
    /// each backend (or no syscall on CPU-only builds), so it's fine
    /// to run at startup or on every heartbeat.
    pub fn detect() -> Self {
        let cpu_threads = std::thread::available_parallelism()
            .map(|n| n.get().min(u16::MAX as usize) as u16)
            .unwrap_or(1);

        // Try CUDA first (Linux/Windows). Compile-time gated so
        // CPU-only builds don't pull in the bindings.
        #[cfg(feature = "cuda")]
        {
            if candle_core::utils::cuda_is_available() {
                let caps = cuda_caps();
                return Self {
                    backend: ComputeBackend::Cuda,
                    has_accelerator: true,
                    cpu_threads,
                    vram_free_bytes: caps.as_ref().and_then(|c| c.vram_free_bytes),
                    vram_total_bytes: caps.as_ref().and_then(|c| c.vram_total_bytes),
                    estimated_fp16_tflops: caps.and_then(|c| c.estimated_fp16_tflops),
                };
            }
        }

        // Then Metal (macOS).
        #[cfg(feature = "metal")]
        {
            if candle_core::utils::metal_is_available() {
                return Self {
                    backend: ComputeBackend::Metal,
                    has_accelerator: true,
                    cpu_threads,
                    vram_free_bytes: None,
