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
            if candle_core::Device::new_cuda(0).is_ok() {
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
            if candle_core::Device::new_metal(0).is_ok() {
                return Self {
                    backend: ComputeBackend::Metal,
                    has_accelerator: true,
                    cpu_threads,
                    vram_free_bytes: None,
                    vram_total_bytes: None,
                    estimated_fp16_tflops: None,
                };
            }
        }

        // Fallback: CPU.
        Self {
            backend: ComputeBackend::Cpu,
            has_accelerator: false,
            vram_free_bytes: None,
            vram_total_bytes: None,
            estimated_fp16_tflops: None,
            cpu_threads,
        }
    }

    /// Best-effort stable string for the host platform (used in the
    /// `NodeHello.platform` field).
    pub fn platform_str() -> &'static str {
        std::env::consts::OS
    }
}

#[cfg(feature = "cuda")]
struct CudaCaps {
    vram_free_bytes: Option<u64>,
    vram_total_bytes: Option<u64>,
    estimated_fp16_tflops: Option<f32>,
}

#[cfg(feature = "cuda")]
fn cuda_caps() -> Option<CudaCaps> {
    use cudarc::driver::CudaDevice;
    let device = CudaDevice::new(0).ok()?;
    let mem = device.mem_get_info().ok();
    let name = device.name().unwrap_or_default();
    Some(CudaCaps {
        vram_free_bytes: mem.as_ref().map(|(f, _)| *f as u64),
        vram_total_bytes: mem.as_ref().map(|(_, t)| *t as u64),
        estimated_fp16_tflops: Some(estimate_tflops(&name)),
    })
}

/// Coarse lookup table. Values are FP16 dense matmul TFLOPS (no
/// sparsity). Replace with a proper micro-benchmark in a follow-up.
#[cfg(feature = "cuda")]
fn estimate_tflops(model_name: &str) -> f32 {
    let upper = model_name.to_ascii_uppercase();
    if upper.contains("H100") {
        989.0
    } else if upper.contains("A100") {
        312.0
    } else if upper.contains("L40S") {
        362.0
    } else if upper.contains("RTX 4090") {
        165.0
    } else if upper.contains("RTX 3090") {
        71.0
    } else if upper.contains("RTX 4080") {
        97.0
    } else if upper.contains("V100") {
        125.0
    } else if upper.contains("T4") {
        65.0
    } else {
        50.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detect_is_cpu_on_unaccelerated_build() {
        let p = HardwareProbe::detect();
        assert_eq!(p.backend, ComputeBackend::Cpu);
        assert!(!p.has_accelerator);
        assert!(p.cpu_threads >= 1);
    }

    #[test]
    fn backend_as_str_is_stable() {
        assert_eq!(ComputeBackend::Cpu.as_str(), "cpu");
        assert_eq!(ComputeBackend::Cuda.as_str(), "cuda");
        assert_eq!(ComputeBackend::Metal.as_str(), "metal");
    }
}
