//! Measured compute scores remain independent from model-specific memory admission.

use crate::hardware::HardwareProbe;
use crate::protocol::MAX_COMPUTE_SCORE;

/// Compatibility constant for the legacy planner. Runtime admission uses
/// `ModelMemoryProfile` and never estimates layers from this value.
pub const MODEL_LAYER_FOOTPRINT_MB: u64 = 50;

/// Preserve actual measured work without a second GPU or RAM multiplier.
/// Zero indicates an unavailable probe or unknown memory and is ineligible.
pub fn blend_compute_score(measured: u64, ram_free_mb: u64, _probe: &HardwareProbe) -> u64 {
    // Measured work already reflects accelerator throughput. Multiplying it
    // by a marketing TFLOPS table counts GPU performance twice. RAM is an
    // independent admission constraint; unknown memory is not a speed bonus.
    if measured == 0 || ram_free_mb == 0 {
        0
    } else {
        measured.min(MAX_COMPUTE_SCORE)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hardware::{ComputeBackend, HardwareProbe};

    fn cpu_probe(threads: u16) -> HardwareProbe {
        HardwareProbe {
            backend: ComputeBackend::Cpu,
            has_accelerator: false,
            vram_free_bytes: None,
            vram_total_bytes: None,
            estimated_fp16_tflops: None,
            cpu_threads: threads,
        }
    }

    fn gpu_probe(tflops: f32) -> HardwareProbe {
        HardwareProbe {
            backend: ComputeBackend::Cuda,
            has_accelerator: true,
            vram_free_bytes: Some(80 * 1024 * 1024 * 1024),
            vram_total_bytes: Some(80 * 1024 * 1024 * 1024),
            estimated_fp16_tflops: Some(tflops),
            cpu_threads: 1,
        }
    }

    #[test]
    fn clamp_to_minimum() {
        let score = blend_compute_score(0, 0, &cpu_probe(1));
        assert_eq!(score, 0);
    }

    #[test]
    fn clamp_to_maximum() {
        let probe = gpu_probe(989.0); // H100
        let score = blend_compute_score(9_999_999, 1_000_000, &probe);
        assert_eq!(score, MAX_COMPUTE_SCORE);
    }

    #[test]
    fn bigger_measured_yields_larger_score() {
        let probe = cpu_probe(4);
        let low = blend_compute_score(50, 16_000, &probe);
        let high = blend_compute_score(200, 16_000, &probe);
        assert!(high > low);
    }

    #[test]
    fn memory_capacity_does_not_fabricate_throughput() {
        let probe = cpu_probe(4);
        let low = blend_compute_score(100, 100, &probe); // < 200 MB → penalty
        let high = blend_compute_score(100, 16_000, &probe);
        assert_eq!(high, low);
    }

    #[test]
    fn identical_measured_work_has_no_gpu_marketing_bonus() {
        let cpu = cpu_probe(16);
        let gpu = gpu_probe(50.0); // modest GPU
        let cpu_score = blend_compute_score(100, 16_000, &cpu);
        let gpu_score = blend_compute_score(100, 16_000, &gpu);
        assert_eq!(gpu_score, cpu_score);
    }

    #[test]
    fn unknown_gpu_tflops_does_not_invent_a_bonus() {
        let gpu_unknown = HardwareProbe {
            backend: ComputeBackend::Metal,
            has_accelerator: true,
            vram_free_bytes: None,
            vram_total_bytes: None,
            estimated_fp16_tflops: None,
            cpu_threads: 8,
        };
        let cpu = cpu_probe(8);
        let gpu_score = blend_compute_score(100, 16_000, &gpu_unknown);
        let cpu_score = blend_compute_score(100, 16_000, &cpu);
        assert_eq!(gpu_score, cpu_score);
    }

    #[test]
    fn capacity_is_checked_by_the_model_memory_profile() {
        let probe = cpu_probe(4);
        let starved = blend_compute_score(100, 50, &probe); // < 200 MB
        let well_fed = blend_compute_score(100, 16_000, &probe);
        assert_eq!(starved, well_fed);
    }

    #[test]
    fn deterministic() {
        let probe = cpu_probe(8);
        let a = blend_compute_score(42, 8_192, &probe);
        let b = blend_compute_score(42, 8_192, &probe);
        assert_eq!(a, b);
    }
}
