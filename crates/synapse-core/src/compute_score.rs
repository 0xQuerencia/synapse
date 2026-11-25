//! Blended compute-score function that combines a real micro-benchmark
//! with available RAM and GPU presence into a single planner-friendly
//! `compute_score`.
//!
//! The output feeds directly into `plan_weighted_shards`, which divides
//! layers proportionally by score. A higher score = more layers assigned.
//!
//! Design: prima.cpp-style blend — real device throughput, RAM ceiling,
//! and a GPU bonus, all clamped to `[1, MAX_COMPUTE_SCORE]`.

use crate::hardware::HardwareProbe;
use crate::protocol::MAX_COMPUTE_SCORE;

/// How many MiB of free RAM we assume one model layer needs including
/// weights, KV cache, and intermediate activations. SmolLM2-135M layers
/// are ~1.5 MB each (weights only), but KV cache can grow to several MB
/// for long sequences, so we budget 50 MB per layer. The `ram_factor`
/// uses `4 * MODEL_LAYER_FOOTPRINT_MB = 200 MB` as the floor — below
/// that the device gets a proportional penalty to avoid over-assignment.
pub const MODEL_LAYER_FOOTPRINT_MB: u64 = 50;

/// Blend a measured micro-benchmark score with available RAM and GPU
/// presence into a planner-friendly `compute_score`.
///
/// # Formula
///
/// 1. `ram_factor = clamp(ram_free_mb / (MODEL_LAYER_FOOTPRINT_MB * 4), 0.1, 1.0)`
///    A device with < 200 MB free RAM is penalised proportionally.
///
/// 2. `gpu_factor` = if GPU present:
///    `max(estimated_fp16_tflops / 10.0, 1.5)` — a slow GPU still
///    gets a 1.5× bonus over CPU; a 4090 (165 TFLOPS) gets ~16.5×.
///    else: `1.0`.
///
/// 3. `blended = measured * ram_factor * gpu_factor`
///
/// 4. Clamp to `[1, MAX_COMPUTE_SCORE]`.
///
/// # Determinism
///
/// The function is pure: same inputs → same output. The caller is
/// responsible for providing a real measured score from
/// `run_micro_benchmark`. Unit-testable.
pub fn blend_compute_score(measured: u64, ram_free_mb: u64, probe: &HardwareProbe) -> u64 {
    let ram_factor = {
        let min_ram = MODEL_LAYER_FOOTPRINT_MB.saturating_mul(4); // 200 MB
        if ram_free_mb >= min_ram {
            1.0_f64
        } else if ram_free_mb == 0 {
            0.1_f64
        } else {
            (ram_free_mb as f64 / min_ram as f64).max(0.1)
        }
    };

    let gpu_factor: f64 = if probe.has_accelerator {
        probe
            .estimated_fp16_tflops
            .map(|tflops| ((tflops as f64) / 10.0).max(1.5))
            .unwrap_or(2.0) // GPU with unknown TFLOPS → 2× bonus
    } else {
        1.0
    };

    let blended = (measured as f64) * ram_factor * gpu_factor;
    (blended.round() as u64).clamp(1, MAX_COMPUTE_SCORE)
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
        assert_eq!(score, 1);
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
    fn more_ram_yields_larger_score() {
        let probe = cpu_probe(4);
        let low = blend_compute_score(100, 100, &probe); // < 200 MB → penalty
        let high = blend_compute_score(100, 16_000, &probe);
        assert!(high > low);
    }

    #[test]
    fn gpu_scores_higher_than_cpu_with_equal_measured() {
        let cpu = cpu_probe(16);
        let gpu = gpu_probe(50.0); // modest GPU
        let cpu_score = blend_compute_score(100, 16_000, &cpu);
        let gpu_score = blend_compute_score(100, 16_000, &gpu);
        assert!(
            gpu_score > cpu_score,
            "GPU score {gpu_score} should exceed CPU score {cpu_score}"
        );
    }

    #[test]
    fn gpu_with_unknown_tflops_still_gets_bonus() {
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
        assert!(gpu_score > cpu_score);
    }

    #[test]
    fn ram_starvation_penalty_applies() {
        let probe = cpu_probe(4);
        let starved = blend_compute_score(100, 50, &probe); // < 200 MB
        let well_fed = blend_compute_score(100, 16_000, &probe);
        assert!(starved < well_fed);
    }

    #[test]
    fn deterministic() {
        let probe = cpu_probe(8);
        let a = blend_compute_score(42, 8_192, &probe);
        let b = blend_compute_score(42, 8_192, &probe);
        assert_eq!(a, b);
    }
}
