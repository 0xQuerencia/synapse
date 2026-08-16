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
///  else: `1.0`.
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

