//! Model-shaped, checked inference reservations. These are tensor allocations,
//! not whole-process RSS: OS/runtime headroom belongs to the process policy.
use super::pipeline::{TensorRoles, ToyLlamaConfig};
use anyhow::{Result, ensure};
use serde::Serialize;

#[derive(Clone, Debug, Serialize)]
pub struct MemoryEstimate {
    pub weights_bytes: u64,
    pub kv_bytes: u64,
    pub scratch_bytes: u64,
    pub load_scratch_bytes: u64,
    pub total_bytes: u64,
}

#[derive(Clone, Debug)]
pub struct ModelMemoryProfile {
    config: ToyLlamaConfig,
    dtype_bytes: u64,
}

fn product(values: &[u64]) -> Result<u64> {
    values.iter().try_fold(1u64, |a, b| {
        a.checked_mul(*b)
            .ok_or_else(|| anyhow::anyhow!("memory calculation overflow"))
    })
}
fn sum(values: &[u64]) -> Result<u64> {
    values.iter().try_fold(0u64, |a, b| {
        a.checked_add(*b)
            .ok_or_else(|| anyhow::anyhow!("memory calculation overflow"))
    })
}
impl ModelMemoryProfile {
    /// GGUF uses an intentionally conservative F32 weight ceiling because
    /// Candle may dequantize tensors and the configured quantization can vary
    /// per tensor. No optimistic advertised bit-width is used for admission.
    pub fn new(config: &ToyLlamaConfig, gguf: bool) -> Result<Self> {
        config.validate()?;
        let dtype_bytes = if gguf || config.torch_dtype == "float32" {
            4
        } else {
            2
        };
        Ok(Self {
            config: config.clone(),
            dtype_bytes,
        })
    }

    pub fn estimate(
        &self,
        layers: usize,
        context_tokens: usize,
        roles: TensorRoles,
    ) -> Result<MemoryEstimate> {
        let c = &self.config;
        ensure!(layers <= c.num_hidden_layers, "layer count exceeds model");
        ensure!(
            context_tokens > 0 && context_tokens <= c.context_limit(),
            "context outside model limits"
        );
        let (h, f, v, n, t, d) = (
            c.hidden_size as u64,
            c.intermediate_size as u64,
            c.vocab_size as u64,
            layers as u64,
            context_tokens as u64,
            self.dtype_bytes,
        );
        let hd = h / c.num_attention_heads as u64;
        let kv_width = product(&[c.num_key_value_heads as u64, hd])?;
        let layer_elements = sum(&[
            product(&[2, h, h])?,
            product(&[2, h, kv_width])?,
            product(&[3, h, f])?,
            product(&[2, h])?,
        ])?;
        let layer_weights = product(&[layer_elements, d])?;
        let embedding = if roles.input { product(&[h, v, d])? } else { 0 };
        let output = if roles.output {
            product(&[h, v, d])?
                .checked_add(h * d)
                .ok_or_else(|| anyhow::anyhow!("head size overflow"))?
        } else {
            0
        };
        // RoPE stores cos and sin in F32, each [context_limit, head_dim/2].
        let rope = product(&[c.context_limit() as u64, hd, 4])?;
        let weights_bytes = sum(&[product(&[n, layer_weights])?, embedding, output, rope])?;
        let kv_bytes = product(&[2, n, t, kv_width, d])?;
        let wire_and_hidden = product(&[t, h, 4, 4])?; // receive, decode, outgoing, residual
        let projection_scratch = if n > 0 {
            sum(&[
                product(&[t, h, d, 12])?,
                product(&[t, f, d, 4])?,
                product(&[t, t, c.num_attention_heads as u64, 4, 4])?,
                product(&[t, h, 4, 4])?, // RMS promotion and GQA materialization
                product(&[2, t, kv_width, d])?, // append reallocates one layer's KV
            ])?
        } else {
            0
        };
        let head_scratch = if roles.output {
            product(&[v, 4, 2])?
        } else {
            0
        };
        let scratch_bytes = sum(&[wire_and_hidden, projection_scratch, head_scratch])?;
        // A transposed/dequantized projection can coexist with its source
        // while loading. Budget one entire layer, or the largest IO tensor.
        let load_scratch_bytes = if n > 0 { layer_weights } else { 0 }
            .max(embedding)
            .max(output);
        let runtime = sum(&[kv_bytes, scratch_bytes])?;
        let total_bytes = sum(&[weights_bytes, runtime.max(load_scratch_bytes)])?;
        Ok(MemoryEstimate {
            weights_bytes,
            kv_bytes,
            scratch_bytes,
            load_scratch_bytes,
            total_bytes,
        })
    }

    /// Largest feasible layer count, including mandatory IO for a coordinator.
    /// Unknown/zero budgets fail closed; no fallback or MB rounding enters this.
    pub fn layer_capacity(
        &self,
        budget_bytes: u64,
        context_tokens: usize,
        roles: TensorRoles,
    ) -> Result<usize> {
        ensure!(budget_bytes > 0, "memory capacity is unknown");
        ensure!(
            self.estimate(0, context_tokens, roles)?.total_bytes <= budget_bytes,
            "memory cannot fit required IO and activation buffers"
        );
        let mut low = 0;
        let mut high = self.config.num_hidden_layers;
        while low < high {
            let mid = (low + high).div_ceil(2);
            if self.estimate(mid, context_tokens, roles)?.total_bytes <= budget_bytes {
                low = mid;
            } else {
                high = mid - 1;
            }
        }
        Ok(low)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::{ShardedModel, ToyLlama};
    fn profile() -> (
        ModelMemoryProfile,
        crate::inference::model_loader::ModelFiles,
        ToyLlamaConfig,
    ) {
        let files = crate::inference::model_loader::test_model_files().unwrap();
        let config = ToyLlamaConfig::from_file(&files.config).unwrap();
        (
            ModelMemoryProfile::new(&config, false).unwrap(),
            files,
            config,
        )
    }
    #[test]
    fn weights_match_real_tensor_residency_for_every_fixture_role_and_range() {
        let (profile, files, c) = profile();
        for roles in [TensorRoles::COORDINATOR, TensorRoles::WORKER] {
            for n in 0..=c.num_hidden_layers {
                let model =
                    ToyLlama::load_range_with_roles(&c, &files.weights, 0..n, roles).unwrap();
                let e = profile.estimate(n, 11, roles).unwrap();
                assert_eq!(e.weights_bytes, model.resident_weight_bytes());
                assert!(e.total_bytes >= model.resident_weight_bytes() + e.kv_bytes);
                if n > 0 {
                    let input = candle_core::Tensor::zeros(
                        (1, 3, c.hidden_size),
                        candle_core::DType::F32,
                        model.model_device(),
                    )
                    .unwrap();
                    model.forward_layers_prefill(&input, 0..n).unwrap();
                    assert_eq!(
                        model.layer_kv_cache(0).unwrap().sequence_len().unwrap(),
                        Some(3)
                    );
                }
            }
        }
    }
    #[test]
    fn exact_sub_megabyte_budget_admits_one_layer_and_rejects_two() {
        let (p, _, _) = profile();
        let one = p.estimate(1, 11, TensorRoles::WORKER).unwrap().total_bytes;
        assert!(one < 1024 * 1024);
        assert_eq!(p.layer_capacity(one, 11, TensorRoles::WORKER).unwrap(), 1);
        assert_eq!(
            p.layer_capacity(one - 1, 11, TensorRoles::WORKER).unwrap(),
            0
        );
        assert!(p.layer_capacity(0, 11, TensorRoles::WORKER).is_err());
        assert!(p.estimate(1, 129, TensorRoles::WORKER).is_err());
    }
    #[test]
    fn kv_accounts_gqa_context_dtype_and_layer_count() {
        let (p, _, _) = profile();
        let a = p.estimate(1, 10, TensorRoles::WORKER).unwrap();
        let b = p.estimate(2, 20, TensorRoles::WORKER).unwrap();
        assert_eq!(a.kv_bytes, 2 * 10 * 2 * 4 * 4);
        assert_eq!(b.kv_bytes, 4 * a.kv_bytes);
        assert!(b.total_bytes > a.total_bytes);
    }
}
