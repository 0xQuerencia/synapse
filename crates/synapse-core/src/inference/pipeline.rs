//! Llama-family pipeline for sharded inference.
//!
//! Supports both tiny-random-LlamaForCausalLM (testing) and SmolLM2-135M-Instruct
//! (real demo). Implements causal self-attention with RoPE and GQA (Grouped-Query
//! Attention) plus the SwiGLU MLP.
//!
//! Architecture fields read from `config.json`:
//!   hidden_size, intermediate_size, num_hidden_layers, vocab_size,
//!   num_attention_heads, num_key_value_heads, rms_norm_eps,
//!   tie_word_embeddings, max_position_embeddings, rope_theta

use std::fs::File;
use std::ops::Range;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::Instant;

use anyhow::{Context, Result, anyhow, bail};
use candle_core::quantized::{QMatMul, gguf_file};
use candle_core::{DType, Device, Module, Tensor};
use candle_nn::VarBuilder;
use rand::{Rng, SeedableRng, rngs::StdRng};
use serde::Deserialize;
use tracing::{info, trace, warn};

// ─── Config ──────────────────────────────────────────────────────────────────

/// Configuration parsed from the model's `config.json`.
#[derive(Debug, Clone, Deserialize)]
pub struct ToyLlamaConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub vocab_size: usize,
    pub rms_norm_eps: f64,
    #[serde(default = "default_num_heads")]
    pub num_attention_heads: usize,
    #[serde(default = "default_num_heads")]
    pub num_key_value_heads: usize,
    #[serde(default)]
    pub tie_word_embeddings: bool,
    #[serde(default = "default_max_pos")]
    pub max_position_embeddings: usize,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f64,
    /// Declared weight dtype (`torch_dtype` in config.json). Real checkpoints
    /// store weights as `bfloat16`/`float16`; we honor that at load time so
    /// large models don't double in RAM when forced to f32.
    #[serde(default = "default_torch_dtype", rename = "torch_dtype")]
    pub torch_dtype: String,
}

fn default_torch_dtype() -> String {
    "float32".to_owned()
}

fn default_num_heads() -> usize {
    1
}
fn default_max_pos() -> usize {
    2048
}
fn default_rope_theta() -> f64 {
    10000.0
}

impl ToyLlamaConfig {
    /// Load config from a `config.json` file.
    pub fn from_file(path: &Path) -> Result<Self> {
        let content = std::fs::read_to_string(path).context("failed to read config.json")?;
        let config: Self = serde_json::from_str(&content).context("failed to parse config.json")?;
        info!(
            hidden_size = config.hidden_size,
            num_layers = config.num_hidden_layers,
            vocab_size = config.vocab_size,
            num_heads = config.num_attention_heads,
            num_kv_heads = config.num_key_value_heads,
            tie_embeddings = config.tie_word_embeddings,
            "loaded model config"
        );
        Ok(config)
    }

    fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }
}

/// Select the best available inference device in priority order:
/// CUDA → Metal → CPU.
pub fn get_best_device() -> Result<Device> {
    if candle_core::utils::cuda_is_available() {
        match Device::new_cuda(0) {
            Ok(device) => {
                info!("using CUDA device 0 for inference");
                return Ok(device);
            }
            Err(error) => {
                warn!(%error, "CUDA backend detected but initialization failed, falling back");
            }
        }
    }

    if candle_core::utils::metal_is_available() {
        match Device::new_metal(0) {
            Ok(device) => {
                info!("using Metal device 0 for inference");
                return Ok(device);
            }
            Err(error) => {
                warn!(%error, "Metal backend detected but initialization failed, falling back");
            }
        }
    }

    info!("using CPU for inference");
    Ok(Device::Cpu)
}

fn micro_benchmark_score(elapsed_ms: f64) -> u64 {
    let safe_elapsed_ms = elapsed_ms.max(0.001);
    (10_000.0 / safe_elapsed_ms).max(1.0).round() as u64
}

/// Run a deterministic micro-benchmark for a device and return its relative score.
pub fn run_micro_benchmark(device: &Device) -> u64 {
    const MATRIX_DIM: usize = 512;
    const BENCHMARK_RUNS: usize = 5;
    const WARMUP_RUNS: usize = 1;
    const RNG_SEED: u64 = 0x5EED_CAFE;

    let element_count = MATRIX_DIM * MATRIX_DIM;
    let mut rng = StdRng::seed_from_u64(RNG_SEED);
    let lhs_data: Vec<f32> = (0..element_count)
        .map(|_| rng.gen_range(-1.0_f32..1.0_f32))
        .collect();
    let rhs_data: Vec<f32> = (0..element_count)
        .map(|_| rng.gen_range(-1.0_f32..1.0_f32))
        .collect();

    let lhs = match Tensor::new(lhs_data.as_slice(), device)
        .and_then(|t| t.reshape((MATRIX_DIM, MATRIX_DIM)))
    {
        Ok(tensor) => tensor,
        Err(error) => {
            warn!(%error, "failed to build lhs tensor for micro-benchmark");
            return 1;
        }
    };
    let rhs = match Tensor::new(rhs_data.as_slice(), device)
        .and_then(|t| t.reshape((MATRIX_DIM, MATRIX_DIM)))
    {
        Ok(tensor) => tensor,
        Err(error) => {
            warn!(%error, "failed to build rhs tensor for micro-benchmark");
            return 1;
        }
    };

    if let Err(error) = device.synchronize() {
        warn!(%error, "failed to synchronize device before micro-benchmark");
        return 1;
    }

    let mut samples_ms = Vec::with_capacity(BENCHMARK_RUNS);
    for run in 0..(WARMUP_RUNS + BENCHMARK_RUNS) {
        let started = Instant::now();
        let matmul = match lhs.broadcast_matmul(&rhs) {
            Ok(output) => output,
            Err(error) => {
                warn!(%error, "micro-benchmark matrix multiplication failed");
                return 1;
            }
        };
        if let Err(error) = matmul.sum_all().and_then(|sum| sum.to_scalar::<f32>()) {
            warn!(%error, "micro-benchmark reduction failed");
            return 1;
        }
        if let Err(error) = device.synchronize() {
            warn!(%error, "failed to synchronize device during micro-benchmark");
            return 1;
        }

        if run >= WARMUP_RUNS {
            samples_ms.push(started.elapsed().as_secs_f64() * 1_000.0);
        }
    }

    samples_ms.sort_by(|a, b| a.total_cmp(b));
    let median_ms = samples_ms[samples_ms.len() / 2];
    micro_benchmark_score(median_ms)
}

// ─── Building blocks ─────────────────────────────────────────────────────────

/// Minimal RMS normalization.
struct RmsNorm {
    weight: Tensor,
    eps: f64,
}

impl RmsNorm {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x_sq = x.sqr()?;
        let mean = x_sq.mean_keepdim(candle_core::D::Minus1)?;
        let rms = (mean + self.eps)?.sqrt()?;
        let normed = x.broadcast_div(&rms)?;
        let out = normed.broadcast_mul(&self.weight)?;
        Ok(out)
    }
}

/// Precomputed RoPE frequency table.
struct RotaryEmbedding {
    cos: Tensor,
    sin: Tensor,
}

impl RotaryEmbedding {
    fn new(head_dim: usize, max_seq_len: usize, theta: f64, device: &Device) -> Result<Self> {
        let half = head_dim / 2;
        let mut inv_freq = Vec::with_capacity(half);
        for i in 0..half {
            inv_freq.push(1.0_f32 / (theta as f32).powf(2.0 * i as f32 / head_dim as f32));
        }
        let inv_freq = Tensor::new(inv_freq.as_slice(), device)?; // [half]

        let positions: Vec<f32> = (0..max_seq_len).map(|p| p as f32).collect();
        let positions = Tensor::new(positions.as_slice(), device)?.unsqueeze(1)?; // [seq, 1]
        let inv_freq = inv_freq.unsqueeze(0)?; // [1, half]

        let freqs = positions.broadcast_matmul(&inv_freq)?; // [seq, half]
        let cos = freqs.cos()?;
        let sin = freqs.sin()?;
        Ok(Self { cos, sin })
    }

    /// Apply RoPE for positions `[0, seq_len)`.
    fn apply(&self, x: &Tensor, seq_len: usize) -> Result<Tensor> {
        self.apply_with_offset(x, 0, seq_len)
    }

    /// Apply RoPE for positions `[position_offset, position_offset + seq_len)`.
    fn apply_with_offset(
        &self,
        x: &Tensor,
        position_offset: usize,
        seq_len: usize,
    ) -> Result<Tensor> {
        let head_dim = x.dim(3)?;
        let half = head_dim / 2;
        let rope_len = self.cos.dim(0)?;
        if position_offset + seq_len > rope_len {
            bail!(
                "RoPE position range {}..{} exceeds precomputed limit {}",
                position_offset,
                position_offset + seq_len,
                rope_len
            );
        }

        // Split into two halves
        let x1 = x.narrow(3, 0, half)?;
        let x2 = x.narrow(3, half, half)?;

        // Get cos/sin for this sequence length: [seq_len, half]
        let cos = self.cos.narrow(0, position_offset, seq_len)?;
        let sin = self.sin.narrow(0, position_offset, seq_len)?;

        // Reshape for broadcasting: [1, 1, seq_len, half]
        let cos = cos.unsqueeze(0)?.unsqueeze(0)?;
        let sin = sin.unsqueeze(0)?.unsqueeze(0)?;

        // Apply rotation: (x1*cos - x2*sin, x1*sin + x2*cos)
        let out1 = (x1.broadcast_mul(&cos)? - x2.broadcast_mul(&sin)?)?;
        let out2 = (x1.broadcast_mul(&sin)? + x2.broadcast_mul(&cos)?)?;

        Tensor::cat(&[&out1, &out2], 3).map_err(Into::into)
    }
}

/// Grouped-Query Attention (GQA) with RoPE.
struct CausalSelfAttention {
    q_proj: Tensor,
    k_proj: Tensor,
    v_proj: Tensor,
    o_proj: Tensor,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
}

struct AttentionForward {
    output: Tensor,
    key: Tensor,
    value: Tensor,
}

impl CausalSelfAttention {
    fn forward_prefill(&self, hidden: &Tensor, rope: &RotaryEmbedding) -> Result<AttentionForward> {
        let (batch, seq_len, _) = hidden.dims3()?;

        // Project Q, K, V
        let q = hidden.broadcast_matmul(&self.q_proj)?;
        let k = hidden.broadcast_matmul(&self.k_proj)?;
        let v = hidden.broadcast_matmul(&self.v_proj)?;

        // Reshape to [batch, seq, num_heads, head_dim] → [batch, num_heads, seq, head_dim]
        let q = q
            .reshape((batch, seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let k = k
            .reshape((batch, seq_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let v = v
            .reshape((batch, seq_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;

        // Apply RoPE to Q and K
        let q = rope.apply(&q, seq_len)?;
        let k = rope.apply(&k, seq_len)?;
        let cache_k = k.clone();
        let cache_v = v.clone();

        // GQA: repeat-interleave K,V heads to match Q heads
        // Each KV head serves a group of Q heads:
        //   [h0,h0,h0, h1,h1,h1, h2,h2,h2] (not [h0,h1,h2,h0,h1,h2,h0,h1,h2])
        let (k, v) = if self.num_kv_heads != self.num_heads {
            let repeats = self.num_heads / self.num_kv_heads;
            let k = repeat_kv(&k, repeats)?;
            let v = repeat_kv(&v, repeats)?;
            (k, v)
        } else {
            (k, v)
        };

        // Scaled dot-product attention
        let scale = (self.head_dim as f64).sqrt();
        let attn_weights = q.broadcast_matmul(&k.transpose(2, 3)?)?;
        let attn_weights = (attn_weights / scale)?;

        // Causal mask: fill upper triangle with -inf
        let attn_weights = apply_causal_mask(&attn_weights, seq_len, 0)?;

        // Softmax
        let attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights)?;

        // Attention output
        let attn_output = attn_weights.broadcast_matmul(&v)?;

        // Reshape back: [batch, num_heads, seq, head_dim] → [batch, seq, hidden_size]
        let attn_output = attn_output.transpose(1, 2)?.reshape((
            batch,
            seq_len,
            self.num_heads * self.head_dim,
        ))?;

        // Output projection
        let output = attn_output.broadcast_matmul(&self.o_proj)?;
        Ok(AttentionForward {
            output,
            key: cache_k,
            value: cache_v,
        })
    }

    fn forward_decode(
        &self,
        hidden: &Tensor,
        rope: &RotaryEmbedding,
        kv_cache: &LayerKvCache,
    ) -> Result<AttentionForward> {
        let (batch, seq_len, _) = hidden.dims3()?;
        if seq_len != 1 {
            bail!(
                "decode attention expects hidden shape [batch, 1, hidden_size], got seq_len={seq_len}"
            );
        }

        let past_seq_len = kv_cache.sequence_len()?.unwrap_or(0);

        // Project Q, K, V for newest token only.
        let q = hidden.broadcast_matmul(&self.q_proj)?;
        let k = hidden.broadcast_matmul(&self.k_proj)?;
        let v = hidden.broadcast_matmul(&self.v_proj)?;

        // [batch, seq=1, heads, head_dim] -> [batch, heads, seq=1, head_dim]
        let q = q
            .reshape((batch, seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let k = k
            .reshape((batch, seq_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let v = v
            .reshape((batch, seq_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;

        // Decode token position starts at current cache length.
        let q = rope.apply_with_offset(&q, past_seq_len, seq_len)?;
        let new_k = rope.apply_with_offset(&k, past_seq_len, seq_len)?;
        let new_v = v;

        // Append newest KV onto cached historical sequence.
        let (cache_k, cache_v) = kv_cache.merge_with_appended(&new_k, &new_v)?;

        // Repeat KV heads for GQA.
        let (k, v) = if self.num_kv_heads != self.num_heads {
            let repeats = self.num_heads / self.num_kv_heads;
            let k = repeat_kv(&cache_k, repeats)?;
            let v = repeat_kv(&cache_v, repeats)?;
            (k, v)
        } else {
            (cache_k.clone(), cache_v.clone())
        };

        // Decode attention: query_len=1 attends across full cached key_len.
        let scale = (self.head_dim as f64).sqrt();
        let attn_weights = q.broadcast_matmul(&k.transpose(2, 3)?)?;
        let attn_weights = (attn_weights / scale)?;
        let attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights)?;
        let attn_output = attn_weights.broadcast_matmul(&v)?;

        let attn_output = attn_output.transpose(1, 2)?.reshape((
            batch,
            seq_len,
            self.num_heads * self.head_dim,
        ))?;

        let output = attn_output.broadcast_matmul(&self.o_proj)?;
        Ok(AttentionForward {
            output,
            key: cache_k,
            value: cache_v,
        })
    }

    /// Multi-token extend: append `seq_len` positions to the existing cache.
    ///
    /// Used by speculative decoding verification passes, which feed
    /// `[last_committed_token, draft_1, ..., draft_k]` in one forward while
    /// keeping every previously cached position visible. Identical to
    /// [`Self::forward_decode`] when `seq_len == 1`.
    fn forward_extend(
        &self,
        hidden: &Tensor,
        rope: &RotaryEmbedding,
        kv_cache: &LayerKvCache,
    ) -> Result<AttentionForward> {
        let (batch, seq_len, _) = hidden.dims3()?;
        if seq_len == 0 {
            bail!("extend attention received an empty sequence");
        }

        let past_seq_len = kv_cache.sequence_len()?.unwrap_or(0);

        // Project Q, K, V for the new block of tokens.
        let q = hidden.broadcast_matmul(&self.q_proj)?;
        let k = hidden.broadcast_matmul(&self.k_proj)?;
        let v = hidden.broadcast_matmul(&self.v_proj)?;

        // [batch, seq, heads, head_dim] -> [batch, heads, seq, head_dim]
        let q = q
            .reshape((batch, seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let k = k
            .reshape((batch, seq_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let v = v
            .reshape((batch, seq_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;

        // New tokens occupy positions [past_seq_len, past_seq_len + seq_len).
        let q = rope.apply_with_offset(&q, past_seq_len, seq_len)?;
        let new_k = rope.apply_with_offset(&k, past_seq_len, seq_len)?;
        let new_v = v;

        // Append the new block onto the cached historical sequence.
        let (cache_k, cache_v) = kv_cache.merge_with_appended(&new_k, &new_v)?;

        // Repeat KV heads for GQA.
        let (k, v) = if self.num_kv_heads != self.num_heads {
            let repeats = self.num_heads / self.num_kv_heads;
            let k = repeat_kv(&cache_k, repeats)?;
            let v = repeat_kv(&cache_v, repeats)?;
            (k, v)
        } else {
            (cache_k.clone(), cache_v.clone())
        };

        // Each new query attends to all cached keys plus block keys up to
        // its own position.
        let scale = (self.head_dim as f64).sqrt();
        let attn_weights = q.broadcast_matmul(&k.transpose(2, 3)?)?;
        let attn_weights = (attn_weights / scale)?;
        let attn_weights = apply_causal_mask(&attn_weights, seq_len, past_seq_len)?;
        let attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights)?;
        let attn_output = attn_weights.broadcast_matmul(&v)?;

        let attn_output = attn_output.transpose(1, 2)?.reshape((
            batch,
            seq_len,
            self.num_heads * self.head_dim,
        ))?;

        let output = attn_output.broadcast_matmul(&self.o_proj)?;
        Ok(AttentionForward {
            output,
            key: cache_k,
            value: cache_v,
        })
    }
}

/// Apply causal mask (upper triangle = -inf) to attention weights.
fn apply_causal_mask(attn: &Tensor, seq_len: usize) -> Result<Tensor> {
    if seq_len <= 1 {
        return Ok(attn.clone());
    }
    // Create a lower-triangular mask
    let mask: Vec<f32> = (0..seq_len)
        .flat_map(|i| (0..seq_len).map(move |j| if j <= i { 0.0_f32 } else { f32::NEG_INFINITY }))
        .collect();
    let mask = Tensor::new(mask.as_slice(), attn.device())?.reshape((1, 1, seq_len, seq_len))?;
    let result = attn.broadcast_add(&mask)?;
    Ok(result)
}

/// Repeat-interleave KV heads for Grouped-Query Attention.
///
/// Input shape: `[batch, num_kv_heads, seq_len, head_dim]`
/// Output shape: `[batch, num_kv_heads * repeats, seq_len, head_dim]`
///
/// Produces `[h0,h0,h0, h1,h1,h1, ...]` ordering (not tiling).
fn repeat_kv(x: &Tensor, repeats: usize) -> Result<Tensor> {
    if repeats == 1 {
        return Ok(x.clone());
    }
    let (batch, num_kv_heads, seq_len, head_dim) = x.dims4()?;
    let x = x
        .unsqueeze(2)? // [B, KV, 1, S, D]
        .broadcast_as((batch, num_kv_heads, repeats, seq_len, head_dim))? // [B, KV, R, S, D]
        .contiguous()? // make memory contiguous
        .reshape((batch, num_kv_heads * repeats, seq_len, head_dim))?; // [B, H, S, D]
    Ok(x)
}

// ─── Transformer Layer ───────────────────────────────────────────────────────

/// Per-layer KV cache for decode-time attention reuse.
#[derive(Clone, Default)]
pub struct LayerKvCache {
    key: Option<Tensor>,
    value: Option<Tensor>,
}

impl LayerKvCache {
    pub fn key(&self) -> Option<Tensor> {
        self.key.clone()
    }

    pub fn value(&self) -> Option<Tensor> {
        self.value.clone()
    }

    pub fn is_empty(&self) -> bool {
        self.key.is_none() && self.value.is_none()
    }

    pub fn sequence_len(&self) -> Result<Option<usize>> {
        match (&self.key, &self.value) {
            (None, None) => Ok(None),
            (Some(_), None) | (None, Some(_)) => {
                bail!("layer KV cache is inconsistent: key and value must both be set");
            }
            (Some(key), Some(value)) => {
                let key_dims = key
                    .dims4()
                    .context("layer KV key must have shape [batch, kv_heads, seq, head_dim]")?;
                let value_dims = value
                    .dims4()
                    .context("layer KV value must have shape [batch, kv_heads, seq, head_dim]")?;
                if key_dims != value_dims {
                    bail!(
                        "layer KV cache shape mismatch: key={:?}, value={:?}",
                        key_dims,
                        value_dims
                    );
                }
                Ok(Some(key_dims.2))
            }
        }
    }

    fn merge_with_appended(&self, key: &Tensor, value: &Tensor) -> Result<(Tensor, Tensor)> {
        let key_dims = key
            .dims4()
            .context("layer KV key must have shape [batch, kv_heads, seq, head_dim]")?;
        let value_dims = value
            .dims4()
            .context("layer KV value must have shape [batch, kv_heads, seq, head_dim]")?;
        if key_dims != value_dims {
            bail!(
                "layer KV cache shape mismatch: key={:?}, value={:?}",
                key_dims,
                value_dims
            );
        }

        match (&self.key, &self.value) {
            (None, None) => Ok((key.clone(), value.clone())),
            (Some(_), None) | (None, Some(_)) => {
                bail!("layer KV cache is inconsistent: key and value must both be set");
            }
            (Some(cached_key), Some(cached_value)) => {
                let cached_key_dims = cached_key
                    .dims4()
                    .context("cached KV key must have shape [batch, kv_heads, seq, head_dim]")?;
                let cached_value_dims = cached_value
                    .dims4()
                    .context("cached KV value must have shape [batch, kv_heads, seq, head_dim]")?;
                if cached_key_dims != cached_value_dims {
                    bail!(
                        "cached KV shape mismatch: key={:?}, value={:?}",
                        cached_key_dims,
                        cached_value_dims
                    );
                }
                if cached_key_dims.0 != key_dims.0
                    || cached_key_dims.1 != key_dims.1
                    || cached_key_dims.3 != key_dims.3
                {
                    bail!(
                        "cached KV append mismatch: cached={:?}, incoming={:?}",
                        cached_key_dims,
                        key_dims
                    );
                }

                let merged_key = Tensor::cat(&[cached_key, key], 2)?;
                let merged_value = Tensor::cat(&[cached_value, value], 2)?;
                Ok((merged_key, merged_value))
            }
        }
    }

    fn set(&mut self, key: Tensor, value: Tensor) {
        self.key = Some(key);
        self.value = Some(value);
    }

    fn clear(&mut self) {
        self.key = None;
        self.value = None;
    }

    /// Truncate the cached sequence to `target_seq_len` tokens.
    ///
    /// Used by speculative decoding to roll back rejected draft tokens.
    fn truncate(&mut self, target_seq_len: usize) -> Result<()> {
        match (&self.key, &self.value) {
            (None, None) => Ok(()),
            (Some(_), None) | (None, Some(_)) => {
                bail!("layer KV cache is inconsistent: key and value must both be set");
            }
            (Some(key), Some(value)) => {
                let current_len = key.dim(2)?;
                if target_seq_len >= current_len {
                    return Ok(()); // nothing to truncate
                }
                self.key = Some(key.narrow(2, 0, target_seq_len)?);
                self.value = Some(value.narrow(2, 0, target_seq_len)?);
                Ok(())
            }
        }
    }
}

/// A single transformer block with attention + MLP.
struct LlamaLayer {
    input_layernorm: RmsNorm,
    self_attn: CausalSelfAttention,
    gate_proj: Tensor,
    up_proj: Tensor,
    down_proj: Tensor,
    post_attention_layernorm: RmsNorm,
    kv_cache: Mutex<LayerKvCache>,
}

impl LlamaLayer {
    fn forward_prefill(&self, hidden: &Tensor, rope: &RotaryEmbedding) -> Result<Tensor> {
        // Pre-norm → Attention → residual
        let normed = self.input_layernorm.forward(hidden)?;
        let attn = self.self_attn.forward_prefill(&normed, rope)?;
        self.set_kv_cache(&attn.key, &attn.value)?;
        let after_attn = (hidden + attn.output)?;

        // Post-norm → MLP → residual
        let normed = self.post_attention_layernorm.forward(&after_attn)?;
        let gate = normed.broadcast_matmul(&self.gate_proj.t()?)?;
        let gate = candle_nn::ops::silu(&gate)?;
        let up = normed.broadcast_matmul(&self.up_proj.t()?)?;
        let mlp_out = (gate * up)?.broadcast_matmul(&self.down_proj.t()?)?;

        let output = (after_attn + mlp_out)?;
        Ok(output)
    }

    fn forward_decode(&self, hidden: &Tensor, rope: &RotaryEmbedding) -> Result<Tensor> {
        // Pre-norm → decode attention with KV append → residual
        let normed = self.input_layernorm.forward(hidden)?;
        let cache = self.kv_cache()?;
        let attn = self.self_attn.forward_decode(&normed, rope, &cache)?;
        self.set_kv_cache(&attn.key, &attn.value)?;
        let after_attn = (hidden + attn.output)?;

        // Post-norm → MLP → residual
        let normed = self.post_attention_layernorm.forward(&after_attn)?;
        let gate = normed.broadcast_matmul(&self.gate_proj.t()?)?;
        let gate = candle_nn::ops::silu(&gate)?;
        let up = normed.broadcast_matmul(&self.up_proj.t()?)?;
        let mlp_out = (gate * up)?.broadcast_matmul(&self.down_proj.t()?)?;

        let output = (after_attn + mlp_out)?;
        Ok(output)
    }

    fn kv_cache(&self) -> Result<LayerKvCache> {
        let cache = self
            .kv_cache
            .lock()
            .map_err(|_| anyhow!("layer KV cache lock poisoned"))?;
        Ok(cache.clone())
    }

    fn clear_kv_cache(&self) -> Result<()> {
        let mut cache = self
            .kv_cache
            .lock()
            .map_err(|_| anyhow!("layer KV cache lock poisoned"))?;
        cache.clear();
        Ok(())
    }

    fn set_kv_cache(&self, key: &Tensor, value: &Tensor) -> Result<()> {
        let key_dims = key
            .dims4()
            .context("layer KV key must have shape [batch, kv_heads, seq, head_dim]")?;
        let value_dims = value
            .dims4()
            .context("layer KV value must have shape [batch, kv_heads, seq, head_dim]")?;
        if key_dims != value_dims {
            bail!(
                "layer KV cache shape mismatch: key={:?}, value={:?}",
                key_dims,
                value_dims
            );
        }

        let mut cache = self
            .kv_cache
            .lock()
            .map_err(|_| anyhow!("layer KV cache lock poisoned"))?;
        cache.set(key.clone(), value.clone());
        Ok(())
    }
}

// ─── Full Model ──────────────────────────────────────────────────────────────

/// The Llama model for sharded inference (supports SmolLM2 and tiny-random).
pub struct ToyLlama {
    embed_tokens: Tensor,
    layers: Vec<LlamaLayer>,
    norm: RmsNorm,
    lm_head: Tensor,
    rope: RotaryEmbedding,
    pub config: ToyLlamaConfig,
}

impl ToyLlama {
    /// Load from safetensors weights and config.
    pub fn load(config: &ToyLlamaConfig, weights_path: &Path) -> Result<Self> {
        info!(?weights_path, "loading Llama weights from safetensors");

        let device = get_best_device().context("failed to select inference device")?;
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[weights_path.to_path_buf()], DType::F32, &device)?
        };

        let embed_tokens = vb
            .get(
                (config.vocab_size, config.hidden_size),
                "model.embed_tokens.weight",
            )
            .context("failed to load embed_tokens")?;

        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for i in 0..config.num_hidden_layers {
            let prefix = format!("model.layers.{i}");

            let input_ln_w = vb
                .get(
                    config.hidden_size,
                    &format!("{prefix}.input_layernorm.weight"),
                )
                .context(format!("failed to load input_layernorm for layer {i}"))?;

            // Attention projections
            let q_size = config.num_attention_heads * config.head_dim();
            let kv_size = config.num_key_value_heads * config.head_dim();

            let q_proj = vb
                .get(
                    (q_size, config.hidden_size),
                    &format!("{prefix}.self_attn.q_proj.weight"),
                )
                .context(format!("failed to load q_proj for layer {i}"))?;
            let k_proj = vb
                .get(
                    (kv_size, config.hidden_size),
                    &format!("{prefix}.self_attn.k_proj.weight"),
                )
                .context(format!("failed to load k_proj for layer {i}"))?;
            let v_proj = vb
                .get(
                    (kv_size, config.hidden_size),
                    &format!("{prefix}.self_attn.v_proj.weight"),
                )
                .context(format!("failed to load v_proj for layer {i}"))?;
            let o_proj = vb
                .get(
                    (config.hidden_size, q_size),
                    &format!("{prefix}.self_attn.o_proj.weight"),
                )
                .context(format!("failed to load o_proj for layer {i}"))?;

            // MLP projections
            let gate = vb
                .get(
                    (config.intermediate_size, config.hidden_size),
                    &format!("{prefix}.mlp.gate_proj.weight"),
                )
                .context(format!("failed to load gate_proj for layer {i}"))?;

            let up = vb
                .get(
                    (config.intermediate_size, config.hidden_size),
                    &format!("{prefix}.mlp.up_proj.weight"),
                )
                .context(format!("failed to load up_proj for layer {i}"))?;

            let down = vb
                .get(
                    (config.hidden_size, config.intermediate_size),
                    &format!("{prefix}.mlp.down_proj.weight"),
                )
                .context(format!("failed to load down_proj for layer {i}"))?;

            let post_ln_w = vb
                .get(
                    config.hidden_size,
                    &format!("{prefix}.post_attention_layernorm.weight"),
                )
                .context(format!(
                    "failed to load post_attention_layernorm for layer {i}"
                ))?;

            layers.push(LlamaLayer {
                input_layernorm: RmsNorm {
                    weight: input_ln_w,
                    eps: config.rms_norm_eps,
                },
                self_attn: CausalSelfAttention {
                    q_proj,
                    k_proj,
                    v_proj,
                    o_proj,
                    num_heads: config.num_attention_heads,
                    num_kv_heads: config.num_key_value_heads,
                    head_dim: config.head_dim(),
                },
                gate_proj: gate,
                up_proj: up,
                down_proj: down,
                post_attention_layernorm: RmsNorm {
                    weight: post_ln_w,
                    eps: config.rms_norm_eps,
                },
                kv_cache: Mutex::new(LayerKvCache::default()),
            });
        }

        let norm_w = vb
            .get(config.hidden_size, "model.norm.weight")
            .context("failed to load final norm")?;

        // Handle tied word embeddings: SmolLM2 doesn't have a separate lm_head
        let lm_head = if config.tie_word_embeddings {
            info!("using tied embeddings for lm_head");
            embed_tokens.clone()
        } else {
            vb.get((config.vocab_size, config.hidden_size), "lm_head.weight")
                .context("failed to load lm_head")?
        };

        let rope = RotaryEmbedding::new(
            config.head_dim(),
            config.max_position_embeddings.min(4096), // cap precompute
            config.rope_theta,
            &device,
        )?;

        info!(
            num_layers = layers.len(),
            vocab_size = config.vocab_size,
            hidden_size = config.hidden_size,
            head_dim = config.head_dim(),
            "Llama weights loaded"
        );

        Ok(Self {
            embed_tokens,
            layers,
            norm: RmsNorm {
                weight: norm_w,
                eps: config.rms_norm_eps,
            },
            lm_head,
            rope,
            config: config.clone(),
        })
    }

    fn model_device(&self) -> &Device {
        self.embed_tokens.device()
    }

    fn to_model_device(&self, tensor: &Tensor, tensor_name: &str) -> Result<Tensor> {
        let device = self.model_device();
        if tensor.device().same_device(device) {
            return Ok(tensor.clone());
        }

        tensor
            .to_device(device)
            .with_context(|| format!("failed to move {tensor_name} tensor to model device"))
    }

    /// Run the embedding lookup for token IDs → hidden states.
    /// Input shape: [batch, seq_len] or [seq_len] of u32 token IDs.
    /// Output shape: [batch, seq_len, hidden_size].
    pub fn embed(&self, token_ids: &Tensor) -> Result<Tensor> {
        let token_ids = self.to_model_device(token_ids, "token_ids")?;
        let dims = token_ids.dims().to_vec();
        let flat_ids = if dims.len() > 1 {
            token_ids.flatten_all()?
        } else {
            token_ids.clone()
        };

        let flat_hidden = self.embed_tokens.embedding(&flat_ids)?;

        let hidden = if dims.len() > 1 {
            let batch = dims[0];
            let seq_len = dims[1];
            flat_hidden.reshape((batch, seq_len, self.config.hidden_size))?
        } else {
            flat_hidden.unsqueeze(0)?
        };

        info!(shape = ?hidden.dims(), "embedding complete");
        Ok(hidden)
    }

    /// Run a subset of transformer layers in prefill mode.
    ///
    /// - Input shape: `[batch, seq_len, hidden_size]`
    /// - Output shape: `[batch, seq_len, hidden_size]`
    /// - Cache behavior: replace per-layer KV cache with the prefill sequence.
    pub fn forward_layers_prefill(
        &self,
        hidden: &Tensor,
        layer_range: Range<usize>,
    ) -> Result<Tensor> {
        let hidden = self.to_model_device(hidden, "prefill hidden state")?;
        self.validate_layer_range(&layer_range)?;
        let (batch, seq_len, hidden_size) = hidden
            .dims3()
            .context("prefill path expects hidden shape [batch, seq_len, hidden_size]")?;
        if hidden_size != self.config.hidden_size {
            bail!(
                "prefill hidden_size mismatch: expected {}, got {}",
                self.config.hidden_size,
                hidden_size
            );
        }
        info!(
            batch,
            seq_len,
            hidden_size,
            layers = format!("{}..{}", layer_range.start, layer_range.end),
            "running prefill layers"
        );

        let mut h = hidden;
        for i in layer_range.clone() {
            h = self.layers[i].forward_prefill(&h, &self.rope)?;
            info!(layer = i, shape = ?h.dims(), "prefill layer forward complete");
        }
        Ok(h)
    }

    /// Run a subset of transformer layers in decode mode.
    ///
    /// - Input shape: `[batch, 1, hidden_size]`
    /// - Output shape: `[batch, 1, hidden_size]`
    /// - Cache behavior: append one token to per-layer KV cache.
    pub fn forward_layers_decode(
        &self,
        hidden: &Tensor,
        layer_range: Range<usize>,
    ) -> Result<Tensor> {
        let hidden = self.to_model_device(hidden, "decode hidden state")?;
        self.validate_layer_range(&layer_range)?;
        let (batch, seq_len, hidden_size) = hidden
            .dims3()
            .context("decode path expects hidden shape [batch, 1, hidden_size]")?;
        if seq_len != 1 {
            bail!(
                "decode path expects hidden shape [batch, 1, hidden_size], got [batch={}, seq_len={}, hidden_size={}]",
                batch,
                seq_len,
                hidden_size
            );
        }
        if hidden_size != self.config.hidden_size {
            bail!(
                "decode hidden_size mismatch: expected {}, got {}",
                self.config.hidden_size,
                hidden_size
            );
        }
        info!(
            batch,
            seq_len,
            hidden_size,
            layers = format!("{}..{}", layer_range.start, layer_range.end),
            "running decode layers"
        );

        let mut h = hidden;
        for i in layer_range.clone() {
            h = self.layers[i].forward_decode(&h, &self.rope)?;
            info!(layer = i, shape = ?h.dims(), "decode layer forward complete");
        }
        Ok(h)
    }

    /// Compatibility wrapper: existing callers keep prefill behavior.
    pub fn forward_layers(&self, hidden: &Tensor, layer_range: Range<usize>) -> Result<Tensor> {
        self.forward_layers_prefill(hidden, layer_range)
    }

    /// Replace one layer's KV cache snapshot.
    pub fn set_layer_kv_cache(&self, layer_idx: usize, key: &Tensor, value: &Tensor) -> Result<()> {
        self.layer(layer_idx)?.set_kv_cache(key, value)
    }

    /// Read one layer's KV cache snapshot.
    pub fn layer_kv_cache(&self, layer_idx: usize) -> Result<LayerKvCache> {
        self.layer(layer_idx)?.kv_cache()
    }

    /// Reset one layer's KV cache.
    pub fn clear_layer_kv_cache(&self, layer_idx: usize) -> Result<()> {
        self.layer(layer_idx)?.clear_kv_cache()
    }

    /// Reset all layer KV caches.
    pub fn clear_kv_cache(&self) -> Result<()> {
        for layer in &self.layers {
            layer.clear_kv_cache()?;
        }
        Ok(())
    }

    /// Reset KV caches for a subset of layers.
    pub fn clear_kv_cache_range(&self, layer_range: Range<usize>) -> Result<()> {
        self.validate_layer_range(&layer_range)?;
        for i in layer_range {
            self.layers[i].clear_kv_cache()?;
        }
        Ok(())
    }

    /// Truncate all layer KV caches to `target_seq_len` tokens.
    ///
    /// Used by speculative decoding to roll back rejected draft tokens
    /// from the cache without a full clear + replay.
    pub fn truncate_kv_cache(&self, target_seq_len: usize) -> Result<()> {
        for layer in &self.layers {
            let mut cache = layer
                .kv_cache
                .lock()
                .map_err(|_| anyhow!("layer KV cache lock poisoned"))?;
            cache.truncate(target_seq_len)?;
        }
        Ok(())
    }

    /// Run final RmsNorm + lm_head projection.
    /// Input shape: [batch, seq_len, hidden_size].
    /// Output shape: [batch, seq_len, vocab_size].
    pub fn head(&self, hidden: &Tensor) -> Result<Tensor> {
        let hidden = self.to_model_device(hidden, "lm_head hidden state")?;
        let normed = self.norm.forward(&hidden)?;
        let logits = normed.broadcast_matmul(&self.lm_head.t()?)?;
        info!(shape = ?logits.dims(), "lm_head complete");
        Ok(logits)
    }

    /// Full prefill pass over all tokens.
    ///
    /// - Input token shape: `[batch, seq_len]` or `[seq_len]`
    /// - Output logits shape: `[batch, seq_len, vocab_size]`
    pub fn forward_prefill(&self, token_ids: &Tensor) -> Result<Tensor> {
        let hidden = self.embed(token_ids)?;
        let hidden = self.forward_layers_prefill(&hidden, 0..self.layers.len())?;
        let logits = self.head(&hidden)?;
        Ok(logits)
    }

    /// Decode pass for newest token only.
    ///
    /// - Input token shape: `[batch, 1]` or `[1]`
    /// - Output logits shape: `[batch, 1, vocab_size]`
    pub fn forward_decode(&self, token_ids: &Tensor) -> Result<Tensor> {
        let hidden = self.embed(token_ids)?;
        let hidden = self.forward_layers_decode(&hidden, 0..self.layers.len())?;
        let logits = self.head(&hidden)?;
        Ok(logits)
    }

    /// Compatibility wrapper for legacy full forward callers.
    pub fn forward_full(&self, token_ids: &Tensor) -> Result<Tensor> {
        self.forward_prefill(token_ids)
    }

    /// Greedy argmax sampling from `[1, seq_len, vocab_size]` logits.
    pub fn sample_argmax(logits: &Tensor) -> Result<u32> {
        let seq_len = logits.dim(1)?;
        let last_logits = logits.narrow(1, seq_len - 1, 1)?.squeeze(1)?;
        let token_id = last_logits
            .argmax(candle_core::D::Minus1)?
            .squeeze(0)?
            .to_scalar::<u32>()?;
        Ok(token_id)
    }

    /// Batched greedy argmax sampling from `[batch, seq_len, vocab_size]` logits.
    ///
    /// Returns one token per batch element, sampled from the last sequence position.
    pub fn sample_argmax_batched(logits: &Tensor) -> Result<Vec<u32>> {
        let (batch_size, seq_len, _vocab) = logits
            .dims3()
            .context("sample_argmax_batched expects [batch, seq_len, vocab] logits")?;
        let last_logits = logits.narrow(1, seq_len - 1, 1)?.squeeze(1)?; // [batch, vocab]
        let argmax_ids = last_logits.argmax(candle_core::D::Minus1)?; // [batch]
        let mut tokens = Vec::with_capacity(batch_size);
        for i in 0..batch_size {
            let token = argmax_ids.get(i)?.to_scalar::<u32>()?;
            tokens.push(token);
        }
        Ok(tokens)
    }

    /// Run autoregressive generation locally.
    pub fn generate(
        &self,
        initial_ids: &[u32],
        max_new_tokens: usize,
        eos_token_id: Option<u32>,
    ) -> Result<Vec<u32>> {
        let device = self.model_device();
        let mut ids = initial_ids.to_vec();
        if ids.is_empty() {
            bail!("generation requires at least one prompt token");
        }
        if max_new_tokens == 0 {
            return Ok(ids);
        }
        self.clear_kv_cache()?;

        // Prefill: process the full prompt once and seed all layer KV caches.
        let prefill = Tensor::new(ids.as_slice(), &device)?.reshape((1, ids.len()))?;
        let mut logits = self.forward_prefill(&prefill)?;

        for step in 0..max_new_tokens {
            let next_token = Self::sample_argmax(&logits)?;

            info!(step, next_token, "generated token");
            ids.push(next_token);

            if eos_token_id == Some(next_token) {
                info!(step, "hit EOS token — stopping generation");
                break;
            }

            if step + 1 < max_new_tokens {
                // Decode: process newest token only, while reusing + appending KV cache.
                let decode = Tensor::new(&[next_token], &device)?.reshape((1, 1))?;
                logits = self.forward_decode(&decode)?;
            }
        }

        Ok(ids)
    }

    fn layer(&self, layer_idx: usize) -> Result<&LlamaLayer> {
        self.layers.get(layer_idx).ok_or_else(|| {
            anyhow!(
                "layer index {} out of bounds (num_layers={})",
                layer_idx,
                self.layers.len()
            )
        })
    }

    fn validate_layer_range(&self, layer_range: &Range<usize>) -> Result<()> {
        if layer_range.start > layer_range.end {
            bail!(
                "invalid layer range {}..{}: start must be <= end",
                layer_range.start,
                layer_range.end
            );
        }
        if layer_range.end > self.layers.len() {
            bail!(
                "layer range {}..{} out of bounds (num_layers={})",
                layer_range.start,
                layer_range.end,
                self.layers.len()
            );
        }
        Ok(())
    }

    /// Run a subset of transformer layers in speculative mode.
    ///
    /// Like prefill, processes multiple tokens at once, but appends to existing
    /// KV cache instead of replacing it. Used for speculative decoding
    /// verification passes where we need to process draft_len tokens in one go
    /// while preserving the cache from previous decode steps.
    ///
    /// - Input shape: `[batch, draft_len, hidden_size]`
    /// - Output shape: `[batch, draft_len, hidden_size]`
    /// - Cache behavior: appends draft_len tokens to existing per-layer KV cache.
    pub fn forward_layers_speculative(
        &self,
        hidden: &Tensor,
        layer_range: Range<usize>,
    ) -> Result<Tensor> {
        // For PoC, speculative verification reuses the prefill path.
        // The KV cache will grow by draft_len tokens. The caller must
        // roll back the cache if draft tokens are rejected.
        self.forward_layers_prefill(hidden, layer_range)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::download_model;

    #[tokio::test]
    async fn full_forward_pass() {
        let files = download_model("hf-internal-testing/tiny-random-LlamaForCausalLM")
            .await
            .expect("download should succeed");

        let config = ToyLlamaConfig::from_file(&files.config).expect("config should parse");

        let model = ToyLlama::load(&config, &files.weights).expect("model should load");

        let token_ids = Tensor::new(&[1_u32, 2, 3], &Device::Cpu)
            .unwrap()
            .reshape((1, 3))
            .unwrap();

        let logits = model
            .forward_full(&token_ids)
            .expect("forward pass should succeed");

        assert_eq!(logits.dims(), &[1, 3, 32000]);
    }

    #[tokio::test]
    async fn sharded_forward_matches_full() {
        let files = download_model("hf-internal-testing/tiny-random-LlamaForCausalLM")
            .await
            .expect("download should succeed");

        let config = ToyLlamaConfig::from_file(&files.config).expect("config should parse");

        let model = ToyLlama::load(&config, &files.weights).expect("model should load");

        let token_ids = Tensor::new(&[1_u32, 2, 3], &Device::Cpu)
            .unwrap()
            .reshape((1, 3))
            .unwrap();

        // Full forward
        let full_logits = model.forward_full(&token_ids).unwrap();

        // Sharded forward: master(embed + layer 0) → worker(layer 1) → master(head)
        let hidden = model.embed(&token_ids).unwrap();
        let after_master = model.forward_layers(&hidden, 0..1).unwrap();
        let after_worker = model.forward_layers(&after_master, 1..2).unwrap();
        let sharded_logits = model.head(&after_worker).unwrap();

        // Both should produce the same shape
        assert_eq!(full_logits.dims(), sharded_logits.dims());

        // Values should be identical (no randomness, same weights, same input)
        let full_vals = full_logits.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let shard_vals = sharded_logits
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        for (a, b) in full_vals.iter().zip(shard_vals.iter()) {
            assert!((a - b).abs() < 1e-4, "mismatch: {a} vs {b}");
        }
    }

    #[tokio::test]
    async fn kv_cache_lifecycle_roundtrip() {
        let files = download_model("hf-internal-testing/tiny-random-LlamaForCausalLM")
            .await
            .expect("download should succeed");

        let config = ToyLlamaConfig::from_file(&files.config).expect("config should parse");
        let model = ToyLlama::load(&config, &files.weights).expect("model should load");

        let token_ids = Tensor::new(&[1_u32, 2, 3], &Device::Cpu)
            .unwrap()
            .reshape((1, 3))
            .unwrap();

        let hidden = model.embed(&token_ids).unwrap();
        let _ = model
            .forward_layers(&hidden, 0..config.num_hidden_layers)
            .expect("forward should succeed");

        let first_cache = model.layer_kv_cache(0).expect("cache read should work");
        let first_key = first_cache.key().expect("key cache should be populated");
        let first_value = first_cache
            .value()
            .expect("value cache should be populated");
        assert_eq!(first_key.dims(), first_value.dims());

        model
            .clear_layer_kv_cache(0)
            .expect("single-layer cache reset should work");
        assert!(
            model
                .layer_kv_cache(0)
                .expect("cache read should work")
                .is_empty()
        );

        model
            .clear_kv_cache()
            .expect("global cache reset should work");
        for layer_idx in 0..config.num_hidden_layers {
            assert!(
                model
                    .layer_kv_cache(layer_idx)
                    .expect("cache read should work")
                    .is_empty()
            );
        }
    }

    #[tokio::test]
    async fn prefill_then_decode_matches_full_last_token() {
        let files = download_model("hf-internal-testing/tiny-random-LlamaForCausalLM")
            .await
            .expect("download should succeed");

        let config = ToyLlamaConfig::from_file(&files.config).expect("config should parse");
        let model = ToyLlama::load(&config, &files.weights).expect("model should load");

        // Compare:
        // 1) Full prefill over [1,2,3,4]
        // 2) Prefill [1,2,3] then decode [4] with KV cache append.
        let full_ids = Tensor::new(&[1_u32, 2, 3, 4], &Device::Cpu)
            .unwrap()
            .reshape((1, 4))
            .unwrap();
        let full_logits = model
            .forward_full(&full_ids)
            .expect("full forward should work");
        let full_last = full_logits.narrow(1, 3, 1).expect("last token slice");

        model.clear_kv_cache().expect("cache reset should work");
        let prompt_ids = Tensor::new(&[1_u32, 2, 3], &Device::Cpu)
            .unwrap()
            .reshape((1, 3))
            .unwrap();
        let prefill_logits = model
            .forward_prefill(&prompt_ids)
            .expect("prefill should work");
        assert_eq!(prefill_logits.dims(), &[1, 3, config.vocab_size]);
        for layer_idx in 0..config.num_hidden_layers {
            let cache = model
                .layer_kv_cache(layer_idx)
                .expect("cache read should work");
            let cache_key = cache.key().expect("prefill cache key should exist");
            let (_, _, seq_len, _) = cache_key.dims4().expect("cache key should be rank-4");
            assert_eq!(seq_len, 3, "prefill should store full prompt in KV cache");
        }

        let decode_ids = Tensor::new(&[4_u32], &Device::Cpu)
            .unwrap()
            .reshape((1, 1))
            .unwrap();
        let decode_logits = model
            .forward_decode(&decode_ids)
            .expect("decode should work");
        assert_eq!(decode_logits.dims(), &[1, 1, config.vocab_size]);
        for layer_idx in 0..config.num_hidden_layers {
            let cache = model
                .layer_kv_cache(layer_idx)
                .expect("cache read should work");
            let cache_key = cache.key().expect("decode cache key should exist");
            let (_, _, seq_len, _) = cache_key.dims4().expect("cache key should be rank-4");
            assert_eq!(seq_len, 4, "decode should append one KV token");
        }

        let expected = full_last.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let actual = decode_logits
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        for (a, b) in expected.iter().zip(actual.iter()) {
            assert!((a - b).abs() < 1e-4, "mismatch: {a} vs {b}");
        }
    }

    #[tokio::test]
    async fn decode_path_rejects_multi_token_hidden() {
        let files = download_model("hf-internal-testing/tiny-random-LlamaForCausalLM")
            .await
            .expect("download should succeed");

        let config = ToyLlamaConfig::from_file(&files.config).expect("config should parse");
        let model = ToyLlama::load(&config, &files.weights).expect("model should load");

        let bad_hidden = Tensor::new(
            vec![0.0_f32; config.hidden_size * 2].as_slice(),
            &Device::Cpu,
        )
        .unwrap()
        .reshape((1, 2, config.hidden_size))
        .unwrap();

        let err = model
            .forward_layers_decode(&bad_hidden, 0..config.num_hidden_layers)
            .expect_err("decode should reject seq_len > 1");
        assert!(
            err.to_string()
                .contains("decode path expects hidden shape [batch, 1, hidden_size]"),
            "unexpected error: {err}"
        );
    }

    #[tokio::test]
    async fn sharded_prefill_decode_ranges_match_full_last_token() {
        let files = download_model("hf-internal-testing/tiny-random-LlamaForCausalLM")
            .await
            .expect("download should succeed");

        let config = ToyLlamaConfig::from_file(&files.config).expect("config should parse");
        let model = ToyLlama::load(&config, &files.weights).expect("model should load");
        let split_point = (config.num_hidden_layers / 2)
            .max(1)
            .min(config.num_hidden_layers);

        let full_ids = Tensor::new(&[1_u32, 2, 3, 4], &Device::Cpu)
            .unwrap()
            .reshape((1, 4))
            .unwrap();
        let full_logits = model
            .forward_full(&full_ids)
            .expect("full forward should work");
        let full_last = full_logits.narrow(1, 3, 1).expect("last token slice");

        model.clear_kv_cache().expect("cache reset should work");

        let prompt_ids = Tensor::new(&[1_u32, 2, 3], &Device::Cpu)
            .unwrap()
            .reshape((1, 3))
            .unwrap();
        let prompt_hidden = model.embed(&prompt_ids).expect("embedding should work");
        assert_eq!(prompt_hidden.dims(), &[1, 3, config.hidden_size]);

        let after_master_prefill = model
            .forward_layers_prefill(&prompt_hidden, 0..split_point)
            .expect("master prefill should work");
        assert_eq!(after_master_prefill.dims(), &[1, 3, config.hidden_size]);

        let after_worker_prefill = model
            .forward_layers_prefill(&after_master_prefill, split_point..config.num_hidden_layers)
            .expect("worker prefill should work");
        assert_eq!(after_worker_prefill.dims(), &[1, 3, config.hidden_size]);

        for layer_idx in 0..config.num_hidden_layers {
            let cache = model
                .layer_kv_cache(layer_idx)
                .expect("cache read should work");
            assert_eq!(
                cache.sequence_len().expect("cache should be consistent"),
                Some(3),
                "prefill should seed full prompt cache for layer {layer_idx}"
            );
        }

        let decode_ids = Tensor::new(&[4_u32], &Device::Cpu)
            .unwrap()
            .reshape((1, 1))
            .unwrap();
        let decode_hidden = model
            .embed(&decode_ids)
            .expect("decode embedding should work");
        assert_eq!(decode_hidden.dims(), &[1, 1, config.hidden_size]);

        let after_master_decode = model
            .forward_layers_decode(&decode_hidden, 0..split_point)
            .expect("master decode should work");
        assert_eq!(after_master_decode.dims(), &[1, 1, config.hidden_size]);

        let after_worker_decode = model
            .forward_layers_decode(&after_master_decode, split_point..config.num_hidden_layers)
            .expect("worker decode should work");
        assert_eq!(after_worker_decode.dims(), &[1, 1, config.hidden_size]);

        for layer_idx in 0..config.num_hidden_layers {
            let cache = model
                .layer_kv_cache(layer_idx)
                .expect("cache read should work");
            assert_eq!(
                cache.sequence_len().expect("cache should be consistent"),
                Some(4),
                "decode should append one cache token for layer {layer_idx}"
            );
        }

        let decode_logits = model.head(&after_worker_decode).expect("head should work");
        assert_eq!(decode_logits.dims(), &[1, 1, config.vocab_size]);

        let expected = full_last.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let actual = decode_logits
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        for (a, b) in expected.iter().zip(actual.iter()) {
            assert!((a - b).abs() < 1e-4, "mismatch: {a} vs {b}");
        }
    }

    #[tokio::test]
    async fn clear_kv_cache_range_only_resets_target_layers() {
        let files = download_model("hf-internal-testing/tiny-random-LlamaForCausalLM")
            .await
            .expect("download should succeed");

        let config = ToyLlamaConfig::from_file(&files.config).expect("config should parse");
        let model = ToyLlama::load(&config, &files.weights).expect("model should load");

        let prompt_ids = Tensor::new(&[1_u32, 2, 3], &Device::Cpu)
            .unwrap()
            .reshape((1, 3))
            .unwrap();
        model
            .forward_prefill(&prompt_ids)
            .expect("prefill should seed cache");

        for layer_idx in 0..config.num_hidden_layers {
            let cache = model
                .layer_kv_cache(layer_idx)
                .expect("cache read should work");
            assert_eq!(
                cache.sequence_len().expect("cache should be consistent"),
                Some(3),
                "all layers should have prompt cache before range clear"
            );
        }

        let reset_start = config.num_hidden_layers.saturating_sub(1);
        let reset_range = reset_start..config.num_hidden_layers;
        model
            .clear_kv_cache_range(reset_range.clone())
            .expect("range clear should work");

        for layer_idx in 0..config.num_hidden_layers {
            let cache = model
                .layer_kv_cache(layer_idx)
                .expect("cache read should work");
            let expected = if reset_range.contains(&layer_idx) {
                None
            } else {
                Some(3)
            };
            assert_eq!(
                cache.sequence_len().expect("cache should be consistent"),
                expected,
                "unexpected cache sequence length for layer {layer_idx}"
            );
        }
    }

    #[tokio::test]
    async fn forward_decode_rejects_multi_token_input_ids() {
        let files = download_model("hf-internal-testing/tiny-random-LlamaForCausalLM")
            .await
            .expect("download should succeed");

        let config = ToyLlamaConfig::from_file(&files.config).expect("config should parse");
        let model = ToyLlama::load(&config, &files.weights).expect("model should load");

        let bad_decode_ids = Tensor::new(&[4_u32, 5], &Device::Cpu)
            .unwrap()
            .reshape((1, 2))
            .unwrap();
        let err = model
            .forward_decode(&bad_decode_ids)
            .expect_err("decode should reject seq_len > 1");
        assert!(
            err.to_string()
                .contains("decode path expects hidden shape [batch, 1, hidden_size]"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn micro_benchmark_score_guards_zero_elapsed_ms() {
        assert!(micro_benchmark_score(0.0) > 0);
        assert_eq!(micro_benchmark_score(50.0), 200);
    }

    #[test]
    fn run_micro_benchmark_returns_non_zero_score() {
        let score = run_micro_benchmark(&Device::Cpu);
        assert!(score > 0);
    }

    #[test]
    fn get_best_device_never_panics() {
        let selection = std::panic::catch_unwind(get_best_device);
        assert!(
            selection.is_ok(),
            "device selection should not panic when acceleration backends are missing"
        );
        assert!(
            selection.unwrap().is_ok(),
            "device selection should always return a usable fallback device"
        );
    }

    #[test]
    fn get_best_device_falls_back_to_cpu_in_cpu_only_environments() {
        if candle_core::utils::cuda_is_available() || candle_core::utils::metal_is_available() {
            return;
        }

        let device = get_best_device().expect("device selection should succeed");
        assert!(device.is_cpu(), "CPU-only environments must return CPU");
    }
}
