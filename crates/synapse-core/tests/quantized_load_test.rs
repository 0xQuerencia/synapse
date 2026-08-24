//! Integration tests for quantized GGUF loading (`ToyLlamaGGUF`).
//!
//! A tiny 2-layer Llama-compatible GGUF fixture is written to a temp file
//! with `gguf_file::write` + `QTensor::quantize`, then loaded through the
//! real pipeline. Verifies full forward + sampling, and that sliced
//! (master 0..1 + worker 1..2) forward equals the full model, exercising
//! `load_range` offset mapping on quantized weights.

use std::collections::BTreeMap;

use candle_core::quantized::{GgmlDType, QTensor, gguf_file};
use candle_core::{Device, Tensor};
use synapse_core::inference::{ShardedModel, ToyLlama, ToyLlamaGGUF};

const VOCAB: usize = 32;
const HIDDEN: usize = 64;
const N_HEADS: usize = 4;
const N_KV_HEADS: usize = 4;
const HEAD_DIM: usize = HIDDEN / N_HEADS;
const N_LAYERS: usize = 2;
const FFN: usize = 128;

/// Metadata a `ToyLlamaGGUF::load_range` expects (llama.cpp key naming).
fn gguf_metadata() -> BTreeMap<String, gguf_file::Value> {
    use gguf_file::Value;
    let mut m = BTreeMap::new();
    m.insert("llama.embedding_length".into(), Value::U32(HIDDEN as u32));
    m.insert("llama.block_count".into(), Value::U32(N_LAYERS as u32));
    m.insert(
        "llama.attention.head_count".into(),
        Value::U32(N_HEADS as u32),
    );
    m.insert(
        "llama.attention.head_count_kv".into(),
        Value::U32(N_KV_HEADS as u32),
    );
    m.insert(
        "llama.attention.layer_norm_rms_epsilon".into(),
        Value::F32(1e-5),
    );
    m.insert("llama.feed_forward_length".into(), Value::U32(FFN as u32));
    m.insert("llama.context_length".into(), Value::U32(64));
    m.insert("llama.rope.freq_base".into(), Value::F32(10_000.0));
    m.insert("llama.vocab_size".into(), Value::U32(VOCAB as u32));
    m
}

/// Deterministic-ish random weight tensor, quantized for the fixture.
fn weight(shape: &[usize], qtype: GgmlDType) -> QTensor {
    let t = Tensor::randn(0.0, 0.5, shape, &Device::Cpu).expect("randn");
    QTensor::quantize(&t, qtype).expect("quantize")
}

/// Write a 2-layer tiny Llama GGUF fixture and return its path.
fn write_fixture() -> std::path::PathBuf {
    // Unique per call: tests run in parallel threads on one process.
    static CALLS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    let n = CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!("synapse-gguf-{}-{n}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("create temp dir");
    let path = dir.join("tiny-llama-q8.gguf");

    let mut tensors: Vec<(String, QTensor)> = Vec::new();
    tensors.push((
        "token_embd.weight".to_owned(),
        weight(&[VOCAB, HIDDEN], GgmlDType::Q4_0),
    ));
    for i in 0..N_LAYERS {
        let blk = format!("blk.{i}");
        tensors.push((
            format!("{blk}.attn_norm.weight"),
            weight(&[HIDDEN], GgmlDType::Q8_0),
        ));
        tensors.push((
            format!("{blk}.attn_q.weight"),
            weight(&[HIDDEN, N_HEADS * HEAD_DIM], GgmlDType::Q8_0),
        ));
        tensors.push((
            format!("{blk}.attn_k.weight"),
            weight(&[HIDDEN, N_KV_HEADS * HEAD_DIM], GgmlDType::Q8_0),
        ));
        tensors.push((
            format!("{blk}.attn_v.weight"),
            weight(&[HIDDEN, N_KV_HEADS * HEAD_DIM], GgmlDType::Q8_0),
        ));
        tensors.push((
            format!("{blk}.attn_output.weight"),
            weight(&[HIDDEN, HIDDEN], GgmlDType::Q8_0),
        ));
        tensors.push((
            format!("{blk}.ffn_norm.weight"),
            weight(&[HIDDEN], GgmlDType::Q8_0),
        ));
        // QMatMul weights are stored [out, in] (llama.cpp convention):
        // qmatmul reads dims[1] as the input dim, dims[0] as the output.
        tensors.push((
            format!("{blk}.ffn_gate.weight"),
            weight(&[FFN, HIDDEN], GgmlDType::Q8_0),
        ));
        tensors.push((
            format!("{blk}.ffn_up.weight"),
            weight(&[FFN, HIDDEN], GgmlDType::Q8_0),
        ));
        tensors.push((
            format!("{blk}.ffn_down.weight"),
            weight(&[HIDDEN, FFN], GgmlDType::Q8_0),
        ));
    }
