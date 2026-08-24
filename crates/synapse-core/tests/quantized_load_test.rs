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
    tensors.push((
        "output_norm.weight".to_owned(),
        weight(&[HIDDEN], GgmlDType::Q8_0),
    ));
    tensors.push((
        "output.weight".to_owned(),
        weight(&[VOCAB, HIDDEN], GgmlDType::Q8_0),
    ));

    let metadata = gguf_metadata();
    let metadata: Vec<(&str, &gguf_file::Value)> =
        metadata.iter().map(|(k, v)| (k.as_str(), v)).collect();
    let tensor_refs: Vec<(&str, &QTensor)> = tensors.iter().map(|(n, t)| (n.as_str(), t)).collect();

    let file = std::fs::File::create(&path).expect("create gguf file");
    let mut writer = std::io::BufWriter::new(file);
    gguf_file::write(&mut writer, &metadata, &tensor_refs).expect("write gguf fixture");

    path
}

fn token_ids() -> Tensor {
    Tensor::new(&[1_u32, 2, 3], &Device::Cpu)
        .unwrap()
        .reshape((1, 3))
        .unwrap()
}

fn assert_logits_equal(full: &Tensor, sharded: &Tensor, what: &str) {
    assert_eq!(full.dims(), sharded.dims(), "{what}: shape mismatch");
    assert_eq!(
        full.to_vec3::<f32>().unwrap(),
        sharded.to_vec3::<f32>().unwrap(),
        "{what}: logits differ"
    );
}

#[test]
fn gguf_full_forward_and_sampling() {
    let path = write_fixture();
    let model = ToyLlamaGGUF::load(&path).expect("gguf model should load");

    let tokens = token_ids();
    let logits = model.forward_prefill(&tokens).expect("prefill");
    assert_eq!(logits.dims(), &[1, 3, VOCAB]);

    // Argmax sampling must produce an in-vocab token, deterministically.
    let token = ToyLlama::sample_argmax(&logits).expect("sample");
    assert!(token < VOCAB as u32, "sampled token in vocab");
    let again = ToyLlama::sample_argmax(&logits).expect("sample");
    assert_eq!(token, again, "argmax is deterministic");

    // Decode path: continue with the sampled token using the KV cache.
    let decoded = model
        .forward_decode(&Tensor::new(&[[token]], &Device::Cpu).unwrap())
        .expect("decode");
    let decoded_token = ToyLlama::sample_argmax(&decoded).expect("sample decode");
    assert!(decoded_token < VOCAB as u32, "decoded token in vocab");
}

#[test]
fn gguf_sharded_slices_match_full() {
    let path = write_fixture();
    let device = Device::Cpu;

    // Compare like-for-like: pin every load to CPU so quantized matmul
    // accumulation matches between the full model and the slices.
    let full = ToyLlamaGGUF::load_range(&path, 0..2, &device).expect("full model should load");
    let master = ToyLlamaGGUF::load_range(&path, 0..1, &device).expect("master slice");
    let worker = ToyLlamaGGUF::load_range(&path, 1..2, &device).expect("worker slice");

    let tokens = token_ids();
    let full_logits = full.forward_prefill(&tokens).expect("full prefill");

    let hidden = master.embed(&tokens).expect("master embed");
    let hidden = master
        .forward_layers_prefill(&hidden, 0..1)
        .expect("master layers");
    let hidden = worker
        .forward_layers_prefill(&hidden, 1..2)
        .expect("worker layers");
    let sharded_logits = master.head(&hidden).expect("master LM head");

    assert_logits_equal(&full_logits, &sharded_logits, "gguf prefill");
}

#[test]
fn gguf_local_layer_index_maps_offsets() {
    let path = write_fixture();
    let device = Device::Cpu;

    let master = ToyLlamaGGUF::load_range(&path, 0..1, &device).expect("master slice");
    let worker = ToyLlamaGGUF::load_range(&path, 1..2, &device).expect("worker slice");

    assert_eq!(master.layer_offset(), 0);
    assert_eq!(master.local_layer_index(0).expect("global 0"), 0);
    assert!(master.local_layer_index(1).is_err(), "1 ∉ 0..1");

    assert_eq!(worker.layer_offset(), 1);
    assert_eq!(worker.local_layer_index(1).expect("global 1"), 0);
    assert!(worker.local_layer_index(0).is_err(), "0 ∉ 1..2");
    assert!(worker.local_layer_index(2).is_err(), "2 ∉ 1..2");

    // Slice construction must fail on an out-of-range window.
    assert!(
        ToyLlamaGGUF::load_range(&path, 3..4, &device).is_err(),
        "range beyond block_count must fail"
    );
}
