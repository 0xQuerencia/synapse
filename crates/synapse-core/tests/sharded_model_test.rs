//! Integration tests for slice-only model loading (`load_range`).
//!
//! A sharded deployment splits the transformer across nodes: the master
//! holds the embedding + lower layers + LM head, workers hold the upper
//! layers. These tests verify that a sliced model produces bit-identical
//! logits to the full model, and that global↔local layer index mapping
//! validates against the slice window.

use std::path::Path;

use candle_core::{Device, Tensor};
use synapse_core::{
    ToyLlama, ToyLlamaConfig,
    inference::{ShardedModel, download_model, shard_files},
};

/// The tiny checkpoint used across the repo's tests (2 layers, single-file).
const TINY_MODEL: &str = "hf-internal-testing/tiny-random-LlamaForCausalLM";

fn token_ids() -> Tensor {
    Tensor::new(&[1_u32, 2, 3], &Device::Cpu)
        .unwrap()
        .reshape((1, 3))
        .unwrap()
}

/// Assert two `[1, seq, vocab]` logits tensors are element-wise identical.
fn assert_logits_equal(full: &Tensor, sharded: &Tensor, what: &str) {
    assert_eq!(full.dims(), sharded.dims(), "{what}: shape mismatch");
    let (full, sharded) = (
        full.to_vec3::<f32>().unwrap(),
        sharded.to_vec3::<f32>().unwrap(),
    );
    for (i, (a, b)) in full.iter().zip(sharded.iter()).enumerate() {
        assert_eq!(a, b, "{what}: batch {i} logits differ");
    }
}

#[tokio::test]
async fn sharded_slices_match_full_forward() {
    let files = download_model(TINY_MODEL)
        .await
        .expect("download should succeed");
    let config = ToyLlamaConfig::from_file(&files.config).expect("config should parse");

    let full = ToyLlama::load(&config, &files.weights).expect("full model should load");
    // master = 0..1, worker = 1..2 (tiny model has exactly 2 layers)
    let master =
        ToyLlama::load_range(&config, &files.weights, 0..1).expect("master slice should load");
    let worker =
        ToyLlama::load_range(&config, &files.weights, 1..2).expect("worker slice should load");

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

    assert_logits_equal(&full_logits, &sharded_logits, "prefill");
}

#[tokio::test]
async fn sharded_head_last_token_matches_full() {
    let files = download_model(TINY_MODEL)
        .await
        .expect("download should succeed");
    let config = ToyLlamaConfig::from_file(&files.config).expect("config should parse");

    let full = ToyLlama::load(&config, &files.weights).expect("full model should load");
    let master =
        ToyLlama::load_range(&config, &files.weights, 0..1).expect("master slice should load");
    let worker =
        ToyLlama::load_range(&config, &files.weights, 1..2).expect("worker slice should load");

    let tokens = token_ids();
    // forward_prefill returns full-sequence logits; take the last position.
    let full_prefill = full.forward_prefill(&tokens).expect("full prefill");
    let seq_len = full_prefill.dim(1).expect("seq dim");
    let full_last = full_prefill
        .narrow(1, seq_len - 1, 1)
        .expect("last position");

    let hidden = master.embed(&tokens).expect("master embed");
    let hidden = master
        .forward_layers_prefill(&hidden, 0..1)
