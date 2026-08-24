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

