//! Integration tests for the speculative decoding compute path.
//!
//! The speculative cycle feeds `[last_committed_token, d1..dk]` through a
//! multi-token **extend** forward (append to existing KV caches) and then
//! truncates rejected positions away. These tests pin down the two invariants
//! that make the distributed protocol correct:
//!
//! 1. A multi-token extend produces the same activations as feeding the same
//!    tokens one-by-one through decode.
//! 2. After appending a verification block and truncating back, the KV state
//!    is indistinguishable from never having appended the rejected tail —
//!    so subsequent decode logits match the reference path exactly.

use candle_core::{Device, Tensor};
use synapse_core::ToyLlamaConfig;
use synapse_core::inference::{ToyLlama, download_model};

const TINY_MODEL: &str = "hf-internal-testing/tiny-random-LlamaForCausalLM";

fn token_tensor(ids: &[u32]) -> Tensor {
    Tensor::new(ids, &Device::Cpu)
        .expect("token tensor")
        .reshape((1, ids.len()))
        .expect("reshape to [1, seq]")
}

fn assert_all_close(a: &Tensor, b: &Tensor, what: &str) {
    assert_eq!(a.dims(), b.dims(), "{what}: shape mismatch");
    let (a, b) = (a.to_vec3::<f32>().unwrap(), b.to_vec3::<f32>().unwrap());
    for (batch, (xa, xb)) in a.iter().zip(b.iter()).enumerate() {
        for (seq, (ya, yb)) in xa.iter().zip(xb.iter()).enumerate() {
            for (elem, (x, y)) in ya.iter().zip(yb.iter()).enumerate() {
                let diff = (x - y).abs();
                assert!(
                    diff < 1e-3,
                    "{what}: [{batch},{seq},{elem}] differs by {diff} ({x} vs {y})"
                );
            }
        }
    }
}

/// Feeding `[a, b]` as one extend block must match sequential decode
/// forwards over `[a]` then `[b]`, both in logits and final cache length.
#[tokio::test]
async fn extend_block_matches_sequential_decode() {
    let files = download_model(TINY_MODEL)
        .await
        .expect("download should succeed");
    let config = ToyLlamaConfig::from_file(&files.config).expect("config should parse");
    let model = ToyLlama::load(&config, &files.weights).expect("model should load");
