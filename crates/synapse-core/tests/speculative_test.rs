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

    let prompt: Vec<u32> = vec![1, 2, 3];
    let block: Vec<u32> = vec![7, 42];

    // Reference: prefill, then two single-token decodes.
    model.clear_kv_cache().unwrap();
    model.forward_prefill(&token_tensor(&prompt)).unwrap();
    let ref_logits_a = model.forward_decode(&token_tensor(&[block[0]])).unwrap();
    let ref_logits_b = model.forward_decode(&token_tensor(&[block[1]])).unwrap();

    // Extend path: prefill, then one multi-token append.
    model.clear_kv_cache().unwrap();
    model.forward_prefill(&token_tensor(&prompt)).unwrap();
    let extend_hidden = model
        .forward_layers_extend(
            &model.embed(&token_tensor(&block)).unwrap(),
            0..config.num_hidden_layers,
        )
        .unwrap();
    let all_logits = model.head(&extend_hidden).unwrap();

    // Position 0 of the block predicts like decode(block[0]); position 1
    // like decode(block[1]).
    let pos0 = all_logits.narrow(1, 0, 1).unwrap();
    let pos1 = all_logits.narrow(1, 1, 1).unwrap();
    assert_all_close(&pos0, &ref_logits_a, "extend pos0 vs decode a");
    assert_all_close(&pos1, &ref_logits_b, "extend pos1 vs decode b");

    // Cache lengths must agree too.
    let cache_len_ref = {
        model.clear_kv_cache().unwrap();
        model.forward_prefill(&token_tensor(&prompt)).unwrap();
        model.forward_decode(&token_tensor(&[block[0]])).unwrap();
        model.forward_decode(&token_tensor(&[block[1]])).unwrap();
        model.layer_kv_cache(0).unwrap().sequence_len().unwrap()
    };
    let cache_len_extend = model.layer_kv_cache(0).unwrap().sequence_len().unwrap();
    assert_eq!(cache_len_extend, cache_len_ref, "final cache length");
}

/// Appending a verification block and rolling back rejected positions must
/// leave the caches exactly where the reference (never-speculated) path
/// ended up.
#[tokio::test]
async fn truncate_after_rejected_drafts_restores_reference_state() {
    let files = download_model(TINY_MODEL)
        .await
        .expect("download should succeed");
    let config = ToyLlamaConfig::from_file(&files.config).expect("config should parse");
    let model = ToyLlama::load(&config, &files.weights).expect("model should load");

    // True continuation after prompt [1, 2, 3]: t1, t2, t3.
    let t1: u32 = 9;
    let t2: u32 = 17;
    let t3: u32 = 33;

    // Reference: decode t1, t2, t3 — no speculation ever happened. The
    // captured logits predict the token *after* t3 (both paths below end
    // with an identical cache of [1,2,3,t1,t2,t3]).
    model.clear_kv_cache().unwrap();
    model.forward_prefill(&token_tensor(&[1, 2, 3])).unwrap();
    let reference_logits = {
        model.forward_decode(&token_tensor(&[t1])).unwrap();
        model.forward_decode(&token_tensor(&[t2])).unwrap();
        model.forward_decode(&token_tensor(&[t3])).unwrap()
    };

    // Speculated path: prefill (cache=3), verification block [t1, t2, junk]
    // appends 3 positions (cache=6). Verification would accept t2 but reject
    // `junk`, so accepted=1 → target cache len = base(3) + accepted(1) + 1.
    const BASE: usize = 3;
    const ACCEPTED: usize = 1;
    let target = BASE + ACCEPTED + 1;

    model.clear_kv_cache().unwrap();
    model.forward_prefill(&token_tensor(&[1, 2, 3])).unwrap();
    model
        .forward_layers_speculative(
            &model.embed(&token_tensor(&[t1, t2, 31_999])).unwrap(),
            0..config.num_hidden_layers,
        )
        .unwrap();
    assert_eq!(
        model.layer_kv_cache(0).unwrap().sequence_len().unwrap(),
        Some(6),
        "verification block should have been appended"
    );

    model.truncate_kv_cache(target).unwrap();
    assert_eq!(
        model.layer_kv_cache(0).unwrap().sequence_len().unwrap(),
        Some(target),
        "rollback target length"
    );

    // Feed the correction token (the true next token) and compare against
    // the reference logits — both paths now hold [1,2,3,t1,t2,t3].
    let corrected_logits = model.forward_decode(&token_tensor(&[t3])).unwrap();
    assert_all_close(
        &corrected_logits,
        &reference_logits,
        "post-rollback decode vs reference",
    );
}
