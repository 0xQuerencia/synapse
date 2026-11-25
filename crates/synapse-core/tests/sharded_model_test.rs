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
    inference::{ShardedModel, shard_files},
};

/// The tiny checkpoint used across the repo's tests (2 layers, single-file).
mod support;

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
    let files = support::fixture_files();
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
    let files = support::fixture_files();
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
        .expect("master layers");
    let hidden = worker
        .forward_layers_prefill(&hidden, 1..2)
        .expect("worker layers");
    let sharded_last = master
        .head_last_token(&hidden)
        .expect("sharded last-token logits");

    assert_logits_equal(&full_last, &sharded_last, "head_last_token");

    // Argmax sampling must pick the same token on both paths.
    assert_eq!(
        ToyLlama::sample_argmax(&full_last).expect("sample full"),
        ToyLlama::sample_argmax(&sharded_last).expect("sample sharded"),
        "argmax should agree between full and sliced forward"
    );
}

#[tokio::test]
async fn local_layer_index_validates_slice_window() {
    let files = support::fixture_files();
    let config = ToyLlamaConfig::from_file(&files.config).expect("config should parse");

    let master = ToyLlama::load_range(&config, &files.weights, 0..1).expect("slice should load");
    let worker = ToyLlama::load_range(&config, &files.weights, 1..2).expect("slice should load");

    // Full window mapping: global == local.
    assert_eq!(master.layer_offset(), 0);
    assert_eq!(master.local_layer_index(0).expect("global 0 in window"), 0);
    assert!(master.local_layer_index(1).is_err(), "1 ∉ 0..1");

    // Worker window: global 1 → local 0; 0 and 2 rejected.
    assert_eq!(worker.layer_offset(), 1);
    assert_eq!(worker.local_layer_index(1).expect("global 1 in window"), 0);
    assert!(worker.local_layer_index(0).is_err(), "0 ∉ 1..2");
    assert!(worker.local_layer_index(2).is_err(), "2 ∉ 1..2");
}

/// Multi-shard safetensors checkpoints must resolve to one path per shard,
/// in sorted order, with unsafe names rejected.
#[test]
fn shard_files_reads_index_and_validates() {
    // Write a synthetic index next to a fake weights path.
    let dir = std::env::temp_dir().join(format!("synapse-shards-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("create temp dir");
    let index_path = dir.join("model.safetensors.index.json");
    std::fs::write(
        &index_path,
        r#"{"weight_map": {
            "model.layers.0.self_attn.q_proj.weight": "model-00001-of-00002.safetensors",
            "model.layers.1.self_attn.q_proj.weight": "model-00002-of-00002.safetensors",
            "model.layers.2.self_attn.q_proj.weight": "model-00002-of-00002.safetensors"
        }}"#,
    )
    .expect("write index");
    for shard in [
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
    ] {
        std::fs::write(dir.join(shard), b"fake").expect("write shard stub");
    }

    let shards = shard_files(Path::new(&index_path)).expect("shards should resolve");
    assert_eq!(shards.len(), 2, "duplicate shard names should dedupe");
    assert!(
        shards[0].ends_with("model-00001-of-00002.safetensors"),
        "shards should be sorted"
    );

    // Unsafe filenames must be rejected.
    std::fs::write(
        &index_path,
        r#"{"weight_map": {"a.weight": "../../etc/passwd"}}"#,
    )
    .expect("write index");
    assert!(
        shard_files(Path::new(&index_path)).is_err(),
        "path traversal must be rejected"
    );

    // A plain single-file weights path returns itself — but only when no
    // sibling index exists, so isolate this case in its own directory.
    let single_dir =
        std::env::temp_dir().join(format!("synapse-shards-single-{}", std::process::id()));
    std::fs::create_dir_all(&single_dir).expect("create temp dir");
    let single = single_dir.join("model.safetensors");
    std::fs::write(&single, b"fake").expect("write stub");
    let shards = shard_files(Path::new(&single)).expect("single file should pass through");
    assert_eq!(shards, vec![single], "no index → single file");
}
