//! Independently calculated full-forward logits and greedy sequence; no network.
mod support;

use candle_core::{Device, Tensor};
use serde::Deserialize;
use synapse_core::inference::{ToyLlama, ToyLlamaConfig};

#[derive(Deserialize)]
struct Reference {
    input_ids: Vec<u32>,
    logits: Vec<Vec<f32>>,
    greedy_ids: Vec<u32>,
    absolute_tolerance: f32,
}

#[test]
fn candle_matches_independent_scalar_reference() {
    let files = support::fixture_files();
    let config = ToyLlamaConfig::from_file(&files.config).unwrap();
    let model = ToyLlama::load(&config, &files.weights).unwrap();
    let golden: Reference =
        serde_json::from_str(include_str!("fixtures/tiny-llama/reference.json")).unwrap();
    let input = Tensor::new(golden.input_ids.as_slice(), &Device::Cpu)
        .unwrap()
        .reshape((1, golden.input_ids.len()))
        .unwrap();
    let logits = model
        .forward_prefill(&input)
        .unwrap()
        .to_vec3::<f32>()
        .unwrap();
    assert_eq!(logits[0].len(), golden.logits.len());
    for (position, (actual, expected)) in logits[0].iter().zip(&golden.logits).enumerate() {
        assert_eq!(actual.len(), expected.len());
        for (token, (a, b)) in actual.iter().zip(expected).enumerate() {
            assert!(
                (a - b).abs() < golden.absolute_tolerance,
                "reference mismatch at position {position} token {token}: {a} vs {b}"
            );
        }
    }
    let generated = model
        .generate(
            &golden.input_ids,
            golden.greedy_ids.len() - golden.input_ids.len(),
            None,
        )
        .unwrap();
    assert_eq!(generated, golden.greedy_ids);
}

#[test]
fn local_manifest_rejects_modified_tokenizer_before_loading() {
    let files = support::fixture_files();
    let directory = tempfile::tempdir().unwrap();
    for artifact in &files.manifest.artifacts {
        std::fs::copy(
            files.config.parent().unwrap().join(&artifact.filename),
            directory.path().join(&artifact.filename),
        )
        .unwrap();
    }
    std::fs::write(
        directory.path().join("manifest.json"),
        serde_json::to_vec(&files.manifest).unwrap(),
    )
    .unwrap();
    std::fs::write(directory.path().join("tokenizer.json"), b"{}").unwrap();
    assert!(
        synapse_core::inference::model_loader::ModelFiles::from_directory(directory.path())
            .is_err()
    );
}

#[tokio::test]
#[ignore = "explicit real Hub integration; network access and model download required"]
async fn hub_download_is_pinned_and_verified() {
    let files =
        synapse_core::inference::download_model("hf-internal-testing/tiny-random-LlamaForCausalLM")
            .await
            .unwrap();
    assert_eq!(files.manifest.revision.len(), 40);
    files
        .manifest
        .verify_directory(files.config.parent().unwrap())
        .unwrap();
    let config = ToyLlamaConfig::from_file(&files.config).unwrap();
    let model = ToyLlama::load(&config, &files.weights).unwrap();
    let ids = model.generate(&[1, 2, 3], 4, None).unwrap();
    assert_eq!(ids.len(), 7);
}
