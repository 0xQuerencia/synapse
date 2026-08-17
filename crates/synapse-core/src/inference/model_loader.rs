//! HF Hub model downloader.
//!
//! Downloads `config.json` and the model weights from a Hugging Face model
//! repo into the default HF cache (`~/.cache/huggingface/hub`). Supports:
//!
//! - single-file safetensors checkpoints (`model.safetensors`),
//! - multi-shard safetensors checkpoints (via `model.safetensors.index.json`),
//! - quantized GGUF files (any `*.gguf` sibling).

use std::{fs, path::Path, path::PathBuf};

use crate::model::sha256_hex;
use crate::protocol::weight_filename_allowed;
use anyhow::{Context, Result};
use hf_hub::api::tokio::ApiRepo;
use tracing::info;

/// Download model files from Hugging Face Hub and return paths to config and weights.
///
/// Uses the default HF cache directory so that models downloaded by other tools
/// (Python, other UIs) are automatically reused. `weights` points at
/// `model.safetensors`, at the `model.safetensors.index.json` for multi-shard
/// checkpoints (all shards are downloaded), or at the first `*.gguf` sibling.
pub async fn download_model(model_id: &str) -> Result<ModelFiles> {
    info!(model_id, "downloading model from Hugging Face Hub");

    let api = hf_hub::api::tokio::Api::new().context("failed to create HF Hub API client")?;
    let repo = api.model(model_id.to_owned());

    let config_path = repo
        .get("config.json")
        .await
        .context("failed to download config.json")?;
    info!(?config_path, "downloaded config.json");

    let tokenizer_path = repo
        .get("tokenizer.json")
        .await
        .context("failed to download tokenizer.json")?;
    info!(?tokenizer_path, "downloaded tokenizer.json");

    let weights_path = resolve_weights(&repo).await?;
    info!(?weights_path, "resolved model weights");

    Ok(ModelFiles {
        config: config_path,
        weights: weights_path,
        tokenizer: tokenizer_path,
    })
}

/// Choose and download the weight files, returning the path `load` should use.
///
/// Order of preference:
/// 1. `model.safetensors.index.json` — multi-shard checkpoint; every shard
///    listed in the `weight_map` is downloaded (names validated against the
///    protocol whitelist). The returned path is the index file, which
///    `shard_files` later re-reads to mmap the shards.
/// 2. `model.safetensors` — single-file checkpoint.
/// 3. The first `*.gguf` sibling — quantized model.
async fn resolve_weights(repo: &ApiRepo) -> Result<PathBuf> {
    if let Ok(index_path) = repo.get("model.safetensors.index.json").await {
        let index_bytes = fs::read(&index_path)
            .with_context(|| format!("failed to read shard index {}", index_path.display()))?;
        let index: serde_json::Value =
            serde_json::from_slice(&index_bytes).context("shard index is not valid JSON")?;
        let weight_map = index
            .get("weight_map")
            .and_then(serde_json::Value::as_object)
            .context("shard index is missing its weight_map")?;
        let mut downloaded = Vec::new();
        for name in weight_map.keys() {
            if !weight_filename_allowed(name) {
                anyhow::bail!("shard index lists an unsafe filename: {name}");
            }
            if !downloaded.contains(name) {
                repo.get(name)
                    .await
                    .with_context(|| format!("failed to download shard {name}"))?;
                downloaded.push(name.clone());
            }
        }
        info!(
            shard_count = downloaded.len(),
            "downloaded multi-shard checkpoint"
        );
        return Ok(index_path);
    }

    if let Ok(weights) = repo.get("model.safetensors").await {
        return Ok(weights);
    }

    let info = repo.info().await.context("failed to list repo files")?;
    let gguf_name = info
        .siblings
        .iter()
        .map(|s| &s.rfilename)
        .find(|name| name.ends_with(".gguf"))
        .cloned();
    if let Some(name) = gguf_name {
        let path = repo
            .get(&name)
            .await
            .with_context(|| format!("failed to download {name}"))?;
        return Ok(path);
    }

    anyhow::bail!("model has no model.safetensors.index.json, model.safetensors, or *.gguf weights")
}

/// Verify file integrity against an expected SHA-256 hex digest.
pub fn verify_file_sha256(path: &Path, expected: &str) -> Result<()> {
    let bytes = fs::read(path)?;
    let actual = sha256_hex(&bytes);
    if actual != expected.to_ascii_lowercase() {
        anyhow::bail!("sha256 mismatch for {}", path.display());
    }
    Ok(())
}

/// Paths to the downloaded model files.
#[derive(Debug, Clone)]
pub struct ModelFiles {
    pub config: PathBuf,
    pub weights: PathBuf,
    pub tokenizer: PathBuf,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn download_tiny_model() {
        // This test requires network access. Skip in CI if needed.
        let files = download_model("hf-internal-testing/tiny-random-LlamaForCausalLM")
            .await
            .expect("should download tiny model");

        assert!(files.config.exists(), "config.json should exist");
        assert!(files.weights.exists(), "model.safetensors should exist");
    }
}
