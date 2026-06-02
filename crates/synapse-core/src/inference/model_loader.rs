//! HF Hub model downloader.
//!
//! Downloads `config.json` and `model.safetensors` from a Hugging Face model repo
//! into the default HF cache (`~/.cache/huggingface/hub`).

use std::{fs, path::Path, path::PathBuf};

use crate::model::sha256_hex;
use anyhow::{Context, Result};
use tracing::info;

/// Download model files from Hugging Face Hub and return paths to config and weights.
///
/// Uses the default HF cache directory so that models downloaded by other tools
/// (Python, other UIs) are automatically reused.
pub async fn download_model(model_id: &str) -> Result<ModelFiles> {
    info!(model_id, "downloading model from Hugging Face Hub");

    let api = hf_hub::api::tokio::Api::new().context("failed to create HF Hub API client")?;

    let repo = api.model(model_id.to_owned());

    let config_path = repo
        .get("config.json")
        .await
        .context("failed to download config.json")?;

    info!(?config_path, "downloaded config.json");

    let weights_path = repo
        .get("model.safetensors")
        .await
        .context("failed to download model.safetensors")?;

    info!(?weights_path, "downloaded model.safetensors");

    let tokenizer_path = repo
        .get("tokenizer.json")
        .await
        .context("failed to download tokenizer.json")?;

    info!(?tokenizer_path, "downloaded tokenizer.json");

    Ok(ModelFiles {
        config: config_path,
        weights: weights_path,
        tokenizer: tokenizer_path,
    })
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
