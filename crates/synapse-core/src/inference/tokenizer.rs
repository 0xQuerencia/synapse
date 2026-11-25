//! Minimal tokenizer wrapper around the `tokenizers` crate.
//!
//! Loads a `tokenizer.json` (HuggingFace format) and provides simple
//! encode/decode methods for the Synapse inference pipeline.

use std::path::Path;

use anyhow::Result;
use tracing::info;

/// Thin wrapper around `tokenizers::Tokenizer` for text ↔ token-ID conversion.
pub struct SimpleTokenizer {
    inner: tokenizers::Tokenizer,
    eos_ids: Vec<u32>,
}

impl SimpleTokenizer {
    /// Load a tokenizer from a `tokenizer.json` file.
    pub fn from_file(path: &Path) -> Result<Self> {
        let inner = tokenizers::Tokenizer::from_file(path)
            .map_err(|e| anyhow::anyhow!("failed to load tokenizer: {e}"))?;

        let mut eos_ids = Vec::new();
        let config_path = path.with_file_name("config.json");
        if config_path.exists() {
            let config: serde_json::Value = serde_json::from_slice(&std::fs::read(config_path)?)?;
            if let Some(value) = config.get("eos_token_id") {
                let values = value
                    .as_array()
                    .cloned()
                    .unwrap_or_else(|| vec![value.clone()]);
                for value in values {
                    if value.is_null() {
                        continue;
                    }
                    let id = value
                        .as_u64()
                        .and_then(|id| u32::try_from(id).ok())
                        .ok_or_else(|| anyhow::anyhow!("invalid configured EOS token"))?;
                    anyhow::ensure!(
                        (id as usize) < inner.get_vocab_size(true),
                        "EOS token outside vocabulary"
                    );
                    if !eos_ids.contains(&id) {
                        eos_ids.push(id);
                    }
                }
            }
        }
        if eos_ids.is_empty() {
            for name in ["<|im_end|>", "</s>", "<|endoftext|>"] {
                if let Some(id) = inner.token_to_id(name) {
                    eos_ids.push(id);
                }
            }
        }
        info!(?path, "tokenizer loaded");
        Ok(Self { inner, eos_ids })
    }

    /// Encode text into token IDs.
    ///
    /// `add_special_tokens` controls whether BOS/EOS tokens are prepended/appended.
    pub fn encode(&self, text: &str, add_special_tokens: bool) -> Result<Vec<u32>> {
        let encoding = self
            .inner
            .encode(text, add_special_tokens)
            .map_err(|e| anyhow::anyhow!("tokenizer encode failed: {e}"))?;

        let ids = encoding.get_ids().to_vec();
        info!(token_count = ids.len(), "text encoded to tokens");
        Ok(ids)
    }

    /// Decode token IDs back into text.
    ///
    /// `skip_special_tokens` controls whether BOS/EOS tokens are stripped.
    pub fn decode(&self, ids: &[u32], skip_special_tokens: bool) -> Result<String> {
        let text = self
            .inner
            .decode(ids, skip_special_tokens)
            .map_err(|e| anyhow::anyhow!("tokenizer decode failed: {e}"))?;

        Ok(text)
    }

    /// Return the EOS token ID, if configured.
    /// Checks common EOS token names used by Llama and SmolLM2 models.
    pub fn eos_token_id(&self) -> Option<u32> {
        self.eos_ids.first().copied()
    }

    pub fn is_eos(&self, token_id: u32) -> bool {
        self.eos_ids.contains(&token_id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::download_model;

    #[tokio::test]
    async fn tokenizer_roundtrip() {
        let files = download_model("hf-internal-testing/tiny-random-LlamaForCausalLM")
            .await
            .expect("download should succeed");

        let tokenizer =
            SimpleTokenizer::from_file(&files.tokenizer).expect("tokenizer should load");

        let text = "Hello world";
        let ids = tokenizer.encode(text, false).expect("encode should work");
        assert!(!ids.is_empty(), "should produce at least one token");

        let decoded = tokenizer.decode(&ids, true).expect("decode should work");
        // Roundtrip may not be exact due to whitespace normalization,
        // but the content should be preserved.
        assert!(
            decoded.contains("Hello") || decoded.contains("hello"),
            "decoded text should contain original content: got '{decoded}'"
        );
    }
}
