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
}

impl SimpleTokenizer {
    /// Load a tokenizer from a `tokenizer.json` file.
    pub fn from_file(path: &Path) -> Result<Self> {
        let inner = tokenizers::Tokenizer::from_file(path)
            .map_err(|e| anyhow::anyhow!("failed to load tokenizer: {e}"))?;

        info!(?path, "tokenizer loaded");
        Ok(Self { inner })
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
        let vocab = self.inner.get_added_vocabulary().get_vocab();
        // Try common EOS token names
        for name in ["</s>", "<|endoftext|>", "<|im_end|>"] {
            if let Some(id) = vocab.get(name) {
                return Some(*id);
            }
        }
        None
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
