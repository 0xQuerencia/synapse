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

/// Owned state for one generation's incremental decoder. The upstream helper
/// retains only the context needed to decode the next stable text chunk, rather
/// than replaying the entire generated sequence or borrowing our tokenizer.
#[derive(Default)]
pub(crate) struct TokenDecodeState {
    ids: Vec<u32>,
    prefix: String,
    prefix_index: usize,
}

// A pathological sequence of undecodable bytes or skipped special tokens must
// not grow the upstream pending buffer indefinitely. This matches the runtime's
// maximum context; ordinary completed text keeps only a small rolling window.
const MAX_DECODE_BUFFER_TOKENS: usize = 4096;

impl TokenDecodeState {
    pub(crate) fn reset(&mut self) {
        self.ids.clear();
        self.prefix.clear();
        self.prefix_index = 0;
    }

    pub(crate) fn push(
        &mut self,
        tokenizer: &SimpleTokenizer,
        token_id: u32,
    ) -> Result<Option<String>> {
        anyhow::ensure!(
            tokenizer.inner.id_to_token(token_id).is_some(),
            "generated token {token_id} is outside the tokenizer vocabulary"
        );
        anyhow::ensure!(
            self.ids.len() < MAX_DECODE_BUFFER_TOKENS,
            "incremental tokenizer buffer exceeds the context limit"
        );
        tokenizers::tokenizer::step_decode_stream(
            &tokenizer.inner,
            token_id,
            true,
            &mut self.ids,
            &mut self.prefix,
            &mut self.prefix_index,
        )
        .map_err(|error| anyhow::anyhow!("incremental tokenizer decode failed: {error}"))
    }
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
    use crate::inference::model_loader::test_model_files;

    fn byte_tokenizer() -> SimpleTokenizer {
        let vocab: [(String, u32); 258] = std::array::from_fn(|id| {
            let token = match id {
                256 => "�".to_owned(),
                257 => "<EOS>".to_owned(),
                _ => format!("<0x{id:02X}>"),
            };
            (token, id as u32)
        });
        let model = tokenizers::models::bpe::BPE::builder()
            .vocab_and_merges(vocab, vec![])
            .byte_fallback(true)
            .build()
            .unwrap();
        let mut inner = tokenizers::Tokenizer::new(model);
        inner.with_decoder(Some(
            tokenizers::decoders::byte_fallback::ByteFallback::default(),
        ));
        inner.add_special_tokens(&[tokenizers::AddedToken::from("<EOS>", true)]);
        SimpleTokenizer {
            inner,
            eos_ids: vec![257],
        }
    }

    #[test]
    fn stream_decoding_buffers_utf8_across_single_tokens_and_accepted_blocks() {
        let tokenizer = byte_tokenizer();
        let mut state = TokenDecodeState::default();
        let mut text = String::new();
        // An ordinary token starts a three-byte character. A later accepted
        // speculative block completes it and starts a four-byte character.
        assert_eq!(state.push(&tokenizer, 0xe2).unwrap(), None);
        for token in [0x82, 0xac, 0xf0, 0x9f] {
            if let Some(chunk) = state.push(&tokenizer, token).unwrap() {
                assert!(!chunk.contains('�'));
                text.push_str(&chunk);
            }
        }
        assert_eq!(text, "€");
        assert_eq!(state.push(&tokenizer, 0x98).unwrap(), None);
        text.push_str(&state.push(&tokenizer, 0x80).unwrap().unwrap());
        assert_eq!(state.push(&tokenizer, 257).unwrap(), None);
        state.reset();
        assert_eq!(text, "€😀");
        assert_eq!(
            text,
            tokenizer
                .decode(&[0xe2, 0x82, 0xac, 0xf0, 0x9f, 0x98, 0x80, 257], true)
                .unwrap()
        );
    }

    #[test]
    fn stream_decoding_reset_isolates_turn_bytes_and_whitespace() {
        let tokenizer = byte_tokenizer();
        let mut state = TokenDecodeState::default();
        assert_eq!(state.push(&tokenizer, 0xe2).unwrap(), None);
        state.reset(); // cancelled turn, followed by a different request
        assert_eq!(state.push(&tokenizer, 0x82).unwrap(), None);
        assert_eq!(state.push(&tokenizer, 0xac).unwrap(), None);
        state.reset();
        assert!(state.ids.is_empty());
        assert_eq!(state.push(&tokenizer, 0xe2).unwrap(), None);
        assert_eq!(state.push(&tokenizer, 0x82).unwrap(), None);
        assert_eq!(state.push(&tokenizer, 0xac).unwrap(), Some("€".to_owned()));
        state.reset();

        // SentencePiece-style initial-space handling also needs per-turn state.
        let model = tokenizers::models::bpe::BPE::builder()
            .vocab_and_merges([("▁Hello".to_owned(), 0)], vec![])
            .build()
            .unwrap();
        let mut inner = tokenizers::Tokenizer::new(model);
        inner.with_decoder(Some(tokenizers::decoders::metaspace::Metaspace::default()));
        let tokenizer = SimpleTokenizer {
            inner,
            eos_ids: vec![],
        };
        assert_eq!(state.push(&tokenizer, 0).unwrap(), Some("Hello".to_owned()));
        assert_eq!(
            state.push(&tokenizer, 0).unwrap(),
            Some(" Hello".to_owned())
        );
        state.reset();
        assert_eq!(state.push(&tokenizer, 0).unwrap(), Some("Hello".to_owned()));
    }

    #[test]
    fn stream_decoding_matches_upstream_special_tokens_and_terminal_boundaries() {
        let tokenizer = byte_tokenizer();
        let mut state = TokenDecodeState::default();
        let mut upstream = tokenizer.inner.decode_stream(true);
        // A literal replacement character is held until another text chunk
        // completes it; skipped special tokens do not change that behavior.
        for token in [256, b'!' as u32, 257, 0xe2, 0x82] {
            assert_eq!(
                state.push(&tokenizer, token).unwrap(),
                upstream.step(token).unwrap()
            );
        }
        // The API backend drops its DecodeStream at a terminal event. Match
        // that contract: no lossy replacement is invented for unfinished bytes.
        state.reset();
        assert!(state.ids.is_empty());
        assert_eq!(
            state.push(&tokenizer, b'A' as u32).unwrap(),
            Some("A".to_owned())
        );
    }

    #[test]
    fn stream_decoding_bounds_pending_state_and_rejects_invalid_ids() {
        let tokenizer = byte_tokenizer();
        let mut state = TokenDecodeState::default();
        assert!(state.push(&tokenizer, u32::MAX).is_err());
        assert!(state.ids.is_empty());
        for _ in 0..MAX_DECODE_BUFFER_TOKENS {
            assert_eq!(state.push(&tokenizer, 257).unwrap(), None);
        }
        assert!(state.push(&tokenizer, 257).is_err());
        assert_eq!(state.ids.len(), MAX_DECODE_BUFFER_TOKENS);
        state.reset();
        for _ in 0..MAX_DECODE_BUFFER_TOKENS {
            assert_eq!(state.push(&tokenizer, 0xc3).unwrap(), None);
            assert_eq!(state.push(&tokenizer, 0xa9).unwrap(), Some("é".to_owned()));
            assert!(
                state.ids.len() <= 4,
                "completed text must use a rolling window"
            );
        }
    }

    #[test]
    fn stream_decoding_propagates_upstream_prefix_errors() {
        let mut tokenizer = byte_tokenizer();
        // This non-prefix-preserving decoder is valid tokenizer data, but it
        // cannot stream: adding "bd" changes already-emitted "a" into "c".
        tokenizer
            .inner
            .with_decoder(Some(tokenizers::decoders::sequence::Sequence::new(vec![
                tokenizers::decoders::byte_fallback::ByteFallback::default().into(),
                tokenizers::decoders::fuse::Fuse::new().into(),
                tokenizers::normalizers::replace::Replace::new("ab", "c")
                    .unwrap()
                    .into(),
            ])));
        let mut state = TokenDecodeState::default();
        assert_eq!(
            state.push(&tokenizer, b'a' as u32).unwrap(),
            Some("a".to_owned())
        );
        assert_eq!(state.push(&tokenizer, b'b' as u32).unwrap(), None);
        let error = state.push(&tokenizer, b'd' as u32).unwrap_err();
        assert!(error.to_string().contains("Invalid prefix"));
        state.reset();
        assert!(state.ids.is_empty());
    }

    #[tokio::test]
    async fn tokenizer_roundtrip() {
        let files = test_model_files().expect("offline fixture should verify");

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
