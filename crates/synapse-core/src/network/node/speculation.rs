/// Speculative decoding support: N-gram draft cache and verification logic.
///
/// The draft cache tracks observed token sequences and predicts the next tokens
/// using frequency-based lookup. The verifier compares draft predictions against
/// actual model logits and accepts the longest matching prefix.
use std::collections::HashMap;

use anyhow::{Result, bail};
use candle_core::Tensor;
use tracing::info;

/// Maximum number of tokens to speculatively draft per round.
pub const MAX_DRAFT_LEN: usize = 3;

/// N-gram context size for the draft cache.
const DEFAULT_NGRAM_SIZE: usize = 3;

/// Simple N-gram draft cache: tracks (context_hash → next_token) frequency.
pub struct NgramDraftCache {
    ngram_size: usize,
    /// Maps context hash → list of (token, count) pairs sorted by count descending.
    table: HashMap<u64, Vec<(u32, u32)>>,
}

impl Default for NgramDraftCache {
    fn default() -> Self {
        Self::new(DEFAULT_NGRAM_SIZE)
    }
}

impl NgramDraftCache {
    pub fn new(ngram_size: usize) -> Self {
        Self {
            ngram_size: ngram_size.max(1),
            table: HashMap::new(),
        }
    }

    /// Record an observed token following a context window.
    pub fn observe(&mut self, context: &[u32], next_token: u32) {
        if context.is_empty() {
            return;
        }

        // Use the last `ngram_size` tokens as the context key.
        let context_start = context.len().saturating_sub(self.ngram_size);
        let key = Self::hash_context(&context[context_start..]);

        let entries = self.table.entry(key).or_default();
        if let Some(entry) = entries.iter_mut().find(|(tok, _)| *tok == next_token) {
            entry.1 += 1;
        } else {
            entries.push((next_token, 1));
        }
        // Keep sorted by count descending for fast draft lookup.
        entries.sort_by(|a, b| b.1.cmp(&a.1));
    }

    /// Draft the next N tokens greedily from the cache.
    ///
    /// Returns up to `max_len` predicted tokens. Returns empty if no prediction
    /// is available for the given context.
    pub fn draft(&self, context: &[u32], max_len: usize) -> Vec<u32> {
        let max_len = max_len.min(MAX_DRAFT_LEN);
        let mut result = Vec::with_capacity(max_len);
        let mut extended_context = context.to_vec();

        for _ in 0..max_len {
            let context_start = extended_context.len().saturating_sub(self.ngram_size);
            let key = Self::hash_context(&extended_context[context_start..]);

            if let Some(entries) = self.table.get(&key) {
                if let Some(&(token, _)) = entries.first() {
                    result.push(token);
                    extended_context.push(token);
                } else {
                    break;
                }
            } else {
                break;
            }
        }

        result
    }

    /// Clear the cache entirely (e.g., on new conversation).
    pub fn clear(&mut self) {
        self.table.clear();
    }

    fn hash_context(tokens: &[u32]) -> u64 {
        // Simple FNV-1a hash for token context.
        let mut hash: u64 = 0xcbf29ce484222325;
        for &token in tokens {
            hash ^= token as u64;
            hash = hash.wrapping_mul(0x100000001b3);
        }
        hash
    }
}

/// Result of verifying a speculative draft against model logits.
#[derive(Debug, Clone)]
pub struct VerificationResult {
    /// Number of draft tokens accepted (0 = all rejected).
    pub accepted_count: usize,
    /// The correctly sampled token for the first rejected position
    /// (or the token after the last accepted draft if all accepted).
    pub correction_token: Option<u32>,
}

/// Verify draft tokens against logits from a multi-token forward pass.
///
/// `draft_tokens`: The speculatively predicted token sequence.
/// `logits`: Tensor of shape `[1, draft_len, vocab_size]` from the verification pass.
///
/// Returns how many draft tokens matched and the correction token for the
/// first mismatch (or None if all matched).
pub fn verify_draft(draft_tokens: &[u32], logits: &Tensor) -> Result<VerificationResult> {
    let dims = logits.dims();
    if dims.len() != 3 {
        bail!(
            "verify_draft expects logits shape [1, draft_len, vocab], got {:?}",
            dims
        );
    }
    let draft_len = dims[1];
    if draft_len == 0 {
        bail!("verify_draft received empty logits");
    }

    let mut accepted = 0;

    for i in 0..draft_tokens.len().min(draft_len) {
        // Get logits for position i → predicts token at position i+1
        // But if we're checking position i, we compare argmax(logits[i]) against draft_tokens[i]
        let position_logits = logits.narrow(1, i, 1)?.squeeze(1)?;
        let predicted = position_logits
            .argmax(candle_core::D::Minus1)?
            .squeeze(0)?
            .to_scalar::<u32>()?;

        if predicted == draft_tokens[i] {
            accepted += 1;
        } else {
            // First mismatch — the correction token is what the model actually predicted.
            return Ok(VerificationResult {
                accepted_count: accepted,
                correction_token: Some(predicted),
            });
        }
    }

    // All draft tokens matched. Get the correction token from the last position
    // (the model's prediction for the position after the last draft token).
    let correction = if draft_len > draft_tokens.len() {
        let last_logits = logits.narrow(1, draft_tokens.len(), 1)?.squeeze(1)?;
        let token = last_logits
            .argmax(candle_core::D::Minus1)?
            .squeeze(0)?
            .to_scalar::<u32>()?;
        Some(token)
    } else if draft_len == draft_tokens.len() {
        // The last logit position corresponds to after the last draft token
        let last_logits = logits.narrow(1, draft_len - 1, 1)?.squeeze(1)?;
        let token = last_logits
            .argmax(candle_core::D::Minus1)?
            .squeeze(0)?
            .to_scalar::<u32>()?;
        Some(token)
    } else {
        None
    };

    info!(
        accepted,
        draft_len = draft_tokens.len(),
        "speculative draft verification complete"
    );

    Ok(VerificationResult {
        accepted_count: accepted,
        correction_token: correction,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ngram_cache_observes_and_drafts() {
        let mut cache = NgramDraftCache::new(2);
        // Observe: [10, 20] → 30
        cache.observe(&[10, 20], 30);
        cache.observe(&[10, 20], 30);
        cache.observe(&[10, 20], 31);

        // Observe: [20, 30] → 40
        cache.observe(&[20, 30], 40);

        let draft = cache.draft(&[10, 20], 2);
        assert_eq!(draft, vec![30, 40], "should draft 30 then 40");
    }

    #[test]
    fn ngram_cache_returns_empty_for_unknown_context() {
        let cache = NgramDraftCache::new(2);
        let draft = cache.draft(&[99, 99], 3);
        assert!(draft.is_empty());
    }

    #[test]
    fn ngram_cache_clears() {
        let mut cache = NgramDraftCache::new(2);
        cache.observe(&[1, 2], 3);
        assert!(!cache.draft(&[1, 2], 1).is_empty());
        cache.clear();
        assert!(cache.draft(&[1, 2], 1).is_empty());
    }

    #[test]
    fn verify_draft_accepts_matching_tokens() {
        // Build logits where argmax matches draft tokens
        let vocab_size = 10;
        let draft_tokens = vec![3_u32, 5, 7];
        let draft_len = draft_tokens.len();

        // Create logits tensor where the highest value at position i is at index draft_tokens[i]
        let mut logits_data = vec![0.0_f32; draft_len * vocab_size];
        for (i, &token) in draft_tokens.iter().enumerate() {
            logits_data[i * vocab_size + token as usize] = 10.0;
        }

        let logits = Tensor::new(logits_data.as_slice(), &candle_core::Device::Cpu)
            .unwrap()
            .reshape((1, draft_len, vocab_size))
            .unwrap();

        let result = verify_draft(&draft_tokens, &logits).expect("verification should work");
        assert_eq!(result.accepted_count, 3);
    }

    #[test]
    fn verify_draft_rejects_on_first_mismatch() {
        let vocab_size = 10;
        let draft_tokens = vec![3_u32, 5, 7];
        let draft_len = draft_tokens.len();

        // Position 0 matches (3), position 1 mismatch (predict 6 instead of 5)
        let mut logits_data = vec![0.0_f32; draft_len * vocab_size];
        logits_data[0 * vocab_size + 3] = 10.0; // position 0 → 3 ✓
        logits_data[1 * vocab_size + 6] = 10.0; // position 1 → 6 ✗ (draft said 5)
        logits_data[2 * vocab_size + 7] = 10.0; // position 2 → 7 (not checked)

        let logits = Tensor::new(logits_data.as_slice(), &candle_core::Device::Cpu)
            .unwrap()
            .reshape((1, draft_len, vocab_size))
            .unwrap();

        let result = verify_draft(&draft_tokens, &logits).expect("verification should work");
        assert_eq!(result.accepted_count, 1);
        assert_eq!(result.correction_token, Some(6));
    }
}
