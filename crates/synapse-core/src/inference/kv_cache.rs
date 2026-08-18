//! Incremental KV-cache transfer for distributed inference.
//!
//! Tracks the KV-cache state per generation hop and provides delta
//! compression: only new KV pairs are sent after the first hop,
//! reducing per-token transfer overhead from O(sequence_length) to O(1).

use std::collections::HashMap;

/// Tracks KV-cache state for incremental transfer across hops.
///
/// The `full_cache` accumulates all KV pairs seen so far in the generation.
/// The `last_sent_length` watermark allows extracting only the delta (new
/// KV pairs) since the last transmission.
#[derive(Clone, Debug)]
pub struct IncrementalKVCache {
    /// Watermark: byte offset of the last successfully sent chunk.
    pub last_sent_length: usize,
    /// Complete KV-cache buffer accumulated so far.
    pub full_cache: Vec<u8>,
}

impl IncrementalKVCache {
    /// Create a new empty KV-cache tracker.
    pub fn new() -> Self {
        Self {
            last_sent_length: 0,
            full_cache: Vec::new(),
        }
    }

    /// Create a tracker initialized with a full KV-cache buffer.
    pub fn from_full(cache: Vec<u8>) -> Self {
        Self {
            last_sent_length: 0,
            full_cache: cache,
        }
    }

    /// Extract the delta (new KV pairs) since the last call to this method.
    ///
    /// Advances the `last_sent_length` watermark to the current cache size.
    /// The first call returns the entire cache; subsequent calls return only
    /// new additions.
