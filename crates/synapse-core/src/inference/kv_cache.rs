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
    pub fn get_delta(&mut self) -> &[u8] {
        let delta = &self.full_cache[self.last_sent_length..];
        self.last_sent_length = self.full_cache.len();
        delta
    }

    /// Append a delta received from another node to the local cache.
    ///
    /// This reconstructs the full KV-cache incrementally as deltas arrive
    /// from upstream hops.
    pub fn apply_delta(&mut self, delta: &[u8]) {
        self.full_cache.extend_from_slice(delta);
    }

    /// Get the full accumulated KV-cache.
    pub fn full_cache(&self) -> &[u8] {
        &self.full_cache
    }

    /// Get the current size of the full cache in bytes.
    pub fn size(&self) -> usize {
        self.full_cache.len()
    }

    /// Reset the cache to empty (used when starting a new generation).
    pub fn reset(&mut self) {
        self.full_cache.clear();
        self.last_sent_length = 0;
    }
}

impl Default for IncrementalKVCache {
    fn default() -> Self {
        Self::new()
    }
}

/// Per-generation KV-cache tracker for the master node.
///
/// Maps generation trace IDs to their incremental KV-cache state.
/// Allows the master to track what has been sent for each active generation
/// and extract deltas for subsequent hops.
#[derive(Default)]
pub struct KVCacheTracker {
    /// Active KV-cache states indexed by trace ID.
    caches: HashMap<String, IncrementalKVCache>,
}

impl KVCacheTracker {
    /// Create a new empty tracker.
    pub fn new() -> Self {
        Self {
            caches: HashMap::new(),
        }
    }

    /// Get or create a KV-cache for the given trace ID.
    pub fn get_or_create(&mut self, trace_id: &str) -> &mut IncrementalKVCache {
        self.caches.entry(trace_id.to_owned()).or_default()
    }

    /// Remove a completed generation's KV-cache.
    pub fn remove(&mut self, trace_id: &str) -> Option<IncrementalKVCache> {
        self.caches.remove(trace_id)
    }

    /// Clear all tracked caches (e.g., on node reset).
    pub fn clear(&mut self) {
        self.caches.clear();
    }

    /// Get the number of active generations being tracked.
    pub fn len(&self) -> usize {
        self.caches.len()
    }

    /// Check if any generations are being tracked.
    pub fn is_empty(&self) -> bool {
        self.caches.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_incremental_kv_cache_delta() {
        let mut cache = IncrementalKVCache::new();

        // Start with initial data
        cache.full_cache = vec![1, 2, 3, 4];

        // First delta returns everything
        let delta1 = cache.get_delta();
        assert_eq!(delta1, &[1, 2, 3, 4]);

        // Add more data
        cache.apply_delta(&[5, 6]);

        // Second delta returns only new data
        let delta2 = cache.get_delta();
        assert_eq!(delta2, &[5, 6]);

        // Full cache contains everything
        assert_eq!(cache.full_cache(), &[1, 2, 3, 4, 5, 6]);
    }

    #[test]
    fn test_kv_cache_tracker() {
        let mut tracker = KVCacheTracker::new();

        // Create cache for trace1
        let cache1 = tracker.get_or_create("trace1");
        cache1.full_cache = vec![1, 2, 3];

        // Create cache for trace2
        let cache2 = tracker.get_or_create("trace2");
        cache2.full_cache = vec![4, 5, 6];

        assert_eq!(tracker.len(), 2);

        // Remove trace1
        let removed = tracker.remove("trace1");
        assert!(removed.is_some());
        assert_eq!(removed.unwrap().full_cache, vec![1, 2, 3]);
        assert_eq!(tracker.len(), 1);
    }
}
