//! DHT peer discovery cache for faster cold starts.
//!
//! Caches discovered providers to disk so workers can reconnect to known
//! masters without waiting for DHT queries (which can take 5-30 seconds).

use libp2p::{Multiaddr, PeerId};
use serde::{Deserialize, Serialize};
use std::{
    collections::HashMap,
    fs,
    path::{Path, PathBuf},
    time::{Duration, SystemTime},
};
use tracing::{info, warn};

const CACHE_FILE_NAME: &str = "peer_cache.json";
const CACHE_TTL: Duration = Duration::from_secs(5 * 60); // 5 minutes

/// Cached peer discovery entry with timestamp.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CachedPeer {
    pub peer_id: String,
    pub addrs: Vec<String>,
    pub model_id: String,
    pub cached_at: u64, // Unix timestamp
}

/// Persistent peer cache stored at ~/.synapse/peer_cache.json
#[derive(Debug, Default, Serialize, Deserialize)]
pub struct PeerCache {
    /// Map: model_id → list of known provider peers
    #[serde(default)]
    pub providers: HashMap<String, Vec<CachedPeer>>,
}

impl PeerCache {
    /// Load cache from disk, or create empty if missing/invalid.
    pub fn load() -> Self {
        match Self::cache_path() {
            Ok(path) => match fs::read_to_string(&path) {
                Ok(contents) => match serde_json::from_str::<PeerCache>(&contents) {
                    Ok(cache) => {
                        info!(path = %path.display(), entries = cache.providers.len(), "loaded peer cache");
                        cache
                    }
                    Err(error) => {
