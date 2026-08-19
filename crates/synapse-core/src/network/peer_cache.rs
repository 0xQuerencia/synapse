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
                        warn!(%error, "failed to parse peer cache, starting fresh");
                        Self::default()
                    }
                },
                Err(_) => {
                    info!("no existing peer cache, starting fresh");
                    Self::default()
                }
            },
            Err(error) => {
                warn!(%error, "failed to determine cache path");
                Self::default()
            }
        }
    }

    /// Save cache to disk.
    pub fn save(&self) {
        match Self::cache_path() {
            Ok(path) => {
                if let Some(parent) = path.parent() {
                    let _ = fs::create_dir_all(parent);
                }
                match serde_json::to_string_pretty(self) {
                    Ok(json) => {
                        if let Err(error) = fs::write(&path, json) {
                            warn!(%error, "failed to write peer cache");
                        } else {
                            info!(path = %path.display(), "saved peer cache");
                        }
                    }
                    Err(error) => {
                        warn!(%error, "failed to serialize peer cache");
                    }
                }
            }
            Err(error) => {
                warn!(%error, "failed to determine cache path");
            }
        }
    }

    /// Get fresh cached peers for a model (TTL < 5 minutes).
    pub fn get_providers(&self, model_id: &str) -> Vec<(PeerId, Vec<Multiaddr>)> {
        let now = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap()
            .as_secs();

        self.providers
            .get(model_id)
            .map(|entries| {
                entries
                    .iter()
                    .filter(|entry| now - entry.cached_at < CACHE_TTL.as_secs())
                    .filter_map(|entry| {
                        let peer_id = entry.peer_id.parse::<PeerId>().ok()?;
                        let addrs: Vec<Multiaddr> =
                            entry.addrs.iter().filter_map(|s| s.parse().ok()).collect();
                        if addrs.is_empty() {
                            None
                        } else {
                            Some((peer_id, addrs))
                        }
                    })
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Add or update a provider entry.
    pub fn add_provider(&mut self, model_id: String, peer_id: PeerId, addrs: Vec<Multiaddr>) {
        let now = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap()
            .as_secs();

        let entry = CachedPeer {
            peer_id: peer_id.to_string(),
            addrs: addrs.iter().map(|a| a.to_string()).collect(),
            model_id: model_id.clone(),
            cached_at: now,
        };

        self.providers
            .entry(model_id)
            .or_default()
            .retain(|e| e.peer_id != entry.peer_id); // Remove old entry if exists
        self.providers.get_mut(&entry.model_id).unwrap().push(entry);
    }

    /// Prune stale entries older than TTL.
    pub fn prune_stale(&mut self) {
        let now = SystemTime::now()
