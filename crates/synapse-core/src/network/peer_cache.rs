//! Bounded, expiring discovery hints. Cached addresses are never authority.

use libp2p::{Multiaddr, PeerId, multiaddr::Protocol};
use serde::{Deserialize, Serialize};
use std::{
    collections::HashMap,
    fs,
    io::Read,
    path::{Path, PathBuf},
    time::{Duration, SystemTime},
};
use tracing::warn;

const CACHE_FILE_NAME: &str = "peer_cache.json";
const CACHE_TTL: Duration = Duration::from_secs(5 * 60);
const MAX_CACHE_BYTES: u64 = 1024 * 1024;
const MAX_MODELS: usize = 64;
const MAX_PEERS_PER_MODEL: usize = 64;
const MAX_ADDRS_PER_PEER: usize = 8;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CachedPeer {
    pub peer_id: String,
    pub addrs: Vec<String>,
    pub model_id: String,
    pub cached_at: u64,
}

#[derive(Debug, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PeerCache {
    #[serde(default)]
    pub providers: HashMap<String, Vec<CachedPeer>>,
}

fn now_seconds() -> u64 {
    SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

fn fresh(timestamp: u64, now: u64) -> bool {
    now.checked_sub(timestamp)
        .is_some_and(|age| age < CACHE_TTL.as_secs())
}

fn valid_model(model: &str) -> bool {
    !model.is_empty()
        && model.len() <= crate::protocol::MAX_MODEL_ID_LEN
        && model.bytes().all(|byte| byte.is_ascii_graphic())
}

fn pinned_address(peer: PeerId, addr: &Multiaddr) -> Option<Multiaddr> {
    let mut candidate = addr.clone();
    if !matches!(candidate.iter().last(), Some(Protocol::P2p(_))) {
        candidate.push(Protocol::P2p(peer));
    }
    (crate::netconfig::validate_direct_peer_addr(&candidate).ok()? == peer).then_some(candidate)
}

impl PeerCache {
    pub fn load() -> Self {
        Self::cache_path()
            .ok()
            .and_then(|path| Self::load_from_path(&path).ok())
            .unwrap_or_default()
    }

    pub fn load_from_path(path: &Path) -> std::io::Result<Self> {
        let file = fs::File::open(path)?;
        let mut bytes = Vec::new();
        file.take(MAX_CACHE_BYTES + 1).read_to_end(&mut bytes)?;
        if bytes.len() as u64 > MAX_CACHE_BYTES {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "peer cache exceeds size limit",
            ));
        }
        let mut cache: Self = serde_json::from_slice(&bytes).map_err(std::io::Error::other)?;
        if cache.providers.len() > MAX_MODELS
            || cache
                .providers
                .values()
                .any(|peers| peers.len() > MAX_PEERS_PER_MODEL)
        {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "peer cache exceeds entry limit",
            ));
        }
        cache.prune_stale();
        Ok(cache)
    }

    pub fn save(&self) {
        if let Err(error) = Self::cache_path().and_then(|path| self.save_to_path(&path)) {
            warn!(%error, "failed to persist peer discovery cache");
        }
    }

    pub fn save_to_path(&self, path: &Path) -> std::io::Result<()> {
        let bytes = serde_json::to_vec(self).map_err(std::io::Error::other)?;
        if bytes.len() as u64 > MAX_CACHE_BYTES {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "peer cache exceeds size limit",
            ));
        }
        crate::identity::write_private_state(path, &bytes)
    }

    pub fn get_providers(&self, model_id: &str) -> Vec<(PeerId, Vec<Multiaddr>)> {
        self.get_providers_at(model_id, now_seconds())
    }

    pub fn get_providers_at(&self, model_id: &str, now: u64) -> Vec<(PeerId, Vec<Multiaddr>)> {
        if !valid_model(model_id) {
            return Vec::new();
        }
        self.providers
            .get(model_id)
            .into_iter()
            .flatten()
            .take(MAX_PEERS_PER_MODEL)
            .filter(|entry| entry.model_id == model_id && fresh(entry.cached_at, now))
            .filter_map(|entry| {
                let peer_id = entry.peer_id.parse::<PeerId>().ok()?;
                let addrs: Vec<_> = entry
                    .addrs
                    .iter()
                    .take(MAX_ADDRS_PER_PEER)
                    .filter_map(|value| value.parse::<Multiaddr>().ok())
                    .filter_map(|addr| pinned_address(peer_id, &addr))
                    .collect();
                (!addrs.is_empty()).then_some((peer_id, addrs))
            })
            .collect()
    }

    pub fn add_provider(&mut self, model_id: String, peer_id: PeerId, addrs: Vec<Multiaddr>) {
        self.prune_stale();
        if !valid_model(&model_id)
            || (!self.providers.contains_key(&model_id) && self.providers.len() >= MAX_MODELS)
        {
            return;
        }
        let addrs: Vec<_> = addrs
            .iter()
            .take(MAX_ADDRS_PER_PEER)
            .filter_map(|addr| pinned_address(peer_id, addr))
            .map(|addr| addr.to_string())
            .collect();
        if addrs.is_empty() {
            return;
        }
        let entry = CachedPeer {
            peer_id: peer_id.to_string(),
            addrs,
            model_id: model_id.clone(),
            cached_at: now_seconds(),
        };
        let peers = self.providers.entry(model_id).or_default();
        peers.retain(|peer| peer.peer_id != entry.peer_id);
        if peers.len() < MAX_PEERS_PER_MODEL {
            peers.push(entry);
        }
    }

    pub fn remove_peer(&mut self, peer: PeerId) {
        let key = peer.to_string();
        for peers in self.providers.values_mut() {
            peers.retain(|entry| entry.peer_id != key);
        }
        self.providers.retain(|_, peers| !peers.is_empty());
    }

    pub fn prune_stale(&mut self) {
        let now = now_seconds();
        for (model, entries) in &mut self.providers {
            entries.retain(|entry| entry.model_id == *model && fresh(entry.cached_at, now));
            entries.truncate(MAX_PEERS_PER_MODEL);
        }
        self.providers
            .retain(|model, entries| valid_model(model) && !entries.is_empty());
    }

    fn cache_path() -> std::io::Result<PathBuf> {
        let home_dir = std::env::var_os("HOME")
            .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::NotFound, "HOME not set"))?;
        Ok(PathBuf::from(home_dir)
            .join(".synapse")
            .join(CACHE_FILE_NAME))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cache() -> (PeerCache, PeerId) {
        let mut cache = PeerCache::default();
        let peer = PeerId::random();
        cache.add_provider(
            "model-a".into(),
            peer,
            vec!["/ip4/127.0.0.1/tcp/4001".parse().unwrap()],
        );
        (cache, peer)
    }

    #[test]
    fn cache_ttl_filtering() {
        let (mut cache, _) = cache();
        assert_eq!(cache.get_providers("model-a").len(), 1);
        cache.providers.get_mut("model-a").unwrap()[0].cached_at = 0;
        assert!(cache.get_providers("model-a").is_empty());
    }

    #[test]
    fn future_timestamps_and_wrong_model_are_not_fresh() {
        let (mut cache, _) = cache();
        cache.providers.get_mut("model-a").unwrap()[0].cached_at = u64::MAX;
        assert!(cache.get_providers_at("model-a", 100).is_empty());
        let entry = &mut cache.providers.get_mut("model-a").unwrap()[0];
        entry.cached_at = 100;
        entry.model_id = "other-model".into();
        assert!(cache.get_providers_at("model-a", 100).is_empty());
    }

    #[test]
    fn cached_addresses_are_bound_to_the_named_peer() {
        let (mut cache, peer) = cache();
        let providers = cache.get_providers("model-a");
        assert_eq!(providers[0].1[0].iter().last(), Some(Protocol::P2p(peer)));
        let other = PeerId::random();
        cache.providers.get_mut("model-a").unwrap()[0].addrs =
            vec![format!("/ip4/127.0.0.1/tcp/4001/p2p/{other}")];
        assert!(cache.get_providers("model-a").is_empty());
    }

    #[test]
    fn bounded_cache_persists_and_revocation_removes_candidates() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("cache.json");
        let (mut cache, peer) = cache();
        cache.save_to_path(&path).unwrap();
        assert_eq!(
            PeerCache::load_from_path(&path)
                .unwrap()
                .get_providers("model-a")
                .len(),
            1
        );
        cache.remove_peer(peer);
        assert!(cache.get_providers("model-a").is_empty());
        fs::write(&path, vec![b' '; MAX_CACHE_BYTES as usize + 1]).unwrap();
        assert!(PeerCache::load_from_path(&path).is_err());
    }
}
