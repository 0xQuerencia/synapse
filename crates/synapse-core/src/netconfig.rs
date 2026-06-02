//! Config-driven bootstrap and relay peer loading for global P2P.
//!
//! Reads `config/bootstrap.toml` (`peers = [...]`) and `config/relays.toml`
//! (`relays = [...]`). Entries must be full multiaddrs including a `/p2p/<id>`
//! component. Placeholder/unparseable entries are skipped with a warning so a
//! stale default never panics the node.

use std::{fs, path::Path};

use libp2p::Multiaddr;
use serde::Deserialize;
use tracing::warn;

#[derive(Debug, Default, Deserialize)]
struct BootstrapFile {
    #[serde(default)]
    peers: Vec<String>,
}

#[derive(Debug, Default, Deserialize)]
struct RelaysFile {
    #[serde(default)]
    relays: Vec<String>,
}

/// Resolved network entry points for global discovery and NAT traversal.
#[derive(Debug, Default, Clone)]
pub struct NetConfig {
    pub bootstrap: Vec<Multiaddr>,
    pub relays: Vec<Multiaddr>,
}

impl NetConfig {
    /// Load from the given config directory, falling back to empty lists.
    pub fn load(config_dir: &Path) -> Self {
        let bootstrap = parse_file::<BootstrapFile>(&config_dir.join("bootstrap.toml"))
            .map(|f| valid_addrs(f.peers))
            .unwrap_or_default();
        let relays = parse_file::<RelaysFile>(&config_dir.join("relays.toml"))
            .map(|f| valid_addrs(f.relays))
            .unwrap_or_default();
        Self { bootstrap, relays }
    }

    /// Load from the default `config/` directory relative to the working dir.
    pub fn load_default() -> Self {
        Self::load(Path::new("config"))
    }
}

fn parse_file<T: serde::de::DeserializeOwned>(path: &Path) -> Option<T> {
    let contents = fs::read_to_string(path).ok()?;
    match toml::from_str::<T>(&contents) {
        Ok(parsed) => Some(parsed),
        Err(error) => {
            warn!(path = %path.display(), %error, "failed to parse network config");
            None
        }
    }
}

/// Parse multiaddrs, skipping entries without a `/p2p/<peer-id>` suffix.
fn valid_addrs(raw: Vec<String>) -> Vec<Multiaddr> {
    raw.into_iter()
        .filter_map(|entry| match entry.parse::<Multiaddr>() {
            Ok(addr) if has_peer_id(&addr) => Some(addr),
            Ok(_) => {
                warn!(%entry, "skipping multiaddr without /p2p peer id");
                None
            }
            Err(error) => {
                warn!(%entry, %error, "skipping unparseable multiaddr");
                None
            }
        })
        .collect()
}

fn has_peer_id(addr: &Multiaddr) -> bool {
    addr.iter()
        .any(|p| matches!(p, libp2p::multiaddr::Protocol::P2p(_)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn skips_placeholder_and_keeps_valid() {
        let dir = tempfile::tempdir().unwrap();
        let valid = "/dns4/relay.example.com/tcp/4001/p2p/12D3KooWA9hp9znQQz3oPjijjbF2bszk9wbStV1Km1AvkV2x2YfA";
        let mut f = fs::File::create(dir.path().join("relays.toml")).unwrap();
        writeln!(
            f,
            "relays = [\n  \"{valid}\",\n  \"/dns4/x/tcp/1/p2p/QmRelayPlaceholder\",\n]"
        )
        .unwrap();

        let cfg = NetConfig::load(dir.path());
        assert_eq!(cfg.relays.len(), 1, "valid relay kept, placeholder dropped");
        assert!(cfg.bootstrap.is_empty());
    }

    #[test]
    fn missing_files_yield_empty() {
        let dir = tempfile::tempdir().unwrap();
        let cfg = NetConfig::load(dir.path());
        assert!(cfg.bootstrap.is_empty() && cfg.relays.is_empty());
    }
}
