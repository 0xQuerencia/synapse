//! Config-driven bootstrap and relay peer loading for global P2P.
//!
//! Reads `config/bootstrap.toml` (`peers = [...]`) and `config/relays.toml`
//! (`relays = [...]`). Entries must be full multiaddrs including a `/p2p/<id>`
//! component. Placeholder/unparseable entries are skipped with a warning so a
//! stale default never panics the node.

use std::{collections::BTreeSet, fs, io::Read, path::Path};

use libp2p::{Multiaddr, multiaddr::Protocol};
use serde::Deserialize;
use tracing::warn;

#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct BootstrapFile {
    #[serde(default)]
    peers: Vec<String>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct RelaysFile {
    #[serde(default)]
    relays: Vec<String>,
}

/// Resolved network entry points for global discovery and NAT traversal.
#[derive(Debug, Default, Clone)]
pub struct NetConfig {
    pub bootstrap: Vec<Multiaddr>,
    pub relays: Vec<Multiaddr>,
    pub discovery: DiscoveryPolicy,
}

pub const MAX_ENTRY_POINTS: usize = 64;
const MAX_CONFIG_BYTES: u64 = 64 * 1024;

#[derive(Debug, thiserror::Error)]
pub enum NetConfigError {
    #[error("cannot read network configuration: {0}")]
    Io(#[from] std::io::Error),
    #[error("invalid network configuration: {0}")]
    Invalid(String),
}

/// Discovery supplies candidates, never membership. Internet/cache candidates
/// must be pinned explicitly; a DHT provider record cannot authorize a peer.
#[derive(Clone, Debug, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct DiscoveryPolicy {
    pub allow_lan_discovery: bool,
    pub allow_dht_discovery: bool,
    pub allowed_peers: BTreeSet<String>,
}

#[derive(Clone, Copy, Debug)]
pub enum DiscoverySource {
    Lan,
    Dht,
    Cache,
}

impl DiscoveryPolicy {
    pub fn allows(&self, peer: &libp2p::PeerId, source: DiscoverySource) -> bool {
        let pinned = self.allowed_peers.contains(&peer.to_string());
        match source {
            DiscoverySource::Lan => self.allow_lan_discovery && pinned,
            DiscoverySource::Dht => self.allow_dht_discovery && pinned,
            DiscoverySource::Cache => pinned,
        }
    }

    pub fn validate(&self) -> Result<(), NetConfigError> {
        if self.allowed_peers.len() > MAX_ENTRY_POINTS
            || self
                .allowed_peers
                .iter()
                .any(|p| p.parse::<libp2p::PeerId>().is_err())
        {
            return Err(NetConfigError::Invalid(
                "invalid or excessive pinned peers".into(),
            ));
        }
        Ok(())
    }
}

/// Relay metadata with region and latency tracking.
#[derive(Debug, Clone)]
pub struct RelayInfo {
    pub addr: Multiaddr,
    pub region: Option<String>,
    pub measured_latency_ms: Option<u64>,
}

impl RelayInfo {
    pub fn new(addr: Multiaddr) -> Self {
        let region = Self::parse_region(&addr);
        Self {
            addr,
            region,
            measured_latency_ms: None,
        }
    }

    /// Extract region from DNS name like /dns4/relay-eu.synapse.network/...
    pub fn parse_region(addr: &Multiaddr) -> Option<String> {
        for component in addr.iter() {
            if let Protocol::Dns4(hostname) | Protocol::Dns6(hostname) = component {
                // Look for "relay-<region>" pattern
                if let Some(region_part) = hostname.strip_prefix("relay-")
                    && let Some(region_code) = region_part.split('.').next()
                {
                    return Some(region_code.to_string());
                }
            }
        }
        None
    }
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
        Self {
            bootstrap,
            relays,
            discovery: DiscoveryPolicy::default(),
        }
    }

    /// Runtime entry point: absent files are valid, malformed supplied files
    /// fail startup. Never quietly turn a configured private network into a
    /// different or unconfigured network.
    pub fn load_strict(config_dir: &Path) -> Result<Self, NetConfigError> {
        let bootstrap = read_optional::<BootstrapFile>(&config_dir.join("bootstrap.toml"))?;
        let relays = read_optional::<RelaysFile>(&config_dir.join("relays.toml"))?;
        let discovery = read_optional::<DiscoveryPolicy>(&config_dir.join("discovery.toml"))?;
        discovery.validate()?;
        Ok(Self {
            bootstrap: strict_addrs(bootstrap.peers)?,
            relays: strict_addrs(relays.relays)?,
            discovery,
        })
    }

    pub fn load_strict_default() -> Result<Self, NetConfigError> {
        Self::load_strict(Path::new("config"))
    }

    /// Load from the default `config/` directory relative to the working dir.
    pub fn load_default() -> Self {
        Self::load(Path::new("config"))
    }
}

fn read_optional<T: serde::de::DeserializeOwned + Default>(
    path: &Path,
) -> Result<T, NetConfigError> {
    let file = match fs::File::open(path) {
        Ok(file) => file,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(T::default()),
        Err(error) => return Err(error.into()),
    };
    let mut contents = String::new();
    file.take(MAX_CONFIG_BYTES + 1)
        .read_to_string(&mut contents)?;
    if contents.len() as u64 > MAX_CONFIG_BYTES {
        return Err(NetConfigError::Invalid("file exceeds 64 KiB".into()));
    }
    toml::from_str(&contents).map_err(|error| NetConfigError::Invalid(error.to_string()))
}

fn strict_addrs(entries: Vec<String>) -> Result<Vec<Multiaddr>, NetConfigError> {
    if entries.len() > MAX_ENTRY_POINTS {
        return Err(NetConfigError::Invalid("too many entry points".into()));
    }
    let mut result = Vec::new();
    for value in entries {
        let addr = value
            .parse::<Multiaddr>()
            .map_err(|_| NetConfigError::Invalid("invalid multiaddr".into()))?;
        validate_direct_peer_addr(&addr)?;
        if !result.contains(&addr) {
            result.push(addr);
        }
    }
    Ok(result)
}

/// Require exactly one host, an enabled transport, and a final pinned peer ID.
/// Relay *endpoints* are direct addresses; circuit routes are constructed only
/// after validating the relay and final target separately.
pub fn validate_direct_peer_addr(addr: &Multiaddr) -> Result<libp2p::PeerId, NetConfigError> {
    let bad = || {
        NetConfigError::Invalid(
            "expected host/tcp or host/udp/quic-v1 followed by /p2p/<peer>".into(),
        )
    };
    let mut parts = addr.iter();
    match parts.next() {
        Some(Protocol::Ip4(ip))
            if !ip.is_unspecified() && !ip.is_multicast() && !ip.is_broadcast() => {}
        Some(Protocol::Ip6(ip)) if !ip.is_unspecified() && !ip.is_multicast() => {}
        Some(Protocol::Dns(host) | Protocol::Dns4(host) | Protocol::Dns6(host))
            if !host.is_empty()
                && host.len() <= 253
                && host
                    .bytes()
                    .all(|b| b.is_ascii_alphanumeric() || b".-".contains(&b)) => {}
        _ => return Err(bad()),
    }
    match parts.next() {
        Some(Protocol::Tcp(port)) if port != 0 => {}
        Some(Protocol::Udp(port)) if port != 0 => {
            if !matches!(parts.next(), Some(Protocol::QuicV1)) {
                return Err(bad());
            }
        }
        _ => return Err(bad()),
    }
    let Some(Protocol::P2p(peer)) = parts.next() else {
        return Err(bad());
    };
    if parts.next().is_some() {
        return Err(bad());
    }
    Ok(peer)
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

    #[test]
    fn strict_loading_rejects_bad_config_instead_of_skipping_it() {
        let dir = tempfile::tempdir().unwrap();
        fs::write(
            dir.path().join("bootstrap.toml"),
            "peers = [\"/ip4/0.0.0.0/tcp/42\"]",
        )
        .unwrap();
        assert!(NetConfig::load_strict(dir.path()).is_err());
        fs::write(dir.path().join("bootstrap.toml"), "typo = true").unwrap();
        assert!(NetConfig::load_strict(dir.path()).is_err());
        fs::write(dir.path().join("bootstrap.toml"), " ".repeat(65537)).unwrap();
        assert!(NetConfig::load_strict(dir.path()).is_err());
    }

    #[test]
    fn validates_direct_transports_and_exact_peer_suffix() {
        let peer = libp2p::PeerId::random();
        for prefix in [
            "/ip4/127.0.0.1/tcp/4001",
            "/ip6/::1/udp/4001/quic-v1",
            "/dns4/relay.example.com/tcp/443",
        ] {
            assert_eq!(
                validate_direct_peer_addr(&format!("{prefix}/p2p/{peer}").parse().unwrap())
                    .unwrap(),
                peer
            );
        }
        for prefix in [
            "/ip4/0.0.0.0/tcp/4001",
            "/ip4/224.0.0.1/tcp/4001",
            "/ip4/127.0.0.1/tcp/0",
            "/ip4/127.0.0.1/udp/4001",
        ] {
            assert!(
                validate_direct_peer_addr(&format!("{prefix}/p2p/{peer}").parse().unwrap())
                    .is_err()
            );
        }
        let addr = format!("/ip4/127.0.0.1/tcp/4001/p2p/{peer}/p2p-circuit")
            .parse()
            .unwrap();
        assert!(validate_direct_peer_addr(&addr).is_err());
    }

    #[test]
    fn candidate_discovery_does_not_authorize_unknown_peers() {
        let peer = libp2p::PeerId::random();
        let mut policy = DiscoveryPolicy::default();
        assert!(!policy.allows(&peer, DiscoverySource::Dht));
        policy.allow_dht_discovery = true;
        assert!(!policy.allows(&peer, DiscoverySource::Dht));
        policy.allowed_peers.insert(peer.to_string());
        assert!(policy.allows(&peer, DiscoverySource::Dht));
        assert!(!policy.allows(&peer, DiscoverySource::Lan));
        assert!(!policy.allows(&libp2p::PeerId::random(), DiscoverySource::Cache));
    }
}
