//! Persistent node identity.
//!
//! Nodes generate an Ed25519 keypair on first run and persist it to disk.
//! This allows the same node to keep its PeerId across restarts, which is
//! required for DHT stability and peer reputation.

use std::fs;
use std::path::{Path, PathBuf};

use libp2p::PeerId;
use libp2p::identity::Keypair;
use tracing::{info, warn};

/// Default directory for Synapse configuration and identity.
const SYNAPSE_DIR: &str = ".synapse";
/// Filename for the persisted Ed25519 keypair.
const IDENTITY_FILE: &str = "identity.key";

/// Errors that can occur during identity operations.
#[derive(Debug, thiserror::Error)]
pub enum IdentityError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("invalid key data: {0}")]
    InvalidKey(String),
    #[error("home directory not found")]
    NoHomeDir,
}

/// A persistent node identity backed by an Ed25519 keypair.
#[derive(Clone)]
pub struct NodeIdentity {
    keypair: Keypair,
    peer_id: PeerId,
    key_path: PathBuf,
}

impl NodeIdentity {
    /// Load an existing identity from disk, or create a new one.
    pub fn load_or_create() -> Result<Self, IdentityError> {
        let key_path = default_key_path_with_override()?;

        if key_path.exists() {
            Self::load_from_path(&key_path)
        } else {
            Self::create_at_path(&key_path)
        }
    }

    /// Load from a specific path.
    pub fn load_from_path(path: &Path) -> Result<Self, IdentityError> {
        let bytes = fs::read(path).map_err(|e| {
            warn!(path = ?path, error = %e, "failed to read identity file");
            e
        })?;

        let keypair = Keypair::from_protobuf_encoding(&bytes)
            .map_err(|e| IdentityError::InvalidKey(e.to_string()))?;

        let peer_id = keypair.public().to_peer_id();
        info!(%peer_id, path = ?path, "loaded persisted identity");

        Ok(Self {
            keypair,
            peer_id,
            key_path: path.to_path_buf(),
        })
    }

    /// Generate a new identity and persist it.
    fn create_at_path(path: &Path) -> Result<Self, IdentityError> {
        // Ensure parent directory exists.
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }

        let keypair = Keypair::generate_ed25519();
        let peer_id = keypair.public().to_peer_id();

        let bytes = keypair
            .to_protobuf_encoding()
            .map_err(|e| IdentityError::InvalidKey(e.to_string()))?;

        fs::write(path, &bytes)?;
        info!(%peer_id, path = ?path, "generated and persisted new identity");

        Ok(Self {
            keypair,
            peer_id,
            key_path: path.to_path_buf(),
        })
    }

    /// Get the libp2p PeerId derived from this identity.
    pub fn peer_id(&self) -> PeerId {
        self.peer_id
    }

    /// Get the Ed25519 public key bytes.
    pub fn public_key_bytes(&self) -> Vec<u8> {
        self.keypair.public().encode_protobuf()
    }

    /// Get a reference to the underlying libp2p Keypair.
    pub fn keypair(&self) -> &Keypair {
        &self.keypair
    }

    /// Get the path to the identity key file.
    pub fn key_path(&self) -> &Path {
        &self.key_path
    }

    /// Sign arbitrary bytes with the node's Ed25519 key.
    ///
    /// Returns the raw signature bytes (64 bytes for Ed25519).
    pub fn sign(&self, data: &[u8]) -> Result<Vec<u8>, IdentityError> {
        self.keypair
            .sign(data)
            .map_err(|e| IdentityError::InvalidKey(e.to_string()))
    }

    /// Verify a signature produced by a remote peer's key.
    ///
    /// `public_key_bytes` must be the protobuf-encoded public key
    /// (as returned by `NodeIdentity::public_key_bytes()`).
    pub fn verify_remote(
        public_key_bytes: &[u8],
        data: &[u8],
        signature: &[u8],
    ) -> Result<(), IdentityError> {
        let public_key = libp2p::identity::PublicKey::try_decode_protobuf(public_key_bytes)
            .map_err(|e| IdentityError::InvalidKey(e.to_string()))?;
        if public_key.verify(data, signature) {
            Ok(())
        } else {
            Err(IdentityError::InvalidKey(
                "signature verification failed".to_owned(),
            ))
        }
    }

    /// Display the identity as a human-readable string.
    pub fn display_short(&self) -> String {
        let peer_str = self.peer_id.to_string();
        // Show first 8 and last 4 chars of the PeerId for readability.
        if peer_str.len() > 12 {
            format!("{}...{}", &peer_str[..8], &peer_str[peer_str.len() - 4..])
        } else {
            peer_str
        }
    }
}

/// Default path: `~/.synapse/identity.key`
///
/// Overridable with the `SYNAPSE_IDENTITY_PATH` env var — useful for
/// running two nodes on the same machine (e.g. master + worker in
/// separate terminals) that need distinct libp2p PeerIds, since
/// libp2p refuses connections to a peer's own ID.
fn default_key_path() -> Result<PathBuf, IdentityError> {
    if let Ok(p) = std::env::var("SYNAPSE_IDENTITY_PATH") {
        if !p.is_empty() {
            return Ok(PathBuf::from(p));
        }
    }
    let home = home_dir().ok_or(IdentityError::NoHomeDir)?;
    Ok(home.join(SYNAPSE_DIR).join(IDENTITY_FILE))
}

/// Process-wide override for the identity key path. Set this before
/// calling `load_or_create` to bypass the env-var path lookup
/// entirely — handy for embedders and tests that want to avoid
/// `unsafe { std::env::set_var(...) }` (which is unsafe in Rust 2024).
///
/// The GUI runs its HTTP handlers on a thread pool separate from
/// `main`, so this has to be a process-global rather than a
/// `thread_local!`.
static IDENTITY_PATH_OVERRIDE: std::sync::Mutex<Option<PathBuf>> =
    std::sync::Mutex::new(None);

/// Install a process-wide override for the identity key path used by
/// `load_or_create`. Pass `None` to clear the override. While set,
/// it takes precedence over the `SYNAPSE_IDENTITY_PATH` env var and
/// the default `~/.synapse/identity.key`.
pub fn set_identity_path_override(path: Option<&Path>) {
    let mut guard = IDENTITY_PATH_OVERRIDE
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    *guard = path.map(|p| p.to_path_buf());
}

fn current_override() -> Option<PathBuf> {
    IDENTITY_PATH_OVERRIDE
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .clone()
}

fn default_key_path_with_override() -> Result<PathBuf, IdentityError> {
    if let Some(p) = current_override() {
        return Ok(p);
    }
    default_key_path()
}

/// Get the user's home directory.
fn home_dir() -> Option<PathBuf> {
    // Try $HOME first (works on Unix and most environments).
    if let Ok(home) = std::env::var("HOME") {
        let path = PathBuf::from(home);
        if path.is_dir() {
            return Some(path);
        }
    }

    // Fallback: platform-specific.
    #[cfg(target_os = "windows")]
    {
        std::env::var("USERPROFILE")
            .ok()
            .map(PathBuf::from)
            .filter(|p| p.is_dir())
    }

    #[cfg(not(target_os = "windows"))]
    {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn creates_new_identity_when_no_file_exists() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("identity.key");

        let identity = NodeIdentity::create_at_path(&path).unwrap();
        assert!(path.exists());
        assert_eq!(identity.key_path(), path);

        // Should be loadable.
        let loaded = NodeIdentity::load_from_path(&path).unwrap();
        assert_eq!(identity.peer_id(), loaded.peer_id());
    }

    #[test]
    fn persists_across_reloads() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("identity.key");

        let id1 = NodeIdentity::create_at_path(&path).unwrap();
        let id2 = NodeIdentity::load_from_path(&path).unwrap();

        assert_eq!(id1.peer_id(), id2.peer_id());
        assert_eq!(id1.public_key_bytes(), id2.public_key_bytes());
    }

    #[test]
    fn rejects_corrupted_key_file() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("identity.key");
        fs::write(&path, b"not a valid key").unwrap();

        let result = NodeIdentity::load_from_path(&path);
        assert!(result.is_err());
    }

    #[test]
    fn display_short_works() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("identity.key");
        let identity = NodeIdentity::create_at_path(&path).unwrap();
        let display = identity.display_short();
        assert!(display.contains("..."));
    }
}
