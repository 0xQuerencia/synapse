//! Persistent node identity.
//!
//! Nodes generate an Ed25519 keypair on first run and persist it to disk.
//! This allows the same node to keep its PeerId across restarts, which is
//! required for DHT stability and peer reputation.

use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use libp2p::PeerId;
use libp2p::identity::Keypair;
use tracing::{info, warn};

/// Default directory for Synapse configuration and identity.
const SYNAPSE_DIR: &str = ".synapse";
/// Filename for the persisted Ed25519 keypair.
const IDENTITY_FILE: &str = "identity.key";
const MAX_KEY_FILE_BYTES: u64 = 4096;

/// Errors that can occur during identity operations.
#[derive(Debug, thiserror::Error)]
pub enum IdentityError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("invalid key data: {0}")]
    InvalidKey(String),
    #[error("home directory not found")]
    NoHomeDir,
    #[error("identity file must be a regular, private file (Unix mode 0600)")]
    UnsafeFile,
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

        Self::load_or_create_at(&key_path)
    }

    /// Use an explicit path without a process-global override. Creation is
    /// exclusive: concurrent starters converge on the same persisted key.
    pub fn load_or_create_at(path: &Path) -> Result<Self, IdentityError> {
        match Self::load_from_path(path) {
            Ok(identity) => Ok(identity),
            Err(IdentityError::Io(error)) if error.kind() == std::io::ErrorKind::NotFound => {
                match Self::create_at_path(path) {
                    Err(IdentityError::Io(error))
                        if error.kind() == std::io::ErrorKind::AlreadyExists =>
                    {
                        Self::load_from_path(path)
                    }
                    result => result,
                }
            }
            error => error,
        }
    }

    /// Load from a specific path.
    pub fn load_from_path(path: &Path) -> Result<Self, IdentityError> {
        let mut file = open_private_file(path).map_err(|e| {
            warn!(path = ?path, error = %e, "failed to read identity file");
            e
        })?;
        let mut bytes = Vec::new();
        Read::by_ref(&mut file)
            .take(MAX_KEY_FILE_BYTES + 1)
            .read_to_end(&mut bytes)?;
        if bytes.len() as u64 > MAX_KEY_FILE_BYTES {
            return Err(IdentityError::InvalidKey(
                "key file exceeds size limit".into(),
            ));
        }

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
            create_private_dir(parent)?;
        }

        let keypair = Keypair::generate_ed25519();
        let peer_id = keypair.public().to_peer_id();

        let bytes = keypair
            .to_protobuf_encoding()
            .map_err(|e| IdentityError::InvalidKey(e.to_string()))?;

        publish_private_state_new(path, &bytes)?;
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

    /// Bind an application signature to the identity authenticated by libp2p.
    /// A valid signature from a different key is never accepted for `peer`.
    pub fn verify_remote_peer(
        peer: PeerId,
        public_key_bytes: &[u8],
        data: &[u8],
        signature: &[u8],
    ) -> Result<(), IdentityError> {
        let public_key = libp2p::identity::PublicKey::try_decode_protobuf(public_key_bytes)
            .map_err(|e| IdentityError::InvalidKey(e.to_string()))?;
        if public_key.to_peer_id() != peer || !public_key.verify(data, signature) {
            return Err(IdentityError::InvalidKey(
                "identity proof did not match peer".into(),
            ));
        }
        Ok(())
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

fn create_private_dir(path: &Path) -> Result<(), std::io::Error> {
    if path.as_os_str().is_empty() {
        return Ok(());
    }
    let mut builder = fs::DirBuilder::new();
    builder.recursive(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::DirBuilderExt;
        builder.mode(0o700);
    }
    builder.create(path)
}

fn private_create_options() -> OpenOptions {
    let mut options = OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600).custom_flags(libc::O_NOFOLLOW);
    }
    options
}

fn open_private_file(path: &Path) -> Result<File, IdentityError> {
    let before = fs::symlink_metadata(path)?;
    if !before.file_type().is_file() || before.len() > MAX_KEY_FILE_BYTES {
        return Err(IdentityError::UnsafeFile);
    }
    let mut options = OpenOptions::new();
    options.read(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.custom_flags(libc::O_NOFOLLOW);
    }
    let file = options.open(path)?;
    let metadata = file.metadata()?;
    if !metadata.is_file() || metadata.len() > MAX_KEY_FILE_BYTES {
        return Err(IdentityError::UnsafeFile);
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::{MetadataExt, PermissionsExt};
        if metadata.permissions().mode() & 0o077 != 0
            // SAFETY: geteuid has no arguments or pointer preconditions.
            || metadata.uid() != unsafe { libc::geteuid() }
        {
            return Err(IdentityError::UnsafeFile);
        }
    }
    Ok(file)
}

fn publish_private_state_new(path: &Path, bytes: &[u8]) -> std::io::Result<()> {
    use rand::RngCore;
    let parent = path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    let mut nonce = [0_u8; 16];
    rand::rngs::OsRng.fill_bytes(&mut nonce);
    let name: String = nonce.iter().map(|b| format!("{b:02x}")).collect();
    let temporary = parent.join(format!(".synapse-key-{name}.tmp"));
    let result = (|| {
        let mut file = private_create_options().open(&temporary)?;
        file.write_all(bytes)?;
        file.sync_all()?;
        // hard_link publishes a complete file atomically and fails if the
        // destination already exists, including when it is a symlink.
        fs::hard_link(&temporary, path)?;
        #[cfg(unix)]
        File::open(parent)?.sync_all()?;
        Ok(())
    })();
    let _ = fs::remove_file(&temporary);
    result
}

/// Atomically replace a small private state file in the same directory.
/// Temp files use exclusive creation, mode 0600 and an unpredictable name.
pub(crate) fn write_private_state(path: &Path, bytes: &[u8]) -> std::io::Result<()> {
    use rand::RngCore;
    let parent = path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    create_private_dir(parent)?;
    if let Ok(metadata) = fs::symlink_metadata(path)
        && !metadata.file_type().is_file()
    {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "state destination is not a regular file",
        ));
    }
    let mut nonce = [0_u8; 16];
    rand::rngs::OsRng.fill_bytes(&mut nonce);
    let name: String = nonce.iter().map(|b| format!("{b:02x}")).collect();
    let temporary = parent.join(format!(".synapse-{name}.tmp"));
    let result = (|| {
        let mut file = private_create_options().open(&temporary)?;
        file.write_all(bytes)?;
        file.sync_all()?;
        fs::rename(&temporary, path)?;
        #[cfg(unix)]
        File::open(parent)?.sync_all()?;
        Ok(())
    })();
    if result.is_err() {
        let _ = fs::remove_file(&temporary);
    }
    result
}

/// Serialize read/modify/write of a state file across processes. The lock file
/// remains in place; deleting it would let another process lock a new inode.
/// OS locks release automatically after crashes or when this File is dropped.
pub(crate) fn lock_private_state(path: &Path) -> std::io::Result<File> {
    let parent = path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    create_private_dir(parent)?;
    let mut lock_name = path.as_os_str().to_os_string();
    lock_name.push(".lock");
    let mut options = OpenOptions::new();
    options.read(true).write(true).create(true).truncate(false);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600).custom_flags(libc::O_NOFOLLOW);
    }
    let file = options.open(PathBuf::from(lock_name))?;
    file.try_lock().map_err(std::io::Error::other)?;
    Ok(file)
}

/// Persisted deny list. Revocation is monotonic; a rotated device must enroll
/// with a new PeerId. Restarts must not restore previously revoked access.
#[derive(Clone, Debug, Default, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RevocationList {
    revoked: std::collections::BTreeSet<String>,
}

impl RevocationList {
    pub fn load(path: &Path) -> Result<Self, IdentityError> {
        let bytes = match fs::read(path) {
            Ok(bytes) => bytes,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                return Ok(Self::default());
            }
            Err(error) => return Err(error.into()),
        };
        if bytes.len() > 1024 * 1024 {
            return Err(IdentityError::InvalidKey(
                "revocation list exceeds size limit".into(),
            ));
        }
        let list: Self = serde_json::from_slice(&bytes)
            .map_err(|e| IdentityError::InvalidKey(format!("invalid revocation list: {e}")))?;
        if list.revoked.len() > 10_000
            || list
                .revoked
                .iter()
                .any(|peer| peer.parse::<PeerId>().is_err())
        {
            return Err(IdentityError::InvalidKey(
                "invalid revoked peer list".into(),
            ));
        }
        Ok(list)
    }

    pub fn contains(&self, peer: &PeerId) -> bool {
        self.revoked.contains(&peer.to_string())
    }

    /// Commit to disk before changing the live deny list.
    pub fn revoke(&mut self, peer: PeerId, path: &Path) -> Result<(), IdentityError> {
        if self.revoked.len() >= 10_000 && !self.contains(&peer) {
            return Err(IdentityError::InvalidKey("revocation list is full".into()));
        }
        let mut next = self.clone();
        next.revoked.insert(peer.to_string());
        let bytes =
            serde_json::to_vec(&next).map_err(|e| IdentityError::InvalidKey(e.to_string()))?;
        write_private_state(path, &bytes)?;
        *self = next;
        Ok(())
    }
}

/// Default path: `~/.synapse/identity.key`
///
/// Overridable with the `SYNAPSE_IDENTITY_PATH` env var — useful for
/// running two nodes on the same machine (e.g. master + worker in
/// separate terminals) that need distinct libp2p PeerIds, since
/// libp2p refuses connections to a peer's own ID.
fn default_key_path() -> Result<PathBuf, IdentityError> {
    if let Ok(p) = std::env::var("SYNAPSE_IDENTITY_PATH")
        && !p.is_empty()
    {
        return Ok(PathBuf::from(p));
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
static IDENTITY_PATH_OVERRIDE: std::sync::Mutex<Option<PathBuf>> = std::sync::Mutex::new(None);

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

    #[test]
    fn creation_never_overwrites_an_identity() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("device.key");
        let original = NodeIdentity::create_at_path(&path).unwrap();
        assert!(NodeIdentity::create_at_path(&path).is_err());
        assert_eq!(
            NodeIdentity::load_or_create_at(&path).unwrap().peer_id(),
            original.peer_id()
        );
    }

    #[test]
    fn concurrent_creation_converges_on_one_complete_key() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("device.key");
        let handles: Vec<_> = (0..8)
            .map(|_| {
                let path = path.clone();
                std::thread::spawn(move || {
                    NodeIdentity::load_or_create_at(&path).unwrap().peer_id()
                })
            })
            .collect();
        let ids: Vec<_> = handles
            .into_iter()
            .map(|handle| handle.join().unwrap())
            .collect();
        assert!(ids.iter().all(|peer| *peer == ids[0]));
    }

    #[cfg(unix)]
    #[test]
    fn private_keys_reject_symlinks_and_insecure_permissions() {
        use std::os::unix::fs::{PermissionsExt, symlink};
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("device.key");
        NodeIdentity::create_at_path(&path).unwrap();
        assert_eq!(
            fs::metadata(&path).unwrap().permissions().mode() & 0o777,
            0o600
        );
        let link = dir.path().join("alias.key");
        symlink(&path, &link).unwrap();
        assert!(matches!(
            NodeIdentity::load_from_path(&link),
            Err(IdentityError::UnsafeFile)
        ));
        fs::set_permissions(&path, fs::Permissions::from_mode(0o644)).unwrap();
        assert!(matches!(
            NodeIdentity::load_from_path(&path),
            Err(IdentityError::UnsafeFile)
        ));
    }

    #[test]
    fn identity_proof_is_bound_to_transport_peer() {
        let dir = TempDir::new().unwrap();
        let identity = NodeIdentity::create_at_path(&dir.path().join("device.key")).unwrap();
        let proof = identity.sign(b"proof").unwrap();
        assert!(
            NodeIdentity::verify_remote_peer(
                identity.peer_id(),
                &identity.public_key_bytes(),
                b"proof",
                &proof
            )
            .is_ok()
        );
        assert!(
            NodeIdentity::verify_remote_peer(
                PeerId::random(),
                &identity.public_key_bytes(),
                b"proof",
                &proof
            )
            .is_err()
        );
        assert!(
            NodeIdentity::verify_remote_peer(
                identity.peer_id(),
                &identity.public_key_bytes(),
                b"other",
                &proof
            )
            .is_err()
        );
    }

    #[test]
    fn revoked_old_key_stays_revoked_after_restart_and_rotation() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("revoked.json");
        let old = NodeIdentity::create_at_path(&dir.path().join("old.key")).unwrap();
        let new = NodeIdentity::create_at_path(&dir.path().join("new.key")).unwrap();
        let mut list = RevocationList::default();
        list.revoke(old.peer_id(), &path).unwrap();
        list.revoke(old.peer_id(), &path).unwrap();
        let loaded = RevocationList::load(&path).unwrap();
        assert!(loaded.contains(&old.peer_id()));
        assert!(!loaded.contains(&new.peer_id()));
    }

    #[test]
    fn failed_revocation_write_does_not_change_memory() {
        let dir = TempDir::new().unwrap();
        let peer = PeerId::random();
        let mut list = RevocationList::default();
        assert!(list.revoke(peer, dir.path()).is_err());
        assert!(!list.contains(&peer));
    }
}
