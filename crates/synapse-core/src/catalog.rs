//! Explicit publisher approval and a verified local, content-addressed model cache.
//!
//! A hash proves integrity, not licensing rights. Every acquisition checks an
//! owner-supplied trust root and a signed, unexpired, purpose-specific grant.
//! This module does not distribute artifacts or trust keys obtained from peers.
//!
//! Cache writes are serialized, bounded by logical unique artifact bytes, and
//! staged on the same filesystem. Model views become visible only after every
//! artifact verifies. Callers must keep `PinnedModel` alive while a backend uses
//! its path. The initial filesystem implementation requires Unix permissions and
//! advisory locks; it is not an isolation boundary against the same OS account.

use std::{
    collections::{BTreeMap, BTreeSet},
    fs::{self, File, OpenOptions},
    io::{Read, Write},
    path::{Component, Path, PathBuf},
    sync::atomic::{AtomicU64, Ordering},
};

use anyhow::{Context, Result, bail, ensure};
use libp2p::{PeerId, identity::PublicKey};
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use sha2::{Digest, Sha256};

use crate::{
    NodeIdentity,
    inference::model_loader::{ModelArtifact, ModelManifest, read_safetensors_index},
};

#[cfg(unix)]
use std::os::{
    fd::AsRawFd,
    unix::fs::{DirBuilderExt, MetadataExt, OpenOptionsExt, PermissionsExt},
};

const MAX_JSON_BYTES: u64 = 4 * 1024 * 1024;
const MAX_CACHE_OBJECTS: usize = 100_000;
const MAX_CACHE_MODELS: usize = 10_000;
const MAX_CATALOG_LIFETIME: u64 = 366 * 24 * 60 * 60;
const CACHE_MARKER: &[u8] = b"synapse-verified-cache-v1\n";
static NEXT_STAGE: AtomicU64 = AtomicU64::new(0);

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelRights {
    pub local_inference: bool,
    pub remote_hosting: bool,
    pub redistribution: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CatalogUse {
    LocalInference,
    RemoteHosting,
    Redistribution,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CatalogStatus {
    Approved,
    Disabled,
    Revoked,
}

/// The signature covers all fields, with artifacts sorted by filename and the
/// signature array empty, prefixed by `synapse-model-catalog-v1\0`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SignedModelRecord {
    pub schema_version: u16,
    pub publisher: String,
    pub public_key: Vec<u8>,
    pub issued_at_unix: u64,
    pub expires_at_unix: u64,
    pub status: CatalogStatus,
    pub license: String,
    pub rights: ModelRights,
    pub provenance: String,
    pub manifest: ModelManifest,
    pub signature: Vec<u8>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TrustedPublisher {
    pub peer_id: String,
    pub public_key: Vec<u8>,
}

/// An explicit local policy input. An empty publisher list trusts nobody.
/// Distributing or updating this file is an owner-controlled operation.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TrustRoot {
    pub schema_version: u16,
    pub publishers: Vec<TrustedPublisher>,
    pub revoked_publishers: Vec<String>,
    pub revoked_models: Vec<String>,
}

impl Default for TrustRoot {
    fn default() -> Self {
        Self {
            schema_version: 1,
            publishers: Vec::new(),
            revoked_publishers: Vec::new(),
            revoked_models: Vec::new(),
        }
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct RecordInspection {
    pub manifest_digest: String,
    pub model_id: String,
    pub revision: String,
    pub publisher: String,
    pub license: String,
    pub rights: ModelRights,
    pub purpose: CatalogUse,
    pub artifact_count: usize,
    pub unique_artifact_bytes: u64,
    pub expires_at_unix: u64,
}

fn publisher_key(peer: &str, bytes: &[u8]) -> Result<PeerId> {
    ensure!(
        !peer.is_empty() && peer.len() <= 128 && !bytes.is_empty() && bytes.len() <= 128,
        "invalid publisher identity size"
    );
    let expected = peer.parse::<PeerId>().context("invalid publisher PeerId")?;
    let key = PublicKey::try_decode_protobuf(bytes).context("invalid publisher public key")?;
    key.clone()
        .try_into_ed25519()
        .context("catalog publishers must use Ed25519")?;
    ensure!(
        key.to_peer_id() == expected,
        "publisher key does not match PeerId"
    );
    Ok(expected)
}

fn digest_name(value: &str) -> Result<()> {
    ensure!(
        value.len() == 64
            && value
                .bytes()
                .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b)),
        "expected lowercase SHA-256 name"
    );
    Ok(())
}

impl TrustRoot {
    pub fn validate(&self) -> Result<()> {
        ensure!(self.schema_version == 1, "unsupported trust root version");
        ensure!(
            self.publishers.len() <= 64
                && self.revoked_publishers.len() <= 4096
                && self.revoked_models.len() <= 10_000,
            "trust root exceeds bounds"
        );
        let mut peers = BTreeSet::new();
        for publisher in &self.publishers {
            publisher_key(&publisher.peer_id, &publisher.public_key)?;
            ensure!(
                peers.insert(&publisher.peer_id),
                "duplicate trusted publisher"
            );
        }
        let mut revoked = BTreeSet::new();
        for peer in &self.revoked_publishers {
            ensure!(
                peer.len() <= 128 && peer.parse::<PeerId>().is_ok() && revoked.insert(peer),
                "invalid or duplicate revoked publisher"
            );
        }
        let mut models = BTreeSet::new();
        for digest in &self.revoked_models {
            digest_name(digest)?;
            ensure!(models.insert(digest), "duplicate revoked model");
        }
        Ok(())
    }
}

impl SignedModelRecord {
    fn validate_unsigned(&self) -> Result<()> {
        ensure!(
            self.schema_version == 1,
            "unsupported signed catalog version"
        );
        publisher_key(&self.publisher, &self.public_key)?;
        ensure!(
            self.expires_at_unix > self.issued_at_unix
                && self.expires_at_unix - self.issued_at_unix <= MAX_CATALOG_LIFETIME,
            "catalog lifetime must be positive and at most 366 days"
        );
        ensure!(
            !self.license.trim().is_empty()
                && self.license.len() <= 256
                && !self.license.chars().any(char::is_control),
            "missing or invalid model license assertion"
        );
        ensure!(
            !self.provenance.trim().is_empty()
                && self.provenance.len() <= 4096
                && !self.provenance.chars().any(char::is_control),
            "missing or invalid approval provenance"
        );
        self.manifest.validate()?;
        unique_artifacts(&self.manifest)?;
        Ok(())
    }

    pub fn signing_bytes(&self) -> Result<Vec<u8>> {
        self.validate_unsigned()?;
        let mut canonical = self.clone();
        canonical.signature.clear();
        canonical
            .manifest
            .artifacts
            .sort_by(|a, b| a.filename.cmp(&b.filename));
        let mut bytes = b"synapse-model-catalog-v1\0".to_vec();
        bytes.extend(serde_json::to_vec(&canonical)?);
        ensure!(
            bytes.len() as u64 <= MAX_JSON_BYTES,
            "catalog record exceeds size limit"
        );
        Ok(bytes)
    }

    /// Signing records a publisher's explicit assertion; it does not add that
    /// key to any trust root or infer permissions from a license string.
    pub fn sign(&mut self, identity: &NodeIdentity) -> Result<()> {
        self.publisher = identity.peer_id().to_string();
        self.public_key = identity.public_key_bytes();
        self.signature = identity.sign(&self.signing_bytes()?)?;
        Ok(())
    }
}

pub fn inspect_record(
    record: &SignedModelRecord,
    trust: &TrustRoot,
    purpose: CatalogUse,
    now: u64,
) -> Result<RecordInspection> {
    trust.validate()?;
    record.validate_unsigned()?;
    ensure!(
        record.signature.len() == 64,
        "invalid catalog signature length"
    );
    ensure!(
        record.status == CatalogStatus::Approved,
        "catalog entry is disabled or revoked"
    );
    ensure!(
        !trust.revoked_publishers.contains(&record.publisher),
        "catalog publisher is revoked"
    );
    ensure!(
        trust
            .publishers
            .iter()
            .any(|key| key.peer_id == record.publisher && key.public_key == record.public_key),
        "unknown catalog publisher"
    );
    ensure!(
        record.issued_at_unix <= now.saturating_add(5) && record.expires_at_unix > now,
        "catalog record is expired or issued in the future"
    );
    let manifest_digest = record.manifest.digest()?;
    ensure!(
        !trust.revoked_models.contains(&manifest_digest),
        "model is revoked by local policy"
    );
    NodeIdentity::verify_remote_peer(
        record.publisher.parse()?,
        &record.public_key,
        &record.signing_bytes()?,
        &record.signature,
    )?;
    let allowed = match purpose {
        CatalogUse::LocalInference => record.rights.local_inference,
        CatalogUse::RemoteHosting => record.rights.remote_hosting,
        CatalogUse::Redistribution => record.rights.redistribution,
    };
    ensure!(allowed, "catalog rights do not permit requested purpose");
    let bytes = unique_artifacts(&record.manifest)?
        .values()
        .try_fold(0u64, |sum, a| {
            sum.checked_add(a.size_bytes)
                .context("artifact size sum overflow")
        })?;
    Ok(RecordInspection {
        manifest_digest,
        model_id: record.manifest.model_id.clone(),
        revision: record.manifest.revision.clone(),
        publisher: record.publisher.clone(),
        license: record.license.clone(),
        rights: record.rights.clone(),
        purpose,
        artifact_count: record.manifest.artifacts.len(),
        unique_artifact_bytes: bytes,
        expires_at_unix: record.expires_at_unix,
    })
}

fn unique_artifacts(manifest: &ModelManifest) -> Result<BTreeMap<String, &ModelArtifact>> {
    let mut unique = BTreeMap::new();
    for artifact in &manifest.artifacts {
        let key = artifact.sha256.to_ascii_lowercase();
        if let Some(previous) = unique.insert(key, artifact) {
            ensure!(
                previous.size_bytes == artifact.size_bytes,
                "one digest has conflicting artifact sizes"
            );
        }
    }
    Ok(unique)
}

/// Reads bounded JSON without following a final symlink. Caller-supplied trust
/// files must be obtained through an authenticated, owner-controlled channel.
pub fn read_catalog_json<T: DeserializeOwned>(path: &Path) -> Result<T> {
    let path = checked_path(path, false)?;
    read_json_file(open_read(&path)?)
}

/// A trust file is policy, so unlike a signed record it must also be owned by
/// this account (or root) and not writable by another account or group.
pub fn read_trust_root(path: &Path) -> Result<TrustRoot> {
    let path = checked_path(path, false)?;
    let file = open_read(&path)?;
    #[cfg(unix)]
    {
        let metadata = file.metadata()?;
        ensure!(
            (metadata.uid() == 0 || metadata.uid() == unsafe { libc::geteuid() })
                && metadata.mode() & 0o022 == 0,
            "trust policy must be owned by this user or root and not group/world writable"
        );
    }
    let trust: TrustRoot = read_json_file(file)?;
    trust.validate()?;
    Ok(trust)
}

fn read_json_file<T: DeserializeOwned>(mut file: File) -> Result<T> {
    ensure!(
        file.metadata()?.len() <= MAX_JSON_BYTES,
        "catalog JSON exceeds size limit"
    );
    let mut bytes = Vec::new();
    Read::by_ref(&mut file)
        .take(MAX_JSON_BYTES + 1)
        .read_to_end(&mut bytes)?;
    ensure!(
        bytes.len() as u64 <= MAX_JSON_BYTES,
        "catalog JSON exceeds size limit"
    );
    serde_json::from_slice(&bytes).context("invalid catalog JSON")
}

#[derive(Clone, Debug, Serialize)]
pub struct CacheEntry {
    pub manifest_digest: String,
    pub directory: PathBuf,
    pub unique_artifact_bytes: u64,
}

pub struct PinnedModel {
    pub entry: CacheEntry,
    _pin: File,
}

#[derive(Clone, Debug, Serialize)]
pub struct CachePruneReport {
    pub removed_models: Vec<String>,
    pub removed_objects: usize,
    pub pinned_models: Vec<String>,
    pub remaining_bytes: u64,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Access {
    last_used_unix: u64,
}

pub struct ModelCache {
    root: PathBuf,
    quota_bytes: u64,
}

impl ModelCache {
    pub fn open(root: &Path, quota_bytes: u64) -> Result<Self> {
        ensure!(
            cfg!(unix),
            "verified cache currently requires Unix filesystem semantics"
        );
        ensure!(quota_bytes > 0, "cache quota must be positive");
        let root = checked_path(root, true)?;
        ensure_private_dir(&root)?;
        let marker = root.join(".synapse-cache-v1");
        if !marker.try_exists()? {
            ensure!(
                fs::read_dir(&root)?.next().is_none(),
                "refusing to adopt a nonempty unmanaged cache directory"
            );
            write_new(&marker, CACHE_MARKER, false)?;
        }
        ensure!(
            read_bytes(&marker, 128)? == CACHE_MARKER,
            "unrecognized cache marker"
        );
        for name in ["objects", "models", "staging", "pins"] {
            create_private_dir(&root.join(name))?;
        }
        let cache = Self { root, quota_bytes };
        let _lock = cache.lock()?;
        sync_dir(&cache.root)?;
        Ok(cache)
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    fn lock(&self) -> Result<File> {
        let file = open_lock(&self.root.join(".lock"))?;
        lock_file(&file, true, false).context("cache is busy")?;
        Ok(file)
    }

    /// Imports regular local files only. An import never downloads, executes or
    /// redistributes model content. It rejects source symlinks, including Hub
    /// snapshot links; first make an explicit regular-file copy of such inputs.
    pub fn import_model(
        &self,
        source: &Path,
        record: &SignedModelRecord,
        trust: &TrustRoot,
        purpose: CatalogUse,
        now: u64,
    ) -> Result<CacheEntry> {
        let inspection = inspect_record(record, trust, purpose, now)?;
        let source = checked_path(source, false)?;
        ensure!(
            fs::symlink_metadata(&source)?.is_dir(),
            "model source must be a directory"
        );
        let _lock = self.lock()?;
        self.clean_staging()?;
        let destination = self.root.join("models").join(&inspection.manifest_digest);
        if fs::symlink_metadata(&destination).is_ok() {
            self.verify_view(&destination, record)?;
            self.replace_json(&destination.join("catalog.json"), record)?;
            self.replace_json(
                &destination.join("access.json"),
                &Access {
                    last_used_unix: now,
                },
            )?;
            return Ok(self.entry(&inspection));
        }
        ensure!(
            self.model_names()?.len() < MAX_CACHE_MODELS,
            "cache model count limit reached"
        );
        let unique = unique_artifacts(&record.manifest)?;
        let mut needed = 0u64;
        let mut new_objects = 0usize;
        for (digest, artifact) in &unique {
            let object = self.root.join("objects").join(digest);
            if fs::symlink_metadata(&object).is_ok() {
                verify_artifact(&object, artifact)?;
            } else {
                new_objects += 1;
                needed = needed
                    .checked_add(artifact.size_bytes)
                    .context("cache reservation overflow")?;
            }
        }
        let existing = self.objects()?;
        ensure!(
            existing
                .len()
                .checked_add(new_objects)
                .is_some_and(|n| n <= MAX_CACHE_OBJECTS),
            "cache object count limit reached"
        );
        let current_bytes = existing.into_iter().try_fold(0u64, |sum, (_, metadata)| {
            sum.checked_add(metadata.len())
                .context("cache size overflow")
        })?;
        ensure!(
            current_bytes
                .checked_add(needed)
                .is_some_and(|n| n <= self.quota_bytes),
            "cache artifact quota exceeded; prune or explicitly increase it"
        );
        let stage = self.stage()?;
        let mut published = Vec::new();
        let result = (|| -> Result<CacheEntry> {
            let view = stage.path.join("view");
            create_private_dir(&view)?;
            for artifact in &record.manifest.artifacts {
                let digest = artifact.sha256.to_ascii_lowercase();
                let object = self.root.join("objects").join(&digest);
                let pending = stage.path.join(&digest);
                let exists = fs::symlink_metadata(&object).is_ok();
                if exists || fs::symlink_metadata(&pending).is_ok() {
                    verify_artifact(&source.join(&artifact.filename), artifact)?;
                } else {
                    copy_verified(&source.join(&artifact.filename), &pending, artifact)?;
                }
                fs::hard_link(
                    if exists { &object } else { &pending },
                    view.join(&artifact.filename),
                )?;
            }
            write_json_new(&view.join("manifest.json"), &record.manifest, true)?;
            write_json_new(&view.join("catalog.json"), record, false)?;
            write_json_new(
                &view.join("access.json"),
                &Access {
                    last_used_unix: now,
                },
                false,
            )?;
            record.manifest.verify_directory(&view)?;
            sync_dir(&view)?;
            for digest in unique.keys() {
                let pending = stage.path.join(digest);
                if fs::symlink_metadata(&pending).is_ok() {
                    let object = self.root.join("objects").join(digest);
                    fs::hard_link(&pending, &object)
                        .context("cannot atomically publish cache object")?;
                    published.push(object);
                }
            }
            sync_dir(&self.root.join("objects"))?;
            fs::rename(&view, &destination).context("cannot atomically publish complete model")?;
            sync_dir(&self.root.join("models"))?;
            Ok(self.entry(&inspection))
        })();
        if result.is_err() && !destination.is_dir() {
            for path in published {
                let _ = fs::remove_file(path);
            }
        }
        result
    }

    /// Rechecks the current trust/revocation policy and every cached byte, then
    /// holds a shared process lock. Retain this value until inference unloads.
    pub fn verify_model(
        &self,
        digest: &str,
        trust: &TrustRoot,
        purpose: CatalogUse,
        now: u64,
    ) -> Result<PinnedModel> {
        digest_name(digest)?;
        let _lock = self.lock()?;
        let directory = self.root.join("models").join(digest);
        ensure_private_dir(&directory)?;
        let record: SignedModelRecord = read_catalog_json(&directory.join("catalog.json"))?;
        let inspection = inspect_record(&record, trust, purpose, now)?;
        ensure!(
            inspection.manifest_digest == digest,
            "cache model name differs from signed manifest"
        );
        self.verify_view(&directory, &record)?;
        let pin = open_lock(&self.root.join("pins").join(format!("{digest}.lock")))?;
        lock_file(&pin, false, false)?;
        self.replace_json(
            &directory.join("access.json"),
            &Access {
                last_used_unix: now,
            },
        )?;
        Ok(PinnedModel {
            entry: self.entry(&inspection),
            _pin: pin,
        })
    }

    /// Removes least-recently-used unpinned model views, then orphan objects.
    /// Reports if pinned models prevent reaching the requested logical byte cap.
    pub fn prune(&self, target_bytes: u64) -> Result<CachePruneReport> {
        let _lock = self.lock()?;
        self.clean_staging()?;
        let mut report = CachePruneReport {
            removed_models: Vec::new(),
            removed_objects: self.remove_orphans()?,
            pinned_models: Vec::new(),
            remaining_bytes: self.object_bytes()?,
        };
        let mut models = Vec::new();
        for digest in self.model_names()? {
            let path = self.root.join("models").join(&digest);
            ensure_private_dir(&path)?;
            // An unreadable/corrupt access receipt must not protect an otherwise
            // unpinned model from explicit pruning.
            let access = read_catalog_json::<Access>(&path.join("access.json"))
                .map(|v| v.last_used_unix)
                .unwrap_or(0);
            models.push((access, digest));
        }
        models.sort();
        for (_, digest) in models {
            if report.remaining_bytes <= target_bytes {
                break;
            }
            let pin_path = self.root.join("pins").join(format!("{digest}.lock"));
            let pin = open_lock(&pin_path)?;
            match lock_file(&pin, true, false) {
                Ok(()) => {}
                Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                    report.pinned_models.push(digest);
                    continue;
                }
                Err(error) => return Err(error.into()),
            }
            fs::remove_dir_all(self.root.join("models").join(&digest))?;
            fs::remove_file(pin_path)?;
            report.removed_models.push(digest);
            report.removed_objects += self.remove_orphans()?;
            report.remaining_bytes = self.object_bytes()?;
        }
        sync_dir(&self.root.join("models"))?;
        sync_dir(&self.root.join("objects"))?;
        Ok(report)
    }

    fn entry(&self, inspection: &RecordInspection) -> CacheEntry {
        CacheEntry {
            manifest_digest: inspection.manifest_digest.clone(),
            directory: self.root.join("models").join(&inspection.manifest_digest),
            unique_artifact_bytes: inspection.unique_artifact_bytes,
        }
    }

    fn verify_view(&self, directory: &Path, record: &SignedModelRecord) -> Result<()> {
        ensure_private_dir(directory)?;
        let manifest: ModelManifest = read_catalog_json(&directory.join("manifest.json"))?;
        ensure!(
            manifest.digest()? == record.manifest.digest()?,
            "cached manifest differs from signed record"
        );
        let mut names: BTreeSet<String> = record
            .manifest
            .artifacts
            .iter()
            .map(|a| a.filename.clone())
            .collect();
        names.extend(["manifest.json", "catalog.json", "access.json"].map(String::from));
        for entry in fs::read_dir(directory)? {
            let entry = entry?;
            let name = entry
                .file_name()
                .into_string()
                .map_err(|_| anyhow::anyhow!("non-UTF8 cache entry"))?;
            ensure!(names.remove(&name), "unexpected file in cached model");
            ensure!(
                fs::symlink_metadata(entry.path())?.is_file(),
                "cache model contains a non-regular file"
            );
        }
        ensure!(names.is_empty(), "cached model is incomplete");
        for (digest, artifact) in unique_artifacts(&record.manifest)? {
            verify_artifact(&self.root.join("objects").join(digest), artifact)?;
        }
        for artifact in &record.manifest.artifacts {
            let source = fs::symlink_metadata(
                self.root
                    .join("objects")
                    .join(artifact.sha256.to_ascii_lowercase()),
            )?;
            let view = fs::symlink_metadata(directory.join(&artifact.filename))?;
            #[cfg(unix)]
            ensure!(
                source.dev() == view.dev()
                    && source.ino() == view.ino()
                    && view.mode() & 0o222 == 0,
                "model view is not an immutable cache object link"
            );
            #[cfg(not(unix))]
            let _ = (source, view);
        }
        if record.manifest.weights == "model.safetensors.index.json" {
            let known: BTreeSet<_> = record
                .manifest
                .artifacts
                .iter()
                .map(|a| a.filename.as_str())
                .collect();
            for name in read_safetensors_index(&directory.join(&record.manifest.weights))? {
                ensure!(
                    known.contains(name.as_str()),
                    "cached index refers to an unapproved shard"
                );
            }
        }
        Ok(())
    }

    fn model_names(&self) -> Result<Vec<String>> {
        let mut names = Vec::new();
        ensure_private_dir(&self.root.join("models"))?;
        for entry in fs::read_dir(self.root.join("models"))? {
            let name = entry?
                .file_name()
                .into_string()
                .map_err(|_| anyhow::anyhow!("invalid cache model name"))?;
            digest_name(&name)?;
            names.push(name);
            ensure!(
                names.len() <= MAX_CACHE_MODELS,
                "cache model count exceeds bounds"
            );
        }
        Ok(names)
    }

    fn objects(&self) -> Result<Vec<(PathBuf, fs::Metadata)>> {
        ensure_private_dir(&self.root.join("objects"))?;
        let mut objects = Vec::new();
        for entry in fs::read_dir(self.root.join("objects"))? {
            let entry = entry?;
            let name = entry
                .file_name()
                .into_string()
                .map_err(|_| anyhow::anyhow!("invalid cache object name"))?;
            digest_name(&name)?;
            let metadata = fs::symlink_metadata(entry.path())?;
            ensure!(metadata.is_file(), "cache object is not a regular file");
            objects.push((entry.path(), metadata));
            ensure!(
                objects.len() <= MAX_CACHE_OBJECTS,
                "cache object count exceeds bounds"
            );
        }
        Ok(objects)
    }

    fn object_bytes(&self) -> Result<u64> {
        self.objects()?.into_iter().try_fold(0u64, |sum, (_, m)| {
            sum.checked_add(m.len()).context("cache size overflow")
        })
    }

    fn remove_orphans(&self) -> Result<usize> {
        let mut removed = 0;
        for (path, metadata) in self.objects()? {
            #[cfg(unix)]
            if metadata.nlink() == 1 {
                fs::remove_file(path)?;
                removed += 1;
            }
            #[cfg(not(unix))]
            let _ = (path, metadata);
        }
        Ok(removed)
    }

    fn clean_staging(&self) -> Result<()> {
        let directory = self.root.join("staging");
        ensure_private_dir(&directory)?;
        for entry in fs::read_dir(directory)? {
            let entry = entry?;
            let name = entry
                .file_name()
                .into_string()
                .map_err(|_| anyhow::anyhow!("invalid stage name"))?;
            ensure!(
                name.starts_with("txn-")
                    && name[4..]
                        .bytes()
                        .all(|b| b.is_ascii_hexdigit() || b == b'-'),
                "unmanaged staging entry"
            );
            ensure_private_dir(&entry.path())?;
            fs::remove_dir_all(entry.path())?;
        }
        Ok(())
    }

    fn stage(&self) -> Result<Stage> {
        for _ in 0..16 {
            let path = self.root.join("staging").join(format!(
                "txn-{:x}-{:x}-{:x}",
                std::process::id(),
                NEXT_STAGE.fetch_add(1, Ordering::Relaxed),
                rand::random::<u64>()
            ));
            match private_dir_new(&path) {
                Ok(()) => return Ok(Stage { path }),
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
                Err(error) => return Err(error.into()),
            }
        }
        bail!("cannot create exclusive cache stage")
    }

    fn replace_json<T: Serialize>(&self, path: &Path, value: &T) -> Result<()> {
        let stage = self.stage()?;
        let temporary = stage.path.join("metadata.json");
        write_json_new(&temporary, value, false)?;
        fs::rename(temporary, path)?;
        sync_dir(path.parent().context("metadata needs parent")?)?;
        Ok(())
    }
}

struct Stage {
    path: PathBuf,
}
impl Drop for Stage {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.path);
    }
}

fn checked_path(path: &Path, create: bool) -> Result<PathBuf> {
    let absolute = if path.is_absolute() {
        path.to_owned()
    } else {
        std::env::current_dir()?.join(path)
    };
    let mut current = PathBuf::new();
    for part in absolute.components() {
        match part {
            Component::RootDir | Component::Prefix(_) => current.push(part.as_os_str()),
            Component::Normal(name) => {
                current.push(name);
                match fs::symlink_metadata(&current) {
                    Ok(metadata) => {
                        ensure!(
                            !metadata.file_type().is_symlink(),
                            "symlink component is not permitted: {}",
                            current.display()
                        );
                        #[cfg(unix)]
                        if metadata.is_dir() {
                            let owned =
                                metadata.uid() == 0 || metadata.uid() == unsafe { libc::geteuid() };
                            let protected =
                                metadata.mode() & 0o022 == 0 || metadata.mode() & 0o1000 != 0;
                            ensure!(
                                owned && protected,
                                "path ancestor is writable or controlled by another account: {}",
                                current.display()
                            );
                        }
                    }
                    Err(error) if create && error.kind() == std::io::ErrorKind::NotFound => {
                        private_dir_new(&current)?
                    }
                    Err(error) => return Err(error.into()),
                }
            }
            Component::CurDir => {}
            Component::ParentDir => bail!("parent traversal is not permitted"),
        }
    }
    Ok(current)
}

fn private_dir_new(path: &Path) -> std::io::Result<()> {
    let mut builder = fs::DirBuilder::new();
    #[cfg(unix)]
    builder.mode(0o700);
    builder.create(path)
}

fn create_private_dir(path: &Path) -> Result<()> {
    match private_dir_new(path) {
        Ok(()) => {}
        Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {}
        Err(error) => return Err(error.into()),
    }
    ensure_private_dir(path)
}

fn ensure_private_dir(path: &Path) -> Result<()> {
    let metadata = fs::symlink_metadata(path)?;
    ensure!(
        metadata.is_dir() && !metadata.file_type().is_symlink(),
        "cache directory must be a real directory"
    );
    #[cfg(unix)]
    ensure!(
        metadata.uid() == unsafe { libc::geteuid() } && metadata.mode() & 0o077 == 0,
        "cache directory must be owned by this user and private (0700)"
    );
    Ok(())
}

fn open_read(path: &Path) -> Result<File> {
    let mut options = OpenOptions::new();
    options.read(true);
    #[cfg(unix)]
    options.custom_flags(libc::O_NOFOLLOW | libc::O_CLOEXEC | libc::O_NONBLOCK);
    let file = options.open(path)?;
    ensure!(
        file.metadata()?.is_file(),
        "artifact must be a regular file"
    );
    Ok(file)
}

fn open_lock(path: &Path) -> Result<File> {
    let mut options = OpenOptions::new();
    options.read(true).write(true).create(true).truncate(false);
    #[cfg(unix)]
    options
        .mode(0o600)
        .custom_flags(libc::O_NOFOLLOW | libc::O_CLOEXEC | libc::O_NONBLOCK);
    let file = options.open(path)?;
    let metadata = file.metadata()?;
    ensure!(metadata.is_file(), "lock must be a regular file");
    #[cfg(unix)]
    ensure!(
        metadata.uid() == unsafe { libc::geteuid() } && metadata.mode() & 0o077 == 0,
        "cache lock must be private and owned"
    );
    Ok(file)
}

fn lock_file(file: &File, exclusive: bool, blocking: bool) -> std::io::Result<()> {
    #[cfg(unix)]
    {
        let operation = if exclusive {
            libc::LOCK_EX
        } else {
            libc::LOCK_SH
        } | if blocking { 0 } else { libc::LOCK_NB };
        if unsafe { libc::flock(file.as_raw_fd(), operation) } != 0 {
            return Err(std::io::Error::last_os_error());
        }
        Ok(())
    }
    #[cfg(not(unix))]
    {
        let _ = (file, exclusive, blocking);
        Err(std::io::Error::new(
            std::io::ErrorKind::Unsupported,
            "Unix cache locking required",
        ))
    }
}

fn read_bytes(path: &Path, limit: u64) -> Result<Vec<u8>> {
    let mut file = open_read(path)?;
    ensure!(file.metadata()?.len() <= limit, "file exceeds byte limit");
    let mut bytes = Vec::new();
    Read::by_ref(&mut file)
        .take(limit + 1)
        .read_to_end(&mut bytes)?;
    ensure!(bytes.len() as u64 <= limit, "file exceeds byte limit");
    Ok(bytes)
}

fn write_new(path: &Path, bytes: &[u8], readonly: bool) -> Result<()> {
    let mut options = OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    options
        .mode(0o600)
        .custom_flags(libc::O_NOFOLLOW | libc::O_CLOEXEC);
    let mut file = options.open(path)?;
    file.write_all(bytes)?;
    #[cfg(unix)]
    if readonly {
        file.set_permissions(fs::Permissions::from_mode(0o400))?;
    }
    #[cfg(not(unix))]
    let _ = readonly;
    file.sync_all()?;
    Ok(())
}

fn write_json_new<T: Serialize>(path: &Path, value: &T, readonly: bool) -> Result<()> {
    let bytes = serde_json::to_vec(value)?;
    ensure!(
        bytes.len() as u64 <= MAX_JSON_BYTES,
        "JSON exceeds byte limit"
    );
    write_new(path, &bytes, readonly)
}

fn verify_artifact(path: &Path, artifact: &ModelArtifact) -> Result<()> {
    stream_verified(path, None, artifact)
}

fn copy_verified(source: &Path, destination: &Path, artifact: &ModelArtifact) -> Result<()> {
    let mut options = OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    options
        .mode(0o600)
        .custom_flags(libc::O_NOFOLLOW | libc::O_CLOEXEC);
    let mut output = options.open(destination)?;
    stream_verified(source, Some(&mut output), artifact)?;
    #[cfg(unix)]
    output.set_permissions(fs::Permissions::from_mode(0o400))?;
    output.sync_all()?;
    Ok(())
}

fn stream_verified(
    source: &Path,
    mut output: Option<&mut File>,
    artifact: &ModelArtifact,
) -> Result<()> {
    let mut input = open_read(source)?;
    ensure!(
        input.metadata()?.len() == artifact.size_bytes,
        "artifact size mismatch: {}",
        artifact.filename
    );
    let mut hash = Sha256::new();
    let mut count = 0u64;
    let mut buffer = [0u8; 64 * 1024];
    loop {
        let n = input.read(&mut buffer)?;
        if n == 0 {
            break;
        }
        count = count
            .checked_add(n as u64)
            .context("artifact size overflow")?;
        ensure!(count <= artifact.size_bytes, "artifact grew while reading");
        hash.update(&buffer[..n]);
        if let Some(writer) = &mut output {
            writer.write_all(&buffer[..n])?;
        }
    }
    ensure!(
        count == artifact.size_bytes
            && format!("{:x}", hash.finalize()).eq_ignore_ascii_case(&artifact.sha256),
        "artifact hash mismatch: {}",
        artifact.filename
    );
    Ok(())
}

fn sync_dir(path: &Path) -> Result<()> {
    let mut options = OpenOptions::new();
    options.read(true);
    #[cfg(unix)]
    options.custom_flags(libc::O_NOFOLLOW | libc::O_DIRECTORY | libc::O_CLOEXEC);
    options.open(path)?.sync_all()?;
    Ok(())
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;
    use std::os::unix::fs::symlink;

    struct Fixture {
        _directory: tempfile::TempDir,
        source: PathBuf,
        cache_path: PathBuf,
        identity: NodeIdentity,
        record: SignedModelRecord,
        trust: TrustRoot,
    }

    impl Fixture {
        fn new() -> Self {
            let directory = tempfile::tempdir().unwrap();
            let source = directory.path().join("source");
            create_private_dir(&source).unwrap();
            // Original deterministic test bytes, deliberately not a trained model.
            // Two identical artifacts exercise content-addressed deduplication.
            let files: [(&str, &[u8]); 3] = [
                ("config.json", b"{}"),
                ("tokenizer.json", b"{}"),
                (
                    "model.safetensors",
                    b"synapse original cache fixture, not inference weights\n",
                ),
            ];
            let artifacts = files
                .iter()
                .map(|(name, bytes)| {
                    fs::write(source.join(name), bytes).unwrap();
                    ModelArtifact {
                        filename: (*name).into(),
                        size_bytes: bytes.len() as u64,
                        sha256: format!("{:x}", Sha256::digest(bytes)),
                    }
                })
                .collect();
            let identity =
                NodeIdentity::load_or_create_at(&directory.path().join("publisher.key")).unwrap();
            let mut record = SignedModelRecord {
                schema_version: 1,
                publisher: String::new(), public_key: Vec::new(),
                issued_at_unix: 1000, expires_at_unix: 2000,
                status: CatalogStatus::Approved, license: "MIT (original synthetic fixture)".into(),
                rights: ModelRights { local_inference: true, remote_hosting: false, redistribution: false },
                provenance: "Original synthetic bytes in the Synapse test suite; not a usable language model".into(),
                manifest: ModelManifest { schema_version: 1, model_id: "synapse/catalog-fixture".into(), revision: "fixture-v1".into(), weights: "model.safetensors".into(), artifacts },
                signature: Vec::new(),
            };
            record.sign(&identity).unwrap();
            let trust = TrustRoot {
                publishers: vec![TrustedPublisher {
                    peer_id: identity.peer_id().to_string(),
                    public_key: identity.public_key_bytes(),
                }],
                ..TrustRoot::default()
            };
            let cache_path = directory.path().join("cache");
            Self {
                _directory: directory,
                source,
                cache_path,
                identity,
                record,
                trust,
            }
        }
        fn bytes(&self) -> u64 {
            inspect_record(&self.record, &self.trust, CatalogUse::LocalInference, 1000)
                .unwrap()
                .unique_artifact_bytes
        }
        fn cache(&self) -> ModelCache {
            ModelCache::open(&self.cache_path, self.bytes()).unwrap()
        }
        fn import(&self, cache: &ModelCache) -> CacheEntry {
            cache
                .import_model(
                    &self.source,
                    &self.record,
                    &self.trust,
                    CatalogUse::LocalInference,
                    1000,
                )
                .unwrap()
        }
        fn signed(&self, change: impl FnOnce(&mut SignedModelRecord)) -> SignedModelRecord {
            let mut record = self.record.clone();
            change(&mut record);
            record.sign(&self.identity).unwrap();
            record
        }
    }

    #[test]
    fn catalog_signature_is_canonical_and_binds_every_assertion() {
        let f = Fixture::new();
        let mut reordered = f.record.clone();
        reordered.manifest.artifacts.reverse();
        assert_eq!(
            f.record.signing_bytes().unwrap(),
            reordered.signing_bytes().unwrap()
        );
        inspect_record(&reordered, &f.trust, CatalogUse::LocalInference, 1000).unwrap();
        let changes: [fn(&mut SignedModelRecord); 5] = [
            |v| v.manifest.revision = "another-revision".into(),
            |v| v.rights.redistribution = true,
            |v| v.license = "unapproved license".into(),
            |v| v.provenance = "unapproved provenance".into(),
            |v| v.expires_at_unix += 1,
        ];
        for change in changes {
            let mut record = f.record.clone();
            change(&mut record);
            assert!(inspect_record(&record, &f.trust, CatalogUse::LocalInference, 1000).is_err());
        }
    }

    #[test]
    fn unknown_revoked_disabled_and_expired_records_are_denied() {
        let f = Fixture::new();
        assert!(
            inspect_record(
                &f.record,
                &TrustRoot::default(),
                CatalogUse::LocalInference,
                1000
            )
            .is_err()
        );
        let mut trust = f.trust.clone();
        trust.revoked_publishers.push(f.record.publisher.clone());
        assert!(inspect_record(&f.record, &trust, CatalogUse::LocalInference, 1000).is_err());
        trust = f.trust.clone();
        trust
            .revoked_models
            .push(f.record.manifest.digest().unwrap());
        assert!(inspect_record(&f.record, &trust, CatalogUse::LocalInference, 1000).is_err());
        for status in [CatalogStatus::Disabled, CatalogStatus::Revoked] {
            let record = f.signed(|v| v.status = status);
            assert!(inspect_record(&record, &f.trust, CatalogUse::LocalInference, 1000).is_err());
        }
        assert!(inspect_record(&f.record, &f.trust, CatalogUse::LocalInference, 994).is_err());
        inspect_record(&f.record, &f.trust, CatalogUse::LocalInference, 995).unwrap();
        inspect_record(&f.record, &f.trust, CatalogUse::LocalInference, 1999).unwrap();
        assert!(inspect_record(&f.record, &f.trust, CatalogUse::LocalInference, 2000).is_err());
    }

    #[test]
    fn rights_are_explicit_and_not_inferred_from_license() {
        let f = Fixture::new();
        for purpose in [CatalogUse::RemoteHosting, CatalogUse::Redistribution] {
            assert!(inspect_record(&f.record, &f.trust, purpose, 1000).is_err());
        }
        let denied = f.signed(|v| v.rights = ModelRights::default());
        for purpose in [
            CatalogUse::LocalInference,
            CatalogUse::RemoteHosting,
            CatalogUse::Redistribution,
        ] {
            assert!(inspect_record(&denied, &f.trust, purpose, 1000).is_err());
        }
        let approved = f.signed(|v| v.rights.redistribution = true);
        inspect_record(&approved, &f.trust, CatalogUse::Redistribution, 1000).unwrap();
    }

    #[test]
    fn publisher_key_must_match_and_policy_duplicates_are_rejected() {
        let f = Fixture::new();
        let other_dir = tempfile::tempdir().unwrap();
        let other = NodeIdentity::load_or_create_at(&other_dir.path().join("key")).unwrap();
        let mut trust = f.trust.clone();
        trust.publishers[0].public_key = other.public_key_bytes();
        assert!(trust.validate().is_err());
        trust = f.trust.clone();
        trust.publishers.push(trust.publishers[0].clone());
        assert!(trust.validate().is_err());
        let mut record = f.record.clone();
        record.public_key = other.public_key_bytes();
        assert!(inspect_record(&record, &f.trust, CatalogUse::LocalInference, 1000).is_err());
    }

    #[test]
    fn trust_file_permissions_and_writable_parent_are_rejected() {
        let f = Fixture::new();
        let path = f._directory.path().join("trust.json");
        fs::write(&path, serde_json::to_vec(&f.trust).unwrap()).unwrap();
        fs::set_permissions(&path, fs::Permissions::from_mode(0o644)).unwrap();
        assert_eq!(read_trust_root(&path).unwrap(), f.trust);
        fs::set_permissions(&path, fs::Permissions::from_mode(0o666)).unwrap();
        assert!(read_trust_root(&path).is_err());
        let parent = f._directory.path().join("writable-parent");
        fs::create_dir(&parent).unwrap();
        fs::set_permissions(&parent, fs::Permissions::from_mode(0o777)).unwrap();
        assert!(ModelCache::open(&parent.join("cache"), 1024).is_err());
        assert!(!parent.join("cache").exists());
    }

    #[test]
    fn pin_symlink_cannot_modify_an_external_file() {
        let f = Fixture::new();
        let cache = f.cache();
        let entry = f.import(&cache);
        let outside = f._directory.path().join("owner-file");
        fs::write(&outside, b"preserve owner file").unwrap();
        symlink(
            &outside,
            cache
                .root
                .join("pins")
                .join(format!("{}.lock", entry.manifest_digest)),
        )
        .unwrap();
        assert!(
            cache
                .verify_model(
                    &entry.manifest_digest,
                    &f.trust,
                    CatalogUse::LocalInference,
                    1000
                )
                .is_err()
        );
        assert!(cache.prune(0).is_err());
        assert_eq!(fs::read(outside).unwrap(), b"preserve owner file");
    }

    #[test]
    fn malformed_and_oversized_metadata_is_rejected_before_io() {
        let f = Fixture::new();
        let mut record = f.record.clone();
        record.manifest.artifacts[0].filename = "../secret.json".into();
        assert!(record.sign(&f.identity).is_err());
        record = f.record.clone();
        record.expires_at_unix = record.issued_at_unix + MAX_CATALOG_LIFETIME + 1;
        assert!(record.sign(&f.identity).is_err());
        record = f.record.clone();
        record.provenance = "x".repeat(4097);
        assert!(record.sign(&f.identity).is_err());
        let mut value = serde_json::to_value(&f.record).unwrap();
        value["unrecognized"] = true.into();
        assert!(serde_json::from_value::<SignedModelRecord>(value).is_err());
        let large = f.source.join("oversized.json");
        fs::write(&large, vec![b' '; MAX_JSON_BYTES as usize + 1]).unwrap();
        assert!(read_catalog_json::<SignedModelRecord>(&large).is_err());
    }

    #[test]
    fn complete_model_is_usable_and_deduplicated_by_content() {
        let f = Fixture::new();
        let cache = f.cache();
        let entry = f.import(&cache);
        assert_eq!(cache.object_bytes().unwrap(), f.bytes());
        assert_eq!(cache.objects().unwrap().len(), 2);
        assert_eq!(fs::metadata(&cache.root).unwrap().mode() & 0o777, 0o700);
        for artifact in &f.record.manifest.artifacts {
            let path = entry.directory.join(&artifact.filename);
            assert_eq!(
                fs::read(path.clone()).unwrap(),
                fs::read(f.source.join(&artifact.filename)).unwrap()
            );
            assert_eq!(fs::metadata(path).unwrap().mode() & 0o777, 0o400);
        }
        f.record
            .manifest
            .verify_directory(&entry.directory)
            .unwrap();
        let pin = cache
            .verify_model(
                &entry.manifest_digest,
                &f.trust,
                CatalogUse::LocalInference,
                1001,
            )
            .unwrap();
        assert_eq!(pin.entry.directory, entry.directory);
        assert_eq!(f.import(&cache).manifest_digest, entry.manifest_digest);
        assert!(
            fs::read_dir(cache.root.join("staging"))
                .unwrap()
                .next()
                .is_none()
        );
    }

    #[test]
    fn quota_reservation_rejects_before_writes_and_detects_overflow() {
        let f = Fixture::new();
        let cache = ModelCache::open(&f.cache_path, f.bytes() - 1).unwrap();
        assert!(
            cache
                .import_model(
                    &f.source,
                    &f.record,
                    &f.trust,
                    CatalogUse::LocalInference,
                    1000
                )
                .is_err()
        );
        assert!(cache.objects().unwrap().is_empty());
        assert!(cache.model_names().unwrap().is_empty());
        let mut record = f.record.clone();
        record.manifest.artifacts[2].size_bytes = u64::MAX;
        record.sign(&f.identity).unwrap();
        assert!(inspect_record(&record, &f.trust, CatalogUse::LocalInference, 1000).is_err());
    }

    #[test]
    fn corrupt_late_source_never_publishes_partial_model_or_objects() {
        let f = Fixture::new();
        let cache = f.cache();
        let artifact = &f.record.manifest.artifacts[2];
        fs::write(
            f.source.join(&artifact.filename),
            vec![b'x'; artifact.size_bytes as usize],
        )
        .unwrap();
        assert!(
            cache
                .import_model(
                    &f.source,
                    &f.record,
                    &f.trust,
                    CatalogUse::LocalInference,
                    1000
                )
                .is_err()
        );
        assert!(cache.objects().unwrap().is_empty());
        assert!(cache.model_names().unwrap().is_empty());
        assert!(
            fs::read_dir(cache.root.join("staging"))
                .unwrap()
                .next()
                .is_none()
        );
    }

    #[test]
    fn missing_source_and_unapproved_index_shards_fail_atomically() {
        let mut f = Fixture::new();
        let cache = ModelCache::open(&f.cache_path, 1024).unwrap();
        fs::remove_file(f.source.join("tokenizer.json")).unwrap();
        assert!(
            cache
                .import_model(
                    &f.source,
                    &f.record,
                    &f.trust,
                    CatalogUse::LocalInference,
                    1000
                )
                .is_err()
        );
        fs::write(f.source.join("tokenizer.json"), b"{}").unwrap();
        let index = br#"{"weight_map":{"tensor":"unapproved.safetensors"}}"#;
        fs::write(f.source.join("model.safetensors.index.json"), index).unwrap();
        f.record.manifest.weights = "model.safetensors.index.json".into();
        f.record.manifest.artifacts.push(ModelArtifact {
            filename: "model.safetensors.index.json".into(),
            size_bytes: index.len() as u64,
            sha256: format!("{:x}", Sha256::digest(index)),
        });
        f.record.sign(&f.identity).unwrap();
        assert!(
            cache
                .import_model(
                    &f.source,
                    &f.record,
                    &f.trust,
                    CatalogUse::LocalInference,
                    1000
                )
                .is_err()
        );
        assert!(cache.objects().unwrap().is_empty());
        assert!(cache.model_names().unwrap().is_empty());
    }

    #[test]
    fn source_and_parent_symlinks_are_rejected_without_touching_targets() {
        let f = Fixture::new();
        let cache = f.cache();
        let original = f.source.join("model.safetensors");
        let outside = f._directory.path().join("outside");
        fs::rename(&original, &outside).unwrap();
        let bytes = fs::read(&outside).unwrap();
        symlink(&outside, &original).unwrap();
        assert!(
            cache
                .import_model(
                    &f.source,
                    &f.record,
                    &f.trust,
                    CatalogUse::LocalInference,
                    1000
                )
                .is_err()
        );
        assert_eq!(fs::read(&outside).unwrap(), bytes);
        let linked = f._directory.path().join("linked-source");
        symlink(&f.source, &linked).unwrap();
        assert!(
            cache
                .import_model(
                    &linked,
                    &f.record,
                    &f.trust,
                    CatalogUse::LocalInference,
                    1000
                )
                .is_err()
        );
        let linked_root = f._directory.path().join("linked-cache");
        symlink(&cache.root, &linked_root).unwrap();
        assert!(ModelCache::open(&linked_root, 1024).is_err());
    }

    #[test]
    fn fifo_source_is_rejected_without_blocking() {
        let f = Fixture::new();
        let cache = f.cache();
        let path = f.source.join("model.safetensors");
        fs::remove_file(&path).unwrap();
        let cpath = std::ffi::CString::new(path.as_os_str().as_encoded_bytes()).unwrap();
        assert_eq!(unsafe { libc::mkfifo(cpath.as_ptr(), 0o600) }, 0);
        assert!(
            cache
                .import_model(
                    &f.source,
                    &f.record,
                    &f.trust,
                    CatalogUse::LocalInference,
                    1000
                )
                .is_err()
        );
        assert!(cache.model_names().unwrap().is_empty());
    }

    #[test]
    fn cache_corruption_is_not_reused_or_silently_overwritten() {
        let f = Fixture::new();
        let cache = f.cache();
        let entry = f.import(&cache);
        let artifact = &f.record.manifest.artifacts[2];
        let object = cache.root.join("objects").join(&artifact.sha256);
        fs::set_permissions(&object, fs::Permissions::from_mode(0o600)).unwrap();
        fs::write(&object, vec![b'x'; artifact.size_bytes as usize]).unwrap();
        fs::set_permissions(&object, fs::Permissions::from_mode(0o400)).unwrap();
        assert!(
            cache
                .verify_model(
                    &entry.manifest_digest,
                    &f.trust,
                    CatalogUse::LocalInference,
                    1001
                )
                .is_err()
        );
        let another = f.signed(|v| v.manifest.revision = "fixture-v2".into());
        assert!(
            cache
                .import_model(
                    &f.source,
                    &another,
                    &f.trust,
                    CatalogUse::LocalInference,
                    1001
                )
                .is_err()
        );
        assert_eq!(cache.prune(0).unwrap().remaining_bytes, 0);
        f.import(&cache);
    }

    #[test]
    fn symlink_model_view_is_rejected_and_pruning_preserves_target() {
        let f = Fixture::new();
        let cache = f.cache();
        let entry = f.import(&cache);
        let artifact = entry.directory.join("model.safetensors");
        fs::remove_file(&artifact).unwrap();
        let outside = f.source.join("model.safetensors");
        let bytes = fs::read(&outside).unwrap();
        symlink(&outside, &artifact).unwrap();
        assert!(
            cache
                .verify_model(
                    &entry.manifest_digest,
                    &f.trust,
                    CatalogUse::LocalInference,
                    1001
                )
                .is_err()
        );
        cache.prune(0).unwrap();
        assert_eq!(fs::read(outside).unwrap(), bytes);
    }

    #[test]
    fn object_symlink_and_mutable_or_detached_view_are_rejected() {
        let f = Fixture::new();
        let cache = f.cache();
        let entry = f.import(&cache);
        let artifact = &f.record.manifest.artifacts[2];
        let object = cache.root.join("objects").join(&artifact.sha256);
        fs::remove_file(&object).unwrap();
        symlink(f.source.join(&artifact.filename), &object).unwrap();
        assert!(
            cache
                .verify_model(
                    &entry.manifest_digest,
                    &f.trust,
                    CatalogUse::LocalInference,
                    1000
                )
                .is_err()
        );
        fs::remove_file(&object).unwrap();
        fs::hard_link(entry.directory.join(&artifact.filename), &object).unwrap();
        fs::set_permissions(&object, fs::Permissions::from_mode(0o600)).unwrap();
        assert!(
            cache
                .verify_model(
                    &entry.manifest_digest,
                    &f.trust,
                    CatalogUse::LocalInference,
                    1000
                )
                .is_err()
        );
    }

    #[test]
    fn import_of_second_revision_reuses_blobs_and_live_pins_block_prune() {
        let f = Fixture::new();
        let cache = f.cache();
        let first = f.import(&cache);
        let second_record = f.signed(|v| v.manifest.revision = "fixture-v2".into());
        let second = cache
            .import_model(
                &f.source,
                &second_record,
                &f.trust,
                CatalogUse::LocalInference,
                1002,
            )
            .unwrap();
        assert_ne!(first.manifest_digest, second.manifest_digest);
        assert_eq!(cache.object_bytes().unwrap(), f.bytes());
        let pin = cache
            .verify_model(
                &first.manifest_digest,
                &f.trust,
                CatalogUse::LocalInference,
                1003,
            )
            .unwrap();
        let result = cache.prune(0).unwrap();
        assert_eq!(result.removed_models, vec![second.manifest_digest]);
        assert_eq!(result.pinned_models, vec![first.manifest_digest.clone()]);
        assert_eq!(result.remaining_bytes, f.bytes());
        assert!(pin.entry.directory.exists());
        drop(pin);
        let result = cache.prune(0).unwrap();
        assert_eq!(result.removed_models, vec![first.manifest_digest]);
        assert_eq!(result.remaining_bytes, 0);
    }

    #[test]
    fn current_policy_is_checked_again_for_cached_model() {
        let f = Fixture::new();
        let cache = f.cache();
        let entry = f.import(&cache);
        let mut trust = f.trust.clone();
        trust.revoked_models.push(entry.manifest_digest.clone());
        assert!(
            cache
                .verify_model(
                    &entry.manifest_digest,
                    &trust,
                    CatalogUse::LocalInference,
                    1001
                )
                .is_err()
        );
        assert!(
            cache
                .verify_model(
                    &entry.manifest_digest,
                    &f.trust,
                    CatalogUse::Redistribution,
                    1001
                )
                .is_err()
        );
        assert!(
            cache
                .verify_model(
                    &entry.manifest_digest,
                    &f.trust,
                    CatalogUse::LocalInference,
                    2000
                )
                .is_err()
        );
        let refreshed = f.signed(|v| {
            v.issued_at_unix = 2000;
            v.expires_at_unix = 3000;
        });
        cache
            .import_model(
                &f.source,
                &refreshed,
                &f.trust,
                CatalogUse::LocalInference,
                2000,
            )
            .unwrap();
        cache
            .verify_model(
                &entry.manifest_digest,
                &f.trust,
                CatalogUse::LocalInference,
                2001,
            )
            .unwrap();
    }

    #[test]
    fn exclusive_writer_lock_refuses_concurrent_mutation() {
        let f = Fixture::new();
        let cache = f.cache();
        let lock = cache.lock().unwrap();
        assert!(
            cache
                .import_model(
                    &f.source,
                    &f.record,
                    &f.trust,
                    CatalogUse::LocalInference,
                    1000
                )
                .is_err()
        );
        assert!(cache.prune(0).is_err());
        drop(lock);
        f.import(&cache);
    }

    #[test]
    fn orphan_objects_and_interrupted_stages_are_reclaimed() {
        let f = Fixture::new();
        let cache = f.cache();
        let stage = cache.stage().unwrap();
        let stage_path = stage.path.clone();
        write_new(&stage.path.join("partial"), b"incomplete", false).unwrap();
        std::mem::forget(stage); // Simulate a prior process exiting before RAII cleanup.
        let object = cache.root.join("objects").join("a".repeat(64));
        write_new(&object, b"orphan", true).unwrap();
        let report = cache.prune(0).unwrap();
        assert!(!stage_path.exists());
        assert_eq!(report.removed_objects, 1);
        assert_eq!(report.remaining_bytes, 0);
    }

    #[test]
    fn unmanaged_or_public_directories_are_not_adopted_or_deleted() {
        let directory = tempfile::tempdir().unwrap();
        fs::write(directory.path().join("keep"), b"owner data").unwrap();
        assert!(ModelCache::open(directory.path(), 1024).is_err());
        assert_eq!(
            fs::read(directory.path().join("keep")).unwrap(),
            b"owner data"
        );
        let public = directory.path().join("public");
        fs::create_dir(&public).unwrap();
        fs::set_permissions(&public, fs::Permissions::from_mode(0o755)).unwrap();
        assert!(ModelCache::open(&public, 1024).is_err());
        let f = Fixture::new();
        let cache = f.cache();
        let foreign = cache.root.join("staging").join("owner-data");
        fs::write(&foreign, b"keep").unwrap();
        assert!(cache.prune(0).is_err());
        assert_eq!(fs::read(foreign).unwrap(), b"keep");
    }
}
