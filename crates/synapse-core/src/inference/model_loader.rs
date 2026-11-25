//! Pinned model acquisition and verification. A downloaded manifest is an
//! integrity receipt, not a publisher signature or permission to redistribute.

use std::{
    collections::BTreeSet,
    fs,
    io::Read,
    path::{Path, PathBuf},
};

use anyhow::{Context, Result, bail, ensure};
use hf_hub::{Repo, RepoType, api::tokio::ApiRepo};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::{ModelIdentity, model::sha256_hex};

const INDEX_FILE: &str = "model.safetensors.index.json";
const MAX_INDEX_BYTES: u64 = 16 * 1024 * 1024;

/// A file that must match before a backend may open this model.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelArtifact {
    pub filename: String,
    pub size_bytes: u64,
    pub sha256: String,
}

/// Stable, versioned identity for every input to inference, including tokenizer.
/// `revision` is a resolved Hub commit or a local content revision; it never
/// means a moving branch such as `main`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelManifest {
    pub schema_version: u32,
    pub model_id: String,
    pub revision: String,
    pub weights: String,
    pub artifacts: Vec<ModelArtifact>,
}

impl ModelManifest {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.schema_version == 1,
            "unsupported model manifest version"
        );
        ModelIdentity::new(&self.model_id).validate()?;
        ensure!(
            !self.revision.is_empty()
                && self.revision.len() <= 128
                && self
                    .revision
                    .bytes()
                    .all(|c| c.is_ascii_alphanumeric() || b"-_.".contains(&c)),
            "invalid immutable model revision"
        );
        ensure!(
            self.revision != "main" && self.revision != "master",
            "model revision must be pinned"
        );
        validate_artifact_filename(&self.weights)?;
        ensure!(
            self.weights.ends_with(".safetensors")
                || self.weights.ends_with(".gguf")
                || self.weights == INDEX_FILE,
            "invalid weights format"
        );
        ensure!(
            self.artifacts.len() >= 3 && self.artifacts.len() <= 10_000,
            "invalid artifact count"
        );
        let mut names = BTreeSet::new();
        for artifact in &self.artifacts {
            validate_artifact_filename(&artifact.filename)?;
            ensure!(
                names.insert(artifact.filename.as_str()),
                "duplicate model artifact"
            );
            ensure!(artifact.size_bytes > 0, "empty model artifact");
            validate_digest(&artifact.sha256)?;
        }
        for required in ["config.json", "tokenizer.json", self.weights.as_str()] {
            ensure!(
                names.contains(required),
                "manifest is missing required artifact {required}"
            );
        }
        Ok(())
    }

    /// Canonical digest: fixed struct field order and lexical artifact order.
    pub fn digest(&self) -> Result<String> {
        self.validate()?;
        let mut canonical = self.clone();
        canonical
            .artifacts
            .sort_by(|a, b| a.filename.cmp(&b.filename));
        Ok(sha256_hex(&serde_json::to_vec(&canonical)?))
    }

    pub fn verify_directory(&self, directory: &Path) -> Result<()> {
        self.validate()?;
        for artifact in &self.artifacts {
            let path = directory.join(&artifact.filename);
            let metadata = fs::metadata(&path)
                .with_context(|| format!("missing model artifact {}", artifact.filename))?;
            ensure!(
                metadata.is_file() && metadata.len() == artifact.size_bytes,
                "model artifact size mismatch: {}",
                artifact.filename
            );
            verify_file_sha256(&path, &artifact.sha256)?;
        }
        if self.weights == INDEX_FILE {
            let expected: BTreeSet<_> =
                self.artifacts.iter().map(|a| a.filename.as_str()).collect();
            for shard in read_safetensors_index(&directory.join(INDEX_FILE))? {
                ensure!(
                    expected.contains(shard.as_str()),
                    "index refers to unmanifested shard {shard}"
                );
            }
        }
        Ok(())
    }
}

fn validate_digest(digest: &str) -> Result<()> {
    ensure!(
        digest.len() == 64 && digest.bytes().all(|c| c.is_ascii_hexdigit()),
        "expected a SHA-256 hex digest"
    );
    Ok(())
}

/// Model artifacts are single names, not paths. Reject before touching disk or
/// the network; a remote index must never choose a directory or arbitrary URL.
pub fn validate_artifact_filename(name: &str) -> Result<()> {
    ensure!(
        !name.is_empty()
            && name.len() <= 255
            && !name.starts_with('.')
            && !name.contains("..")
            && name
                .bytes()
                .all(|c| c.is_ascii_alphanumeric() || b"-_.".contains(&c)),
        "unsafe model artifact filename"
    );
    ensure!(
        name.ends_with(".safetensors")
            || name.ends_with(".gguf")
            || matches!(
                name,
                "config.json" | "tokenizer.json" | "tokenizer_config.json" | INDEX_FILE
            ),
        "unsupported model artifact format"
    );
    Ok(())
}

/// Parse **values**, not tensor keys, and reject every malformed entry. Sorting
/// and deduplication make download order and manifest identity deterministic.
pub fn safetensors_index_filenames(bytes: &[u8]) -> Result<Vec<String>> {
    ensure!(
        bytes.len() as u64 <= MAX_INDEX_BYTES,
        "safetensors index too large"
    );
    let index: serde_json::Value =
        serde_json::from_slice(bytes).context("invalid shard index JSON")?;
    let map = index
        .get("weight_map")
        .and_then(serde_json::Value::as_object)
        .context("shard index is missing its weight_map")?;
    ensure!(
        !map.is_empty() && map.len() <= 100_000,
        "invalid weight_map size"
    );
    let mut names = BTreeSet::new();
    for (tensor, value) in map {
        ensure!(
            !tensor.is_empty() && tensor.len() <= 1024,
            "invalid tensor name"
        );
        let name = value
            .as_str()
            .context("weight_map filenames must be strings")?;
        validate_artifact_filename(name)?;
        ensure!(
            name.ends_with(".safetensors"),
            "index contains a non-safetensors shard"
        );
        names.insert(name.to_owned());
    }
    Ok(names.into_iter().collect())
}

pub fn read_safetensors_index(path: &Path) -> Result<Vec<String>> {
    ensure!(
        fs::metadata(path)?.len() <= MAX_INDEX_BYTES,
        "safetensors index too large"
    );
    safetensors_index_filenames(&fs::read(path)?)
}

pub async fn download_model(model_id: &str) -> Result<ModelFiles> {
    download_model_revision(&ModelIdentity::new(model_id)).await
}

/// Resolve a requested revision exactly once, then fetch every file at that
/// immutable commit. Listing files first avoids treating auth/network errors as
/// evidence that a format is absent and silently trying a different format.
pub async fn download_model_revision(identity: &ModelIdentity) -> Result<ModelFiles> {
    identity.validate()?;
    let api = hf_hub::api::tokio::Api::new().context("failed to create HF Hub API client")?;
    let requested = api.repo(Repo::with_revision(
        identity.id.clone(),
        RepoType::Model,
        identity
            .revision
            .clone()
            .unwrap_or_else(|| "main".to_owned()),
    ));
    let info = requested
        .info()
        .await
        .context("failed to resolve model revision")?;
    ensure!(
        info.sha.len() == 40 && info.sha.bytes().all(|c| c.is_ascii_hexdigit()),
        "Hub did not return an immutable model commit"
    );
    let repo = api.repo(Repo::with_revision(
        identity.id.clone(),
        RepoType::Model,
        info.sha.clone(),
    ));
    let siblings: BTreeSet<_> = info.siblings.iter().map(|s| s.rfilename.as_str()).collect();
    ensure!(
        siblings.contains("config.json") && siblings.contains("tokenizer.json"),
        "model requires config.json and tokenizer.json"
    );
    let config = repo
        .get("config.json")
        .await
        .context("failed to download config.json")?;
    let tokenizer = repo
        .get("tokenizer.json")
        .await
        .context("failed to download tokenizer.json")?;
    let weights = resolve_weights(&repo, &siblings).await?;
    let files = ModelFiles::inspect(identity.id.clone(), info.sha, config, weights, tokenizer)?;
    if let Some(expected) = &identity.sha256 {
        verify_file_sha256(&files.weights, expected)?;
    }
    Ok(files)
}

async fn resolve_weights(repo: &ApiRepo, siblings: &BTreeSet<&str>) -> Result<PathBuf> {
    if siblings.contains(INDEX_FILE) {
        let index = repo
            .get(INDEX_FILE)
            .await
            .context("failed to download shard index")?;
        for filename in read_safetensors_index(&index)? {
            ensure!(
                siblings.contains(filename.as_str()),
                "index references missing shard {filename}"
            );
            repo.get(&filename)
                .await
                .with_context(|| format!("failed to download shard {filename}"))?;
        }
        return Ok(index);
    }
    if siblings.contains("model.safetensors") {
        return repo
            .get("model.safetensors")
            .await
            .context("failed to download safetensors");
    }
    let candidates: Vec<_> = siblings.iter().filter(|n| n.ends_with(".gguf")).collect();
    if candidates.len() == 1 {
        let filename = **candidates.first().expect("one candidate");
        validate_artifact_filename(filename)?;
        return repo.get(filename).await.context("failed to download GGUF");
    }
    if candidates.len() > 1 {
        bail!("ambiguous GGUF repository: select a single approved artifact");
    }
    bail!("model has no supported safetensors or GGUF checkpoint")
}

pub fn file_sha256(path: &Path) -> Result<String> {
    let mut file = fs::File::open(path)?;
    let mut digest = Sha256::new();
    let mut buffer = [0_u8; 64 * 1024];
    loop {
        let count = file.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        digest.update(&buffer[..count]);
    }
    Ok(format!("{:x}", digest.finalize()))
}

pub fn verify_file_sha256(path: &Path, expected: &str) -> Result<()> {
    validate_digest(expected)?;
    ensure!(
        file_sha256(path)? == expected.to_ascii_lowercase(),
        "sha256 mismatch for {}",
        path.display()
    );
    Ok(())
}

#[derive(Debug, Clone)]
pub struct ModelFiles {
    pub config: PathBuf,
    pub weights: PathBuf,
    pub tokenizer: PathBuf,
    pub manifest: ModelManifest,
}

impl ModelFiles {
    fn inspect(
        model_id: String,
        revision: String,
        config: PathBuf,
        weights: PathBuf,
        tokenizer: PathBuf,
    ) -> Result<Self> {
        let mut paths = vec![config.clone(), tokenizer.clone(), weights.clone()];
        if weights.file_name().and_then(|n| n.to_str()) == Some(INDEX_FILE) {
            let directory = weights
                .parent()
                .context("weights have no parent directory")?;
            for filename in read_safetensors_index(&weights)? {
                paths.push(directory.join(filename));
            }
        }
        paths.sort();
        paths.dedup();
        let artifacts = paths
            .iter()
            .map(|path| -> Result<ModelArtifact> {
                let filename = path
                    .file_name()
                    .and_then(|n| n.to_str())
                    .context("invalid artifact name")?
                    .to_owned();
                validate_artifact_filename(&filename)?;
                Ok(ModelArtifact {
                    filename,
                    size_bytes: fs::metadata(path)?.len(),
                    sha256: file_sha256(path)?,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let manifest = ModelManifest {
            schema_version: 1,
            model_id,
            revision,
            weights: weights
                .file_name()
                .and_then(|n| n.to_str())
                .context("invalid weights filename")?
                .to_owned(),
            artifacts,
        };
        manifest.validate()?;
        Ok(Self {
            config,
            weights,
            tokenizer,
            manifest,
        })
    }

    /// Open an explicitly selected local model, requiring its complete pinned
    /// manifest. Used by offline CI fixtures and air-gapped private clusters.
    pub fn from_directory(directory: &Path) -> Result<Self> {
        let manifest_path = directory.join("manifest.json");
        ensure!(
            fs::metadata(&manifest_path)?.len() <= MAX_INDEX_BYTES,
            "model manifest too large"
        );
        let manifest: ModelManifest = serde_json::from_slice(&fs::read(manifest_path)?)?;
        manifest.verify_directory(directory)?;
        Ok(Self {
            config: directory.join("config.json"),
            tokenizer: directory.join("tokenizer.json"),
            weights: directory.join(&manifest.weights),
            manifest,
        })
    }
}

#[cfg(test)]
pub(crate) fn test_model_files() -> Result<ModelFiles> {
    ModelFiles::from_directory(
        &Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tiny-llama"),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn index_uses_values_and_deduplicates() {
        let bytes = br#"{"weight_map":{"layer.0":"part-2.safetensors","layer.1":"part-1.safetensors","layer.2":"part-2.safetensors"}}"#;
        assert_eq!(
            safetensors_index_filenames(bytes).unwrap(),
            ["part-1.safetensors", "part-2.safetensors"]
        );
    }

    #[test]
    fn index_rejects_every_malformed_entry() {
        for value in [
            serde_json::json!(17),
            serde_json::json!("../model.safetensors"),
            serde_json::json!("/x.safetensors"),
            serde_json::json!("x\\evil.safetensors"),
            serde_json::json!(""),
            serde_json::json!("weights.bin"),
            serde_json::json!("https://evil/x.safetensors"),
        ] {
            let bytes = serde_json::to_vec(
                &serde_json::json!({"weight_map":{"valid":"model.safetensors","bad":value}}),
            )
            .unwrap();
            assert!(safetensors_index_filenames(&bytes).is_err());
        }
        assert!(safetensors_index_filenames(br#"{"weight_map":{}}"#).is_err());
        assert!(safetensors_index_filenames(b"{}").is_err());
    }

    #[test]
    fn streaming_hash_rejects_corruption_and_bad_digest() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("model.safetensors");
        fs::write(&path, b"abc").unwrap();
        let digest = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad";
        verify_file_sha256(&path, digest).unwrap();
        fs::write(&path, b"abd").unwrap();
        assert!(verify_file_sha256(&path, digest).is_err());
        assert!(verify_file_sha256(&path, "bad").is_err());
    }

    #[test]
    fn complete_manifest_verifies_tokenizer_and_all_shards() {
        let directory = tempfile::tempdir().unwrap();
        for name in [
            "config.json",
            "tokenizer.json",
            "part-1.safetensors",
            "part-2.safetensors",
        ] {
            fs::write(directory.path().join(name), b"fixture").unwrap();
        }
        fs::write(
            directory.path().join(INDEX_FILE),
            br#"{"weight_map":{"a":"part-1.safetensors","b":"part-2.safetensors"}}"#,
        )
        .unwrap();
        let files = ModelFiles::inspect(
            "synapse/test".into(),
            "fixture-v1".into(),
            directory.path().join("config.json"),
            directory.path().join(INDEX_FILE),
            directory.path().join("tokenizer.json"),
        )
        .unwrap();
        files.manifest.verify_directory(directory.path()).unwrap();
        let mut reordered = files.manifest.clone();
        reordered.artifacts.reverse();
        assert_eq!(
            files.manifest.digest().unwrap(),
            reordered.digest().unwrap()
        );
        fs::write(directory.path().join("part-2.safetensors"), b"corrupt").unwrap();
        assert!(files.manifest.verify_directory(directory.path()).is_err());
        fs::remove_file(directory.path().join("tokenizer.json")).unwrap();
        assert!(files.manifest.verify_directory(directory.path()).is_err());
        let mut missing = files.manifest;
        missing.artifacts.retain(|a| a.filename != "tokenizer.json");
        assert!(missing.validate().is_err());
    }
}
