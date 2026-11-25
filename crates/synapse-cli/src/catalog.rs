use std::{
    path::PathBuf,
    time::{SystemTime, UNIX_EPOCH},
};

use anyhow::Result;
use clap::{Args, Subcommand, ValueEnum};
use synapse_core::catalog::{
    CatalogUse, ModelCache, SignedModelRecord, inspect_record, read_catalog_json, read_trust_root,
};

#[derive(Clone, Copy, Debug, ValueEnum)]
pub enum Purpose {
    Local,
    Host,
    Redistribute,
}

impl From<Purpose> for CatalogUse {
    fn from(value: Purpose) -> Self {
        match value {
            Purpose::Local => Self::LocalInference,
            Purpose::Host => Self::RemoteHosting,
            Purpose::Redistribute => Self::Redistribution,
        }
    }
}

#[derive(Debug, Args)]
pub struct Policy {
    /// Explicit owner-controlled publisher keys and revocation policy.
    #[arg(long)]
    trust_root: PathBuf,
    /// The exact use that the signed record must permit.
    #[arg(long, value_enum)]
    purpose: Purpose,
}

#[derive(Debug, Args)]
pub struct Cache {
    /// A private managed cache directory on a local Unix filesystem.
    #[arg(long)]
    cache: PathBuf,
    /// Maximum unique cached artifact bytes; no process-memory claim.
    #[arg(long)]
    quota_bytes: u64,
}

impl Cache {
    fn open(&self) -> Result<ModelCache> {
        ModelCache::open(&self.cache, self.quota_bytes)
    }
}

#[derive(Debug, Subcommand)]
pub enum CatalogCommand {
    /// Verify the publisher signature, expiry, revocations and selected rights.
    Inspect {
        #[arg(long)]
        record: PathBuf,
        #[command(flatten)]
        policy: Policy,
    },
    /// Import verified regular local files into the quota-bound cache.
    Import {
        #[arg(long)]
        source: PathBuf,
        #[arg(long)]
        record: PathBuf,
        #[command(flatten)]
        policy: Policy,
        #[command(flatten)]
        cache: Cache,
    },
    /// Recheck the current policy and every cached byte.
    Verify {
        #[arg(long)]
        digest: String,
        #[command(flatten)]
        policy: Policy,
        #[command(flatten)]
        cache: Cache,
    },
    /// Remove least-recently-used, unpinned model data toward a byte target.
    Prune {
        #[arg(long)]
        target_bytes: u64,
        #[command(flatten)]
        cache: Cache,
    },
    /// Run bounded local inference while keeping the verified model pinned.
    Infer {
        #[arg(long)]
        digest: String,
        #[arg(long)]
        trust_root: PathBuf,
        #[command(flatten)]
        cache: Cache,
        #[arg(long)]
        prompt: String,
        #[arg(long, default_value_t = 8)]
        max_tokens: usize,
        #[arg(long)]
        events_jsonl: Option<PathBuf>,
    },
}

fn output(value: &impl serde::Serialize) -> Result<()> {
    println!("{}", serde_json::to_string_pretty(value)?);
    Ok(())
}

pub fn run(command: CatalogCommand) -> Result<()> {
    let now = SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs();
    match command {
        CatalogCommand::Inspect { record, policy } => {
            let record: SignedModelRecord = read_catalog_json(&record)?;
            let trust = read_trust_root(&policy.trust_root)?;
            output(&inspect_record(
                &record,
                &trust,
                policy.purpose.into(),
                now,
            )?)
        }
        CatalogCommand::Import {
            source,
            record,
            policy,
            cache,
        } => {
            let record: SignedModelRecord = read_catalog_json(&record)?;
            let trust = read_trust_root(&policy.trust_root)?;
            output(&cache.open()?.import_model(
                &source,
                &record,
                &trust,
                policy.purpose.into(),
                now,
            )?)
        }
        CatalogCommand::Verify {
            digest,
            policy,
            cache,
        } => {
            let trust = read_trust_root(&policy.trust_root)?;
            let cache = cache.open()?;
            let pin = cache.verify_model(&digest, &trust, policy.purpose.into(), now)?;
            output(&pin.entry)
        }
        CatalogCommand::Prune {
            target_bytes,
            cache,
        } => output(&cache.open()?.prune(target_bytes)?),
        CatalogCommand::Infer {
            digest,
            trust_root,
            cache,
            prompt,
            max_tokens,
            events_jsonl,
        } => {
            let trust = read_trust_root(&trust_root)?;
            let cache = cache.open()?;
            let pin = cache.verify_model(&digest, &trust, CatalogUse::LocalInference, now)?;
            let result = super::infer_local(
                pin.entry.directory.clone(),
                prompt,
                max_tokens,
                events_jsonl,
            );
            // The OS pin must outlive backend loading, generation and unloading.
            drop(pin);
            result
        }
    }
}
