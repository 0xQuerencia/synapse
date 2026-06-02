//! Discovery of cached Hugging Face models on disk.
//!
//! Synapse stores models in the standard HF cache layout
//! (`~/.cache/huggingface/hub/models--<org>--<name>` by default,
//! or `$HF_HOME/hub` if `HF_HOME` is set). Listing what's already
//! downloaded lets a GUI offer them as a picker before doing any
//! network I/O.

use std::{fs, path::PathBuf};

use anyhow::{Context, Result, anyhow};

/// Return the list of model ids currently cached on disk.
///
/// The returned ids use the canonical HF slash form
/// (`<org>/<name>`), regardless of how the on-disk directory names
/// encode them (`models--<org>--<name>`).
///
/// An empty `HF_HOME` or missing cache directory yields an empty
/// list, never an error.
pub fn list_cached_hf_models() -> Result<Vec<String>> {
    let hub_dir = locate_hub_dir()?;

    let mut models = Vec::new();
    if !hub_dir.exists() {
        return Ok(models);
    }

    for entry in fs::read_dir(&hub_dir)
        .with_context(|| format!("failed to read HF hub dir {}", hub_dir.display()))?
    {
        let entry = entry?;
        if !entry.file_type()?.is_dir() {
            continue;
        }

        let name = entry.file_name();
        let name = name.to_string_lossy();
        if let Some(rest) = name.strip_prefix("models--") {
            models.push(rest.replace("--", "/"));
        }
    }

    models.sort();
    models.dedup();
    Ok(models)
}

fn locate_hub_dir() -> Result<PathBuf> {
    if let Ok(hf_home) = std::env::var("HF_HOME") {
        if !hf_home.is_empty() {
            return Ok(PathBuf::from(hf_home).join("hub"));
        }
    }

    let home = std::env::var("HOME").map_err(|_| anyhow!("HOME is not set"))?;
    Ok(PathBuf::from(home).join(".cache/huggingface/hub"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn list_returns_empty_when_hub_missing() {
        // Point HF_HOME at a guaranteed-missing path. Should not error.
        let prev = std::env::var("HF_HOME").ok();
        // SAFETY: tests in this module are single-threaded for env var mutation.
        unsafe {
            std::env::set_var("HF_HOME", "/tmp/synapse-hf-test-missing");
        }
        let result = list_cached_hf_models();
        match prev {
            Some(v) => unsafe { std::env::set_var("HF_HOME", v) },
            None => unsafe { std::env::remove_var("HF_HOME") },
        }
        let models = result.expect("missing hub dir should yield empty list");
        assert!(models.is_empty());
    }
}
