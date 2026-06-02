//! Read/write of `config/*.toml` for the GUI's config editor.
//!
//! The shapes are intentionally stringly-typed: the GUI just round-trips
//! text the user edited in a textarea and we trust the operator not to
//! write garbage. Validation matches the CLI's load-time rule
//! (entries must end in `/p2p/<peer-id>` or be skipped).

use std::{fs, path::PathBuf};

use anyhow::{Context, Result, anyhow};

const RELAYS_FILE: &str = "config/relays.toml";
const BOOTSTRAP_FILE: &str = "config/bootstrap.toml";
const MODELS_FILE: &str = "config/models.toml";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConfigFile {
    Relays,
    Bootstrap,
    Models,
}

impl ConfigFile {
    pub const fn path(self) -> &'static str {
        match self {
            ConfigFile::Relays => RELAYS_FILE,
            ConfigFile::Bootstrap => BOOTSTRAP_FILE,
            ConfigFile::Models => MODELS_FILE,
        }
    }

    pub const fn label(self) -> &'static str {
        match self {
            ConfigFile::Relays => "relays",
            ConfigFile::Bootstrap => "bootstrap",
            ConfigFile::Models => "models",
        }
    }

    pub fn from_label(label: &str) -> Option<Self> {
        match label {
            "relays" => Some(Self::Relays),
            "bootstrap" => Some(Self::Bootstrap),
            "models" => Some(Self::Models),
            _ => None,
        }
    }
}

/// Read a config file as text, creating an empty placeholder if missing.
pub fn read(file: ConfigFile) -> Result<String> {
    let path = PathBuf::from(file.path());
    if !path.exists() {
        return Ok(default_for(file));
    }
    fs::read_to_string(&path).with_context(|| format!("failed to read {}", path.display()))
}

/// Atomically write a config file.
pub fn write(file: ConfigFile, contents: &str) -> Result<()> {
    let path = PathBuf::from(file.path());
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("failed to create {}", parent.display()))?;
    }
    // Validate TOML before writing.
    let _: toml::Value =
        toml::from_str(contents).map_err(|e| anyhow!("invalid TOML in {}: {e}", file.path()))?;
    fs::write(&path, contents).with_context(|| format!("failed to write {}", path.display()))?;
    Ok(())
}

fn default_for(file: ConfigFile) -> String {
    match file {
        ConfigFile::Relays => "# Public relay peers reachable as `<addr>/p2p/<peer-id>`.\n\
             # Lines starting with `#` and entries missing `/p2p/` are skipped at load time.\n\
             relays = []\n"
            .to_owned(),
        ConfigFile::Bootstrap => {
            "# Kademlia DHT bootstrap peers. Same multiaddr format as relays.\n\
             # bootstrap = [\"/dns4/bootstrap.example.com/udp/4001/quic-v1/p2p/12D3KooW...\"]\n\
             bootstrap = []\n"
                .to_owned()
        }
        ConfigFile::Models => "# Default model id, plus any pre-warmed shortcuts.\n\
             default_model = \"hf-internal-testing/tiny-random-LlamaForCausalLM\"\n"
            .to_owned(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_valid_toml() {
        let original = "relays = [\"/dns4/example/udp/4001/quic-v1/p2p/12D3\"]\n";
        let parsed: toml::Value = toml::from_str(original).expect("should parse");
        assert!(parsed.get("relays").is_some());
    }

    #[test]
    fn rejects_invalid_toml() {
        let bad = "this is not = toml ===";
        let parsed: Result<toml::Value, _> = toml::from_str(bad);
        assert!(parsed.is_err());
    }

    #[test]
    fn config_file_path_mapping() {
        assert_eq!(ConfigFile::Relays.path(), "config/relays.toml");
        assert_eq!(ConfigFile::Bootstrap.path(), "config/bootstrap.toml");
        assert_eq!(ConfigFile::Models.path(), "config/models.toml");
    }
}
