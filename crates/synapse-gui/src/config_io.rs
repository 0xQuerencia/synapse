//! Bounded, validated configuration editing with atomic private replacement.
//! File labels are a fixed allowlist; callers cannot select arbitrary paths.

use std::{fs, path::Path};

use anyhow::{Context, Result, bail};
use serde::Deserialize;

const MAX_CONFIG_BYTES: usize = 64 * 1024;
const RELAYS_FILE: &str = "config/relays.toml";
const BOOTSTRAP_FILE: &str = "config/bootstrap.toml";
const DISCOVERY_FILE: &str = "config/discovery.toml";
const MODELS_FILE: &str = "config/models.toml";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConfigFile {
    Relays,
    Bootstrap,
    Discovery,
    Models,
}

impl ConfigFile {
    pub const fn path(self) -> &'static str {
        match self {
            Self::Relays => RELAYS_FILE,
            Self::Bootstrap => BOOTSTRAP_FILE,
            Self::Discovery => DISCOVERY_FILE,
            Self::Models => MODELS_FILE,
        }
    }
    pub const fn label(self) -> &'static str {
        match self {
            Self::Relays => "relays",
            Self::Bootstrap => "bootstrap",
            Self::Discovery => "discovery",
            Self::Models => "models",
        }
    }
    pub fn from_label(label: &str) -> Option<Self> {
        match label {
            "relays" => Some(Self::Relays),
            "bootstrap" => Some(Self::Bootstrap),
            "discovery" => Some(Self::Discovery),
            "models" => Some(Self::Models),
            _ => None,
        }
    }
}

pub fn read(file: ConfigFile) -> Result<String> {
    read_in(&std::env::current_dir()?, file)
}

pub fn write(file: ConfigFile, contents: &str) -> Result<()> {
    write_in(&std::env::current_dir()?, file, contents)
}

fn read_in(root: &Path, file: ConfigFile) -> Result<String> {
    let directory = root.join("config");
    match synapse_core::identity::validate_owned_directory(&directory) {
        Ok(()) => {}
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(default_for(file)),
        Err(error) => return Err(error.into()),
    }
    let bytes = match synapse_core::identity::read_owned_config(
        &root.join(file.path()),
        MAX_CONFIG_BYTES as u64,
    ) {
        Ok(bytes) => bytes,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(default_for(file)),
        Err(error) => return Err(error.into()),
    };
    let contents = String::from_utf8(bytes).context("configuration must be UTF-8")?;
    validate(file, &contents)?;
    Ok(contents)
}

fn write_in(root: &Path, file: ConfigFile, contents: &str) -> Result<()> {
    // Validation precedes directory creation and every write.
    validate(file, contents)?;
    let directory = root.join("config");
    let mut builder = fs::DirBuilder::new();
    builder.recursive(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::DirBuilderExt;
        builder.mode(0o700);
    }
    builder
        .create(&directory)
        .context("create configuration directory")?;
    synapse_core::identity::validate_owned_directory(&directory)?;
    synapse_core::identity::write_private_state(&root.join(file.path()), contents.as_bytes())
        .context("atomically replace configuration")
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RelayConfig {
    relays: Vec<String>,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct BootstrapConfig {
    peers: Vec<String>,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ModelShortcut {
    id: String,
    #[serde(default)]
    purpose: String,
}

fn validate(file: ConfigFile, contents: &str) -> Result<()> {
    if contents.len() > MAX_CONFIG_BYTES {
        bail!("configuration exceeds 64 KiB");
    }
    match file {
        ConfigFile::Relays => {
            let config: RelayConfig =
                toml::from_str(contents).context("invalid relay configuration")?;
            validate_addresses(config.relays)
        }
        ConfigFile::Bootstrap => {
            let config: BootstrapConfig = toml::from_str(contents)
                .context("invalid bootstrap configuration (expected peers)")?;
            validate_addresses(config.peers)
        }
        ConfigFile::Discovery => {
            let config: synapse_core::netconfig::DiscoveryPolicy =
                toml::from_str(contents).context("invalid discovery configuration")?;
            config.validate().map_err(Into::into)
        }
        ConfigFile::Models => {
            let table: toml::Table =
                toml::from_str(contents).context("invalid model configuration")?;
            if table.is_empty() || table.len() > 64 {
                bail!("expected 1..64 model configuration entries");
            }
            for (name, value) in table {
                if name == "default_model" {
                    validate_model(value.as_str().context("default_model must be a string")?)?;
                } else {
                    if name.is_empty()
                        || name.len() > 64
                        || !name
                            .bytes()
                            .all(|b| b.is_ascii_alphanumeric() || b"_-".contains(&b))
                    {
                        bail!("invalid model shortcut name");
                    }
                    let shortcut: ModelShortcut = value
                        .try_into()
                        .context("model shortcut requires id and optional purpose")?;
                    validate_model(&shortcut.id)?;
                    if shortcut.purpose.len() > 1024
                        || shortcut.purpose.chars().any(char::is_control)
                    {
                        bail!("invalid model shortcut purpose");
                    }
                }
            }
            Ok(())
        }
    }
}

fn validate_addresses(entries: Vec<String>) -> Result<()> {
    if entries.len() > synapse_core::netconfig::MAX_ENTRY_POINTS {
        bail!("too many network entry points");
    }
    for entry in entries {
        let address = entry.parse().context("invalid peer multiaddr")?;
        synapse_core::netconfig::validate_direct_peer_addr(&address)?;
    }
    Ok(())
}

fn validate_model(model: &str) -> Result<()> {
    synapse_core::ModelIdentity::new(model).validate()?;
    if model.starts_with('/')
        || model.split('/').any(|part| matches!(part, "" | "." | ".."))
        || !model
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b"-_./:".contains(&b))
    {
        bail!("model identifier must not contain a path traversal or URL");
    }
    Ok(())
}

fn default_for(file: ConfigFile) -> String {
    match file {
        ConfigFile::Relays => "# Approved relay endpoints require a final /p2p/<peer-id>.\nrelays = []\n",
        ConfigFile::Bootstrap => "# Approved Kademlia bootstrap endpoints.\npeers = []\n",
        ConfigFile::Discovery => "# Discovery supplies candidates; it never grants membership.\nallow_lan_discovery = false\nallow_dht_discovery = false\nallowed_peers = []\n",
        ConfigFile::Models => "default_model = \"hf-internal-testing/tiny-random-LlamaForCausalLM\"\n",
    }.to_owned()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    struct TestRoot(PathBuf);
    impl TestRoot {
        fn new() -> Self {
            let path = std::env::temp_dir().join(format!(
                "synapse-config-test-{:032x}",
                rand::random::<u128>()
            ));
            fs::create_dir(&path).unwrap();
            Self(path)
        }
    }
    impl Drop for TestRoot {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn defaults_match_strict_runtime_shapes() {
        let root = TestRoot::new();
        for file in [
            ConfigFile::Relays,
            ConfigFile::Bootstrap,
            ConfigFile::Discovery,
            ConfigFile::Models,
        ] {
            let default = read_in(&root.0, file).unwrap();
            validate(file, &default).unwrap();
            write_in(&root.0, file, &default).unwrap();
            assert_eq!(read_in(&root.0, file).unwrap(), default);
        }
        synapse_core::NetConfig::load_strict(&root.0.join("config")).unwrap();
    }

    #[test]
    fn failed_validation_preserves_previous_configuration() {
        let root = TestRoot::new();
        let valid = "relays = []\n";
        write_in(&root.0, ConfigFile::Relays, valid).unwrap();
        for invalid in [
            "relays = [123]",
            "relays = []\nunknown = true",
            "relays = [\"/ip4/127.0.0.1/tcp/1\"]",
            &"x".repeat(MAX_CONFIG_BYTES + 1),
        ] {
            assert!(write_in(&root.0, ConfigFile::Relays, invalid).is_err());
            assert_eq!(read_in(&root.0, ConfigFile::Relays).unwrap(), valid);
        }
        assert!(validate(ConfigFile::Bootstrap, "bootstrap = []").is_err());
        assert!(validate(ConfigFile::Discovery, "allow_internet = true").is_err());
        assert!(validate(ConfigFile::Models, "[dev]\nid = 'safe'\ncommand = 'bad'").is_err());
        assert!(validate(ConfigFile::Models, "default_model = '../escape'").is_err());
    }

    #[test]
    fn label_allowlist_and_reads_are_bounded() {
        assert!(ConfigFile::from_label("../identity.key").is_none());
        assert!(ConfigFile::from_label("/tmp/relays").is_none());
        assert!(ConfigFile::from_label("relays.toml").is_none());
        let root = TestRoot::new();
        write_in(&root.0, ConfigFile::Bootstrap, "peers = []").unwrap();
        fs::write(
            root.0.join(BOOTSTRAP_FILE),
            "x".repeat(MAX_CONFIG_BYTES + 1),
        )
        .unwrap();
        assert!(read_in(&root.0, ConfigFile::Bootstrap).is_err());
    }

    #[cfg(unix)]
    #[test]
    fn symlink_destinations_and_directories_are_rejected_without_overwriting_targets() {
        use std::os::unix::fs::{PermissionsExt, symlink};
        let root = TestRoot::new();
        let target = root.0.join("outside");
        fs::write(&target, "keep me").unwrap();
        fs::create_dir(root.0.join("config")).unwrap();
        symlink(&target, root.0.join(RELAYS_FILE)).unwrap();
        assert!(read_in(&root.0, ConfigFile::Relays).is_err());
        assert!(write_in(&root.0, ConfigFile::Relays, "relays = []").is_err());
        assert_eq!(fs::read_to_string(&target).unwrap(), "keep me");
        fs::remove_dir_all(root.0.join("config")).unwrap();
        let external = root.0.join("external");
        fs::create_dir(&external).unwrap();
        symlink(&external, root.0.join("config")).unwrap();
        assert!(write_in(&root.0, ConfigFile::Bootstrap, "peers = []").is_err());
        assert!(!external.join("bootstrap.toml").exists());
        fs::remove_file(root.0.join("config")).unwrap();
        write_in(&root.0, ConfigFile::Bootstrap, "peers = []").unwrap();
        assert_eq!(
            fs::metadata(root.0.join(BOOTSTRAP_FILE))
                .unwrap()
                .permissions()
                .mode()
                & 0o777,
            0o600
        );
    }
}
