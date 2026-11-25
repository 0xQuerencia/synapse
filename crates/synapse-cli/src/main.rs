use std::{
    fs,
    io::{self, IsTerminal, Write},
    path::PathBuf,
    time::{Duration, Instant},
};

use anyhow::{Context, Result, anyhow, bail};
use clap::{ArgAction, Parser, Subcommand};
use synapse_core::{
    ModelIdentity, NodeIdentity, SimpleTokenizer, ToyLlama, ToyLlamaConfig, download_model,
    generate_pairing_code,
    network::{ChatEvent, StopReason, SynapseNode, SynapseRole, parse_listen_addr},
    split_layers, tensor_to_payload, validate_pairing_code,
};
use tracing::{debug, info};
use tracing_subscriber::EnvFilter;

#[derive(Debug, Parser)]
#[command(name = "synapse")]
#[command(about = "LAN sharded AI inference proof of concept")]
struct Cli {
    #[arg(short, long, global = true, action = ArgAction::Count)]
    verbose: u8,

    /// Path to the node's identity key (Ed25519 keypair, libp2p PeerId).
    /// Defaults to `~/.synapse/identity.key`. Override to run multiple
    /// nodes on the same machine with distinct PeerIds.
    #[arg(long, global = true, env = "SYNAPSE_IDENTITY_PATH")]
    identity_path: Option<String>,

    #[command(subcommand)]
    command: Option<Command>,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Start a master node and chat with the AI.
    Host {
        #[arg(long)]
        master: bool,

        #[arg(long)]
        model: String,

        /// Optional initial prompt (omit for interactive REPL).
        #[arg(long)]
        prompt: Option<String>,

        #[arg(long, default_value = "200")]
        max_tokens: usize,

        #[arg(long, default_value = "0.0.0.0:0")]
        listen: String,

        /// Run one supplied prompt and exit with failure on generation errors.
        #[arg(long, requires = "prompt")]
        once: bool,
        /// Send raw text; use for base models and deterministic fixtures.
        #[arg(long)]
        raw_prompt: bool,
        /// Wait for this many authenticated workers before dispatch.
        #[arg(long, default_value_t = 1)]
        min_workers: usize,
        /// Bound worker discovery and generation independently.
        #[arg(long, default_value_t = 120)]
        timeout_secs: u64,
        /// Create a private JSONL record of status, token IDs and actual timings.
        #[arg(long)]
        events_jsonl: Option<PathBuf>,
        /// Per-node inference memory limit (MiB), enforced before shard loading.
        #[arg(long)]
        memory_mib: Option<u64>,
        /// Exact inference reservation limit, useful for reproducible small models.
        #[arg(long, conflicts_with = "memory_mib")]
        memory_bytes: Option<u64>,
        /// Keep transformer layers entirely on the authenticated workers.
        #[arg(long)]
        coordinator_only: bool,
    },
    /// Join an existing master as a worker node.
    Join {
        #[arg(value_name = "MASTER_ADDR")]
        master_addr: Option<String>,

        #[arg(long)]
        model: String,

        #[arg(long)]
        code: String,
        #[arg(long)]
        memory_mib: Option<u64>,
        #[arg(long, conflicts_with = "memory_mib")]
        memory_bytes: Option<u64>,
        /// Linux process virtual address-space cap; distinct from inference memory.
        #[arg(long)]
        address_space_mib: Option<u64>,
        #[arg(long)]
        events_jsonl: Option<PathBuf>,
        #[arg(long, default_value = "0.0.0.0:0")]
        listen: String,
    },
    /// Run a localhost demo (no networking).
    Demo {
        #[arg(long)]
        model: String,

        #[arg(long)]
        prompt: String,

        #[arg(long, default_value = "50")]
        max_tokens: usize,
    },
    /// Stream deterministic local inference from a verified model directory.
    Infer {
        #[arg(long)]
        model_dir: PathBuf,
        #[arg(long)]
        prompt: String,
        #[arg(long, default_value_t = 8)]
        max_tokens: usize,
        #[arg(long)]
        events_jsonl: Option<PathBuf>,
    },
    /// Check local readiness without downloading a model or starting a session.
    Doctor {
        /// Verify every artifact in an already downloaded model manifest.
        #[arg(long)]
        model_dir: Option<PathBuf>,
        #[arg(long, default_value = "config")]
        config_dir: PathBuf,
        /// Total input plus output context used for the inference reservation report.
        #[arg(long, default_value_t = 32)]
        context_tokens: usize,
    },
    /// Connect to a model swarm via DHT (alias for join without manual address).
    Connect {
        #[arg(long)]
        model: String,

        #[arg(long)]
        code: String,
    },
    /// Run as a public relay + bootstrap node (for global NAT traversal).
    Relay {
        /// Listen address (default binds QUIC + TCP on all interfaces, port 4001).
        #[arg(long, default_value = "0.0.0.0:4001")]
        listen: String,
    },
    /// List connected peers and capabilities.
    Peers,
    /// Show DHT and network status.
    Status,
    /// Show this node's persistent identity.
    Id,
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    init_tracing(cli.verbose);

    // Route the identity-path override through synapse-core's
    // process-wide setter so the CLI doesn't have to call unsafe
    // `set_var` (unsafe in Rust 2024) just to swap the key path.
    if let Some(p) = &cli.identity_path {
        synapse_core::set_identity_path_override(Some(std::path::Path::new(p)));
    }

    match cli.command {
        Some(Command::Host {
            master,
            model,
            prompt,
            max_tokens,
            listen,
            once,
            raw_prompt,
            min_workers,
            timeout_secs,
            events_jsonl,
            memory_mib,
            memory_bytes,
            coordinator_only,
        }) => {
            host(
                master,
                model,
                prompt,
                max_tokens,
                listen,
                HostOptions {
                    once,
                    raw_prompt,
                    min_workers,
                    timeout_secs,
                    events_jsonl,
                    memory_mib,
                    memory_bytes,
                    coordinator_only,
                },
            )
            .await
        }
        Some(Command::Join {
            master_addr,
            model,
            code,
            memory_mib,
            memory_bytes,
            address_space_mib,
            events_jsonl,
            listen,
        }) => {
            join(
                master_addr,
                model,
                code,
                JoinOptions {
                    memory_mib,
                    memory_bytes,
                    address_space_mib,
                    events_jsonl,
                    listen,
                },
            )
            .await
        }
        Some(Command::Connect { model, code }) => {
            join(None, model, code, JoinOptions::default()).await
        }
        Some(Command::Relay { listen }) => relay_server(listen).await,
        Some(Command::Peers) => {
            let identity = NodeIdentity::load_or_create().context("failed to load identity")?;
            print_panel(
                "Synapse Peers",
                &[
                    format!("local peer id: {}", identity.peer_id()),
                    "peer listing is exposed through the running node logs for now".to_owned(),
                ],
            );
            Ok(())
        }
        Some(Command::Status) => {
            let identity = NodeIdentity::load_or_create().context("failed to load identity")?;
            let model_count = discover_cached_models()?.len();
            let net = synapse_core::NetConfig::load_default();
            print_panel(
                "Synapse Status",
                &[
                    format!("peer id: {}", identity.peer_id()),
                    format!("cached models: {model_count}"),
                    format!("bootstrap peers: {}", net.bootstrap.len()),
                    format!("relay servers: {}", net.relays.len()),
                    "DHT/relay status is emitted by the node at runtime".to_owned(),
                ],
            );
            Ok(())
        }
        Some(Command::Demo {
            model,
            prompt,
            max_tokens,
        }) => demo(model, prompt, max_tokens).await,
        Some(Command::Infer {
            model_dir,
            prompt,
            max_tokens,
            events_jsonl,
        }) => infer_local(model_dir, prompt, max_tokens, events_jsonl),
        Some(Command::Id) => show_identity(),
        Some(Command::Doctor {
            model_dir,
            config_dir,
            context_tokens,
        }) => doctor(
            model_dir,
            config_dir,
            cli.identity_path.as_deref(),
            context_tokens,
        ),
        None => onboarding_gateway().await,
    }
}

fn doctor(
    model_dir: Option<PathBuf>,
    config_dir: PathBuf,
    identity_path: Option<&str>,
    context_tokens: usize,
) -> Result<()> {
    let mut checks = Vec::new();
    let mut failed = false;
    let mut check = |name: &str, result: Result<serde_json::Value>| match result {
        Ok(details) => {
            checks.push(serde_json::json!({"name":name,"status":"passed","details":details}))
        }
        Err(error) => {
            failed = true;
            checks
                .push(serde_json::json!({"name":name,"status":"failed","error":error.to_string()}));
        }
    };
    check("network_config", synapse_core::NetConfig::load_strict(&config_dir).map(|net| {
        serde_json::json!({"bootstrap_peers":net.bootstrap.len(),"relay_peers":net.relays.len(),
            "pinned_discovery_peers":net.discovery.allowed_peers.len(),
            "lan_discovery":net.discovery.allow_lan_discovery,"dht_discovery":net.discovery.allow_dht_discovery})
    }).map_err(Into::into));
    check(
        "loopback_tcp",
        std::net::TcpListener::bind("127.0.0.1:0")
            .map(|_| serde_json::json!({"bind":"available"}))
            .map_err(Into::into),
    );
    check(
        "loopback_udp",
        std::net::UdpSocket::bind("127.0.0.1:0")
            .map(|_| serde_json::json!({"bind":"available"}))
            .map_err(Into::into),
    );
    if let Some(path) = identity_path {
        check(
            "existing_identity",
            NodeIdentity::load_from_path(std::path::Path::new(path))
                .map(|identity| serde_json::json!({"peer_id":identity.peer_id().to_string()}))
                .map_err(Into::into),
        );
    }
    if let Some(directory) = model_dir {
        check(
            "model_integrity",
            (|| {
                let files =
                    synapse_core::inference::model_loader::ModelFiles::from_directory(&directory)?;
                let config = ToyLlamaConfig::from_file(&files.config)?;
                let profile = synapse_core::inference::memory::ModelMemoryProfile::new(
                    &config,
                    files.weights.extension().is_some_and(|ext| ext == "gguf"),
                )?;
                let full = profile.estimate(
                    config.num_hidden_layers,
                    context_tokens,
                    synapse_core::inference::pipeline::TensorRoles::COORDINATOR,
                )?;
                let one_worker = profile.estimate(
                    1,
                    context_tokens,
                    synapse_core::inference::pipeline::TensorRoles::WORKER,
                )?;
                Ok(
                    serde_json::json!({"model_id":files.manifest.model_id,"revision":files.manifest.revision,
                "manifest_digest":files.manifest.digest()?,"layers":config.num_hidden_layers,
                "context":config.max_position_embeddings,"artifacts":files.manifest.artifacts.len(),
                "reservation_context_tokens":context_tokens,"full_model_reservation":full,
                "one_layer_worker_reservation":one_worker,"reservation_is_process_rss":false}),
                )
            })(),
        );
    }
    println!(
        "{}",
        serde_json::to_string_pretty(&serde_json::json!({
            "schema_version":1,"ready_for_local_checks":!failed,
            "platform":std::env::consts::OS,"architecture":std::env::consts::ARCH,
            "hardware":synapse_core::HardwareProbe::detect(),"checks":checks,
            "not_tested":["remote reachability","multicast discovery","physical device sharding","WAN"]
        }))?
    );
    if failed {
        bail!("readiness checks failed; resolve the named failures before starting")
    }
    Ok(())
}

fn show_identity() -> Result<()> {
    let identity =
        NodeIdentity::load_or_create().context("failed to load or create node identity")?;

    let pub_bytes = identity.public_key_bytes();
    let pub_hex: String = pub_bytes.iter().map(|b| format!("{b:02x}")).collect();

    println!("Peer ID:      {}", identity.peer_id());
    println!("Public key:   {}", truncate_hex(&pub_hex, 16));
    println!("Key file:     {}", identity.key_path().display());
    println!("Short ID:     {}", identity.display_short());

    Ok(())
}

fn truncate_hex(hex: &str, show_bytes: usize) -> String {
    let chars = show_bytes * 2;
    if hex.len() <= chars + 6 {
        hex.to_owned()
    } else {
        format!("{}...{}", &hex[..chars], &hex[hex.len() - 6..])
    }
}

/// Interpret C-style escape sequences in prompt strings.
fn unescape_prompt(s: &str) -> String {
    s.replace("\\n", "\n").replace("\\t", "\t")
}

const SYNAPSE_ASCII_LOGO: &str = r#"   _____
  / ___/ __  __ ____   ____ _ ____   _____ ___
  \__ \ / / / // __ \ / __ `// __ \ / ___// _ \
 ___/ // /_/ // / / // /_/ // /_/ /(__  )/  __/
/____/ \__, //_/ /_/ \__,_// .___//____/ \___/
      /____/              /_/"#;
const ANSI_VIOLET: &str = "\x1b[38;5;141m";
const ANSI_RESET: &str = "\x1b[0m";

fn use_color() -> bool {
    std::env::var_os("NO_COLOR").is_none()
        && std::env::var("TERM").map(|v| v != "dumb").unwrap_or(true)
        && io::stdout().is_terminal()
}

fn violet(text: &str) -> String {
    if use_color() {
        format!("{ANSI_VIOLET}{text}{ANSI_RESET}")
    } else {
        text.to_owned()
    }
}

fn print_panel(title: &str, lines: &[String]) {
    let rendered_lines: Vec<&str> = lines.iter().flat_map(|line| line.split('\n')).collect();

    let width = std::iter::once(title.len())
        .chain(rendered_lines.iter().map(|line| line.len()))
        .max()
        .unwrap_or(0);
    let border = format!("+{}+", "-".repeat(width + 2));
    let title_line = format!("| {:<width$} |", title, width = width);

    println!("{}", violet(SYNAPSE_ASCII_LOGO));
    println!("{}", violet(&border));
    println!("{}", violet(&title_line));
    println!("{}", violet(&border));
    for line in rendered_lines {
        println!("| {:<width$} |", line, width = width);
    }
    println!("{}", violet(&border));
}

async fn onboarding_gateway() -> Result<()> {
    let cached_models = discover_cached_models()?;
    print_panel(
        "Synapse Gateway",
        &[
            "interactive onboarding".to_owned(),
            "1) host (master)".to_owned(),
            "   -> generates pairing code".to_owned(),
            "2) join (manual master / mDNS)".to_owned(),
            "   -> uses code from host".to_owned(),
            "3) connect (DHT discovery)".to_owned(),
            "   -> auto-discovers master".to_owned(),
            "4) demo (localhost)".to_owned(),
            "   -> no pairing code".to_owned(),
            "5) exit".to_owned(),
        ],
    );

    let mode = prompt_menu(
        "Select mode",
        &[
            "Host (master)",
            "Join (manual master / mDNS)",
            "Connect (DHT discovery)",
            "Demo (localhost)",
            "Exit",
        ],
    )?;

    match mode {
        0 => {
            let model = prompt_model_choice(&cached_models)?;
            let prompt = prompt_optional("Initial prompt (optional, press Enter to skip)")?;
            let max_tokens = prompt_usize_default("Max tokens", 200)?;
            let listen = prompt_default("Listen address", "0.0.0.0:0")?;
            host(
                true,
                model,
                prompt,
                max_tokens,
                listen,
                HostOptions::default(),
            )
            .await
        }
        1 => {
            println!("\n[info] join requires pairing code from an active host session");
            let model = prompt_model_choice(&cached_models)?;
            let master_addr = prompt_optional("Master address (optional, press Enter for mDNS)")?;
            let code = prompt_required("Pairing code (shown by host)")?;
            join(master_addr, model, code, JoinOptions::default()).await
        }
        2 => {
            println!("\n[info] connect discovers master via DHT");
            let model = prompt_model_choice(&cached_models)?;
            let code = prompt_required("Pairing code (shown by host)")?;
            join(None, model, code, JoinOptions::default()).await
        }
        3 => {
            let model = prompt_model_choice(&cached_models)?;
            let prompt = prompt_required("Prompt")?;
            let max_tokens = prompt_usize_default("Max new tokens", 50)?;
            demo(model, prompt, max_tokens).await
        }
        _ => {
            println!("[bye] gateway closed");
            Ok(())
        }
    }
}

fn discover_cached_models() -> Result<Vec<String>> {
    let home = std::env::var("HOME").map_err(|_| anyhow!("HOME is not set"))?;
    let hub_dir = if let Ok(hf_home) = std::env::var("HF_HOME") {
        PathBuf::from(hf_home).join("hub")
    } else {
        PathBuf::from(home).join(".cache/huggingface/hub")
    };

    let mut models = Vec::new();
    if !hub_dir.exists() {
        return Ok(models);
    }

    for entry in fs::read_dir(&hub_dir)? {
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

fn prompt_model_choice(cached_models: &[String]) -> Result<String> {
    if cached_models.is_empty() {
        println!("[info] no downloaded models found in HF cache yet");
        return prompt_required("Model ID (e.g. hf-internal-testing/tiny-random-LlamaForCausalLM)");
    }

    println!("\nDownloaded models:");
    let mut options: Vec<String> = cached_models.to_vec();
    options.push("Enter custom model ID".to_owned());
    let refs: Vec<&str> = options.iter().map(|s| s.as_str()).collect();
    let idx = prompt_menu("Choose model", &refs)?;

    if idx == options.len() - 1 {
        prompt_required("Model ID")
    } else {
        Ok(options[idx].clone())
    }
}

fn prompt_menu(title: &str, options: &[&str]) -> Result<usize> {
    println!("\n{title}:");
    for (index, option) in options.iter().enumerate() {
        println!("  {}) {}", index + 1, option);
    }

    loop {
        let raw = prompt_required("Enter number")?;
        let parsed = raw.parse::<usize>();
        match parsed {
            Ok(choice) if (1..=options.len()).contains(&choice) => return Ok(choice - 1),
            _ => println!("[error] enter a number between 1 and {}", options.len()),
        }
    }
}

fn prompt_required(label: &str) -> Result<String> {
    loop {
        let value = prompt_line(label)?;
        if !value.is_empty() {
            return Ok(value);
        }
        println!("[error] value is required");
    }
}

fn prompt_optional(label: &str) -> Result<Option<String>> {
    let value = prompt_line(label)?;
    if value.is_empty() {
        Ok(None)
    } else {
        Ok(Some(value))
    }
}

fn prompt_default(label: &str, default: &str) -> Result<String> {
    let value = prompt_line(&format!("{label} [{default}]"))?;
    if value.is_empty() {
        Ok(default.to_owned())
    } else {
        Ok(value)
    }
}

fn prompt_usize_default(label: &str, default: usize) -> Result<usize> {
    loop {
        let raw = prompt_default(label, &default.to_string())?;
        match raw.parse::<usize>() {
            Ok(value) => return Ok(value),
            Err(_) => println!("[error] enter a valid positive number"),
        }
    }
}

fn prompt_line(label: &str) -> Result<String> {
    print!("{label}: ");
    io::stdout().flush()?;
    let mut input = String::new();
    let read = io::stdin().read_line(&mut input)?;
    if read == 0 {
        bail!("input stream closed");
    }
    Ok(input.trim().to_owned())
}

#[derive(Debug)]
struct HostOptions {
    once: bool,
    raw_prompt: bool,
    min_workers: usize,
    timeout_secs: u64,
    events_jsonl: Option<PathBuf>,
    memory_mib: Option<u64>,
    memory_bytes: Option<u64>,
    coordinator_only: bool,
}
impl Default for HostOptions {
    fn default() -> Self {
        Self {
            once: false,
            raw_prompt: false,
            min_workers: 1,
            timeout_secs: 120,
            events_jsonl: None,
            memory_mib: None,
            memory_bytes: None,
            coordinator_only: false,
        }
    }
}

struct EventLog {
    file: Option<std::fs::File>,
    started: Instant,
}
impl EventLog {
    fn new(path: Option<&PathBuf>) -> Result<Self> {
        let file = path
            .map(|path| {
                let mut options = std::fs::OpenOptions::new();
                options.write(true).create_new(true);
                #[cfg(unix)]
                {
                    use std::os::unix::fs::OpenOptionsExt;
                    options.mode(0o600);
                }
                options.open(path)
            })
            .transpose()?;
        Ok(Self {
            file,
            started: Instant::now(),
        })
    }
    fn record(&mut self, event: &str, value: serde_json::Value) -> Result<()> {
        if let Some(file) = &mut self.file {
            serde_json::to_writer(
                &mut *file,
                &serde_json::json!({"event":event,"elapsed_us":self.started.elapsed().as_micros(),"data":value}),
            )?;
            writeln!(file)?;
            file.flush()?;
        }
        Ok(())
    }
}

fn inference_budget(mib: Option<u64>, bytes: Option<u64>) -> Result<Option<u64>> {
    if let Some(bytes) = bytes {
        if mib.is_some() || bytes == 0 {
            bail!("choose one positive inference memory limit")
        }
        return Ok(Some(bytes));
    }
    mib.map(|value| {
        value
            .checked_mul(1024 * 1024)
            .filter(|bytes| *bytes > 0)
            .context("memory limit must be positive and must not overflow")
    })
    .transpose()
}

// ─── Host (Master) ───────────────────────────────────────────────────────────

async fn host(
    master: bool,
    model: String,
    prompt: Option<String>,
    max_tokens: usize,
    listen: String,
    options: HostOptions,
) -> Result<()> {
    if !master {
        bail!("phase 1 only supports `synapse host --master`");
    }

    if !(1..=64).contains(&options.min_workers)
        || options.timeout_secs == 0
        || !(1..=4096).contains(&max_tokens)
    {
        bail!("workers must be 1..64, max tokens 1..4096, and timeout positive");
    }
    let prompt_model_id = if std::path::Path::new(&model).is_dir() {
        synapse_core::inference::model_loader::ModelFiles::from_directory(std::path::Path::new(
            &model,
        ))?
        .manifest
        .model_id
    } else {
        model.clone()
    };
    let format_prompt = |text: &str| {
        synapse_core::inference::prompt_format::format_prompt_for_model(
            &prompt_model_id,
            text,
            options.raw_prompt,
        )
    };
    let mut events = EventLog::new(options.events_jsonl.as_ref())?;
    let model = ModelIdentity::new(model);
    model.validate()?;

    let pairing_code = generate_pairing_code();
    let listen_addr = parse_listen_addr(&listen).map_err(|error| anyhow!(error))?;

    print_panel(
        "Synapse Master Node",
        &[
            format!("model: {}", model.id),
            format!("max tokens: {max_tokens}"),
            format!("pairing code: {pairing_code}"),
            format!(
                "join cmd: synapse join --model {} --code {pairing_code}",
                model.id
            ),
        ],
    );

    println!("\n[status] downloading model...");
    let mut node = SynapseNode::new(
        SynapseRole::Master {
            expected_pairing_code: pairing_code,
            max_tokens,
        },
        model,
        listen_addr,
    )
    .await?;

    if let Some(bytes) = inference_budget(options.memory_mib, options.memory_bytes)? {
        node.set_memory_limit_bytes(bytes)?;
    }
    node.set_coordinator_only(options.coordinator_only)?;
    node.load_model().await?;
    println!("[ok] model metadata verified\n");
    let mut status_rx = node.status_receiver();
    events.record("node_started", serde_json::json!({"peer_id":node.local_peer_id().to_string(),"model":prompt_model_id,"min_workers":options.min_workers,"memory_mib":options.memory_mib,"coordinator_only":options.coordinator_only}))?;

    // Set up interactive chat channels
    let (prompt_tx, mut event_rx) = node.chat_channels();

    println!("peer id: {}", node.local_peer_id());
    println!("[status] waiting for worker connection...\n");

    // Spawn the swarm event loop in a background task
    let node_handle = tokio::spawn(async move { node.run().await });

    // A first connection does not mean the requested topology is ready.
    let ready = tokio::time::timeout(Duration::from_secs(options.timeout_secs), async {
        loop {
            if status_rx.borrow().authenticated_count >= options.min_workers { break Ok::<(), anyhow::Error>(()); }
            tokio::select! {
                changed = status_rx.changed() => { changed.context("node status channel closed")?; }
                event = event_rx.recv() => { if event.is_none() { bail!("node stopped before worker authentication"); } }
            }
        }
    }).await;
    if !matches!(ready, Ok(Ok(()))) {
        node_handle.abort();
        bail!("workers did not authenticate within the configured deadline");
    }
    events.record(
        "workers_authenticated",
        serde_json::to_value(status_rx.borrow().clone())?,
    )?;

    // ─── Interactive REPL ────────────────────────────────────────────────
    print_panel(
        "Worker connected - chat active",
        &[
            "type a message and press Enter".to_owned(),
            "commands: /help  /clear  /quit".to_owned(),
            "press Ctrl+D to exit".to_owned(),
        ],
    );
    println!();

    // If a --prompt was provided, use it as the first message
    let first_prompt = prompt.map(|p| unescape_prompt(&p));
    let has_first = first_prompt.is_some();

    if let Some(ref first) = first_prompt {
        println!("[you] {first}");
        let formatted = format_prompt(first)?;
        events.record(
            "request_started",
            serde_json::json!({"max_tokens":max_tokens}),
        )?;
        if let Err(error) = prompt_tx.try_send(formatted) {
            bail!("failed to send initial prompt to node: {error}");
        }
    }

    let mut need_readline = !has_first;

    loop {
        if need_readline {
            // Spawn a blocking readline
            let readline_result = tokio::task::spawn_blocking(|| {
                let mut rl = rustyline::DefaultEditor::new().ok()?;
                let line = rl.readline("\n[you] ").ok()?;
                Some(line)
            })
            .await?;

            match readline_result {
                Some(line) => {
                    let trimmed = line.trim();
                    if trimmed.is_empty() {
                        continue;
                    }
                    if trimmed == "/quit" || trimmed == "/exit" {
                        println!("\n[bye] session closed");
                        break;
                    }
                    if trimmed == "/help" {
                        println!("\ncommands: /help  /clear  /quit");
                        continue;
                    }
                    if trimmed == "/clear" {
                        print!("\x1B[2J\x1B[H");
                        let _ = io::stdout().flush();
                        continue;
                    }
                    let formatted = format_prompt(trimmed)?;
                    events.record(
                        "request_started",
                        serde_json::json!({"max_tokens":max_tokens}),
                    )?;
                    if let Err(error) = prompt_tx.try_send(formatted) {
                        eprintln!("\n[error] failed to send prompt to node: {error}");
                        continue;
                    }
                }
                None => {
                    // Ctrl+D or error
                    println!("\n[bye] session closed");
                    break;
                }
            }
        }

        // Stream every real generated token and retain exact completion counts.
        print!("\n[assistant] ");
        let _ = io::stdout().flush();
        let deadline = tokio::time::Instant::now() + Duration::from_secs(options.timeout_secs);
        let mut generation_failed = false;
        loop {
            tokio::select! {
                _ = tokio::time::sleep_until(deadline) => {
                    events.record("generation_timeout", serde_json::json!({}))?;
                    node_handle.abort(); bail!("generation deadline exceeded");
                }
                changed = status_rx.changed() => {
                    if changed.is_ok() { events.record("status", serde_json::to_value(status_rx.borrow().clone())?)?; }
                }
                event = event_rx.recv() => match event {
                    Some(ChatEvent::TokenGenerated { token_id, position }) => {
                        events.record("token", serde_json::json!({"token_id":token_id,"position":position}))?;
                    }
                    Some(ChatEvent::Token(text)) => { print!("{text}"); let _ = io::stdout().flush(); }
                    Some(ChatEvent::GenerationDone { tokens_generated, reason }) => {
                        let reason = match reason { StopReason::Eos => "eos", StopReason::MaxTokens => "length", StopReason::Error => { generation_failed = true; "error" } };
                        events.record("generation_done", serde_json::json!({"tokens_generated":tokens_generated,"reason":reason}))?;
                        println!("\n[complete] tokens={tokens_generated} reason={reason}");
                        break;
                    }
                    Some(ChatEvent::WorkerConnected) => {}
                    None => { generation_failed = true; break; }
                }
            }
        }
        if generation_failed {
            node_handle.abort();
            bail!("distributed generation failed; see retained node logs");
        }
        if options.once {
            break;
        }

        // After generation, always readline next
        need_readline = true;
    }

    // Node will shut down when prompt_tx is dropped
    drop(prompt_tx);
    let mut node_handle = node_handle;
    if tokio::time::timeout(Duration::from_secs(3), &mut node_handle)
        .await
        .is_err()
    {
        node_handle.abort();
        let _ = node_handle.await;
    }

    Ok(())
}

// ─── Join (Worker) ───────────────────────────────────────────────────────────

struct JoinOptions {
    memory_mib: Option<u64>,
    memory_bytes: Option<u64>,
    address_space_mib: Option<u64>,
    events_jsonl: Option<PathBuf>,
    listen: String,
}
impl Default for JoinOptions {
    fn default() -> Self {
        Self {
            memory_mib: None,
            memory_bytes: None,
            address_space_mib: None,
            events_jsonl: None,
            listen: "0.0.0.0:0".into(),
        }
    }
}

async fn join(
    master_addr: Option<String>,
    model: String,
    code: String,
    options: JoinOptions,
) -> Result<()> {
    if let Some(cap) = inference_budget(options.address_space_mib, None)? {
        let actual = synapse_core::network::worker::apply_address_space_limit_bytes(cap)?;
        println!(
            "[resource] Linux virtual address-space ceiling: {actual} bytes (not an RSS limit)"
        );
    }
    validate_pairing_code(&code).context("pairing code format validation failed")?;

    let model = ModelIdentity::new(model);
    model.validate()?;

    let manual_master = master_addr
        .as_deref()
        .map(parse_listen_addr)
        .transpose()
        .map_err(|error| anyhow!("invalid master address: {error}"))?;

    print_panel(
        "Synapse Worker Node",
        &[
            format!(
                "master discovery: {}",
                manual_master
                    .as_ref()
                    .map(|addr| addr.to_string())
                    .unwrap_or_else(|| "pinned discovery configuration".to_owned())
            ),
            format!("model: {}", model.id),
        ],
    );

    println!("\n[status] verifying model artifacts...");
    let mut node = SynapseNode::new(
        SynapseRole::Worker {
            pairing_code: code,
            manual_master,
        },
        model,
        parse_listen_addr(&options.listen).map_err(|error| anyhow!(error))?,
    )
    .await?;

    if let Some(bytes) = inference_budget(options.memory_mib, options.memory_bytes)? {
        node.set_memory_limit_bytes(bytes)?;
    }
    node.load_model().await?;
    println!("[ok] model loaded\n");

    println!("peer id: {}", node.local_peer_id());
    println!("[status] waiting for an authenticated configured master...\n");
    let mut status = node.status_receiver();
    let mut events = EventLog::new(options.events_jsonl.as_ref())?;
    events.record(
        "node_started",
        serde_json::json!({"peer_id":node.local_peer_id().to_string(),"memory_budget_bytes":inference_budget(options.memory_mib, options.memory_bytes)?}),
    )?;
    let monitor = tokio::spawn(async move {
        while status.changed().await.is_ok() {
            if let Ok(value) = serde_json::to_value(status.borrow().clone()) {
                let _ = events.record("status", value);
            }
        }
    });
    let result = node.run().await;
    monitor.abort();
    result?;

    Ok(())
}

// ─── Relay Server (public entry point) ───────────────────────────────────────

async fn relay_server(listen: String) -> Result<()> {
    let listen_addr = parse_listen_addr(&listen).map_err(|error| anyhow!(error))?;

    // Relay nodes don't run inference; the model id is a placeholder.
    let model = ModelIdentity::new("synapse-relay");

    let node = SynapseNode::new(SynapseRole::RelayServer, model, listen_addr).await?;

    print_panel(
        "Synapse Relay Server",
        &[
            format!("peer id: {}", node.local_peer_id()),
            "share a /p2p-circuit relay multiaddr with peers via:".to_owned(),
            "  config/relays.toml -> relays = [\"<this-addr>/p2p/<peer-id>\"]".to_owned(),
            "routes circuits + seeds the DHT for global discovery".to_owned(),
            "press Ctrl+C to stop".to_owned(),
        ],
    );

    node.run().await?;
    Ok(())
}

// ─── Demo (Localhost) ────────────────────────────────────────────────────────

async fn demo(model: String, prompt: String, max_tokens: usize) -> Result<()> {
    let prompt = unescape_prompt(&prompt);
    let model_id = ModelIdentity::new(&model);
    model_id.validate()?;

    print_panel(
        "Synapse Localhost Demo",
        &[
            format!("model: {model}"),
            format!("prompt: {prompt}"),
            format!("max new tokens: {max_tokens}"),
        ],
    );

    // Download model
    println!("\n[status] downloading model...");
    let files = if std::path::Path::new(&model).is_dir() {
        synapse_core::inference::model_loader::ModelFiles::from_directory(std::path::Path::new(
            &model,
        ))?
    } else {
        download_model(&model).await?
    };
    println!("[ok] model downloaded\n");

    // Load config, weights, and tokenizer
    let config = ToyLlamaConfig::from_file(&files.config)?;
    let pipeline = ToyLlama::load(&config, &files.weights)?;
    let tokenizer = SimpleTokenizer::from_file(&files.tokenizer)?;

    let total_layers = config.num_hidden_layers;
    let split = total_layers.div_ceil(2);

    let (master_layers, worker_layers) = split_layers(total_layers as u32)?;
    debug!(
        ?master_layers,
        ?worker_layers,
        "computed localhost shard plan"
    );

    // Tokenize the prompt
    let token_ids = tokenizer.encode(&prompt, true)?;
    println!(
        "[tokens] prompt ids: {token_ids:?} ({} tokens)",
        token_ids.len()
    );

    // --- Sharded single-pass demo ---
    println!("\n[run] sharded single-pass inference...");
    let input_tensor = candle_core::Tensor::new(token_ids.as_slice(), &candle_core::Device::Cpu)?
        .reshape((1, token_ids.len()))?;

    println!("  Step 1: Embedding (master)");
    let hidden = pipeline.embed(&input_tensor)?;
    info!(shape = ?hidden.dims(), "embedding done");

    println!("  Step 2: Layers 0..{split} (master)");
    let after_master = pipeline.forward_layers(&hidden, 0..split)?;

    println!("  Step 3: Serialize → [NETWORK] → Deserialize");
    let payload = tensor_to_payload(&after_master)?;
    let deserialized = synapse_core::payload_to_tensor(&payload)?;

    println!("  Step 4: Layers {split}..{total_layers} (worker)");
    let after_worker = pipeline.forward_layers(&deserialized, split..total_layers)?;

    println!("  Step 5: LM Head + Sampling (master)");
    let logits = pipeline.head(&after_worker)?;
    let sampled_token = ToyLlama::sample_argmax(&logits)?;
    let sampled_text = tokenizer.decode(&[sampled_token], true)?;

    println!();
    print_panel(
        "Sharded Single-Pass Result",
        &[
            format!("logits shape: {:?}", logits.dims()),
            format!("sampled token: {sampled_token} -> \"{sampled_text}\""),
        ],
    );

    // --- Autoregressive generation ---
    println!("\n[run] autoregressive generation ({max_tokens} tokens max)...");
    let eos = tokenizer.eos_token_id();
    let generated_ids = pipeline.generate(&token_ids, max_tokens, eos)?;

    let generated_text = tokenizer.decode(&generated_ids, true)?;
    let new_tokens = generated_ids.len() - token_ids.len();

    println!();
    print_panel(
        "Generation Complete",
        &[
            format!("new tokens: {new_tokens}"),
            "full output:".to_owned(),
            generated_text,
        ],
    );

    Ok(())
}

fn infer_local(
    directory: PathBuf,
    prompt: String,
    max_tokens: usize,
    event_path: Option<PathBuf>,
) -> Result<()> {
    use synapse_core::backend::{BackendLimits, Cancellation, CandleBackend, PromptFormat};
    let files = synapse_core::inference::model_loader::ModelFiles::from_directory(&directory)?;
    let mut backend = CandleBackend::load(
        files.manifest.model_id.clone(),
        &files,
        BackendLimits::default(),
        PromptFormat::Raw,
    )?;
    let mut log = EventLog::new(event_path.as_ref())?;
    log.record("request_started", serde_json::json!({"model_digest":files.manifest.digest()?,"max_tokens":max_tokens,"topology":"local"}))?;
    let usage = backend.generate(&prompt, max_tokens, &Cancellation::default(), |token| {
        log.record(
            "token",
            serde_json::json!({"token_id":token.token_id,"position":token.position}),
        )?;
        print!("{}", token.text);
        io::stdout().flush()?;
        Ok(true)
    })?;
    log.record(
        "generation_done",
        serde_json::json!({"tokens_generated":usage.output_tokens,"reason":usage.finish_reason}),
    )?;
    println!();
    Ok(())
}

fn init_tracing(verbose: u8) {
    let default_filter = match verbose {
        0 => "warn",
        1 => "info",
        2 => "debug",
        _ => "trace",
    };

    let filter =
        EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new(default_filter));
    tracing_subscriber::fmt().with_env_filter(filter).init();
}
