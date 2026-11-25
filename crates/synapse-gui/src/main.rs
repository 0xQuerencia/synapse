//! `synapse-gui` — local web GUI for the Synapse distributed inference network.
//!
//! Default: serves a React SPA on `http://127.0.0.1:8080` with a
//! REST + WebSocket backend. The GUI process is the swarm process —
//! no IPC overhead between inference and the browser.
//!
//! All binary distribution is single-file: the React bundle is
//! embedded by `rust-embed` (see `assets.rs`).

use std::net::SocketAddr;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use clap::Parser;
use tower_http::trace::TraceLayer;
use tracing_subscriber::EnvFilter;

mod api;
mod assets;
mod auth;
mod compatible_api;
mod config_io;
mod ollama;
mod session;
mod ws;

#[derive(Debug, Parser)]
#[command(name = "synapse-gui", about = "Local web GUI for the Synapse network")]
struct Cli {
    /// Address to bind the web server on.
    #[arg(long, default_value = "127.0.0.1:8080")]
    addr: String,

    /// Path to the node's identity key (libp2p Ed25519 keypair).
    /// Defaults to `~/.synapse/identity.key`. Override to run multiple
    /// nodes on the same machine with distinct PeerIds.
    #[arg(long, env = "SYNAPSE_IDENTITY_PATH")]
    identity_path: Option<String>,

    /// Open the GUI in the default browser on startup.
    #[arg(long, default_value_t = false)]
    open: bool,

    /// Access token (prefer SYNAPSE_TOKEN or a private --token-file).
    #[arg(long, env = "SYNAPSE_TOKEN")]
    token: Option<String>,

    /// Private access-token file. Created with a random secret if absent.
    #[arg(long)]
    token_file: Option<PathBuf>,

    /// Explicitly approved local directory containing a verified model manifest.
    #[arg(long)]
    model_dir: Option<PathBuf>,

    /// Public model ID used by the compatible API.
    #[arg(long, default_value = "synapse-local")]
    api_model_id: String,

    /// Prompt format for the approved model: raw or chatml.
    #[arg(long, default_value = "raw", value_parser = ["raw", "chatml"])]
    chat_template: String,

    /// Maximum total input and output tokens per request.
    #[arg(long, default_value_t = 4096)]
    max_context: usize,

    #[arg(long, default_value_t = 512)]
    max_output_tokens: usize,

    /// Deadline checked between model forward passes.
    #[arg(long, default_value_t = 120)]
    request_timeout_secs: u64,

    /// Refuse model weights larger than this budget before loading.
    #[arg(long, default_value_t = 8192)]
    max_weight_mib: u64,

    /// Verbosity (repeat for more detail).
    #[arg(short, long, action = clap::ArgAction::Count)]
    verbose: u8,
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    init_tracing(cli.verbose);

    // Route the identity-path override through synapse-core's
    // thread-local setter so we never need `unsafe { set_var }`
    // (which is unsafe in Rust 2024) and the GUI crate stays
    // `forbid(unsafe_code)`.
    if let Some(p) = &cli.identity_path {
        synapse_core::set_identity_path_override(Some(Path::new(p)));
    }

    let bind_addr: SocketAddr = cli
        .addr
        .parse()
        .with_context(|| format!("invalid --addr `{}`", cli.addr))?;

    // Make sure the working directory contains the config/ folder the
    // GUI edits. If invoked from a different cwd, fall back to the
    // nearest ancestor that has one — same behaviour the CLI already
    // relies on.
    if let Err(e) = ensure_config_dir() {
        tracing::warn!(error = %e, "could not locate config/ directory");
    }

    if !bind_addr.ip().is_loopback() {
        anyhow::bail!(
            "the local admin/API server requires a loopback address; use an authenticated encrypted tunnel for remote private access"
        );
    }
    let listener = tokio::net::TcpListener::bind(&bind_addr)
        .await
        .with_context(|| format!("failed to bind {bind_addr}"))?;
    let local_addr = listener
        .local_addr()
        .context("listener has no local addr")?;
    let token_path = cli.token_file.clone().unwrap_or_else(|| {
        std::env::var_os("HOME")
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from("."))
            .join(".synapse/gui-token")
    });
    let token = match &cli.token {
        Some(token) => token.clone(),
        None => auth::token_file(&token_path)?,
    };
    let local_auth = auth::LocalAuth::new(token, local_addr)?;
    let inference = if let Some(directory) = &cli.model_dir {
        use synapse_core::backend::{BackendLimits, CandleBackend, PromptFormat};
        let files = synapse_core::inference::model_loader::ModelFiles::from_directory(directory)?;
        let limits = BackendLimits {
            max_context_tokens: cli.max_context,
            max_output_tokens: cli.max_output_tokens,
            max_duration: std::time::Duration::from_secs(cli.request_timeout_secs),
            max_weight_bytes: cli
                .max_weight_mib
                .checked_mul(1024 * 1024)
                .context("weight limit overflow")?,
        };
        let format = if cli.chat_template == "chatml" {
            PromptFormat::ChatMl
        } else {
            PromptFormat::Raw
        };
        let backend = CandleBackend::load(cli.api_model_id.clone(), &files, limits, format)?;
        Some(compatible_api::InferenceService::new(backend))
    } else {
        None
    };
    let state = api::AppState {
        sessions: session::SessionStore::new(),
        auth: std::sync::Arc::new(local_auth),
        inference,
    };
    let app = api::router()
        .merge(compatible_api::router())
        .route("/api/auth", axum::routing::post(auth::login))
        .route("/api/logout", axum::routing::post(auth::logout))
        .route("/login", axum::routing::get(auth::page))
        .route("/ws", axum::routing::get(ws::ws_handler))
        .route("/", axum::routing::get(assets::serve_root))
        .route("/contribute", axum::routing::get(assets::serve_contribute))
        .fallback(assets::spa_fallback)
        .layer(axum::extract::DefaultBodyLimit::max(512 * 1024))
        .layer(axum::middleware::from_fn_with_state(state.clone(), auth::guard))
        .layer(TraceLayer::new_for_http().make_span_with(|request: &axum::http::Request<_>| {
            tracing::info_span!("http", method = %request.method(), path = request.uri().path())
        }))
        .with_state(state);

    eprintln!("╭───────────────────────────────────────────────╮");
    eprintln!("│ Synapse GUI                                   │");
    eprintln!("├───────────────────────────────────────────────┤");
    eprintln!("│ Listening on:  http://{:<27} │", local_addr);
    eprintln!("│ Auth:          ENABLED                       │");
    if cli.token.is_none() {
        eprintln!("Private token file: {}", token_path.display());
    }
    eprintln!("Sign in at http://{local_addr}/login");
    eprintln!("│ Press Ctrl+C to stop.                         │");
    eprintln!("╰───────────────────────────────────────────────╯");

    if cli.open {
        let url = format!("http://{local_addr}");
        // Best-effort: if we can't open, just print the URL.
        if let Err(e) = open_in_browser(&url) {
            tracing::debug!(error = %e, "could not auto-open browser");
        }
    }

    axum::serve(listener, app)
        .with_graceful_shutdown(async {
            let _ = tokio::signal::ctrl_c().await;
        })
        .await
        .context("axum server failed")?;
    Ok(())
}

fn ensure_config_dir() -> Result<()> {
    // Look up the tree for a config/ dir; if missing, create it next to CWD.
    let cwd = std::env::current_dir().context("no cwd")?;
    let mut here: PathBuf = cwd.clone();
    loop {
        if here.join("config").is_dir() {
            std::env::set_current_dir(&here).ok();
            return Ok(());
        }
        if !here.pop() {
            // No config/ in any ancestor — create one in CWD.
            std::fs::create_dir_all(cwd.join("config"))?;
            return Ok(());
        }
    }
}

fn open_in_browser(url: &str) -> Result<()> {
    #[cfg(target_os = "linux")]
    {
        std::process::Command::new("xdg-open")
            .arg(url)
            .spawn()
            .context("xdg-open not available")?;
    }
    #[cfg(target_os = "macos")]
    {
        std::process::Command::new("open")
            .arg(url)
            .spawn()
            .context("`open` failed")?;
    }
    #[cfg(target_os = "windows")]
    {
        std::process::Command::new("cmd")
            .args(["/C", "start", "", url])
            .spawn()
            .context("`start` failed")?;
    }
    Ok(())
}

fn init_tracing(verbose: u8) {
    let default_filter = match verbose {
        0 => "warn,synapse_gui=info",
        1 => "info,synapse_gui=debug",
        _ => "debug,synapse_gui=trace",
    };
    let filter =
        EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new(default_filter));
    let _ = tracing_subscriber::fmt().with_env_filter(filter).try_init();
}
