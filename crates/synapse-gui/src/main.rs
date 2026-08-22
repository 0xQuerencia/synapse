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
    #[arg(long, default_value_t = true)]
    open: bool,

    /// Shared access token. When set, every `/api/*`, `/ws`, and page
    /// request must present it (via `?token=`, an `x-synapse-token`
    /// header, `Authorization: Bearer`, or the `synapse_token` cookie).
    /// REQUIRED before exposing the server publicly (e.g. via a tunnel) —
    /// the API includes a `PUT /api/config/:file` that writes to disk.
    /// When unset, no auth is enforced (safe only for `127.0.0.1`).
    #[arg(long, env = "SYNAPSE_TOKEN")]
    token: Option<String>,

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

    let state = api::AppState {
        sessions: session::SessionStore::new(),
        token: cli.token.clone().map(std::sync::Arc::new),
    };

    let app = api::router()
        .route("/ws", axum::routing::get(ws::ws_handler))
        .route("/", axum::routing::get(assets::serve_root))
        .route("/contribute", axum::routing::get(assets::serve_contribute))
        .fallback(assets::spa_fallback)
        // Auth gate runs first (outermost). With a token set, every route —
        // pages, assets, /api, /ws — requires it. No token = open (localhost).
        .layer(axum::middleware::from_fn_with_state(
            state.clone(),
            api::auth_guard,
        ))
        .layer(TraceLayer::new_for_http())
        .with_state(state);

    let listener = tokio::net::TcpListener::bind(&bind_addr)
        .await
        .with_context(|| format!("failed to bind {bind_addr}"))?;
    let local_addr = listener
        .local_addr()
        .context("listener has no local addr")?;

    eprintln!("╭───────────────────────────────────────────────╮");
    eprintln!("│ Synapse GUI                                   │");
    eprintln!("├───────────────────────────────────────────────┤");
    eprintln!("│ Listening on:  http://{:<27} │", local_addr);
    if let Some(tok) = &cli.token {
        eprintln!("│ Auth:          ENABLED (token required)       │");
        eprintln!("│ Share URL:     http://{local_addr}/?token={tok}");
    } else {
        eprintln!("│ Auth:          OFF (localhost only — do NOT   │");
        eprintln!("│                expose without --token)        │");
    }
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
