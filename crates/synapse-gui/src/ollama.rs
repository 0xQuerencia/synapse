//! Thin client over ollama's `POST /api/generate` (NDJSON streaming).
//!
//! Used by the GUI's master role when the requested model id is
//! prefixed with `ollama:` (e.g. `ollama:gemma4:e2b`). The existing
//! candle-based sharded-inference path stays untouched for the
//! multi-node demo; ollama is the fast single-node backend.
//!
//! The client speaks the standard ollama wire format and emits
//! `synapse_core::ChatEvent::Token` events for every `response`
//! chunk, then `ChatEvent::GenerationDone` on the `done: true`
//! frame. The same event format the WebSocket bridge already
//! forwards, so the HTML GUI needs zero changes.

use std::time::Duration;

use anyhow::{Context, Result, anyhow};
use futures::StreamExt;
use reqwest::Client;
use serde::Deserialize;
use tokio::sync::{broadcast, mpsc};
use tracing::{debug, info, warn};

use synapse_core::network::{ChatEvent, StopReason};

/// Thin ollama client. Holds a single `reqwest::Client` and a base
/// URL. Cheap to clone (both fields are `Arc`-y internally).
#[derive(Clone)]
pub struct OllamaClient {
    base_url: String,
    http: Client,
}

impl OllamaClient {
    pub fn new(base_url: impl Into<String>) -> Result<Self> {
        let http = Client::builder()
            .timeout(Duration::from_secs(60 * 60)) // generation can be slow
            .build()
            .context("build reqwest client")?;
        Ok(Self {
            base_url: base_url.into().trim_end_matches('/').to_owned(),
            http,
        })
    }

    /// Return the list of locally-available model names (the bit
    /// after the last `:` in `name`, since ollama returns names like
    /// `gemma4:e2b` already).
    pub async fn list_models(&self) -> Result<Vec<String>> {
