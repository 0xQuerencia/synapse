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
use tokio::sync::mpsc;
use tracing::{debug, info};

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
            .timeout(Duration::from_secs(120))
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
        let url = format!("{}/api/tags", self.base_url);
        let resp = self
            .http
            .get(&url)
            .send()
            .await
            .with_context(|| format!("GET {url}"))?;
        if !resp.status().is_success() {
            return Err(anyhow!("ollama /api/tags returned {}", resp.status()));
        }
        let mut chunks = resp.bytes_stream();
        let mut raw = Vec::new();
        while let Some(chunk) = chunks.next().await {
            let chunk = chunk?;
            anyhow::ensure!(
                raw.len().saturating_add(chunk.len()) <= 1024 * 1024,
                "Ollama model listing exceeds 1 MiB"
            );
            raw.extend_from_slice(&chunk);
        }
        let body: TagsResponse = serde_json::from_slice(&raw).context("decode /api/tags")?;
        Ok(body.models.into_iter().map(|m| m.name).collect())
    }

    /// Pump a single prompt through ollama and stream `ChatEvent`s
    /// into the broadcast. The function returns when the stream
    /// ends (terminal `done: true` frame or transport error).
    pub async fn stream_generate(
        &self,
        model: &str,
        prompt: &str,
        max_tokens: u32,
        bcast: &mpsc::Sender<ChatEvent>,
        mut cancel: mpsc::Receiver<()>,
    ) -> Result<()> {
        let url = format!("{}/api/generate", self.base_url);
        let body = serde_json::json!({
            "model": model,
            "prompt": prompt,
            "stream": true,
            // Disable "thinking" mode. Models like gemma4:e2b-it-qat emit
            // hidden chain-of-thought into a separate `thinking` field that we
            // don't surface; left on, the model spends its entire num_predict
            // budget reasoning and the visible `response` comes back empty
            // (done_reason="length", tokens=0). For a chat GUI we want answers,
            // not the scratchpad. Ollama ignores this flag for non-thinking
            // models, so it's safe to always send.
            "think": false,
            "options": { "num_predict": max_tokens },
            "keep_alive": "5m",
        });
        debug!(%url, %model, len = prompt.len(), "ollama generate request");
        let resp = self
            .http
            .post(&url)
            .json(&body)
            .send()
            .await
            .with_context(|| format!("POST {url}"))?;
        if !resp.status().is_success() {
            let status = resp.status();
            return Err(anyhow!("ollama /api/generate returned {status}"));
        }

        let mut stream = resp.bytes_stream();
        let mut buf: Vec<u8> = Vec::new();
        let mut output_bytes = 0usize;

        loop {
            tokio::select! {
                biased;
                _ = cancel.recv() => {
                    info!("ollama stream cancelled by caller");
                    return Ok(());
                }
                chunk = stream.next() => {
                    let Some(chunk) = chunk else { break };
                    let chunk = chunk.context("read ollama chunk")?;
                    anyhow::ensure!(buf.len().saturating_add(chunk.len()) <= 1024 * 1024, "Ollama frame exceeds 1 MiB");
                    buf.extend_from_slice(&chunk);

                    // ollama's stream emits one JSON object per line
                    // (NDJSON). Parse whatever complete lines are in
                    // the buffer.
                    while let Some(pos) = buf.iter().position(|b| *b == b'\n') {
                        let mut line: Vec<u8> = buf.drain(..=pos).collect();
                        // drop the trailing \n
                        let _ = line.pop();
                        if line.is_empty() {
                            continue;
                        }
                        let frame: GenerateChunk = serde_json::from_slice(&line).context("malformed Ollama frame")?;
                        if let Some(t) = frame.response
                            && !t.is_empty() {
                                output_bytes = output_bytes.saturating_add(t.len());
                                anyhow::ensure!(output_bytes <= 4 * 1024 * 1024, "Ollama output exceeds 4 MiB");
                                tokio::time::timeout(Duration::from_secs(1), bcast.send(ChatEvent::Token(t))).await.context("Ollama consumer stalled")?.context("Ollama consumer disconnected")?;
                            }
                        if frame.done {
                            let reason = frame
                                .done_reason
                                .unwrap_or_else(|| "stop".to_owned());
                            let total_tokens = frame.eval_count.context("Ollama omitted authoritative output token count")?;
                            anyhow::ensure!(total_tokens <= max_tokens as usize, "Ollama exceeded output token limit");
                            info!(tokens = total_tokens, %reason, "ollama done");
                            // Map ollama's done_reason to our StopReason.
                            // ollama emits "stop" on a natural end (EOS/stop
                            // token) and "length" when num_predict is hit.
                            let stop = match reason.as_str() {
                                "length" => StopReason::MaxTokens,
                                _ => StopReason::Eos,
                            };
                            tokio::time::timeout(Duration::from_secs(1), bcast.send(ChatEvent::GenerationDone {
                                tokens_generated: total_tokens,
                                reason: stop,
                            })).await.context("Ollama consumer stalled")?.context("Ollama consumer disconnected")?;
                            return Ok(());
                        }
                    }
                }
            }
        }

        Err(anyhow!(
            "Ollama stream ended without a completed generation"
        ))
    }
}

#[derive(Deserialize)]
struct TagsResponse {
    models: Vec<TagModel>,
}

#[derive(Deserialize)]
struct TagModel {
    name: String,
}

#[derive(Deserialize)]
struct GenerateChunk {
    #[serde(default)]
    response: Option<String>,
    #[serde(default)]
    done: bool,
    #[serde(default)]
    done_reason: Option<String>,
    #[serde(default)]
    eval_count: Option<usize>,
}
