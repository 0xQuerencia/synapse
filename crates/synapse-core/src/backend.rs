//! Bounded full-model inference for a single authenticated host.
//!
//! The caller owns admission and transport. A mutable backend admits one request
//! at a time; KV state is cleared on every terminal path. Cancellation is checked
//! between forward passes, rather than pretending a CPU kernel is preemptible.

use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};
use candle_core::Tensor;
use serde::{Deserialize, Serialize};

use crate::inference::model_loader::ModelFiles;
use crate::inference::{ShardedModel, ToyLlama, ToyLlamaConfig, ToyLlamaGGUF};

#[derive(Clone, Default)]
pub struct Cancellation(Arc<AtomicBool>);

impl Cancellation {
    pub fn cancel(&self) {
        self.0.store(true, Ordering::Release);
    }
    pub fn is_cancelled(&self) -> bool {
        self.0.load(Ordering::Acquire)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PromptFormat {
    Raw,
    ChatMl,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

impl PromptFormat {
    pub fn render(self, messages: &[ChatMessage]) -> Result<String> {
        if messages.is_empty() || messages.len() > 128 {
            bail!("expected 1..128 messages");
        }
        if messages
            .iter()
            .any(|m| !matches!(m.role.as_str(), "system" | "user" | "assistant"))
        {
            bail!("only system, user, and assistant messages are supported");
        }
        if messages.last().is_none_or(|m| m.role != "user") {
            bail!("last message must be from the user");
        }
        let size = messages
            .iter()
            .try_fold(0usize, |n, m| n.checked_add(m.content.len()))
            .context("message length overflow")?;
        if size == 0 || size > 256 * 1024 {
            bail!("message content must contain 1..262144 bytes");
        }
        match self {
            Self::Raw => {
                if messages.len() != 1 || messages[0].role != "user" {
                    bail!(
                        "raw models accept exactly one user message; configure an approved chat template for conversations"
                    );
                }
                Ok(messages[0].content.clone())
            }
            Self::ChatMl => {
                let mut prompt = String::new();
                for message in messages {
                    prompt.push_str(&format!(
                        "<|im_start|>{}\n{}<|im_end|>\n",
                        message.role, message.content
                    ));
                }
                prompt.push_str("<|im_start|>assistant\n");
                Ok(prompt)
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct BackendLimits {
    pub max_context_tokens: usize,
    pub max_output_tokens: usize,
    pub max_duration: Duration,
    pub max_weight_bytes: u64,
}

impl Default for BackendLimits {
    fn default() -> Self {
        Self {
            max_context_tokens: 4096,
            max_output_tokens: 512,
            max_duration: Duration::from_secs(120),
            max_weight_bytes: 8 * 1024 * 1024 * 1024,
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum FinishReason {
    Eos,
    Length,
    Cancelled,
    Deadline,
}

#[derive(Debug, Clone, Serialize)]
pub struct GeneratedToken {
    pub position: usize,
    pub token_id: u32,
    pub text: String,
    pub elapsed_us: u64,
}

#[derive(Debug, Clone, Serialize)]
pub struct GenerationUsage {
    pub input_tokens: usize,
    pub output_tokens: usize,
    pub elapsed_us: u64,
    pub finish_reason: FinishReason,
}

pub struct CandleBackend {
    model: Box<dyn ShardedModel>,
    tokenizer: tokenizers::Tokenizer,
    eos: Vec<u32>,
    limits: BackendLimits,
    model_id: String,
    prompt_format: PromptFormat,
}

impl CandleBackend {
    pub fn load(
        model_id: String,
        files: &ModelFiles,
        mut limits: BackendLimits,
        prompt_format: PromptFormat,
    ) -> Result<Self> {
        if model_id.is_empty() || model_id.len() > 256 {
            bail!("invalid model id");
        }
        if limits.max_context_tokens == 0
            || limits.max_output_tokens == 0
            || limits.max_duration.is_zero()
        {
            bail!("context, output, and duration limits must be positive");
        }
        let config = ToyLlamaConfig::from_file(&files.config)?;
        limits.max_context_tokens = limits
            .max_context_tokens
            .min(config.max_position_embeddings);
        let bytes = crate::inference::shard_files(&files.weights)?
            .iter()
            .try_fold(0u64, |total, file| {
                total
                    .checked_add(std::fs::metadata(file)?.len())
                    .context("weight size overflow")
            })?;
        if bytes > limits.max_weight_bytes {
            bail!("weights exceed the configured host limit");
        }
        let model: Box<dyn ShardedModel> = if files.weights.extension().is_some_and(|e| e == "gguf")
        {
            crate::inference::pipeline::validate_gguf_config(&files.weights, &config)?;
            Box::new(ToyLlamaGGUF::load(&files.weights)?)
        } else {
            Box::new(ToyLlama::load(&config, &files.weights)?)
        };
        let tokenizer = tokenizers::Tokenizer::from_file(&files.tokenizer)
            .map_err(|e| anyhow::anyhow!("tokenizer load failed: {e}"))?;
        let config_json: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&files.config)?)?;
        let mut eos = Vec::new();
        if let Some(value) = config_json.get("eos_token_id") {
            let values = value
                .as_array()
                .cloned()
                .unwrap_or_else(|| vec![value.clone()]);
            for value in values {
                let id = value
                    .as_u64()
                    .filter(|id| *id < config.vocab_size as u64)
                    .context("invalid eos_token_id")?;
                eos.push(id as u32);
            }
        }
        if eos.is_empty() {
            let names = match prompt_format {
                PromptFormat::ChatMl => vec!["<|im_end|>"],
                PromptFormat::Raw => vec!["</s>", "<|endoftext|>"],
            };
            eos.extend(
                names
                    .into_iter()
                    .filter_map(|name| tokenizer.token_to_id(name)),
            );
        }
        Ok(Self {
            model,
            tokenizer,
            eos,
            limits,
            model_id,
            prompt_format,
        })
    }

    pub fn model_id(&self) -> &str {
        &self.model_id
    }
    pub fn backend_id(&self) -> &'static str {
        self.model.name()
    }
    pub fn limits(&self) -> &BackendLimits {
        &self.limits
    }
    pub fn render_chat(&self, messages: &[ChatMessage]) -> Result<String> {
        self.prompt_format.render(messages)
    }

    pub fn validate_prompt(&self, prompt: &str, max_tokens: usize) -> Result<Vec<u32>> {
        if prompt.is_empty() || prompt.len() > 256 * 1024 {
            bail!("prompt must contain 1..262144 bytes");
        }
        if max_tokens == 0 || max_tokens > self.limits.max_output_tokens {
            bail!("max_tokens exceeds the host output limit");
        }
        let ids = self
            .tokenizer
            .encode(prompt, true)
            .map_err(|e| anyhow::anyhow!("tokenization failed: {e}"))?
            .get_ids()
            .to_vec();
        if ids.is_empty() {
            bail!("prompt produced no tokens");
        }
        if ids
            .len()
            .checked_add(max_tokens)
            .is_none_or(|n| n > self.limits.max_context_tokens)
        {
            bail!("prompt and output exceed the model context limit");
        }
        Ok(ids)
    }

    /// The callback returns false when the transport no longer accepts output.
    /// Such a token is not included in accepted output usage.
    pub fn generate(
        &mut self,
        prompt: &str,
        max_tokens: usize,
        cancellation: &Cancellation,
        mut emit: impl FnMut(GeneratedToken) -> Result<bool>,
    ) -> Result<GenerationUsage> {
        let ids = self.validate_prompt(prompt, max_tokens)?;
        self.model.clear_kv_cache()?;
        let result = self.generate_inner(&ids, max_tokens, cancellation, &mut emit);
        let cleanup = self.model.clear_kv_cache();
        match result {
            Ok(usage) => {
                cleanup?;
                Ok(usage)
            }
            Err(error) => {
                let _ = cleanup;
                Err(error)
            }
        }
    }

    fn generate_inner(
        &self,
        ids: &[u32],
        max_tokens: usize,
        cancellation: &Cancellation,
        emit: &mut impl FnMut(GeneratedToken) -> Result<bool>,
    ) -> Result<GenerationUsage> {
        let started = Instant::now();
        let mut usage = GenerationUsage {
            input_tokens: ids.len(),
            output_tokens: 0,
            elapsed_us: 0,
            finish_reason: FinishReason::Length,
        };
        if cancellation.is_cancelled() {
            usage.finish_reason = FinishReason::Cancelled;
            return Ok(usage);
        }
        let input = Tensor::new(ids, self.model.model_device())?.reshape((1, ids.len()))?;
        let mut logits = self.model.forward_prefill(&input)?;
        let mut decoder = self.tokenizer.decode_stream(true);
        for position in 0..max_tokens {
            if cancellation.is_cancelled() {
                usage.finish_reason = FinishReason::Cancelled;
                break;
            }
            if started.elapsed() >= self.limits.max_duration {
                usage.finish_reason = FinishReason::Deadline;
                break;
            }
            let token_id = ToyLlama::sample_argmax(&logits)?;
            let text = decoder
                .step(token_id)
                .map_err(|e| anyhow::anyhow!("stream decoding failed: {e}"))?
                .unwrap_or_default();
            if !emit(GeneratedToken {
                position,
                token_id,
                text,
                elapsed_us: started.elapsed().as_micros().min(u64::MAX as u128) as u64,
            })? {
                usage.finish_reason = FinishReason::Cancelled;
                break;
            }
            usage.output_tokens += 1;
            if self.eos.contains(&token_id) {
                usage.finish_reason = FinishReason::Eos;
                break;
            }
            if position + 1 < max_tokens {
                let next = Tensor::new(&[[token_id]], self.model.model_device())?;
                logits = self.model.forward_decode(&next)?;
            }
        }
        usage.elapsed_us = started.elapsed().as_micros().min(u64::MAX as u128) as u64;
        Ok(usage)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> CandleBackend {
        let path =
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tiny-llama");
        let files = ModelFiles::from_directory(&path).unwrap();
        CandleBackend::load(
            "tiny".into(),
            &files,
            BackendLimits {
                max_output_tokens: 32,
                ..BackendLimits::default()
            },
            PromptFormat::Raw,
        )
        .unwrap()
    }

    #[test]
    fn full_host_stream_matches_independent_reference() {
        let mut backend = fixture();
        let mut ids = Vec::new();
        let mut text = String::new();
        let usage = backend
            .generate("<s> </s> Hello", 8, &Cancellation::default(), |token| {
                assert_eq!(token.position, ids.len());
                ids.push(token.token_id);
                text.push_str(&token.text);
                Ok(true)
            })
            .unwrap();
        assert_eq!(ids, [45, 20, 61, 20, 46, 8, 60, 34]);
        assert_eq!(usage.input_tokens, 3);
        assert_eq!(usage.output_tokens, 8);
        assert_eq!(usage.finish_reason, FinishReason::Length);
        assert_eq!(text, backend.tokenizer.decode(&ids, true).unwrap());
    }

    #[test]
    fn cancelled_or_failed_request_cannot_contaminate_next_session() {
        let mut backend = fixture();
        let cancel = Cancellation::default();
        let usage = backend
            .generate("<s> </s> Hello", 8, &cancel, |_| {
                cancel.cancel();
                Ok(true)
            })
            .unwrap();
        assert_eq!(usage.output_tokens, 1);
        assert_eq!(usage.finish_reason, FinishReason::Cancelled);
        assert!(
            backend
                .generate("Hello world", 8, &Cancellation::default(), |_| bail!(
                    "disconnected"
                ))
                .is_err()
        );
        let mut ids = Vec::new();
        backend
            .generate("<s> </s> Hello", 8, &Cancellation::default(), |token| {
                ids.push(token.token_id);
                Ok(true)
            })
            .unwrap();
        assert_eq!(ids, [45, 20, 61, 20, 46, 8, 60, 34]);
    }

    #[test]
    fn rejects_impossible_limits_and_counts_only_accepted_output() {
        let mut backend = fixture();
        assert!(backend.validate_prompt("", 1).is_err());
        assert!(backend.validate_prompt("Hello", 0).is_err());
        assert!(backend.validate_prompt("Hello", 33).is_err());
        assert!(backend.validate_prompt(&"Hello ".repeat(128), 1).is_err());
        let usage = backend
            .generate("Hello", 4, &Cancellation::default(), |_| Ok(false))
            .unwrap();
        assert_eq!(usage.output_tokens, 0);
        assert_eq!(usage.finish_reason, FinishReason::Cancelled);
        backend.limits.max_duration = Duration::from_nanos(1);
        let usage = backend
            .generate("Hello", 4, &Cancellation::default(), |_| Ok(true))
            .unwrap();
        assert_eq!(usage.finish_reason, FinishReason::Deadline);
        assert_eq!(usage.output_tokens, 0);
    }
    #[test]
    fn templates_reject_unsupported_conversations() {
        let messages = vec![ChatMessage {
            role: "user".into(),
            content: "Hello".into(),
        }];
        assert_eq!(PromptFormat::Raw.render(&messages).unwrap(), "Hello");
        assert_eq!(
            PromptFormat::ChatMl.render(&messages).unwrap(),
            "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n"
        );
        assert!(PromptFormat::Raw.render(&[]).is_err());
        assert!(
            PromptFormat::Raw
                .render(&[ChatMessage {
                    role: "tool".into(),
                    content: "x".into()
                }])
                .is_err()
        );
        assert!(
            PromptFormat::Raw
                .render(&[messages[0].clone(), messages[0].clone()])
                .is_err()
        );
    }
    #[test]
    fn cancellation_is_shared() {
        let cancel = Cancellation::default();
        let clone = cancel.clone();
        clone.cancel();
        assert!(cancel.is_cancelled());
    }
}
