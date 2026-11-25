//! Compare an explicitly selected pinned local model with Candle's separate
//! upstream Llama implementation. No weights or prompt data are uploaded.
//! Usage: verify_model_reference DIRECTORY [PROMPT] [TOKENS]
use anyhow::{Result, ensure};
use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::llama::{Cache, Llama, LlamaConfig};
use synapse_core::inference::{SimpleTokenizer, ToyLlama, ToyLlamaConfig, shard_files};
use synapse_core::inference::{model_loader::ModelFiles, prompt_format::format_prompt_for_model};
#[tokio::main]
async fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    ensure!(
        !args.is_empty(),
        "usage: verify_model_reference DIRECTORY [PROMPT] [TOKENS]"
    );
    let files = ModelFiles::from_directory(std::path::Path::new(&args[0]))?;
    let config = ToyLlamaConfig::from_file(&files.config)?;
    let count = args
        .get(2)
        .map(|s| s.parse())
        .transpose()?
        .unwrap_or(16usize);
    ensure!((1..=64).contains(&count), "verification token limit 1..=64");
    let prompt = args
        .get(1)
        .map(String::as_str)
        .unwrap_or("What is the capital of France? Answer in one short sentence.");
    let prompt = format_prompt_for_model(&files.manifest.model_id, prompt, false)?;
    let tokenizer = SimpleTokenizer::from_file(&files.tokenizer)?;
    let ids = tokenizer.encode(&prompt, false)?;
    ensure!(
        !ids.is_empty() && ids.len() + count <= config.context_limit(),
        "invalid verification context"
    );
    let device = Device::Cpu;
    let dtype = synapse_core::inference::pipeline::execution_dtype(&config, &device);
    let ours = ToyLlama::load_range(&config, &files.weights, 0..config.num_hidden_layers)?;
    let reference_config: LlamaConfig = serde_json::from_slice(&std::fs::read(&files.config)?)?;
    let reference_config = reference_config.into_config(false);
    let shards = shard_files(&files.weights)?;
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&shards, dtype, &device)? };
    let reference = Llama::load(vb, &reference_config)?;
    let mut cache = Cache::new(true, dtype, &reference_config, &device)?;
    let input = Tensor::new(ids.as_slice(), &device)?.reshape((1, ids.len()))?;
    let mut actual = ours.forward_prefill(&input)?;
    let mut expected = reference.forward(&input, 0, &mut cache)?;
    let mut generated = Vec::new();
    let mut reference_greedy = Vec::new();
    let mut max_error = 0.0f32;
    let mut matches = 0usize;
    let started = std::time::Instant::now();
    for step in 0..count {
        let length = actual.dim(1)?;
        let actual_last = actual.i((.., length - 1, ..))?.to_dtype(DType::F32)?;
        let a = actual_last.flatten_all()?.to_vec1::<f32>()?;
        let b = expected.flatten_all()?.to_vec1::<f32>()?;
        ensure!(a.len() == b.len(), "reference vocabulary mismatch");
        for (a, b) in a.into_iter().zip(b) {
            ensure!(
                a.is_finite() && b.is_finite(),
                "non-finite reference logits"
            );
            max_error = max_error.max((a - b).abs());
        }
        let token = ToyLlama::sample_argmax(&actual)?;
        let expected_token = expected
            .argmax(candle_core::D::Minus1)?
            .squeeze(0)?
            .to_scalar::<u32>()?;
        if token == expected_token {
            matches += 1;
        }
        generated.push(token);
        reference_greedy.push(expected_token);
        if tokenizer.is_eos(token) || step + 1 == count {
            break;
        }
        // Teacher-force identical accepted tokens into both implementations so
        // numerical error remains comparable even if a marginal argmax differs.
        let input = Tensor::new(&[[token]], &device)?;
        actual = ours.forward_decode(&input)?;
        expected = reference.forward(&input, ids.len() + step, &mut cache)?;
    }
    let comparison_decode_ms = started.elapsed().as_secs_f64() * 1000.0;
    let memory =
        synapse_core::inference::memory::ModelMemoryProfile::from_verified_files(&config, &files)?;
    let full_memory = memory.estimate(
        config.num_hidden_layers,
        32,
        synapse_core::inference::pipeline::TensorRoles::COORDINATOR,
    )?;
    let host_half_memory = memory.estimate(
        config.num_hidden_layers / 2,
        32,
        synapse_core::inference::pipeline::TensorRoles::COORDINATOR,
    )?;
    let worker_half_memory = memory.estimate(
        config.num_hidden_layers - config.num_hidden_layers / 2,
        32,
        synapse_core::inference::pipeline::TensorRoles::WORKER,
    )?;
    let report = serde_json::json!({"full_memory_context32":full_memory,"host_half_memory_context32":host_half_memory,"worker_half_memory_context32":worker_half_memory,"stored_dtype":config.torch_dtype,"model_id":files.manifest.model_id,"revision":files.manifest.revision,"manifest_sha256":files.manifest.digest()?,"dtype":format!("{dtype:?}"),"prompt_token_count":ids.len(),"tokens":generated,"reference_greedy_tokens_on_same_prefixes":reference_greedy,"greedy_matches":matches,"max_abs_logit_error":max_error,"text":tokenizer.decode(&generated,true)?,"comparison_decode_ms":comparison_decode_ms,"resident_weight_bytes":ours.resident_weight_bytes(),"reference":"candle-transformers 0.8.4 models::llama","note":"This is a concrete sample and numerical comparison, not a general language-quality benchmark."});
    println!("{}", serde_json::to_string_pretty(&report)?);
    let tolerance = if dtype == DType::F32 { 0.005 } else { 1.0 };
    ensure!(
        max_error <= tolerance,
        "reference logit error exceeds declared dtype tolerance"
    );
    ensure!(
        matches == generated.len(),
        "greedy output diverged from independent implementation"
    );
    Ok(())
}
