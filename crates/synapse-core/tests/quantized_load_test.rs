//! Original deterministic weights exercise the real F32/Q8_0/Q4_0 GGUF
//! loaders, GQA, slice-only residency and multi-step KV decoding. These error
//! bounds test numerical machinery; they make no language-quality claim.
use candle_core::quantized::{GgmlDType, QTensor, gguf_file};
use candle_core::{Device, Tensor};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, HashMap},
    io::Write,
    path::{Path, PathBuf},
};
use synapse_core::inference::pipeline::{TensorRoles, validate_gguf_config};
use synapse_core::inference::{ToyLlama, ToyLlamaConfig, ToyLlamaGGUF};
const VOCAB: usize = 64;
const HIDDEN: usize = 64;
const HEADS: usize = 4;
const KV_HEADS: usize = 2;
const LAYERS: usize = 2;
const FFN: usize = 128;

fn config() -> ToyLlamaConfig {
    ToyLlamaConfig {
        hidden_size: HIDDEN,
        intermediate_size: FFN,
        num_hidden_layers: LAYERS,
        vocab_size: VOCAB,
        rms_norm_eps: 1e-6,
        num_attention_heads: HEADS,
        num_key_value_heads: KV_HEADS,
        tie_word_embeddings: false,
        max_position_embeddings: 64,
        rope_theta: 5000.0,
        torch_dtype: "float32".into(),
    }
}
fn metadata() -> BTreeMap<String, gguf_file::Value> {
    use gguf_file::Value as V;
    BTreeMap::from([
        ("general.architecture".into(), V::String("llama".into())),
        ("llama.embedding_length".into(), V::U32(HIDDEN as u32)),
        ("llama.block_count".into(), V::U32(LAYERS as u32)),
        ("llama.attention.head_count".into(), V::U32(HEADS as u32)),
        (
            "llama.attention.head_count_kv".into(),
            V::U32(KV_HEADS as u32),
        ),
        (
            "llama.attention.layer_norm_rms_epsilon".into(),
            V::F32(1e-6),
        ),
        ("llama.feed_forward_length".into(), V::U32(FFN as u32)),
        ("llama.context_length".into(), V::U32(64)),
        ("llama.rope.freq_base".into(), V::F32(5000.0)),
        ("llama.vocab_size".into(), V::U32(VOCAB as u32)),
    ])
}
fn tensor(name: &str, shape: &[usize]) -> Tensor {
    let hash = Sha256::digest(name.as_bytes());
    let mut state = u64::from_le_bytes(hash[..8].try_into().unwrap());
    let is_norm = name.contains("norm");
    let values: Vec<f32> = (0..shape.iter().product())
        .map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let value = (((state >> 32) % 2001) as i64 - 1000) as f32 / 1000.0;
            if is_norm {
                1.0 + value * 0.03
            } else {
                value * 0.07
            }
        })
        .collect();
    Tensor::from_vec(values, shape, &Device::Cpu).unwrap()
}
fn tensors(tied: bool) -> Vec<(String, Tensor)> {
    let mut out = vec![(
        "token_embd.weight".into(),
        tensor("token_embd.weight", &[VOCAB, HIDDEN]),
    )];
    for i in 0..LAYERS {
        let prefix = format!("blk.{i}");
        let kv = KV_HEADS * HIDDEN / HEADS;
        for (suffix, shape) in [
            ("attn_norm.weight", vec![HIDDEN]),
            ("attn_q.weight", vec![HIDDEN, HIDDEN]),
            ("attn_k.weight", vec![kv, HIDDEN]),
            ("attn_v.weight", vec![kv, HIDDEN]),
            ("attn_output.weight", vec![HIDDEN, HIDDEN]),
            ("ffn_norm.weight", vec![HIDDEN]),
            ("ffn_gate.weight", vec![FFN, HIDDEN]),
            ("ffn_up.weight", vec![FFN, HIDDEN]),
            ("ffn_down.weight", vec![HIDDEN, FFN]),
        ] {
            let name = format!("{prefix}.{suffix}");
            out.push((name.clone(), tensor(&name, &shape)));
        }
    }
    out.push((
        "output_norm.weight".into(),
        tensor("output_norm.weight", &[HIDDEN]),
    ));
    if !tied {
        out.push((
            "output.weight".into(),
            tensor("output.weight", &[VOCAB, HIDDEN]),
        ));
    }
    out
}
fn write_gguf(
    path: &Path,
    dtype: GgmlDType,
    metadata: &BTreeMap<String, gguf_file::Value>,
    weights: &[(String, Tensor)],
) {
    let q: Vec<_> = weights
        .iter()
        .map(|(name, value)| {
            (
                name,
                QTensor::quantize(
                    value,
                    if name.contains("norm") {
                        GgmlDType::F32
                    } else {
                        dtype
                    },
                )
                .unwrap(),
            )
        })
        .collect();
    let refs: Vec<_> = q
        .iter()
        .map(|(name, value)| (name.as_str(), value))
        .collect();
    let meta: Vec<_> = metadata.iter().map(|(k, v)| (k.as_str(), v)).collect();
    let mut file = std::io::BufWriter::new(std::fs::File::create(path).unwrap());
    gguf_file::write(&mut file, &meta, &refs).unwrap();
    file.flush().unwrap();
}
fn hf_name(name: &str) -> String {
    match name {
        "token_embd.weight" => "model.embed_tokens.weight".into(),
        "output_norm.weight" => "model.norm.weight".into(),
        "output.weight" => "lm_head.weight".into(),
        _ => {
            let parts: Vec<_> = name.split('.').collect();
            let suffix = match parts[2] {
                "attn_norm" => "input_layernorm",
                "ffn_norm" => "post_attention_layernorm",
                "attn_q" => "self_attn.q_proj",
                "attn_k" => "self_attn.k_proj",
                "attn_v" => "self_attn.v_proj",
                "attn_output" => "self_attn.o_proj",
                "ffn_gate" => "mlp.gate_proj",
                "ffn_up" => "mlp.up_proj",
                "ffn_down" => "mlp.down_proj",
                _ => panic!("unknown fixture tensor"),
            };
            format!("model.layers.{}.{}.weight", parts[1], suffix)
        }
    }
}
struct Fixture {
    _dir: tempfile::TempDir,
    f32: PathBuf,
    q8: PathBuf,
    q4: PathBuf,
    safetensors: PathBuf,
}
fn fixture() -> Fixture {
    let dir = tempfile::tempdir().unwrap();
    let f32 = dir.path().join("f32.gguf");
    let q8 = dir.path().join("q8.gguf");
    let q4 = dir.path().join("q4.gguf");
    let safetensors = dir.path().join("model.safetensors");
    let weights = tensors(false);
    for (path, dtype) in [
        (&f32, GgmlDType::F32),
        (&q8, GgmlDType::Q8_0),
        (&q4, GgmlDType::Q4_0),
    ] {
        write_gguf(path, dtype, &metadata(), &weights);
    }
    let map: HashMap<_, _> = weights
        .into_iter()
        .map(|(name, t)| (hf_name(&name), t))
        .collect();
    candle_core::safetensors::save(&map, &safetensors).unwrap();
    Fixture {
        _dir: dir,
        f32,
        q8,
        q4,
        safetensors,
    }
}
fn input(ids: &[u32]) -> Tensor {
    Tensor::new(ids, &Device::Cpu)
        .unwrap()
        .reshape((1, ids.len()))
        .unwrap()
}
fn max_error(a: &Tensor, b: &Tensor) -> f32 {
    assert_eq!(a.dims(), b.dims());
    a.flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap()
        .into_iter()
        .zip(b.flatten_all().unwrap().to_vec1::<f32>().unwrap())
        .map(|(x, y)| {
            assert!(x.is_finite() && y.is_finite());
            (x - y).abs()
        })
        .fold(0.0, f32::max)
}
#[test]
fn deterministic_f32_gguf_matches_safetensors_with_nondefault_f32_metadata() {
    let f = fixture();
    validate_gguf_config(&f.f32, &config()).unwrap();
    let gguf = ToyLlamaGGUF::load_range(&f.f32, 0..LAYERS, &Device::Cpu).unwrap();
    let st = ToyLlama::load(&config(), &f.safetensors).unwrap();
    let mut a = gguf.forward_prefill(&input(&[1, 2, 3])).unwrap();
    let mut b = st.forward_prefill(&input(&[1, 2, 3])).unwrap();
    for step in 0..8 {
        let error = max_error(&a, &b);
        assert!(error <= 2e-5, "step{step}: F32 format mismatch {error}");
        let token = ToyLlama::sample_argmax(&a).unwrap();
        assert_eq!(token, ToyLlama::sample_argmax(&b).unwrap());
        a = gguf.forward_decode(&input(&[token])).unwrap();
        b = st.forward_decode(&input(&[token])).unwrap();
    }
}
#[test]
fn each_quantization_matches_slice_only_prefill_and_eight_decode_steps() {
    let f = fixture();
    let mut previous = u64::MAX;
    for path in [&f.f32, &f.q8, &f.q4] {
        let full = ToyLlamaGGUF::load_range(path, 0..2, &Device::Cpu).unwrap();
        let coordinator =
            ToyLlamaGGUF::load_range_with_roles(path, 0..0, &Device::Cpu, TensorRoles::COORDINATOR)
                .unwrap();
        let a = ToyLlamaGGUF::load_range_with_roles(path, 0..1, &Device::Cpu, TensorRoles::WORKER)
            .unwrap();
        let b = ToyLlamaGGUF::load_range_with_roles(path, 1..2, &Device::Cpu, TensorRoles::WORKER)
            .unwrap();
        assert!(a.embed(&input(&[1])).is_err());
        assert!(b.local_layer_index(0).is_err());
        assert_eq!(b.local_layer_index(1).unwrap(), 0);
        assert!(a.resident_weight_bytes() < full.resident_weight_bytes());
        assert!(full.resident_weight_bytes() < previous);
        previous = full.resident_weight_bytes();
        let tokens = input(&[1, 2, 3]);
        let mut expected = full.forward_prefill(&tokens).unwrap();
        let hidden = coordinator.embed(&tokens).unwrap();
        let hidden = a.forward_layers_prefill(&hidden, 0..1).unwrap();
        let hidden = b.forward_layers_prefill(&hidden, 1..2).unwrap();
        let mut actual = coordinator.head(&hidden).unwrap();
        for step in 0..8 {
            assert_eq!(
                max_error(&expected, &actual),
                0.0,
                "{} step{step}",
                path.display()
            );
            let token = ToyLlama::sample_argmax(&expected).unwrap();
            assert_eq!(token, ToyLlama::sample_argmax(&actual).unwrap());
            expected = full.forward_decode(&input(&[token])).unwrap();
            let hidden = coordinator.embed(&input(&[token])).unwrap();
            let hidden = a.forward_layers_decode(&hidden, 0..1).unwrap();
            let hidden = b.forward_layers_decode(&hidden, 1..2).unwrap();
            actual = coordinator.head(&hidden).unwrap();
        }
        eprintln!(
            "{} resident_weight_bytes={}",
            path.file_name().unwrap().to_string_lossy(),
            full.resident_weight_bytes()
        );
    }
}
#[test]
fn quantization_error_is_bounded_against_same_original_f32_weights() {
    let f = fixture();
    let reference = ToyLlamaGGUF::load_range(&f.f32, 0..2, &Device::Cpu).unwrap();
    for (path, tolerance) in [(&f.q8, 0.03f32), (&f.q4, 0.3f32)] {
        reference.clear_kv_cache().unwrap();
        let quantized = ToyLlamaGGUF::load_range(path, 0..2, &Device::Cpu).unwrap();
        let mut expected = reference.forward_prefill(&input(&[1, 2, 3])).unwrap();
        let mut actual = quantized.forward_prefill(&input(&[1, 2, 3])).unwrap();
        let mut worst = 0.0f32;
        for _ in 0..8 {
            worst = worst.max(max_error(&expected, &actual));
            let token = ToyLlama::sample_argmax(&expected).unwrap();
            expected = reference.forward_decode(&input(&[token])).unwrap();
            actual = quantized.forward_decode(&input(&[token])).unwrap();
        }
        eprintln!(
            "{} max_abs_logit_error={worst}",
            path.file_name().unwrap().to_string_lossy()
        );
        assert!(
            worst < tolerance,
            "quantization error{worst} exceeds{tolerance}"
        );
    }
}
#[test]
fn tied_head_alias_does_not_duplicate_embedding_and_matches_slices() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("tied.gguf");
    write_gguf(&path, GgmlDType::Q4_0, &metadata(), &tensors(true));
    let full = ToyLlamaGGUF::load_range(&path, 0..2, &Device::Cpu).unwrap();
    let coordinator =
        ToyLlamaGGUF::load_range_with_roles(&path, 0..0, &Device::Cpu, TensorRoles::COORDINATOR)
            .unwrap();
    let worker =
        ToyLlamaGGUF::load_range_with_roles(&path, 0..2, &Device::Cpu, TensorRoles::WORKER)
            .unwrap();
    let expected = full.forward_prefill(&input(&[1, 2, 3])).unwrap();
    let hidden = coordinator.embed(&input(&[1, 2, 3])).unwrap();
    let actual = coordinator
        .head(&worker.forward_layers_prefill(&hidden, 0..2).unwrap())
        .unwrap();
    assert_eq!(max_error(&expected, &actual), 0.0);
    assert_eq!(
        coordinator.resident_weight_bytes(),
        (VOCAB * HIDDEN * 4 + HIDDEN * 4 + 64 * (HIDDEN / HEADS) * 4) as u64
    );
}
#[test]
fn unsupported_metadata_bad_shapes_sidecars_ranges_and_truncation_fail_closed() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("bad.gguf");
    for (key, value) in [
        (
            "general.architecture",
            gguf_file::Value::String("gpt2".into()),
        ),
        ("llama.attention.head_count", gguf_file::Value::U32(0)),
        (
            "llama.rope.freq_base",
            gguf_file::Value::String("5000".into()),
        ),
        (
            "llama.rope.scaling.type",
            gguf_file::Value::String("linear".into()),
        ),
        ("llama.rope.dimension_count", gguf_file::Value::U32(8)),
    ] {
        let mut meta = metadata();
        meta.insert(key.into(), value);
        write_gguf(&path, GgmlDType::Q4_0, &meta, &tensors(false));
        assert!(
            ToyLlamaGGUF::load_range(&path, 0..2, &Device::Cpu).is_err(),
            "accepted {key}"
        );
    }
    let mut wrong = tensors(false);
    let index = wrong
        .iter()
        .position(|(name, _)| name == "blk.0.attn_q.weight")
        .unwrap();
    wrong[index].1 = tensor("bad", &[32, HIDDEN]);
    write_gguf(&path, GgmlDType::Q4_0, &metadata(), &wrong);
    assert!(ToyLlamaGGUF::load_range(&path, 0..2, &Device::Cpu).is_err());
    write_gguf(&path, GgmlDType::Q4_0, &metadata(), &tensors(false));
    let mut wrong = config();
    wrong.hidden_size = 32;
    assert!(validate_gguf_config(&path, &wrong).is_err());
    assert!(ToyLlamaGGUF::load_range(&path, 2..3, &Device::Cpu).is_err());
    let file = std::fs::OpenOptions::new().write(true).open(&path).unwrap();
    let len = file.metadata().unwrap().len();
    file.set_len(len - 16).unwrap();
    assert!(ToyLlamaGGUF::load_range(&path, 0..2, &Device::Cpu).is_err());
}
