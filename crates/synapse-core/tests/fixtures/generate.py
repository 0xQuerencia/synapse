#!/usr/bin/env python3
"""Rebuild the original deterministic tiny Llama fixture (Python stdlib only).

All weights are generated here, with no downloaded checkpoint or training data.
The scalar reference uses Python doubles and an independent full-sequence
forward pass; Rust tests compare Candle F32 logits against its golden output.
"""
import hashlib
import json
import math
import struct
from pathlib import Path

ROOT = Path(__file__).resolve().parent / "tiny-llama"
ROOT.mkdir(parents=True, exist_ok=True)
HIDDEN, FFN, LAYERS, VOCAB, HEADS, KV_HEADS = 16, 32, 2, 64, 4, 2
HEAD_DIM = HIDDEN // HEADS
weights = {}


def add(name, shape, norm=False):
    count = math.prod(shape)
    data = []
    for index in range(count):
        raw = hashlib.sha256(f"synapse-fixture-v1:{name}:{index}".encode()).digest()
        value = (int.from_bytes(raw[:4], "little") / 2**32 - 0.5) * 0.2
        value = value + 1.0 if norm else value
        data.append(struct.unpack("<f", struct.pack("<f", value))[0])
    weights[name] = (shape, data)


add("model.embed_tokens.weight", [VOCAB, HIDDEN])
for layer in range(LAYERS):
    prefix = f"model.layers.{layer}"
    for name in ["input_layernorm", "post_attention_layernorm"]:
        add(f"{prefix}.{name}.weight", [HIDDEN], norm=True)
    for name, outputs in [("q_proj", HIDDEN), ("k_proj", KV_HEADS * HEAD_DIM),
                          ("v_proj", KV_HEADS * HEAD_DIM), ("o_proj", HIDDEN)]:
        add(f"{prefix}.self_attn.{name}.weight", [outputs, HIDDEN])
    for name, shape in [("gate_proj", [FFN, HIDDEN]), ("up_proj", [FFN, HIDDEN]),
                        ("down_proj", [HIDDEN, FFN])]:
        add(f"{prefix}.mlp.{name}.weight", shape)
add("model.norm.weight", [HIDDEN], norm=True)
add("lm_head.weight", [VOCAB, HIDDEN])

header, payload = {}, bytearray()
for name in sorted(weights):
    shape, data = weights[name]
    start = len(payload)
    payload.extend(struct.pack("<" + "f" * len(data), *data))
    header[name] = {"dtype": "F32", "shape": shape, "data_offsets": [start, len(payload)]}
encoded = json.dumps(header, separators=(",", ":")).encode()
encoded += b" " * ((8 - len(encoded) % 8) % 8)
(ROOT / "model.safetensors").write_bytes(struct.pack("<Q", len(encoded)) + encoded + payload)

config = {"model_type": "llama", "architectures": ["LlamaForCausalLM"],
          "hidden_size": HIDDEN, "intermediate_size": FFN, "num_hidden_layers": LAYERS,
          "vocab_size": VOCAB, "rms_norm_eps": 1e-5, "num_attention_heads": HEADS,
          "num_key_value_heads": KV_HEADS, "max_position_embeddings": 128,
          "rope_theta": 10000.0, "torch_dtype": "float32", "tie_word_embeddings": False,
          "bos_token_id": 1, "eos_token_id": 2}
(ROOT / "config.json").write_text(json.dumps(config, indent=2) + "\n")
words = ["[UNK]", "<s>", "</s>", "Hello", "world", "Once", "upon", "a", "time"]
words += [f"token_{index}" for index in range(len(words), VOCAB)]
tokenizer = {"version": "1.0", "truncation": None, "padding": None,
             "added_tokens": [{"id": index, "content": word, "single_word": False,
                               "lstrip": False, "rstrip": False, "normalized": False,
                               "special": True} for index, word in enumerate(words[:3])],
             "normalizer": None, "pre_tokenizer": {"type": "Whitespace"},
             "post_processor": None, "decoder": None,
             "model": {"type": "WordLevel", "vocab": dict(zip(words, range(VOCAB))), "unk_token": "[UNK]"}}
(ROOT / "tokenizer.json").write_text(json.dumps(tokenizer, indent=2) + "\n")


def linear(vector, name):
    (rows, columns), data = weights[name]
    assert len(vector) == columns
    return [sum(vector[j] * data[i * columns + j] for j in range(columns)) for i in range(rows)]


def norm(vector, name):
    scale = 1.0 / math.sqrt(sum(x*x for x in vector) / len(vector) + config["rms_norm_eps"])
    return [x * scale * weight for x, weight in zip(vector, weights[name][1])]


def rotate(vector, position):
    output = list(vector)
    half = HEAD_DIM // 2
    for head in range(len(vector) // HEAD_DIM):
        for index in range(half):
            angle = position / (config["rope_theta"] ** (2 * index / HEAD_DIM))
            left, right = head * HEAD_DIM + index, head * HEAD_DIM + index + half
            output[left] = vector[left] * math.cos(angle) - vector[right] * math.sin(angle)
            output[right] = vector[left] * math.sin(angle) + vector[right] * math.cos(angle)
    return output


def forward(ids):
    embedding = weights["model.embed_tokens.weight"][1]
    hidden = [embedding[token * HIDDEN:(token + 1) * HIDDEN] for token in ids]
    for layer in range(LAYERS):
        prefix = f"model.layers.{layer}"
        normalized = [norm(row, f"{prefix}.input_layernorm.weight") for row in hidden]
        query = [rotate(linear(row, f"{prefix}.self_attn.q_proj.weight"), pos) for pos, row in enumerate(normalized)]
        key = [rotate(linear(row, f"{prefix}.self_attn.k_proj.weight"), pos) for pos, row in enumerate(normalized)]
        value = [linear(row, f"{prefix}.self_attn.v_proj.weight") for row in normalized]
        attended = []
        for position in range(len(ids)):
            combined = []
            for head in range(HEADS):
                qoffset, koffset = head * HEAD_DIM, (head // (HEADS // KV_HEADS)) * HEAD_DIM
                scores = [sum(query[position][qoffset+d] * key[j][koffset+d] for d in range(HEAD_DIM)) / math.sqrt(HEAD_DIM)
                          for j in range(position + 1)]
                exponentials = [math.exp(score-max(scores)) for score in scores]
                probs = [value/sum(exponentials) for value in exponentials]
                combined += [sum(probs[j] * value[j][koffset+d] for j in range(position+1)) for d in range(HEAD_DIM)]
            attended.append(linear(combined, f"{prefix}.self_attn.o_proj.weight"))
        after_attn = [[x+y for x, y in zip(row, delta)] for row, delta in zip(hidden, attended)]
        hidden = []
        for row in after_attn:
            normalized = norm(row, f"{prefix}.post_attention_layernorm.weight")
            gate = linear(normalized, f"{prefix}.mlp.gate_proj.weight")
            up = linear(normalized, f"{prefix}.mlp.up_proj.weight")
            delta = linear([g / (1 + math.exp(-g)) * u for g, u in zip(gate, up)], f"{prefix}.mlp.down_proj.weight")
            hidden.append([x+y for x, y in zip(row, delta)])
    return [linear(norm(row, "model.norm.weight"), "lm_head.weight") for row in hidden]


ids = [1, 2, 3]
logits = forward(ids)
greedy = list(ids)
for _ in range(8):
    row = forward(greedy)[-1]
    greedy.append(max(range(len(row)), key=row.__getitem__))
(ROOT / "reference.json").write_text(json.dumps({"input_ids": ids, "logits": logits,
    "greedy_ids": greedy, "absolute_tolerance": 1e-5, "source": "generate.py scalar Python reference"}, indent=2) + "\n")
artifacts = []
for name in ["config.json", "model.safetensors", "tokenizer.json"]:
    data = (ROOT / name).read_bytes()
    artifacts.append({"filename": name, "size_bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()})
(ROOT / "manifest.json").write_text(json.dumps({"schema_version": 1, "model_id": "synapse/tiny-llama-fixture",
    "revision": "fixture-v1", "weights": "model.safetensors", "artifacts": artifacts}, indent=2) + "\n")
print(f"Wrote deterministic fixture, {len(payload)} tensor bytes, greedy IDs {greedy}")
