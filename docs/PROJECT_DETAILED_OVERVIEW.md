> **Historical source overview:** This document predates the v2 execution plan. Its implementation and performance claims are not current acceptance evidence. Use [the master plan](MASTER_PLAN.md), [status](IMPLEMENTATION_STATUS.md) and revision-specific test artifacts.

# Synapse Project Deep Dive

## 1. What Synapse Is

Synapse is a **LAN-first distributed inference proof-of-concept**.
It splits a causal language model across machines on the same local network:

- **Master node**: prompt/token flow, orchestration, final LM head + sampling.
- **Worker nodes**: routed transformer layer hops.

The current implementation now includes **Phase 7 multi-worker planning and orchestration** while staying in LAN PoC scope.

---

## 2. Phase 7 Capability Snapshot

Current runtime capabilities include:

- Real Candle inference path for `hf-internal-testing/tiny-random-LlamaForCausalLM`.
- libp2p QUIC request/response tensor exchange with strict payload validation.
- **Worker capability registry** on master:
  - tracks worker connection/authentication lifecycle,
  - stores per-worker capability snapshots (`compute_score`, `available_ram_mb`) from `NodeHello`,
  - preserves last known capability even after disconnect/expiry transitions.
- **Static shard planner** (`plan_static_shards`): deterministic, contiguous layer assignment in participant order.
- **Weighted heterogeneous planner** (`plan_weighted_shards`): compute-weighted assignment with RAM-based caps.
- **Multi-hop master-routed orchestrator**:
  - builds per-generation route plan,
  - runs master prefix layers,
  - forwards activation through ordered worker hops,
  - validates hop identity/trace before continuing,
  - returns final activation to master for LM head + sampling.
- Cache-reset barrier before first hop (`ClearCache`/`ClearCacheAck`) per generation round.
- Local-only execution fallback when no worker hop is available.

---

## 3. Repository Layout

### Workspace

- `crates/synapse-core`: protocol, planners, network runtime, inference primitives.
- `crates/synapse-cli`: CLI UX and operator flows.

### Important Docs

- `README.md`: top-level usage/status.
- `docs/AI_CONTEXT.md`: scope boundaries.
- `docs/PROJECT_DETAILED_OVERVIEW.md` (this file): detailed runtime/architecture behavior.

---

## 4. Execution Modes

1. **Host (master)**
   - Generates pairing code.
   - Authenticates workers.
   - Accepts prompts and orchestrates generation route.

2. **Join (worker)**
   - Discovers/dials master.
   - Sends `NodeHello` + capability hints.
   - Executes routed layer ranges.

3. **Demo (localhost)**
   - No network dependency.
   - Validates model/load + generation behavior in one process.

Running `synapse` with no subcommand opens the interactive gateway.

---

## 5. Worker Capability Registry (Master)

Master maintains an internal worker registry keyed by `PeerId` with state:

- `Connected`
- `Authenticated`
- `Disconnected`
- `Expired`

Stored capability snapshot fields:

- `compute_score: Option<u64>`
- `available_ram_mb: Option<u64>`

Population rules:

- Worker sends capability fields via `NodeHello`.
- `available_ram_mb` can be derived from `memory_bytes` fallback if needed.
- Current `compute_score` production path is a lightweight CPU-thread heuristic; a micro-benchmark helper exists in inference for future handshake wiring.
- Snapshot is recorded at authentication and reused for planner input.

### Operator note

- Capability-aware planning is only used when master + every routed worker provide both compute and RAM fields.
- If capability data is incomplete, routing **automatically falls back to static planning**.

---

## 6. Shard Planning

### 6.1 Static planner (`plan_static_shards`)

Behavior:

- Requires `total_layers > 0` and at least one participant.
- Splits layers contiguously in participant order.
- Deterministic for same input ordering.
- Omits zero-layer assignments when participants exceed layer count.

### 6.2 Weighted planner (`plan_weighted_shards`)

Behavior:

- Requires non-zero total compute across participants.
- Starts from compute-proportional rounded targets.
- Applies RAM ceiling using `RAM_PER_LAYER_HEURISTIC_MB = 50`.
- Rebalances until assigned layers exactly cover `total_layers`.
- Returns contiguous ranges; omits participants assigned zero layers.
- Deterministic tie-breaks keep routes stable for identical inputs.

### Operator note

Weighted plans can fail when aggregate RAM capacity is insufficient for requested layer count. In that case generation route construction fails rather than silently overcommitting.

---

## 7. Multi-Hop Master-Routed Generation

For each prompt:

1. Master tokenizes prompt and clears local KV cache.
2. Master gathers authenticated workers + capabilities from registry.
3. Master builds a `GenerationRoutePlan`:
   - `master_layer_range`
   - ordered `worker_hops` with per-hop layer ranges.
4. Master sends `ClearCache` to each routed worker and waits for all matching `ClearCacheAck`.
5. Per step:
   - step 0 uses **prefill** tokens (full current prompt window),
   - later steps use **decode** token (last token only).
6. Master runs embed + master layers, serializes hidden state.
7. Master dispatches `ActivationRequest` hop-by-hop (`gen-step-{n}-hop-{k}` trace IDs).
8. Each worker executes requested layer range and returns `ActivationResponse`.
9. Master verifies response trace, sequence position, and expected peer before chaining next hop.
10. After final hop (or directly if no worker hop), master runs LM head + argmax sample and emits token.

### Operator note

- If no eligible worker hop exists, the route becomes master-only for that generation round.
- If a required worker disconnects mid-round or a hop response is inconsistent, generation fails fast with explicit error signaling.

---

## 8. Tensor Wire Format and Safety

Tensor payloads use validated structured format:

- `dtype`
- `shape`
- `bytes`

Validation includes:

- max size limit,
- dimension constraints,
- overflow-safe element math,
- exact dtype/shape byte-length match.

This protects runtime against malformed payload input.

---

## 9. Network Protocol and Limits

Core constants (`synapse-core::protocol`):

- `PROTOCOL_VERSION = "synapse-lan/0.1.0"`
- `MAX_TENSOR_BYTES = 64 MiB`
- `MAX_TENSOR_DIMS = 8`
- `MAX_SEQUENCE_LEN = 4096`
- `MAX_WIRE_MESSAGE_BYTES = MAX_TENSOR_BYTES + 4096`

Transport stack:

- libp2p QUIC
- request/response codec
- mDNS discovery

Hardening includes request timeout, stream concurrency cap, and protocol-level message validation before use.

---

## 10. Fallback and Failure Behavior (Operator-Facing)

- **Weighted -> static fallback**: triggered when capability inputs are incomplete.
- **Route with zero workers**: master executes locally for that generation round.
- **Unavailable expected worker**: generation fails (`expected worker ... unavailable`).
- **Mismatched hop response** (trace/peer/sequence): generation fails to avoid cross-hop corruption.
- **Cache reset barrier**: first activation is delayed until all expected `ClearCacheAck` arrive.
- **Worker-side role guarding**: workers reject invalid request types; masters reject worker-only messages.

---

## 11. Regression Testing Notes

Phase 7 behavior is covered by focused unit tests in `synapse-core`, including:

- Worker registry lifecycle + capability snapshot updates/staleness handling (`network/node/mod.rs`).
- Static planner coverage/determinism/edge-cases (`shard.rs`).
- Weighted planner proportionality, RAM-cap enforcement, rebalance, determinism (`shard.rs`).
- Multi-hop route ordering/contiguity and per-hop assignment (`network/node/generation.rs`).
- Weighted route selection + static fallback when capability data is missing (`network/node/generation.rs`).
- Worker availability checks for routed hops (`network/node/generation.rs`).

These tests are intended to catch planning drift and routing regressions before runtime integration testing.

---

## 12. Typical Usage

```bash
synapse
```

Or explicit commands:

```bash
synapse host --master --model hf-internal-testing/tiny-random-LlamaForCausalLM --prompt "Hello"
synapse join --model hf-internal-testing/tiny-random-LlamaForCausalLM --code <pairing-code>
synapse demo --model hf-internal-testing/tiny-random-LlamaForCausalLM --prompt "Hello"
```

---

## 13. Development / Verification

```bash
cargo fmt
cargo test --workspace
```

Install CLI from this repo:

```bash
cargo install --path crates/synapse-cli --force
```

---

## 14. Current Limitations

- LAN PoC scope only (no global/NAT-traversed production mesh behavior).
- KV-cache state is local to each node instance and reset per prompt; cache handoff/migration across route changes is not implemented.
- Route plans are per-generation-round snapshots (not continuously re-optimized mid-round).
- Planner capability scoring is heuristic (CPU thread-derived compute + RAM heuristic), not hardware benchmarking.

---

## 15. Practical Mental Model

> Synapse currently behaves like a master-driven distributed forward pass: the master builds a deterministic route, executes its shard, relays activations across ordered worker hops, then finishes sampling locally.
