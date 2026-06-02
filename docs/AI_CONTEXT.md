# Synapse AI Agent Context

Canonical handoff for AI tools working in this repo. Read before changing scope.

## Mission

Synapse runs a real causal language model split across machines, reachable **globally** over P2P (DHT discovery + relay/DCUtR/AutoNAT NAT traversal), with a LAN mDNS fast path.

## Current Phase

Global P2P: relay client transport is wired into the SwarmBuilder (`with_relay_client`), masters reserve relay circuits and listen on `<relay>/p2p-circuit`, workers discover masters via the model-keyed DHT, and AutoNAT-detected public addresses are advertised. A dedicated `synapse relay` server routes circuits and seeds the DHT. Bootstrap/relay peers load from `config/*.toml`.

## Scope Boundaries (Still In Effect)

Global P2P and NAT traversal are now **in scope**. Do **not** expand into credits/payments, public HTTP APIs, desktop product features, or token economics in this phase.

## Implemented State

- Rust workspace with `synapse-core` and `synapse-cli`.
- Real Candle inference path for `hf-internal-testing/tiny-random-LlamaForCausalLM`.
- Model download from Hugging Face Hub (`config.json`, `model.safetensors`, `tokenizer.json`) on host/join/demo paths.
- Real sharded forward exchange over network:
  - master runs tokenization, embedding, and lower layers
  - worker runs upper layers
  - master runs LM head + greedy sampling and streams tokens
- Structured tensor wire payloads via `TensorPayload { dtype, shape, bytes }`.
- Tensor/wire safety checks:
  - dtype/shape/byte-length validation
  - overflow-aware size calculations
  - decode behavior that safely handles unaligned byte buffers
  - protocol codec message-size limit aligned to `MAX_WIRE_MESSAGE_BYTES`
- Security/resilience checks:
  - pairing-code auth + model identity match
  - bounded prompt/event channels
  - capped request concurrency and request timeout
  - explicit reject paths for invalid messages
  - generation failure handling on disconnects and inference errors
- CLI UX for `host --master`, `join`, `demo` now includes ASCII-panel output and interactive commands (`/help`, `/clear`, `/quit`) in host chat mode.
- Localhost demo remains available for single-machine validation.
- Continuous Batching (Phase 8):
  - `BatchPromptRequest` queue with per-client event channels
  - Sequential processing of queued requests through existing pipeline
  - `batch_queue_sender()` API for multi-client integration (HTTP/WebSocket)
- Self-Healing Orchestrator (Phase 9):
  - Worker hop timeout detection (`WORKER_HOP_TIMEOUT = 2s`)
  - Dynamic route recomputation on worker disconnect/timeout
  - Automatic KV cache clear + prefill replay through new route
  - `ConnectionClosed` triggers healing instead of fail-fast
- Speculative Decoding (Phase 10):
  - N-gram draft cache with frequency-based token prediction
  - Draft verification against model logits with prefix matching
  - KV cache truncation for rejected draft token rollback
  - Draft cache observation wired into generation token loop

## Not Implemented Yet

- True tensor-batched multi-sequence forward passes (batch_size > 1).
- Full speculative decode loop integration (draft → multi-token forward → verify → rollback cycle).
- Multi-worker scheduling/load balancing.
- Global/private swarm traversal beyond LAN mDNS/manual-address flows.
- Production-grade orchestration beyond current two-role LAN PoC.

## Active CLI Contract

- `synapse host --master --model <hf-id-or-path> --prompt <text>`
- `synapse join <master-addr> --model <same-hf-id-or-path> --code <pairing-code>`
- `synapse demo --model <hf-id-or-path> --prompt <text>`

## Technical Direction (Current)

- Language/runtime: Rust + Tokio.
- Inference: Candle (CPU-first), safetensors weights, Hugging Face tokenizer.
- Network: libp2p QUIC + mDNS + request/response protocol.
- Logging: tracing.

## Security Rules

- Treat every peer as untrusted (even on LAN).
- Never accept remote model paths for execution.
- Keep strict validation on protocol version, model identity, tensor payloads, trace IDs, and sequence bounds.
- Keep payload/message limits and bounded channels in place.
- Never log secrets or pairing codes after auth.

## Collaboration Rules

- Keep changes scoped and compiling.
- Update this file when implementation status materially changes.
- Add durable context in `docs/` instead of large source comments.
- Do not commit secrets, model weights, `.env` files, or Hugging Face tokens.
- Preserve unrelated work from other tools.
