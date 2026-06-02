# Synapse

Decentralized AI compute: split a real model across machines and run inference together — now reachable **globally** over P2P, not just on a LAN.

Read `docs/AI_CONTEXT.md` before changing scope.

## How It Works

- **Master**: orchestrates a session (embedding, lower layers, LM head + sampling) and recruits workers.
- **Worker**: contributes compute by running routed transformer layer ranges.
- **Relay**: optional public node (`synapse relay`) that routes circuits and seeds the DHT so NAT'd peers can connect across the internet.

Discovery and connectivity:

- **mDNS** for the LAN fast path.
- **Kademlia DHT** keyed by model id, so a worker anywhere can find a master serving the same model.
- **Relay v2 + DCUtR + AutoNAT** for NAT traversal: a NAT'd master reserves a relay circuit and listens on `<relay>/p2p-circuit`, becoming dialable by remote workers; AutoNAT promotes a detected public address automatically.

## Commands

```bash
# Run a public relay/bootstrap node (on a VPS with a public IP)
synapse relay --listen 0.0.0.0:4001

# Host a session (master). Generates a pairing code.
synapse host --master --model <hf-id> --prompt "Explain P2P clustering"

# Join a known master directly
synapse join <master-addr> --model <hf-id> --code <pairing-code>

# Join by discovering a master globally via the DHT (no address needed)
synapse connect --model <hf-id> --code <pairing-code>

# Localhost single-process demo (no networking)
synapse demo --model <hf-id> --prompt "Hello"
```

## Global Setup

1. Run `synapse relay` on a machine with a public IP. Note its printed peer id + address.
2. Add that relay to `config/relays.toml` on master/worker machines:
   ```toml
   relays = ["/dns4/your-relay.example.com/udp/4001/quic-v1/p2p/12D3KooW..."]
   ```
   Optionally add DHT bootstrap peers to `config/bootstrap.toml`.
3. Start a master with `synapse host --master ...`; it dials the relay and listens on a circuit address.
4. Remote workers run `synapse connect --model <same-id> --code <code>` to discover and join via the DHT/relay.

Entries in `config/*.toml` must be full multiaddrs ending in `/p2p/<peer-id>`; placeholders are skipped at load time.

## Security

- Pairing remains required: discovery (DHT) is open, but joining a session needs the pairing code. Auth is challenge-response (Ed25519 signature + peer-id binding + hashed pairing code) with brute-force rate limiting.
- Treat every peer as untrusted. Remote model paths are never executed; only Hugging Face model ids are accepted.

## Inference Status

- Real Candle inference path for `hf-internal-testing/tiny-random-LlamaForCausalLM`.
- Master/worker run a real sharded forward exchange with structured `TensorPayload` (`dtype`, `shape`, `bytes`) and strict validation.
- Static + weighted (capability-aware) shard planning, multi-hop routing, continuous-batching queue, self-healing re-route, and speculative-decode scaffolding.

## Development

```bash
cargo fmt
cargo test --workspace
cargo run -p synapse-cli -- demo --model hf-internal-testing/tiny-random-LlamaForCausalLM --prompt "Hello"
```

## Web GUI

A browser-based GUI is provided as a single Rust binary that embeds a
React/Vite SPA. It opens `http://127.0.0.1:8080` in your default
browser and exposes the same `host` / `join` / `relay` / `config` flows
without any CLI.

```bash
# First time: installs npm deps and builds the React frontend.
cargo run -p synapse-gui

# Override defaults
cargo run -p synapse-gui -- --addr 0.0.0.0:9000 --no-open
```

The GUI process *is* the swarm process — tokens stream from the
inference loop to the browser over a single WebSocket, with no IPC
hop. The React build is embedded via `rust-embed`, so the
release binary is a single self-contained executable.
