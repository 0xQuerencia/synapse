# Developer quickstart

This guide provides reproducible commands for the experimental code. A command appearing here does not establish a successful run, supported hardware or a released binary. Record the checkout SHA, toolchain, command, exit status and log for each run.

## Requirements

Start with Linux x86_64 CPU, Git, the repository's Rust toolchain and a native C/C++ compiler/linker. On Debian-family systems the build may require `build-essential`, `pkg-config` and the TLS development package used by the locked dependency graph. Check the exact CI dependency list before preparing a fresh machine. Model fetching needs access to the approved origin, enough disk and a reviewed model revision. Do not copy gated model credentials into commands or logs.

```sh
rustc --version
cargo --version
git rev-parse HEAD
cargo build --release --locked --bin synapse
./target/release/synapse --help
```

Run the independent checks used by the repository:

```sh
cargo fmt --all -- --check
cargo clippy --workspace --all-targets --locked -- -D warnings
cargo test --workspace --locked
```

Network-dependent model tests are separate from deterministic offline tests. A download failure is not evidence of model correctness or a successful integration test.

## Small local model

```sh
./target/release/synapse demo \
  --model hf-internal-testing/tiny-random-LlamaForCausalLM \
  --prompt "Hello" --max-tokens 8
```

The model is a mechanical fixture. For meaningful output, a candidate is `HuggingFaceTB/SmolLM2-135M-Instruct`; its exact revision, license, tokenizer/template and numerical parity must be checked before adding it to a supported catalog. Do not interpret a repository name as a compatibility guarantee.

## Trusted two-process experiment

Use separate identity files for every process. In one terminal:

```sh
./target/release/synapse --identity-path /tmp/synapse-host-test.key host --master \
  --model HuggingFaceTB/SmolLM2-135M-Instruct \
  --listen 127.0.0.1:34001 --max-tokens 32
```

Read the pairing code locally. In another terminal, substitute that code:

```sh
./target/release/synapse --identity-path /tmp/synapse-worker-test.key join 127.0.0.1:34001 \
  --model HuggingFaceTB/SmolLM2-135M-Instruct --code YOUR_PAIRING_CODE
```

A loopback run is two processes on one computer. It does not establish two-device memory savings or internet reachability. Preserve per-process logs, exact resident ranges and measured memory; do not include pairing codes or private keys in shared evidence. Stop both processes with Ctrl+C. Dedicated test identities should not be reused for real clusters.

For physical LAN tests, bind the host to the intended trusted interface and use its private address. Record both machines, model digest, configured memory limits and route evidence. Separate-router and relay tests belong to the WAN gate, with consenting participants; avoid exposing the local GUI or raw inference/admin ports publicly.

## Local GUI

```sh
cargo run --locked -p synapse-gui -- --addr 127.0.0.1:8080
```

Keep the experimental GUI on loopback. The Ollama option, when available, is a local full-model backend; its presence does not demonstrate swarm inference. Use existing CLI help for actual options; planned `serve`, `worker` or credit commands must not be documented as available until integrated and tested.

## Useful failure reports

Include revision, OS, backend, exact command, a redacted error and model revision. Distinguish build failures, missing artifacts, insufficient memory, unsupported format and unreachable peers. Never attach prompts, activations, keys, invitation codes or access tokens. See [security](../SECURITY.md) for sensitive reports.
