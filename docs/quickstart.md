# Developer quickstart

These commands exercise the experimental implementation. Record the checkout SHA, toolchain, command, exit status and artifact directory for each run. The [final tranche evidence](evidence/final-tranche.md) identifies the actual integrated checks; this guide is not a claim that an arbitrary computer or model has been qualified.

## Requirements and build

Start with Linux x86_64 CPU, Git, the pinned Rust toolchain and a native C/C++ compiler/linker. Use the locked dependency graph. Native compiler, `pkg-config` and TLS/build prerequisites are listed in the CI workflow. Python 3 runs the standard-library fixture/harness tools. Contract validation uses the pinned requirements below and OpenSSL for independent Ed25519 checks. The landing uses Node.js without downloaded runtime packages.

```sh
rustc --version
cargo --version
git rev-parse HEAD
cargo build --workspace --locked
./target/debug/synapse --help
python3 -m pip install -r schemas/requirements.txt
```

If using a custom `CARGO_TARGET_DIR`, pass the actual binary path to every harness. Do not rebuild or overwrite a shared binary while another process is testing that artifact.

## Offline local inference

The committed `tiny-llama` fixture is original deterministic test data with an independent scalar reference. No third-party checkpoint or model credential is needed, and the generated text has no language-quality meaning.

```sh
./target/debug/synapse doctor \
  --model-dir crates/synapse-core/tests/fixtures/tiny-llama \
  --context-tokens 32
./target/debug/synapse infer \
  --model-dir crates/synapse-core/tests/fixtures/tiny-llama \
  --prompt "<s> </s> Hello" --max-tokens 8
```

`doctor` reports model-shaped weight/KV/scratch/load reservations. Those numbers are distinct from measured process RSS, operating-system address-space ceilings and GPU memory. Unsupported model shapes, missing files or corrupt bytes should fail before model use.

## Real local coordinator and worker processes

```sh
./scripts/demo_two_nodes.sh \
  --binary target/debug/synapse \
  --workers 1 --coordinator-only
```

The harness creates distinct test identities, launches actual processes, waits for authenticated readiness, verifies exact route coverage and independent reference output, records token timestamps/observed memory, removes test secrets and cleans up its processes. It writes a new artifact directory and retains failures. The transport defaults to QUIC; `--transport tcp` selects the corresponding local transport experiment.

A forced three-process case for the original fixture uses exact inference reservations:

```sh
./scripts/demo_two_nodes.sh \
  --binary target/debug/synapse \
  --workers 2 --coordinator-only \
  --host-memory-bytes 16000 --worker-memory-bytes 45000
```

These are model-shaped admission budgets for this specific small fixture, not limits on total process RSS. The final report records the observed ranges, resident bytes, output and reservation calculation. Successful execution on one host does not establish a model exceeding each physical machine's RAM or a useful model running faster.

`python3 scripts/wan_matrix.py --help` describes delayed-transport and worker-loss experiments. An injected delay or jitter is simulated WAN evidence. A loopback relay is local relay evidence. For physical LAN/WAN tests, use consenting devices and the documented runbook, record model digest/hardware/topology and actual route/residency, and keep admin/API listeners private. No inter-city or CGNAT success is inferred from these local commands.

## Authenticated local GUI and chat API

```sh
cargo run --locked -p synapse-gui -- \
  --addr 127.0.0.1:8080 \
  --token-file /absolute/private/synapse-gui-token \
  --model-dir crates/synapse-core/tests/fixtures/tiny-llama \
  --api-model-id synapse-fixture \
  --chat-template raw
```

Use an owned private directory for the token file. The server creates a private random token if absent. Read it locally, open the login page and enter it in the form; do not put credentials in a URL, screenshot, shared log or source file. The server requires loopback and validates Host/Origin. Bearer-token API clients use the same protected local credential. See [the API contract](protocols/api.md) for the supported deterministic subset and streaming errors.

The workspace can start a coordinator or worker using an already verified local model directory. Pairing controls authenticate the actual peers; status and controls report observed node state. Signed invitation helpers are separate from the current pairing UI. The local Ollama option, where configured, is a separate full-model backend and does not establish distributed execution.

The actual HTTP/WebSocket/process smoke harness and observed results are documented in [GUI/API evidence](evidence/gui-http-smoke.md). Browser layout/keyboard checks have a [separate record](evidence/device-layout-qa.md), including the labeled layout fixture and its limits. Fresh-user, physical-device and assistive-technology acceptance remain outstanding.

## Model policy, cache and packages

Use the [signed catalog workflow](model-catalog.md) for explicit trusted publishers, purpose-specific rights, atomic import, verified pins and pruning. Its example fixture is only cache-test content; the inference fixture above is different. No third-party model is approved merely because it downloads successfully or has a valid manifest hash. A raw local model-directory development flow does not automatically enforce a separate production catalog policy.

[Package tooling](release-installation.md) builds local candidates and verifies checksums/signatures before staged activation. It refuses active-process updates and unsafe archives. It does not publish a release, enroll a production key, run a service or make unsupported-platform promises.

## Checks and local landing

```sh
./scripts/verify_all.sh
python3 schemas/check.py --source-root . --wire-root .
python3 schemas/check_catalog.py --source-root .
python3 docs/evidence/check_gates.py
python3 -m unittest discover -s scripts/security -p 'test_*.py' -v
python3 -m unittest discover -s scripts/release -p 'test_*.py' -v
npm --prefix web/landing test
npm --prefix web/landing run check
npm --prefix web/landing run build
```

The static landing builds to `web/landing/dist`. It shows a recorded two-node fixture run. The separate memory simulator uses fictional profiles and timing estimates. Run `npm --prefix web/landing run dev` from the repository root to preview both pages.

Network model downloads and physical tests are separate from deterministic offline tests. A failed or unavailable download is not evidence of compatibility. For an issue, include source/model revision, OS/backend, exact command, redacted error and artifact location. Do not attach prompts, activations, keys, pairing/invitation codes or access tokens. Use the [security process](../SECURITY.md) for sensitive reports.
