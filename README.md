# Synapse

**Your devices. One AI.**

Synapse is an experimental Rust application for running compatible language models locally and across explicitly trusted computers. It has a Candle inference runtime, authenticated libp2p transport, a command line, a local device workspace, and a small compatible chat API.

The current implementation tranche covers **34 planned work packages**: 01–21, 25–26, 33, 39–45, and 51–53. The code and local evidence include real independent processes exchanging tensors, exact transformer-layer placement, verified model artifacts, authenticated streaming, worker controls, and a signed model cache. **Separate physical-device memory savings and real WAN reliability remain unaccepted gates.** The original offline fixture tests numerical mechanics; it does not produce useful language or establish compatibility with arbitrary checkpoints.

Work-package labels such as `[PR-07]` identify planned scope. They are not GitHub pull-request numbers or acceptance stamps. This tranche has **zero published GitHub PRs, zero published release artifacts, and no production deployment**. Exact integrated revision, commands, results and remaining failures belong to the [final tranche evidence](docs/evidence/final-tranche.md).

## Start here

- [Implementation, local evidence and acceptance status](docs/IMPLEMENTATION_STATUS.md)
- [Developer quickstart](docs/quickstart.md)
- [Master plan: architecture, 56 packages and exit gates](docs/MASTER_PLAN.md)
- [Remaining gaps](docs/GAPS.md) and [machine-readable gates](docs/evidence/gates.json)
- [Trust and privacy](SECURITY.md), [model approval/cache](docs/model-catalog.md), and [package installation](docs/release-installation.md)
- [License and provenance record](docs/upstream-provenance.md)

## Run the original offline fixture

Initial validated development environment: Linux x86_64 CPU with the repository's pinned Rust toolchain. Build dependencies still need to be available; the fixture itself requires no model download.

```sh
cargo build --workspace --locked
./target/debug/synapse doctor \
  --model-dir crates/synapse-core/tests/fixtures/tiny-llama
./target/debug/synapse infer \
  --model-dir crates/synapse-core/tests/fixtures/tiny-llama \
  --prompt "<s> </s> Hello" --max-tokens 8
```

Run actual coordinator/worker processes with separate test identities and retained evidence:

```sh
./scripts/demo_two_nodes.sh \
  --binary target/debug/synapse \
  --workers 1 --coordinator-only
```

The harness checks token positions, exact layer coverage, worker execution and independent reference output. It removes test secrets and terminates only the processes it started. This is a same-host loopback experiment. The [quickstart](docs/quickstart.md) includes a forced three-process memory-budget experiment, local GUI authentication and the limits of these results.

## What is available, and what remains gated

| Surface | Current implementation | Remaining acceptance |
| --- | --- | --- |
| Local inference | Manifest verification, supported Llama fixture, bounded generation, explicit prompt/template handling | Useful model and hardware compatibility matrix; rights review |
| Private cluster | Authenticated prepare/ready/release, partial tensor loading, exact ranges, admission, correlated transport and worker controls | Two physical devices, measured memory benefit and separate-network direct/relay inference |
| Local HTTP and device workspace | Protected login, Host/Origin checks, real status, streaming and documented chat subset | Independent usability, broader platform qualification and external security review |
| Model cache and native package tools | Explicit trusted publishers and rights, atomic verified cache, signed package checks, staged update/rollback/uninstall | Production key ownership, approved third-party models and fresh-machine release validation |
| Public landing | Responsive static site with an interactive, clearly labeled local resource simulator | Publication is a separate action; no user, earnings or benchmark claims are fabricated |
| Community compute and credits | Planned architecture only | Hosted membership, transactional accounting, verification, operator budget and pilot approval |
| Open public swarm | Research only | Independent adversarial, privacy and economic review |

Sharing model files and executing inference are distinct operations. A full-model host can be more practical than a chain of distant workers. A hash or signature does not establish model quality, sufficient memory, permission to redistribute, or a trustworthy computing peer.

## Trust and resource controls

Private participation is explicit. Pause, drain and stop must reflect acknowledgements from the actual worker. The runtime distinguishes model-shaped inference reservations from operating-system limits; Linux address-space limiting is a dedicated worker option, not a portable sandbox or a resident-memory guarantee.

Transport encryption protects data in transit. Computing peers may see prompts or infer information from activations. Use trusted devices and appropriate data. Local/private inference makes no community billing calls and must not silently switch to unapproved external compute. Planned credits are noncash usage accounting, with no promised earnings, transfer value or guaranteed available capacity.

The compatible local subset supports `GET /v1/models` and `POST /v1/chat/completions`, including bounded streaming and usage. Unsupported tools, modalities and sampling options fail explicitly. See the [API contract](docs/protocols/api.md); this is not a claim of complete compatibility with every chat client.

## Development and release status

```sh
./scripts/verify_all.sh
python3 schemas/check.py --source-root . --wire-root .
python3 schemas/check_catalog.py --source-root .
```

The contract checks need the pinned Python dependency in `schemas/requirements.txt`; the catalog signature check also uses OpenSSL. Other concrete harnesses and their scope are listed in the quickstart. Local checks do not stand in for remote CI, an independent audit or physical tests. Preserve failed and blocked evidence when repeating a run.

Read [contributing](CONTRIBUTING.md) before modifying the runtime. The root [LICENSE](LICENSE) contains MIT terms; workspace package metadata says Apache-2.0. That discrepancy remains unresolved. This work preserves the root license and does not decide ownership or relicense prior contributions. Resolve the [provenance questions](docs/upstream-provenance.md) before releasing packages or approving third-party distribution.
