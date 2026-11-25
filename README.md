# Synapse

**Your devices. One AI.**

Synapse is an experimental Rust project for cooperative inference with open-weight language models. The goal is to run a model locally, split its layers across trusted computers when one device cannot fit it, and eventually let approved contributors exchange useful compute for noncash usage credits.

The repository contains an inference runtime built with Candle, libp2p networking, a CLI and a local web interface. **Code presence is not evidence of a validated public network.** Physical two-device memory savings, separate-network reliability, supported hardware and community accounting each have their own acceptance gates.

## Start here

- [Master plan: architecture, 56 proposed work packages and acceptance gates](docs/MASTER_PLAN.md)
- [Implementation and evidence status](docs/IMPLEMENTATION_STATUS.md)
- [Developer quickstart](docs/quickstart.md)
- [Trust and privacy](SECURITY.md)
- [License and provenance record](docs/upstream-provenance.md)

## What is being built

| Mode | Purpose | Release condition |
| --- | --- | --- |
| Solo | Run one compatible model on your own machine | Reproducible local model parity |
| Private cluster | Run across explicitly trusted devices, on LAN or internet | Real placement, memory and network evidence |
| Approved community hosts | Send a bounded request to an approved full-model host | Security, consent, accounting and pilot gates |
| Community layer sharding | Split a request across approved contributors | Separate correctness, recovery and settlement review |
| Open public swarm | Research into untrusted global participation | Not an enabled product promise |

Model distribution and inference are different operations. Sharing verified model files does not execute a model. Layer sharding also needs compatible weights, exact coverage, session state and a usable network path. A remote device that fits the entire model can be a better route than a chain of distant workers.

## Build and inspect

Initial target: Linux CPU. Use the repository toolchain file when available; workspace metadata declares the minimum supported Rust version. Native compiler/linker and dependency prerequisites are described in the quickstart.

```sh
git clone https://github.com/0xQuerencia/synapse.git
cd synapse
cargo build --release --locked --bin synapse
./target/release/synapse --help
```

These are reproduction commands, not a claim that your platform has been tested. Check the recorded command results for the exact revision you build. First-time model downloads additionally require network access and disk space.

```sh
./target/release/synapse demo \
  --model hf-internal-testing/tiny-random-LlamaForCausalLM \
  --prompt "Hello" --max-tokens 8
```

The tiny random model tests mechanics; its output is not meaningful language. Only use model revisions, formats and templates that have actually passed compatibility tests. See [quickstart](docs/quickstart.md) for trusted two-process experiments and local GUI instructions.

## Contribute compute, use compute

The proposed community loop is: accept eligible work, validate it, earn credits, then spend those credits on available community inference. Credits are planned noncash usage accounting. They are not cryptocurrency, transferable money, a guaranteed reward or a guarantee that a model will be available. Private inference must not depend on a hosted account or credit service.

Contribution must be explicit, with resource caps and visible pause/stop controls. Transport encryption protects traffic in transit; computing peers may see prompts or infer information from activations. Use trusted devices and appropriate data. Local mode must never silently fall back to external compute.

## Development

Read [contributing](CONTRIBUTING.md) before modifying the runtime. Work-package IDs in commit messages map to the master plan. Build, test and benchmark outcomes belong to a specific revision and environment; report missing evidence rather than assuming success.

The root [LICENSE](LICENSE) contains MIT terms. Workspace package metadata currently says Apache-2.0. That discrepancy is explicitly unresolved; this implementation does not choose a new license or assert rights over prior contributions. See the [provenance record](docs/upstream-provenance.md) before distributing packages or importing upstream code.
