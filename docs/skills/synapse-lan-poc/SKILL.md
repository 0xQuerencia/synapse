---
name: synapse-lan-poc
description: Use when working on Project Synapse Phase 1, the Rust LAN clustering proof of concept for sharded local AI inference across two machines.
---

# Synapse LAN PoC Skill

## Start Here

Read `docs/AI_CONTEXT.md` before changing code. It is the canonical cross-agent handoff.

## Goal

Build only the Phase 1 LAN proof of concept: a Rust CLI that can split a tiny real causal language model across one master and one worker on the same LAN.

## Non-Goals

Do not implement NAT traversal, public grid behavior, compute credits, Stripe, a Tauri desktop app, enterprise auth, or a global scheduler in this phase.

## Working Rules

1. Keep the repo in a compiling state after each meaningful change.
2. Put source in `crates/`, tests in the relevant Rust test locations, docs in `docs/`, and configuration in `config/`.
3. Preserve work from other tools. Do not reset, checkout, or overwrite unrelated files.
4. Treat LAN peers as untrusted. Validate all wire data before use.
5. Prefer deterministic localhost tests before two-machine manual demos.

## Implementation Order

1. Stabilize protocol contracts and shard planning in `synapse-core`.
2. Build the CLI command surface in `synapse-cli`.
3. Add a localhost demo path.
4. Add the real LAN transport using libp2p QUIC plus mDNS and manual address fallback.
5. Add Candle model loading and a tiny sharded forward pass.
6. Add failure tests and manual demo docs.

## Done Criteria

The milestone is done when `cargo test` passes and two machines can run `synapse host --master ...` plus `synapse join ...` to generate text from the master terminal.
