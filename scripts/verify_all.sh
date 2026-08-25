#!/usr/bin/env bash
# Plan-gate verification for the Synapse swarm implementation (plan: goofy-bubbling-wreath.md).
# Runs every gate the plan's Verification section prescribes. Safe to re-run any time.
set -uo pipefail

cd "$(dirname "$0")/.." || exit 1

echo "── 1/4 cargo fmt check ─────────────────────────────────"
cargo fmt --all --check && echo "fmt: OK" || echo "fmt: FAIL (run: cargo fmt --all)"

echo "── 2/4 clippy (warnings are errors) ────────────────────"
cargo clippy --workspace --all-targets -- -D warnings && echo "clippy: OK" || echo "clippy: FAIL"

echo "── 3/4 full test suite ────────────────────────────────"
cargo test --workspace && echo "tests: OK" || echo "tests: FAIL"

echo "── 4/4 release build ──────────────────────────────────"
cargo build --release && echo "build: OK" || echo "build: FAIL"

echo "── new-feature tests (must exist per plan) ────────────"
for t in sharded_model_test quantized_load_test; do
  if grep -rq "fn ${t}_" crates/synapse-core/tests/ 2>/dev/null; then
    echo "test file for ${t}: present"
  else
    echo "test file for ${t}: MISSING (expected by plan)"
  fi
done

echo "── done. summary above; all four gates must say OK ────"