#!/usr/bin/env bash
# Two node smoke test on one machine: run the master in one terminal,
# join from another with the pairing code the master prints.
set -euo pipefail

MODEL="${1:-hf-internal-testing/tiny-random-LlamaForCausalLM}"

if ! ls target/debug/synapse-cli >/dev/null 2>&1; then
  echo "build first: cargo build -p synapse-cli"
  exit 1
fi

echo "terminal 1 (master):"
echo "  ./target/debug/synapse-cli host --master --model $MODEL --prompt \"hello swarm\""
echo
echo "terminal 2 (worker), use the addr + pairing code printed by the master:"
echo "  ./target/debug/synapse-cli join <master-addr> --model $MODEL --code <code>"
echo
echo "same machine? the master is reachable on localhost, so any addr it"
echo "prints with 127.0.0.1 works."
