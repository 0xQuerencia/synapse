#!/usr/bin/env bash
# Measure real accepted token events and keep all raw evidence.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
SYNAPSE_BENCH_OUTPUT="${SYNAPSE_BENCH_OUTPUT:-artifacts/benchmark-$(date +%s)}"
python3 scripts/local_cluster.py --output "$SYNAPSE_BENCH_OUTPUT" "$@"
python3 scripts/benchmark_report.py "$SYNAPSE_BENCH_OUTPUT" --output "$SYNAPSE_BENCH_OUTPUT/summary.json" --csv "$SYNAPSE_BENCH_OUTPUT/summary.csv"
