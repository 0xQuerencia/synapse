#!/usr/bin/env bash
# Real independent-process smoke test. Logs and evidence are retained.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
exec python3 scripts/local_cluster.py "$@"
