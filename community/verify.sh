#!/bin/sh
set -eu
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/community${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
python3 -m unittest discover -s community/tests -v
python3 -m synapse_testnet.demo
python3 -m synapse_testnet.demo --full-model
python3 -m synapse_testnet.economy
if command -v node >/dev/null 2>&1; then node --check community/synapse_testnet/static/app.js; fi
python3 community/tools/release_gate.py
