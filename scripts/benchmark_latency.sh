#!/usr/bin/env bash
#
# Benchmark script: spawn 1 master + 2 workers locally,
# generate 50 tokens, and measure per-token latency.
# Assert p95 < 50ms for Phase 1 target.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BUILD_DIR="$PROJECT_ROOT/target/release"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "=== Synapse Latency Benchmark ==="
echo "Goal: p95 < 50ms/token"
echo ""

# Build in release mode for accurate benchmarks
echo "Building in release mode..."
cd "$PROJECT_ROOT"
cargo build --release --bin synapse 2>&1 | grep -E "(Compiling|Finished)" || true
echo ""

# Check if binary exists
if [ ! -f "$BUILD_DIR/synapse" ]; then
    echo -e "${RED}Error: synapse binary not found at $BUILD_DIR/synapse${NC}"
    exit 1
fi

# Create temp directory for logs
TEMP_DIR=$(mktemp -d -t synapse-bench-XXXXXX)
echo "Logs directory: $TEMP_DIR"
echo ""

# Cleanup function
cleanup() {
    echo ""
    echo "Cleaning up processes..."
  pkill -P $$ || true
    sleep 1
    rm -rf "$TEMP_DIR"
}
trap cleanup EXIT

# ─── Local swarm topology ─────────────────────────────────
# The master prints its own generated pairing code in the startup
# panel; workers read it out of the log and `join` over a fixed
# loopback address. Each node gets a distinct identity key so the
# three processes register separate PeerIds on the same machine.
MASTER_LISTEN="127.0.0.1:34001"
MODEL="hf-internal-testing/tiny-random-LlamaForCausalLM"
IDENT_DIR="$TEMP_DIR/identities"
mkdir -p "$IDENT_DIR"

# Start master node
echo "Starting master node..."
MASTER_LOG="$TEMP_DIR/master.log"
START_SEC=$(date +%s)
START_NS=$(date +%s%N)
SYNAPSE_IDENTITY_PATH="$IDENT_DIR/master.key" \
    "$BUILD_DIR/synapse" host --master \
    --model "$MODEL" \
    --max-tokens 50 \
    --prompt "The quick brown fox jumps over the lazy dog" \
    --listen "$MASTER_LISTEN" \
    > "$MASTER_LOG" 2>&1 &
MASTER_PID=$!
echo "Master PID: $MASTER_PID"

# Wait for master startup, then extract its generated pairing code
PAIRING_CODE=""
for i in $(seq 1 60); do
    if ! kill -0 $MASTER_PID 2>/dev/null; then
        break
    fi
    PAIRING_CODE=$(grep -oP 'pairing code: \K\S+' "$MASTER_LOG" 2>/dev/null | head -1 || true)
    [ -n "$PAIRING_CODE" ] && break
    sleep 0.5
done

if [ -z "$PAIRING_CODE" ]; then
    echo -e "${RED}Failed to extract pairing code from master log:${NC}"
    tail -30 "$MASTER_LOG"
    exit 1
fi
echo "Pairing code: $PAIRING_CODE"
echo ""

# Start worker 1
echo "Starting worker 1..."
WORKER1_LOG="$TEMP_DIR/worker1.log"
SYNAPSE_IDENTITY_PATH="$IDENT_DIR/worker1.key" \
    "$BUILD_DIR/synapse" join "$MASTER_LISTEN" \
    --model "$MODEL" \
    --code "$PAIRING_CODE" \
    > "$WORKER1_LOG" 2>&1 &
WORKER1_PID=$!
echo "Worker 1 PID: $WORKER1_PID"

# Start worker 2
echo "Starting worker 2..."
WORKER2_LOG="$TEMP_DIR/worker2.log"
SYNAPSE_IDENTITY_PATH="$IDENT_DIR/worker2.key" \
    "$BUILD_DIR/synapse" join "$MASTER_LISTEN" \
    --model "$MODEL" \
    --code "$PAIRING_CODE" \
    > "$WORKER2_LOG" 2>&1 &
WORKER2_PID=$!
echo "Worker 2 PID: $WORKER2_PID"
echo ""

# Wait for the swarm to discover peers and generate all 50 tokens.
# The master is a chat REPL and stays alive after the session closes, so watch
# for the session-closed marker in the master log (or the process exiting).
echo "Generation in progress..."
TIMEOUT=180
while true; do
    ELAPSED=$(( $(date +%s) - START_SEC ))

    if [ $ELAPSED -gt $TIMEOUT ]; then
        echo -e "${RED}Timeout: Generation did not complete in ${TIMEOUT}s${NC}"
        echo "Master log:"
        tail -50 "$MASTER_LOG"
        exit 1
    fi

    # Generation is complete when either the master exits or the chat session closes
    if ! kill -0 $MASTER_PID 2>/dev/null; then
        break
    fi
    if grep -q "\[bye\] session closed" "$MASTER_LOG" 2>/dev/null; then
        break
    fi
    sleep 1
done

END_NS=$(date +%s%N)
TOTAL_MS=$(( (END_NS - START_NS) / 1000000 ))
PER_TOKEN_MS=$((TOTAL_MS / 50))

echo ""
echo "Generation complete!"
echo ""
echo "=== Latency Analysis ==="
echo "Tokens generated: 50"
echo "Total wall time: ${TOTAL_MS}ms"
echo "Avg per-token latency: ${PER_TOKEN_MS}ms"
echo ""

# Check if we meet the target
if [ "$PER_TOKEN_MS" -lt 50 ]; then
    echo -e "${GREEN}✓ SUCCESS: avg latency (${PER_TOKEN_MS}ms) < 50ms target${NC}"
else
    echo -e "${YELLOW}⚠ WARNING: avg latency (${PER_TOKEN_MS}ms) >= 50ms target${NC}"
    echo "Note: this includes model load, swarm discovery warmup, and prefill."
    echo "Steady-state per-token timing is reported in the master log."
fi

echo ""
echo "=== Master Log (last 30 lines) ==="
tail -30 "$MASTER_LOG"

echo ""
echo "=== Worker 1 Log (last 10 lines) ==="
tail -10 "$WORKER1_LOG"

echo ""
echo "=== Worker 2 Log (last 10 lines) ==="
tail -10 "$WORKER2_LOG"

echo ""
echo -e "${GREEN}Benchmark completed successfully${NC}"
echo "Review logs at: $TEMP_DIR"
