# Private local and LAN validation

The automated harness starts real processes with distinct private identities,
waits for authenticated membership, runs a verified model, checks every output
token against a separate full-model process, and checks the live layer route.
It is a loopback experiment on one physical machine. It does not satisfy the
two-physical-device or WAN acceptance gates.

## Reproduce the offline fixture

```sh
cargo build --locked -p synapse-cli
cargo run --locked -p synapse-cli -- doctor --model-dir crates/synapse-core/tests/fixtures/tiny-llama
python3 scripts/local_cluster.py --workers 1 --transport quic --host-memory-bytes 52000 --worker-memory-bytes 45000
python3 scripts/local_cluster.py --workers 2 --coordinator-only --transport tcp --host-memory-bytes 16000 --worker-memory-bytes 45000
```

The fixture is original deterministic test data, not a useful language model.
It requires no Hub download. `--binary` selects a separately built executable;
`CARGO_TARGET_DIR` is respected. `--model-dir`, `--prompt`, and `--tokens` allow
an explicitly chosen compatible real model to use the same validation path.
For such a model, choose a raw prompt that matches its template, and retain
its full manifest and license evidence.

Without constrained budgets, the production planner may correctly choose a local
full-model route and the distributed-only harness rejects that run. The limits
above are calibrated for this exact fixture and eight output tokens. They are
inference reservations, not process RSS limits.

Each run creates a new private directory under `artifacts/`. It retains source
commit/dirty state, executable hash, complete model manifest, transport, actual
token IDs/counts/timestamps, node routes, resident shard tensor bytes, and sampled
Linux process memory. `VmHWM` is sampled while the process is alive; the result is
an observed lower bound, not a final process peak. Missing measurements remain
unknown. Identity files are temporary; pairing secrets and the supplied prompt
are redacted from retained logs. The harness terminates only its own processes.
Every timeout, early exit, missing route, missing residency, token mismatch or
uncompleted generation produces a nonzero exit and a retained failure record.

## Two physical computers

Use matching source builds and verified model directories on two consenting
devices. Run `synapse doctor --model-dir /absolute/model/path` on each first.
Choose a host interface/address reachable by the worker. On the host:

```sh
synapse host --master --model /absolute/model/path --listen 0.0.0.0:4001 \
  --prompt '<s> </s> Hello' --raw-prompt --max-tokens 8 --once \
  --min-workers 1 --timeout-secs 120 --events-jsonl host-events.jsonl
```

Copy the actual host Peer ID and pairing code through a trusted private channel.
On the worker, substitute the real host LAN address and displayed values:

```sh
synapse join /ip4/HOST_LAN_IP/udp/4001/quic-v1/p2p/HOST_PEER_ID \
  --model /absolute/model/path --code HOST_PAIRING_CODE \
  --events-jsonl worker-events.jsonl
```

Pairing is mutual, model-bound, and does not rely on trusting multicast or DHT
advertisements. An explicit peer address works when multicast is unavailable.
Discovery defaults to disabled; enabling discovery also requires a peer allowlist.
Do not publish pairing codes, identity files, prompts, or private addresses in a
public evidence report. `host --once` returns an error for failed generation.

For TCP, use a host listen multiaddr `/ip4/0.0.0.0/tcp/4001` and the corresponding
worker address `/ip4/HOST_LAN_IP/tcp/4001/p2p/HOST_PEER_ID`. Check firewall rules
for the actual chosen transport. A UDP-only rule cannot enable TCP fallback.
Wrong identity, pairing failure, model mismatch, exhausted memory, and missing
coverage must be resolved rather than treated as a successful connection.

Record both hardware/OS descriptions, independent process identities, identical
manifest digests, exact assigned ranges, complete output equality to `infer`,
and the raw timing files. Also interrupt a worker mid-generation and verify
bounded failure, no duplicate output, release of reservations, and a clean next
request. A physical-device run has not been performed in the current workspace.
