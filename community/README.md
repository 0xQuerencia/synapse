# Synapse community testnet (experimental source candidate)

This additive Python service demonstrates the second half of the Synapse roadmap in an isolated environment. **It is a synthetic testnet, not a hosted AI service.** It intentionally refuses all model identities except its original four-stage synthetic fixture. The native Rust runtime remains independent. The service does not send prompts or activations to strangers and does not implement real credits, public registration, open swarm admission, cryptocurrency, or production payments.

## Quick start

Linux with Python 3.11+ and `cryptography` (tested on Python 3.13 and cryptography 46.0.4); a local filesystem with SQLite and private directory permissions. Run from the repository root. Install requirements into a virtual environment on your own machine if needed. No production credentials are needed.

```sh
python3 -m pip install -r community/requirements.txt
export PYTHONPATH="$PWD/community"
community/verify.sh
```

A self-contained test runs the full synthetic HTTP exchange and destroys the temporary credentials and DB after exit:

```sh
python3 -m synapse_testnet.demo
python3 -m synapse_testnet.demo --full-model
```

The full-model test includes earned-credit redemption by one worker on another worker, not a self-request. Both use fake credits. No actual transformer is invoked by these tests.

## Live local testnet

Pick a new private directory, e.g. `TESTNET=$(mktemp -d)`. Run the following with the same `PYTHONPATH` and directory on one machine. The init command creates an isolated ledger and test identities and explicitly enables only the synthetic pool. Worker devices begin stopped; starting a worker explicitly proves its key and makes it eligible.

```sh
python3 -m synapse_testnet.pilot --root "$TESTNET" init
python3 -m synapse_testnet.http_api --db "$TESTNET/ledger.db" --port 8787 --testnet
```

In three separate terminals:

```sh
python3 -m synapse_testnet.pilot --root "$TESTNET" worker worker-a
python3 -m synapse_testnet.pilot --root "$TESTNET" worker worker-b
python3 -m synapse_testnet.pilot --root "$TESTNET" verifier
```

Then issue a synthetic request:

```sh
python3 -m synapse_testnet.pilot --root "$TESTNET" request
python3 -m synapse_testnet.pilot --root "$TESTNET" balance worker-a
python3 -m synapse_testnet.pilot --root "$TESTNET" control worker-a pause
python3 -m synapse_testnet.pilot --root "$TESTNET" control worker-a resume
```

The dashboard is served at `http://127.0.0.1:8787/`. Read the local `client.json` credential file on your own machine and enter its token in the local dashboard; do not paste credentials into chat, URLs, shell command arguments, screenshots, or shared logs. The token stays in page memory only. The dashboard has live ledger balance, quote, reserve, job status, settle/cancel, history, and worker statistics. The browser QA harness is included; browser rendering in the authoring environment was blocked by administrator policy. Static HTTP, same-origin and JS checks passed. The CLI process harness passed end to end.

Stop workers with Ctrl+C; the worker requests stop and refunds interrupted jobs. Expired heartbeats and job deadlines refund on the server. Do not expose this HTTP listener, token files, or database through public port forwarding or a tunnel. No cross-device TLS or hosted authentication is provided.

## Accounting and trust model

One test credit = 1,000,000 integer microcredits. Initial grants are transfers from a capped treasury. A quote fixes tariff, input units, maximum output and expiry. Reservation moves funds to escrow before assignment. Only approved live devices with complete range coverage and capacity qualify; self-dealing across the same member is excluded. At most four stages are selected, preferring a full-model worker. Worker receipts bind exact model, job, stage, epoch, units and evidence; Ed25519 proves device attribution. The distinct verifier role independently recomputes the built-in synthetic challenge, and the service also recomputes it before accepting. There is no general proof of remote LLM computation. A job settles only once after all stages verify the same output prefix; deterministic largest-remainder splitting conserves the exact charge. Cancelled, rejected or expired test jobs receive a complete refund, not paid partial work. Disputes use a separate reviewer and a capped treasury transfer; test provisioning spends the full treasury on the requester, so compensation is denied until a different funded test configuration exists. No idle earnings or transferable credits.

SQLite WAL with synchronous FULL and BEGIN IMMEDIATE serializes all mutations. Journal and entry update/delete are blocked; each transaction is checked for zero-sum, projection consistency, pool isolation, supply conservation and escrow consistency. Database write access is reserved to the local operator. This is not a hardened hosted database role boundary. PostgreSQL migration and psycopg adapter are supplied as an **untested candidate**; no PostgreSQL server or driver was available here. Do not enable hosted settlement from this code.

## Recovery and simulation

```sh
python3 -m synapse_testnet.ops check "$TESTNET/ledger.db"
python3 -m synapse_testnet.ops backup "$TESTNET/ledger.db" /path/to/new/private/backup
python3 -m synapse_testnet.ops restore /path/to/new/private/backup /path/to/new/private/ledger.db
python3 -m synapse_testnet.economy --seed 42
```

Backups contain test member/device identifiers, token hashes and journal metadata, never real prompts. Keep them private. Restored snapshots must not be treated as simultaneous active copies of the same ledger. The simulator models fictional arrivals, supply, churn, fraud and grant exhaustion; it is not measured demand or actual earnings. Supply conservation is asserted on every scenario transition.

## Native runtime and artifact work

`native_probe.py` calls the existing authenticated local Rust chat API using an original fixture prompt, validates actual usage and produces hashes and timing without awarding credits. Its contract tests used a local HTTP double; **live Rust execution was not possible in this environment**. Once the previous native checkout is restored and built, follow its `docs/quickstart.md` and run:

```sh
python3 -m synapse_testnet.native_probe --port 8080 --token-file /absolute/private/synapse-gui-token --model synapse-fixture
```

`artifacts.py` implements explicitly authorized signed synthetic chunk manifests, SHA-256 addressing, authenticated loopback seeders, bounded egress, resume, hash checks, local cache quotas, multiple configured peers, and rejection of tampered/expired/revoked manifests. Its isolated format is not the native Rust catalog format. Do not use it to redistribute third-party model files or expose it publicly; native integration and licensed model delivery remain open.

See [the stage-two status and review](../docs/community/STAGE2.md), [policy draft](../docs/community/POLICY.md) and [release gates](../docs/community/gates.json). The production enablement check deliberately fails:

```sh
python3 community/tools/release_gate.py --assert-release
```
