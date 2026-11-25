# Stage 3: local qualification and operational follow-ups

The first 44 roadmap slices are merged in `main` at `a127c8907113862379e597a84ec4cc16ee98bd5a`. The following review stack extends that baseline; its individual pull requests are drafts until their checks and dependencies qualify. The earlier stage records describe the historical baseline, not the current PR stack.

| PR | Scope | Acceptance boundary |
| --- | --- | --- |
| [46](https://github.com/0xQuerencia/synapse/pull/46) | PostgreSQL qualification | Disposable PostgreSQL 16 restricted role and isolated schema; no hosted deployment. [Qualification run](https://github.com/0xQuerencia/synapse/actions/runs/45712144388). |
| [47](https://github.com/0xQuerencia/synapse/pull/47) | Native full-model fake-credit experiment | Known-answer original fixture and trusted local process verifier; no arbitrary remote model verification. |
| [48](https://github.com/0xQuerencia/synapse/pull/48) | Pin-aware cache eviction, verified artifact export and native fixture delivery | Original fixture only, authenticated loopback transport; no third-party distribution rights or multi-host transfer claim. |
| [49](https://github.com/0xQuerencia/synapse/pull/49) | Recipient-key-bound enrollment and rotation | Local membership tokens, not hosted identity or native peer invitations. |
| [50](https://github.com/0xQuerencia/synapse/pull/50) | Sharded fixture accounting | Trusted operator-controlled two-worker harness, fake credits, exact split, failure refund. Native process CI must establish the experiment; signatures are not independent remote attestations. |
| This slice | Recoverable event export, opt-in ranking, dispute interface | Local operator sink; no external integrations, public ranking, payment or hosted operations. |

## Operator event export

The operator-only HTTP endpoints expose bounded pending batches, reconciliation status and fingerprint-bound idempotent acknowledgements. The exporter commits each event to a private local SQLite sink before acknowledging the ledger. Restart after a committed insert but before acknowledgement repeats the delivery safely. Concurrent exporters targeting the **same sink** deduplicate by event ID and reject mismatched payloads. This is not exactly-once arbitrary side-effect delivery, fan-out, multi-sink synchronization, or a replacement for ledger backups. Retain the sink with the same ledger; restoring an older sink requires operator reconciliation. The service never returns credentials or prompts in outbox events.

```sh
PYTHONPATH=community python3 -m synapse_testnet.outbox \
  --port 8787 --operator-file /private/testnet/operator.json \
  --sink /private/testnet/events.sqlite --limit 50
```

Use the existing private operator credential file (adjust the example paths to the local testnet). Rankings are disabled by default. A worker can opt in under a lowercase alias and remove it later; only settled verified job counts and chosen aliases appear to consenting members of the same enabled pool. Do not choose an alias containing identifying information. The dispute table is owner-scoped; disputes still require independent verifier review and operator authorization before compensation. Compensation is fake credits only.

## Still required for acceptance

- Physical multi-machine useful-model measurements, separate-router/ISP connectivity and security checks.
- Native GPU execution and actual macOS/Windows/mobile builds and installation testing.
- Native worker-signed and independently verified remote receipts, secure recovery/replay across machines, and model redistribution rights.
- Hosted identity, deployment and backup qualification, production budgets and independent abuse/security review.
- Release ownership and licensing decisions, publisher signing keys and signed packages; open swarm remains research, not a release claim.

No gate is accepted by this note. Existing release guards and fake-credit-only API behavior remain unchanged.
