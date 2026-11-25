# Draft policy, security and operating boundaries (PR-32, PR-38, PR-54)

**Not a binding terms document or legal sign-off. No production service is approved.** The executable stage is limited to an opt-in, loopback-only synthetic testnet. No user prompts, LLM activations, arbitrary jobs, public enrollment, credits with monetary value, or third-party model redistribution are authorized. Private inference remains outside accounting and should not silently fall back to community compute.

## Data register

| Data | Testnet behavior | Production gate |
|---|---|---|
| Device/member ID and Ed25519 public key | Operator provisions; scoped role; device proof, heartbeat and revocation | Established identity provider, owner/device linking and deletion/export process |
| Test access token | Generated random, hash in DB, plain token only in protected local role file or volatile page memory | Short-lived OIDC-backed tokens, keychain, recovery and rotation, independent auth review |
| Request content | No actual prompt or activation accepted by the testnet; only fixture ID and bounded synthetic units | Explicit trust and consent, no raw prompts in accounting, no sensitive external prompts; approved retention and isolation |
| Credits, quote, stage, journal and receipt | Integer fake balances, immutable accounting and signed synthetic evidence; scoped history | Reviewed terms, funded caps, audited tariff, retention/deletion policy, independent anti-fraud review |
| Operations and backup | Local hashes, health, events, metadata-only backup; no credential or prompt logging | Named operator, protected storage, tested recovery, retention and deletion, incident handling |
| Artifact cache | Explicit publisher keys, original synthetic bytes and digest-addressed chunks | Signed native catalog integration, verified redistribution rights, quotas, peer security and revocation |

## Threats and stop conditions

Transport authentication and signatures do not make remote model inference confidential from workers or prove arbitrary results. Collusion, Sybil identities, compromised devices and activation reconstruction are unresolved production threats. The synthetic verifier is not a model audit. The local HTTP service rejects foreign origins/hosts, unapproved roles, malformed scopes, oversized messages and replayed receipts; it is not public-grade network security. Stop admission and refund active jobs on an operator pool suspension, revoked device/model or failed synthetic validation. Accounting invariant failure stops mutations; isolate the database, preserve evidence privately, investigate and reconcile before restarting. Do not expose database writers, use a public tunnel or deploy this service on the internet.

## Required owner and independent approvals

Before any real pilot: name the operator and legal entity; confirm root license and model rights; configure identity with role/tenant isolation; approve consent, privacy/retention/export/deletion, acceptable use and contributor terms; determine participant eligibility and geographic scope; approve operating cash/egress budget, tariff, grant caps, refund and dispute policy; establish incident/disclosure/support contact; approve signed native releases and key rotation; run independent security/privacy/economic review. Real PostgreSQL crash/recovery, native execution and physical WAN evidence are mandatory. No public swarm or browser/mobile compute until separately validated. No promise of cash, profit, free unlimited AI, or guaranteed capacity.

## Unresolved operational choices

SQLite is used as a constrained local reference because PostgreSQL was unavailable; the proposed production transaction substrate remains PostgreSQL. The testnet uses manually provisioned random bearer credentials instead of hosted OIDC; it is not a replacement for the latter. The service uses Python to avoid changing the unbuildable-in-this-session Rust checkout. It is a separate experimental control-plane candidate, not a rewrite of inference. Final framework, operator, pricing, deployment and identity provider are owner decisions. None has been silently approved.
