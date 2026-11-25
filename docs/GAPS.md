# Remaining gaps and evidence boundaries

This register distinguishes software work from requirements that cannot be completed by a source change alone. It accompanies the master plan and the machine-readable [gate record](evidence/gates.json). No gate is marked accepted by this file. Update evidence at the actual integrated revision, and preserve failed or blocked runs.

## Required external evidence

| Gap | Why local code/tests are insufficient | Concrete next evidence |
| --- | --- | --- |
| Physical memory benefit | Two loopback processes share one physical host and OS page cache | Two consenting machines, resident-range traces, real memory measurements and correct output beyond each configured individual budget |
| Separate-network connectivity | Simulated delay and a local relay do not exercise real NAT/router/ISP conditions | Actual direct/relay runs across separate routers, bounded failure tests, coarse consented topology |
| Model/hardware compatibility | A tiny fixture does not certify a useful model, quantized format or accelerator | Exact approved model revision/template/backend, numerical parity and actual target device |
| License ownership | Root MIT and package Apache-2.0 metadata disagree | Authorized maintainer rights decision and consistent metadata before release |
| Signed release trust | A locally generated key only proves possession of that test key | Identified signing-key owner, independently distributed trusted public key and revocation/rotation procedure |
| Pilot economics and operation | Simulated credits cannot pay relay/DB/electricity costs or establish demand | Approved cash budget, actual workload/capacity measurements and accountable operator |
| Community integrity/privacy | Signed claims and encryption do not prove correct computation or hide prompts from workers | Independent review, approved hosts, explicit consent, durable accounting and operational drills |
| Independent usability | Developer-run tests cannot establish a fresh user's experience | A separate tester installs, connects, runs, cancels and removes the supported package |

## Implementation details that require explicit contracts

- **Accepted output and cancellation:** bind authoritative output intervals to a request/epoch; specify which event establishes delivery and where cancellation stops further chargeable work. Retried computation must not appear or settle twice.
- **Recovery and persistence:** a restarted coordinator cannot safely reconstruct a session without authorized token history. Fail explicitly when that state is unavailable; do not persist prompts by default merely to claim recovery.
- **Trust roots:** model integrity hashes are not publisher approval. Key enrollment, distribution, revocation and model rights remain distinct from a manifest digest.
- **Actual enforcement:** expose which CPU/RAM/GPU/network limits the current backend and OS enforce. Configuration fields and estimated metrics are not proof of isolation.
- **Compatibility:** every binary protocol change requires negotiated version and golden vectors. JSON fixture schemas describe data shapes, not the bincode wire format.
- **Live status:** labels such as connected, ready, paused or verified must come from actual daemon/ledger state. The public landing explorer remains a clearly labeled local simulation.

## Product verification scope

The static site has deterministic resource-model tests, offline semantic/link/contrast checks and browser QA. Browser iframe viewport checks exercise mobile/tablet CSS layouts; they do not establish physical-phone, touch or screen-reader usability. Reduced-motion CSS is checked; the QA report distinguishes that from OS preference emulation.

The local application has a separate QA owner and must exercise real authentication, same-origin protections, error states, streaming cancellation, resource controls and keyboard flow against the integrated backend. Landing-page checks do not certify the application.

## Rule for closing a gap

Attach revision, environment, exact command/run, actual result, limitations and reviewer where required. Use `loopback`, `physical_lan`, `physical_wan` or `simulated_wan` explicitly. Keep implementation, local test, accepted and released states separate. No fabricated speed, users, geography, earnings or independent approval.
