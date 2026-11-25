# ADR 0001: preserve the runtime and separate implementation from acceptance

Status: accepted for this implementation work; release gates remain open.

Keep Rust, Candle and libp2p as the starting architecture. Prioritize a reproducible local path, exact layer placement, request-scoped state and trusted private connections. A full-model remote route and a layer-sharded route are complementary choices. Private use must not depend on community accounts or credits.

The owner has authorized building the plan. That authorizes reversible code, documents and local tests, not invented evidence, production deployment, a license change or monetary features. Work-package IDs label changes; acceptance requires the real evidence in the plan. Later features stay unavailable until their checks and user consent are satisfied.

Local-only must never silently become community inference. Computing peers may observe prompts/activations; encryption in transit is not worker confidentiality. Community credits remain noncash usage accounting and are not required for private mode.

Alternative rejected: replace the existing runtime with unrelated frameworks before measuring correctness. Alternative deferred: permissionless swarm, browser workers, token trading and speculative performance optimizations.
