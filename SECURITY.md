# Security and trust

Synapse is an experimental inference runtime. Use trusted devices and synthetic or otherwise appropriate data during development. Do not expose raw worker/admin interfaces or the local GUI publicly.

## Trust boundaries

Local computation runs on your own machine. Private clusters add explicitly trusted devices. A remote full-model host may see the prompt; a layer host may infer information from activations. Encrypted transport does not hide computation from its endpoint. Public untrusted participation and community spending require separate gates and are not established by this repository's code presence.

Workers must accept narrowly defined model operations, verified artifacts and authorized assignments. They must not execute arbitrary submitted programs, model repository code or shell commands. Cap messages, memory, context, disk and concurrency, and keep pause/stop available without a hosted control service.

## Reporting a vulnerability

Use the repository's **Security → Report a vulnerability** channel when private vulnerability reporting is enabled. If that channel is unavailable, first open an issue asking for a private contact without including exploit details, private data or credentials. No security email address or response-time guarantee is invented here.

Include affected revision, component, minimal synthetic reproduction and impact. Do not post real prompts, activation contents, private keys, invitations, access tokens or personal network identifiers. Coordinated reports should identify any mitigation and affected trust mode.

## Release requirements

A public/private WAN release requires the project's threat model, protocol limits, model provenance, authorization tests and review. A community pilot additionally requires independently reviewed accounting, privacy/consent, worker isolation, budget, backup/reconciliation and incident ownership. Critical unresolved findings block the affected feature. See [master plan](docs/MASTER_PLAN.md) and [status](docs/IMPLEMENTATION_STATUS.md).
