# Private transport evidence

Work package: PR-15. This is local transport evidence, not a WAN release approval.

Command:

```sh
cargo test --locked -p synapse-core --test private_transport --no-default-features
```

Observed on Linux CPU with Rust 1.94.0: **3 passed, 0 failed**.

- TCP + Noise connection authenticates the configured target PeerId.
- A three-swarm Circuit Relay v2 connection retains the worker's end-to-end Noise identity; a Ping traverses the established circuit.
- The same TCP endpoint with a wrong pinned PeerId is rejected.

Each swarm uses a distinct generated identity and an ephemeral loopback TCP port. Relay reservation must advertise a validated reachable external address; merely listening does not establish one. The test explicitly registers its loopback address because every participant is local.

`RelayLimits` feeds concrete reservation, circuit, byte and duration limits into libp2p. `connection_limits` provides a behaviour enforcing total, per-peer and pending connection caps; it must be composed into the runtime behaviour, as it is in the transport tests. `circuit_address` rejects unpinned or nested relay routes. Relay ranking treats an absent RTT sample as unknown; enqueueing a dial is never a latency measurement.

## Reservation capacity regression (PR-44)

The targeted test below passed on the same Linux CPU environment with Rust 1.94.0:

```sh
cargo test --locked -p synapse-core --test private_transport relay_reservation_capacity_refuses_extra_peer_without_evicting_existing_one
```

Observed: **1 passed, 0 failed, 3 filtered out**. A real loopback relay with a one-reservation limit accepts the first independent Noise peer, then refuses the second with `ResourceLimitExceeded`. The second client receives a failed reservation event; the first remains connected and retains its circuit listener. The test bounds the entire exchange to 20 seconds. This covers reservation exhaustion, without asserting unmeasured circuit byte or duration behavior.

Remaining release gates: connect actual private devices on separate networks, UDP/CGNAT constraints, packet loss, firewall and multi-relay failure tests, an operator-approved relay configuration and measured egress budget. These local checks do not establish those results.
