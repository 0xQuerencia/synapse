# Transport experiments and WAN acceptance

## What the automated matrix measures

```sh
cargo build --locked -p synapse-cli
python3 scripts/wan_matrix.py --output artifacts/transport-matrix
```

The matrix runs real host/worker inference over encrypted TCP on one physical
computer. It includes an unmodified baseline, per-chunk bidirectional delay and
jitter, a bandwidth ceiling, and a worker terminated after output has begun.
Every case retains the exact executable hash, model manifest, token events,
route, timings, error and process cleanup evidence from `local_cluster.py`.
Fault cases pass only if the host reports an explicit generation error within
the deadline and emitted output positions have no duplicates or gaps. A failed
inference is always recorded as failed inference, even when expected by a fault
test. The matrix returns nonzero if any expected property is absent.

The proxy operates on TCP read chunks. Its configured delay is not an observed
network RTT, and it does not simulate IP packet loss, NAT, Wi-Fi or geography.
It forwards encrypted bytes without terminating Synapse's transport identity.
Actual runtime ping and matched activation measurements can be inspected in
node status. Missing bandwidth/calibration stays unknown; the route planner
does not invent WAN rates. Proxy-delivered byte counts are a transport total,
not model payload bytes or proof of a billing quantity.

## Required physical test matrix

| Case | Required setup | Evidence to retain | Current acceptance |
| --- | --- | --- | --- |
| Wired LAN | Two independently identified physical computers | Both hardware/OS profiles, route, memory, exact reference parity | Not performed |
| Wi-Fi LAN | Same devices, Wi-Fi link with signal/channel recorded | Actual RTT distribution, dropped connections, throughput and parity | Not performed |
| Direct WAN | Consenting devices on separate networks, direct pinned endpoint | Public/private address classification, NAT/firewall setup, RTT/loss and parity | Not performed |
| Relayed WAN | Separate networks and an explicitly controlled relay | Relay identity, reservation limits, end-to-end peer authentication and parity | Not performed |
| Inter-city | Two physical sites with independently recorded location/distance | Device ownership consent, route type, actual dates and measurements | Not performed |
| Packet impairment | Isolated test interface or namespace under the operator's control | Exact netem settings, interface, counters, RTT/loss and cleanup | Not performed |

No loopback or proxy run can change the acceptance column above. Do not label
a local run as a 1,000 km result. Physical testers must run the commands in
`LOCAL_CLUSTER.md`, use the identical immutable model manifest on both sides,
and retain the full local reference and distributed output records.

## Network impairment on a controlled test host

An operator with permission over an isolated interface can apply Linux netem
latency/loss/rate profiles. Record the existing qdisc and ensure a separate
management path before changing it. Example profiles to evaluate, not measured
results: 20/100/200 ms RTT, 0/0.1/1% packet loss, and 10/50/100 Mbit/s. Apply half
the target delay to each direction and measure actual RTT; queues and device
conditions make nominal settings insufficient evidence. Preserve qdisc output
and restore the original interface configuration when finished. The local
matrix never modifies host network settings or requires elevated privileges.

Test worker loss during preparation, prefill and decode; wrong peer identity;
revoked membership; full capacity; stale readiness; and a slow/disconnected
consumer. Current recovery guarantees are bounded explicit failure and clean
resource release. Cross-worker resumption is not an accepted capability.

## Capacity interpretation

A route must cover every transformer layer exactly once, and all workers must
acknowledge a model/session/epoch-bound assignment before inference starts.
The coordinator also budgets embeddings/output projection. KV reservation grows
with requested context and grouped-query dimensions; working tensors and load
scratch add further headroom. Process RSS and file-backed virtual mappings are
separate measurements. Use OS-enforced process/container memory limits in the
physical constrained-memory acceptance run, in addition to Synapse's inference
reservation. Actual measurements, not CPU count or advertised TFLOPS, must
support a performance conclusion.
