# Measurement method

`scripts/local_cluster.py` creates raw events from actual independent full-model,
host, and worker processes. `scripts/benchmark_report.py RUN_DIR... --output
report.json --csv report.csv` aggregates those preserved runs. The convenience
wrapper `scripts/benchmark_latency.sh` runs the same experiment. There are no
assumed token counts, inferred successes, or speed targets.

Time to first token starts at `request_started`; startup/model loading is a
separate concern. Inter-token latency uses successive actual token events,
including tokens whose text decoder emits an empty string. End-to-end latency
ends at the terminal generation event. Percentiles use the nearest-rank method
(`ceil(p × N)` in sorted samples), and missing samples are JSON `null`.
Completion counts and consecutive output positions are validated before timing
is accepted. Failed runs remain in the report denominator and raw artifacts.

Sampled Linux `/proc` VmHWM/RSS values are explicitly an observed lower bound;
they are not final process peaks. Resident shard tensor bytes, inference memory
reservations, virtual address space limits, and operating-system RSS measure
different quantities. A tensor-memory reservation does not cap total process
RSS. Use a process/container memory controller on a physical test host when
reporting whole-process constrained-memory acceptance.

## Actual API load

Start the GUI/API with a verified model and private token file, then run:

```sh
scripts/stress_test.sh --base-url http://127.0.0.1:3000 \
  --token-file /absolute/private/gui-token --model synapse-local \
  --requests 20 --concurrency 4 --output artifacts/api-load.json
```

The load test sends real `/v1/chat/completions` requests. It validates completed
responses and usage counts, records each status and latency, and reports busy
HTTP 429 responses separately. Only successfully completed requests contribute
to completion throughput and latency percentiles. A run with no completed
request or any unexpected failure exits unsuccessfully. Secrets and response
content are omitted from the report; the endpoint and model are retained.
The API currently admits one generation at a time, so 429s under concurrency
are expected and are never counted as successful inference.

All measurements from this workspace are loopback evidence. Hardware, model
revision, executable hash, source state, transport, and raw run files must
accompany any claimed performance comparison. Physical LAN and separate-network
WAN acceptance remain unperformed until independently documented.
