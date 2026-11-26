# Low-memory Rust development on Linux

Use the opt-in runner with the pinned Rust toolchain and Python 3.9 or later. Run scoped commands serially to limit the number of compilers active on the machine.

```sh
python3 scripts/low_memory.py check -p synapse-core
python3 scripts/low_memory.py build --workspace
python3 scripts/low_memory.py test --workspace --lib network::node::
```

The runner uses locked dependencies, one Cargo job, disabled incremental compilation, and no dev/test debug information. Commands share `target/low-memory` in this checkout. Reuse it for the same toolchain and compatible settings. Keep package selection consistent when moving from a workspace build to a filtered test. Feature changes can cause Cargo to rebuild dependencies in the same target directory. `cargo check` produces metadata and can still leave code generation for a subsequent build. A single Cargo job can run a multithreaded compiler or linker. The settings provide no hard memory ceiling.

A JSON record on stderr reports elapsed time, Cargo status, Linux `wait4` maximum RSS, and the sampled sum of RSS across the owned Cargo process group. `wait4` covers Cargo and the maximum of descendants reaped through it. The group sum can double-count shared pages and miss spikes shorter than the 100 ms sample interval. These measurements exclude other workloads and filesystem cache. The runner reports and fails on a process group left behind by Cargo, even when cleanup succeeds. Ctrl+C, SIGTERM and SIGHUP are forwarded to the owned group. The runner escalates to SIGKILL after five seconds if needed and checks for survivors before exiting. Detached processes, supervisor SIGKILL and host termination fall outside this contract.

After building the GUI once, profile a persistent GUI process through repeated loopback host start/stop cycles with the original tiny fixture. Each cycle also checks JSON and streaming inference through the local gateway and verifies that the host returns to idle.

```sh
python3 scripts/runtime_memory.py --binary target/low-memory/debug/synapse-gui --cycles 32
```

The runtime report includes Linux `VmHWM`, post-stop RSS and anonymous resident memory, file descriptor and thread counts, and process-group cleanup. A short rise followed by stable idle measurements is consistent with warm-up or allocator retention. RSS alone cannot attribute retained pages to the allocator or establish a leak. Continued growth after identical, fully completed cycles warrants a longer run and allocation or ownership evidence. Compare runtime measurements separately from compiler peaks.
