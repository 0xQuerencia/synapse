#!/usr/bin/env python3
"""Exercise actual private inference processes; preserve evidence on every exit.

This is a loopback test, never evidence of two physical computers or WAN reach.
Only processes started by this invocation are terminated. Pairing secrets and
private identity files are removed before the artifact directory is reported.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_events(path: Path) -> list[dict]:
    if not path.exists():
        return []
    events = []
    for line in path.read_text().splitlines():
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            # An active writer may have not completed its last line yet.
            continue
    return events


def source_metadata() -> dict:
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT))
    return {"commit": commit, "working_tree_dirty": dirty}


def observed_memory(pid: int) -> dict | None:
    try:
        values = {}
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith(("VmRSS:", "VmHWM:")):
                key, amount, unit = line.split()
                if unit == "kB":
                    values[key.rstrip(":")] = int(amount) * 1024
        return values or None
    except (OSError, ValueError):
        return None


def verify_generation(events: list[dict]) -> list[int]:
    tokens = [event for event in events if event["event"] == "token"]
    done = [event["data"] for event in events if event["event"] == "generation_done"]
    if len(done) != 1 or done[0]["reason"] not in ("length", "eos"):
        raise RuntimeError("generation did not complete successfully exactly once")
    count = done[0]["tokens_generated"]
    if not count or len(tokens) != count:
        raise RuntimeError("actual token events do not match reported output count")
    if [event["data"]["position"] for event in tokens] != list(range(count)):
        raise RuntimeError("output token positions contain a duplicate or gap")
    if any(a["elapsed_us"] > b["elapsed_us"] for a, b in zip(tokens, tokens[1:])):
        raise RuntimeError("token timestamps are not monotonic")
    return [event["data"]["token_id"] for event in tokens]


def verify_route(events: list[dict], total_layers: int, worker_count: int, coordinator_only: bool) -> dict:
    routes = [event["data"].get("route_plan") for event in events if event["event"] == "status"]
    routes = [route for route in routes if route]
    if not routes:
        raise RuntimeError("no live route evidence was recorded")
    route = routes[-1]
    ranges = [route["master_layer_range"]] + [hop["layer_range"] for hop in route["worker_hops"]]
    ranges = sorted((item["start"], item["end"]) for item in ranges if item["start"] != item["end"])
    cursor = 0
    for start, end in ranges:
        if start != cursor or end <= start or end > total_layers:
            raise RuntimeError("live route has missing, overlapping, or invalid layers")
        cursor = end
    if cursor != total_layers:
        raise RuntimeError("live route does not cover the model")
    if len(route["worker_hops"]) != worker_count:
        raise RuntimeError("not every requested worker participated in the route")
    if coordinator_only and route["master_layer_range"] != {"start": 0, "end": 0}:
        raise RuntimeError("coordinator-only run loaded transformer layers on the coordinator")
    return route


def run(args: argparse.Namespace) -> dict:
    binary = Path(args.binary).resolve()
    directory = Path(args.model_dir).resolve()
    if not binary.is_file() or not os.access(binary, os.X_OK):
        raise RuntimeError(f"build the synapse binary first: {binary}")
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False, mode=0o700)
    config = json.loads((directory / "config.json").read_text())
    manifest = json.loads((directory / "manifest.json").read_text())
    run_record = {"schema_version": 1, "topology": "loopback", "transport": args.transport,
        "physical_machine_count": 1, "worker_process_count": args.workers,
        "source": source_metadata(), "binary_sha256": sha256(binary),
        "model": manifest, "started_at_unix": int(time.time()), "success": False,
        "memory_method": "Linux /proc VmHWM sampled while each process is alive; observed lower bound, not final process peak",
        "observed_process_memory": {}, "release_gates": {"physical_sharding": "not_tested", "wan": "not_tested"}}
    processes: dict[str, subprocess.Popen] = {}
    handles = []
    pairing_code = None
    started = time.monotonic()
    environment = os.environ.copy()
    environment.update({"NO_COLOR": "1", "RUST_LOG": "warn,synapse_core=info"})

    def spawn(name: str, command: list[str]) -> subprocess.Popen:
        handle = (output / f"{name}.log").open("w")
        handles.append(handle)
        process = subprocess.Popen(command, cwd=ROOT, env=environment, stdin=subprocess.DEVNULL,
            stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
        processes[name] = process
        return process

    def sample() -> None:
        for name, process in processes.items():
            current = observed_memory(process.pid)
            if current:
                previous = run_record["observed_process_memory"].setdefault(name, {})
                for key, amount in current.items():
                    previous[key] = max(amount, previous.get(key, 0))

    def wait_for(condition, deadline: float, process: subprocess.Popen | None = None):
        while time.monotonic() < deadline:
            sample()
            result = condition()
            if result:
                return result
            if process is not None and process.poll() is not None:
                raise RuntimeError(f"process exited before readiness (status {process.returncode})")
            time.sleep(0.02)
        raise TimeoutError("configured process deadline exceeded")

    try:
        with tempfile.TemporaryDirectory(prefix="synapse-identities-") as identity_dir:
            # Check deterministic local output through the real bounded backend.
            reference = spawn("reference", [str(binary), "infer", "--model-dir", str(directory), "--prompt", args.prompt,
                "--max-tokens", str(args.tokens), "--events-jsonl", str(output / "reference.jsonl")])
            wait_for(lambda: reference.poll() is not None, time.monotonic() + args.timeout)
            if reference.returncode:
                raise RuntimeError("local reference process failed")
            expected = verify_generation(read_events(output / "reference.jsonl"))
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM if args.transport == "quic" else socket.SOCK_STREAM) as reservation:
                reservation.bind(("127.0.0.1", 0)); port = reservation.getsockname()[1]
            address = f"/ip4/127.0.0.1/udp/{port}/quic-v1" if args.transport == "quic" else f"/ip4/127.0.0.1/tcp/{port}"
            command = [str(binary), "--identity-path", str(Path(identity_dir) / "host.key"), "host", "--master",
                "--model", str(directory), "--prompt", args.prompt, "--max-tokens", str(args.tokens),
                "--listen", address, "--once", "--raw-prompt", "--min-workers", str(args.workers),
                "--timeout-secs", str(args.timeout), "--events-jsonl", str(output / "host.jsonl")]
            if args.coordinator_only:
                command.append("--coordinator-only")
            if args.memory_mib is not None:
                command.extend(["--memory-mib", str(args.memory_mib)])
            host = spawn("host", command)

            def host_ready():
                nonlocal pairing_code
                text = (output / "host.log").read_text()
                match = re.search(r"pairing code:\s+([A-Za-z0-9-]+)", text)
                if match:
                    pairing_code = match.group(1)
                records = read_events(output / "host.jsonl")
                return next((record["data"]["peer_id"] for record in records if record["event"] == "node_started"), None) if pairing_code else None

            host_peer = wait_for(host_ready, time.monotonic() + args.timeout, host)
            for index in range(args.workers):
                name = f"worker-{index + 1}"
                command = [str(binary), "--identity-path", str(Path(identity_dir) / f"{name}.key"), "join",
                    address + "/p2p/" + host_peer, "--model", str(directory), "--code", pairing_code,
                    "--events-jsonl", str(output / f"{name}.jsonl"), "--listen", "127.0.0.1:0"]
                if args.memory_mib is not None:
                    command.extend(["--memory-mib", str(args.memory_mib)])
                spawn(name, command)
            def host_completed():
                if host.poll() is not None:
                    return True
                for name, process in processes.items():
                    if name.startswith("worker-") and process.poll() is not None:
                        raise RuntimeError(f"{name} exited before generation completed ({process.returncode})")
                return False
            wait_for(host_completed, time.monotonic() + args.timeout + 5)
            if host.returncode:
                raise RuntimeError(f"distributed host exited unsuccessfully ({host.returncode})")
            events = read_events(output / "host.jsonl")
            actual = verify_generation(events)
            if actual != expected:
                raise RuntimeError("distributed token sequence differs from local reference")
            run_record["route"] = verify_route(events, config["num_hidden_layers"], args.workers, args.coordinator_only)
            peer_ids = [host_peer]
            for index in range(args.workers):
                worker_events = read_events(output / f"worker-{index + 1}.jsonl")
                peer_ids.extend(record["data"]["peer_id"] for record in worker_events if record["event"] == "node_started")
                residents = [record["data"] for record in worker_events if record["event"] == "status" and record["data"].get("resident_weight_bytes")]
                if not residents:
                    raise RuntimeError("worker resident-range evidence is missing")
                run_record.setdefault("worker_residency", {})[f"worker-{index + 1}"] = residents[-1]
            if len(set(peer_ids)) != args.workers + 1:
                raise RuntimeError("processes did not use distinct device identities")
            run_record.update({"success": True, "actual_output_tokens": len(actual), "reference_parity": True,
                "token_ids": actual, "peer_ids": peer_ids})
    except Exception as error:
        run_record["error"] = f"{type(error).__name__}: {error}"
    finally:
        sample()
        for process in processes.values():
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
        for process in processes.values():
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL); process.wait(timeout=5)
        for handle in handles:
            handle.close()
        for path in output.glob("*.log"):
            contents = path.read_text(errors="replace")
            if pairing_code:
                contents = contents.replace(pairing_code, "[pairing-code-redacted]")
            contents = contents.replace(args.prompt, "[prompt-redacted]")
            path.write_text(contents)
        run_record["wall_seconds"] = time.monotonic() - started
        (output / "run.json").write_text(json.dumps(run_record, indent=2) + "\n")
    return run_record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    target = Path(os.environ.get("CARGO_TARGET_DIR", ROOT / "target"))
    parser.add_argument("--binary", default=str(target / "debug/synapse"))
    parser.add_argument("--model-dir", default=str(ROOT / "crates/synapse-core/tests/fixtures/tiny-llama"))
    parser.add_argument("--workers", type=int, choices=(1, 2), default=1)
    parser.add_argument("--coordinator-only", action="store_true")
    parser.add_argument("--transport", choices=("quic", "tcp"), default="quic")
    parser.add_argument("--tokens", type=int, default=8)
    parser.add_argument("--prompt", default="<s> </s> Hello")
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--memory-mib", type=int)
    parser.add_argument("--output", default=str(ROOT / "artifacts" / f"local-cluster-{time.time_ns()}"))
    args = parser.parse_args()
    if args.timeout <= 0 or args.tokens <= 0:
        parser.error("timeout and token limit must be positive")
    result = run(args)
    print(json.dumps({"success": result["success"], "output": str(Path(args.output).resolve()), "error": result.get("error")}))
    return 0 if result["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
