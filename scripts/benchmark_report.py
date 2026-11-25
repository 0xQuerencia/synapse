#!/usr/bin/env python3
"""Summarize retained actual token timestamps; never infer counts from a target."""
from __future__ import annotations
import argparse
import csv
import json
import math
from pathlib import Path
import statistics
from local_cluster import read_events, verify_generation


def percentile(samples: list[float], probability: float) -> float | None:
    """Nearest-rank percentile, with a declared sample count in every result."""
    if not 0 < probability <= 1:
        raise ValueError("probability must be in (0,1]")
    if any(not math.isfinite(value) or value < 0 for value in samples):
        raise ValueError("latency samples must be finite and nonnegative")
    return sorted(samples)[math.ceil(len(samples) * probability) - 1] if samples else None


def describe(samples: list[float]) -> dict:
    return {"samples": len(samples), "mean_ms": statistics.fmean(samples) if samples else None,
        "p50_ms": percentile(samples, .50), "p95_ms": percentile(samples, .95),
        "quantile_method": "nearest-rank", "units": "milliseconds"}


def timings(events: list[dict]) -> dict:
    ids = verify_generation(events)
    starts = [record["elapsed_us"] for record in events if record["event"] == "request_started"]
    tokens = [record["elapsed_us"] for record in events if record["event"] == "token"]
    if len(starts) != 1 or tokens[0] < starts[0]:
        raise ValueError("exactly one request-start timestamp must precede output")
    completion = next(record["elapsed_us"] for record in events if record["event"] == "generation_done")
    if completion < tokens[-1]:
        raise ValueError("completion preceded accepted output")
    return {"actual_tokens": len(ids), "ttft_ms": (tokens[0] - starts[0]) / 1000,
        "generation_ms": (completion - starts[0]) / 1000,
        "inter_token_ms": [(b-a) / 1000 for a, b in zip(tokens, tokens[1:])]}


def summarize(directories: list[Path]) -> dict:
    completed, failures = [], []
    for directory in directories:
        record = json.loads((directory / "run.json").read_text())
        if not record.get("success"):
            failures.append({"directory": str(directory), "error": record.get("error", "run failed")})
            continue
        try:
            host = timings(read_events(directory / "host.jsonl"))
            reference = timings(read_events(directory / "reference.jsonl"))
            if host["actual_tokens"] != record["actual_output_tokens"]:
                raise ValueError("run metadata and actual output disagree")
            completed.append({"directory": str(directory), "topology": record["topology"],
                "transport": record["transport"], "source": record["source"],
                "model_id": record["model"]["model_id"], "revision": record["model"]["revision"],
                "host": host, "local_reference": reference,
                "memory": record.get("observed_process_memory", {}), "memory_method": record["memory_method"],
                "route": record["route"], "binary_sha256": record["binary_sha256"]})
        except (KeyError, ValueError, RuntimeError) as error:
            failures.append({"directory": str(directory), "error": str(error)})
    itl = [sample for run in completed for sample in run["host"]["inter_token_ms"]]
    local_itl = [sample for run in completed for sample in run["local_reference"]["inter_token_ms"]]
    topologies = sorted(set(run["topology"] for run in completed))
    return {"schema_version": 1, "attempts": len(directories), "completed": len(completed),
        "failed": len(failures), "topologies": topologies,
        "ttft": describe([run["host"]["ttft_ms"] for run in completed]),
        "inter_token": describe(itl), "local_reference_inter_token": describe(local_itl),
        "actual_output_tokens": sum(run["host"]["actual_tokens"] for run in completed),
        "runs": completed, "failures": failures,
        "limitations": ["Loopback runs do not establish physical-device or WAN results.",
            "Observed Linux VmHWM samples can miss a process's final peak.",
            "Small sample percentiles are descriptive; no latency SLO or speedup is asserted."]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directories", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--csv", type=Path)
    args = parser.parse_args()
    result = summarize(args.directories)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    if args.csv:
        with args.csv.open("w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["run", "topology", "transport", "actual_tokens", "ttft_ms", "generation_ms", "itl_samples", "itl_p50_ms", "itl_p95_ms"])
            for run in result["runs"]:
                latency = describe(run["host"]["inter_token_ms"])
                writer.writerow([run["directory"], run["topology"], run["transport"], run["host"]["actual_tokens"], run["host"]["ttft_ms"], run["host"]["generation_ms"], latency["samples"], latency["p50_ms"], latency["p95_ms"]])
    print(json.dumps({key: result[key] for key in ["attempts", "completed", "failed", "actual_output_tokens", "ttft", "inter_token"]}))
    return 0 if result["completed"] and not result["failed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
