#!/usr/bin/env python3
"""Measure real authenticated chat requests; keep rejected work separate."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import math
from pathlib import Path
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

from benchmark_report import percentile


def one_request(index: int, endpoint: str, token: str, body: bytes, timeout: float) -> dict:
    started = time.monotonic()
    record = {"index": index, "status": None, "success": False, "completion_tokens": None}
    request = Request(endpoint, data=body, method="POST", headers={
        "Content-Type": "application/json", "Authorization": "Bearer " + token})
    try:
        with urlopen(request, timeout=timeout) as response:
            record["status"] = response.status
            raw = response.read(4 * 1024 * 1024 + 1)
            if len(raw) > 4 * 1024 * 1024:
                raise ValueError("response exceeded 4 MiB")
            result = json.loads(raw)
            usage = result["usage"]
            choices = result["choices"]
            if (result.get("object") != "chat.completion" or len(choices) != 1
                    or choices[0].get("finish_reason") not in ("stop", "length")
                    or not isinstance(choices[0]["message"].get("content"), str)
                    or any(type(usage.get(key)) is not int or usage[key] < 0
                           for key in ("prompt_tokens", "completion_tokens", "total_tokens"))
                    or usage["completion_tokens"] == 0
                    or usage["total_tokens"] != usage["prompt_tokens"] + usage["completion_tokens"]):
                raise ValueError("invalid completion or usage response")
            record.update(success=True, completion_tokens=usage["completion_tokens"])
    except HTTPError as error:
        record["status"] = error.code
        record["error"] = "busy" if error.code == 429 else "http_error"
        error.close()
    except (URLError, TimeoutError, OSError) as error:
        record["error"] = type(error).__name__
    except (ValueError, KeyError, TypeError, IndexError):
        record["error"] = "invalid_completion_response"
    record["elapsed_ms"] = (time.monotonic() - started) * 1000
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:3000")
    parser.add_argument("--token-file", type=Path, required=True)
    parser.add_argument("--model", default="synapse-local")
    parser.add_argument("--prompt", default="<s> </s> Hello")
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument("--requests", type=int, default=20)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=30)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    parsed = urlsplit(args.base_url)
    if (parsed.username or parsed.password or parsed.query or parsed.fragment
            or parsed.path not in ("", "/") or parsed.scheme not in ("http", "https")
            or not parsed.hostname or (parsed.scheme == "http" and parsed.hostname not in ("localhost", "127.0.0.1", "::1"))):
        parser.error("use a loopback HTTP origin or an HTTPS origin without credentials, query or path")
    if not (1 <= args.requests <= 10_000 and 1 <= args.concurrency <= 64
            and 1 <= args.max_tokens <= 4096 and math.isfinite(args.timeout) and 0 < args.timeout <= 300):
        parser.error("invalid request count, concurrency, token limit or timeout")
    token = args.token_file.read_text().strip()
    if not 32 <= len(token) <= 256 or not all(c.isascii() and (c.isalnum() or c in "_-") for c in token):
        parser.error("invalid token file")
    body = json.dumps({"model": args.model, "messages": [{"role": "user", "content": args.prompt}],
        "max_tokens": args.max_tokens, "temperature": 0, "stream": False}).encode()
    started = time.monotonic()
    endpoint = args.base_url.rstrip("/") + "/v1/chat/completions"
    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        records = list(executor.map(lambda index: one_request(index, endpoint, token, body, args.timeout), range(args.requests)))
    duration = time.monotonic() - started
    successful = [r for r in records if r["success"]]
    busy = sum(r["status"] == 429 for r in records)
    latencies = [r["elapsed_ms"] for r in successful]
    report = {"schema_version": 1, "endpoint": endpoint, "model": args.model,
        "attempted": len(records), "completed": len(successful), "busy": busy,
        "failed": len(records) - len(successful) - busy, "concurrency": args.concurrency,
        "elapsed_seconds": duration, "completed_requests_per_second": len(successful) / duration,
        "actual_completion_tokens": sum(r["completion_tokens"] for r in successful),
        "successful_latency_ms": {"p50": percentile(latencies, .5), "p95": percentile(latencies, .95)},
        "records": records}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as file:
        json.dump(report, file, indent=2); file.write("\n")
    print(json.dumps({key: value for key, value in report.items() if key != "records"}))
    return 0 if successful and report["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
