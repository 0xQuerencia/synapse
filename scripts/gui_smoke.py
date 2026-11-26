#!/usr/bin/env python3
"""Run live GUI/API/worker acceptance checks against an explicit built binary.

No binary is built, no model is downloaded, and no credentials are printed.
The only optional dependency is websockets for distributed token verification.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import tempfile
import time
from typing import Any, Callable
import urllib.error
import urllib.request

PROMPT = "<s> </s> Hello"
MAX_RESPONSE_BYTES = 2 * 1024 * 1024
ANSI = re.compile(r"\x1b\[[0-9;]*m")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


class GuiServer:
    def __init__(self, label: str, binary: Path, fixture: Path, root: Path):
        self.label = label
        self.directory = root / label
        self.directory.mkdir(mode=0o700)
        (self.directory / "config").mkdir(mode=0o700)
        (self.directory / "home").mkdir(mode=0o700)
        self.identity = self.directory / "device.key"
        self.token_path = self.directory / "access.token"
        self.log_path = self.directory / "server.log"
        self.log_handle = self.log_path.open("wb")
        self.process: subprocess.Popen[bytes] | None = None
        self.base_url: str | None = None
        self.token: str | None = None
        self.secrets: set[str] = set()
        self.cleanup_result = "not_started"
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        environment = os.environ.copy()
        environment.update({
            "HOME": str(self.directory / "home"),
            "HF_HOME": str(self.directory / "home" / "hf"),
            "RUST_LOG": "warn,synapse_core=info,synapse_gui=info",
            "NO_COLOR": "1",
        })
        # Explicit flags are part of the tested contract. Do not let inherited
        # credentials, identity overrides or discovery settings hide failures.
        for name in ("SYNAPSE_TOKEN", "SYNAPSE_IDENTITY_PATH", "SYNAPSE_OLLAMA_URL"):
            environment.pop(name, None)
        self.process = subprocess.Popen([
            str(binary), "--addr", "127.0.0.1:0",
            "--identity-path", str(self.identity),
            "--token-file", str(self.token_path),
            "--model-dir", str(fixture), "--api-model-id", "tiny",
            "--chat-template", "raw", "--max-context", "128",
            "--max-output-tokens", "32", "--request-timeout-secs", "30",
        ], cwd=self.directory, env=environment, stdout=self.log_handle,
            stderr=subprocess.STDOUT, start_new_session=True)

    def log_text(self) -> str:
        self.log_handle.flush()
        return ANSI.sub("", self.log_path.read_text(errors="replace"))

    def ready(self, timeout: float = 30.0) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            require(self.process is not None and self.process.poll() is None,
                    f"{self.label} server exited before readiness; see redacted log")
            match = re.search(r"Listening on:\s+http://(127\.0\.0\.1:\d+)", self.log_text())
            if match and self.token_path.exists():
                self.base_url = "http://" + match.group(1)
                self.token = self.token_path.read_text().strip()
                self.secrets.add(self.token)
                status, _, _ = self.request("GET", "/v1/models")
                if status == 200:
                    return
            time.sleep(0.05)
        raise RuntimeError(f"{self.label} readiness deadline exceeded")

    def request(self, method: str, path: str, body: Any = None, *,
                authenticated: bool = True, headers: dict[str, str] | None = None,
                timeout: float = 30.0) -> tuple[int, dict[str, str], bytes]:
        require(self.base_url is not None, "server URL unavailable")
        request_headers = dict(headers or {})
        if authenticated:
            require(self.token is not None, "token unavailable")
            request_headers["Authorization"] = "Bearer " + self.token
        encoded = None if body is None else json.dumps(body).encode()
        if encoded is not None:
            request_headers["Content-Type"] = "application/json"
        request = urllib.request.Request(self.base_url + path, data=encoded,
                                         headers=request_headers, method=method)
        try:
            response = self.opener.open(request, timeout=timeout)
        except urllib.error.HTTPError as error:
            response = error
        with response:
            result = response.read(MAX_RESPONSE_BYTES + 1)
            require(len(result) <= MAX_RESPONSE_BYTES, "HTTP response exceeded smoke limit")
            return response.status, dict(response.headers), result

    def json(self, method: str, path: str, body: Any = None,
             expected: int = 200, **kwargs: Any) -> Any:
        status, _, result = self.request(method, path, body, **kwargs)
        require(status == expected, f"{self.label} {method} {path} returned {status}, expected {expected}")
        return json.loads(result)

    def wait_status(self, predicate: Callable[[dict[str, Any]], bool],
                    label: str, timeout: float = 15.0) -> dict[str, Any]:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            status = self.json("GET", "/api/status")
            if predicate(status):
                return status
            time.sleep(0.05)
        raise RuntimeError(f"{self.label} deadline waiting for {label}")

    def stop(self) -> None:
        if self.process is None:
            return
        if self.process.poll() is None and self.base_url:
            try:
                self.request("POST", "/api/session/stop", {}, timeout=10)
            except (OSError, RuntimeError):
                pass
        for sig, timeout in [(signal.SIGINT, 5), (signal.SIGTERM, 2), (signal.SIGKILL, 2)]:
            if self.process.poll() is not None:
                break
            os.killpg(self.process.pid, sig)
            try:
                self.process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                continue
        self.cleanup_result = "exited" if self.process.poll() is not None else "failed"
        self.log_handle.flush()

    def save_redacted_log(self, output: Path, secrets: set[str]) -> bool:
        text = self.log_text()
        leaked = any(secret and secret in text for secret in secrets)
        for secret in secrets:
            if secret:
                text = text.replace(secret, "[REDACTED]")
        (output / f"{self.label}.log").write_text(text)
        self.log_handle.close()
        return leaked


def check_auth_and_configuration(server: GuiServer) -> dict[str, Any]:
    require(server.request("GET", "/v1/models", authenticated=False)[0] == 401,
            "unauthenticated API request was not rejected")
    require(server.request("GET", "/v1/models", headers={"Origin": "https://untrusted.invalid"})[0] == 403,
            "foreign origin was not rejected")
    require(server.request("GET", "/v1/models", headers={"Host": "untrusted.invalid"})[0] == 403,
            "foreign Host was not rejected")
    require(server.request("GET", "/v1/models?token=never-a-secret")[0] == 400,
            "URL credential channel was not rejected")
    identity = server.json("GET", "/api/identity")
    require(Path(identity["key_path"]) == server.identity, "explicit --identity-path was not honored by HTTP handlers")
    require(server.json("GET", "/v1/models")["data"][0]["id"] == "tiny", "hosted model list is incorrect")
    original = server.json("GET", "/api/config/bootstrap")["contents"]
    require("peers = []" in original, "bootstrap default does not use runtime peers key")
    valid = "peers = []\n"
    server.json("PUT", "/api/config/bootstrap", {"contents": valid})
    server.json("PUT", "/api/config/bootstrap", {"contents": "bootstrap = []\n"}, expected=400)
    require(server.json("GET", "/api/config/bootstrap")["contents"] == valid,
            "invalid configuration replaced the valid file")
    require(server.request("POST", "/api/worker/control", {"action": "pause"})[0] == 400,
            "worker control falsely succeeded without a worker")
    return {"unauthorized": 401, "origin_rejected": 403, "host_rejected": 403,
            "query_token_rejected": 400, "identity_path_honored": True,
            "invalid_config_preserved_prior": True, "inactive_worker_control_rejected": True}


def check_compatible_api(server: GuiServer) -> dict[str, Any]:
    request = {"model": "tiny", "messages": [{"role": "user", "content": PROMPT}],
               "max_tokens": 8, "temperature": 0}
    ordinary = server.json("POST", "/v1/chat/completions", request)
    require(ordinary["usage"] == {"prompt_tokens": 3, "completion_tokens": 8, "total_tokens": 11},
            "nonstream completion token counts do not match the fixture")
    require(ordinary["choices"][0]["finish_reason"] == "length", "nonstream termination incorrect")
    status, headers, body = server.request("POST", "/v1/chat/completions",
        {**request, "stream": True, "stream_options": {"include_usage": True}})
    require(status == 200 and "text/event-stream" in headers.get("content-type", headers.get("Content-Type", "")),
            "streaming API did not return SSE")
    text = ""
    emitted = 0
    usage = None
    finish = None
    done = 0
    for raw in body.decode().splitlines():
        if raw.startswith("event:"):
            require(raw != "event: error", "SSE reported inference failure")
        if not raw.startswith("data:"):
            continue
        data = raw[5:].lstrip()
        if data == "[DONE]":
            done += 1
            require(finish == "length" and usage is not None, "DONE preceded required terminal information")
            continue
        event = json.loads(data)
        if event.get("usage") is not None:
            usage = event["usage"]
        for choice in event.get("choices", []):
            if "content" in choice["delta"]:
                text += choice["delta"]["content"]
                emitted += 1
            if choice.get("finish_reason"):
                finish = choice["finish_reason"]
    require(done == 1 and emitted == 8, "SSE output or terminal event count is incorrect")
    require(usage == ordinary["usage"] and text == ordinary["choices"][0]["message"]["content"],
            "streaming and nonstream output/usage diverged")
    return {"prompt_tokens": 3, "completion_tokens": 8, "stream_content_events": emitted,
            "stream_matches_json": True, "finish_reason": finish, "done_events": done,
            "completion_text": text}


def distributed_generation(server: GuiServer, reference_ids: list[int], reference_text: str,
                           required: bool) -> dict[str, Any]:
    if importlib.util.find_spec("websockets") is None:
        require(not required, "websockets is required for distributed GUI generation")
        return {"status": "skipped", "reason": "optional websockets package unavailable"}
    try:
        from websockets.sync.client import connect
    except ImportError as error:
        if required:
            raise RuntimeError("websockets.sync is required for distributed GUI generation") from error
        return {"status": "skipped", "reason": "installed websockets lacks sync client"}
    kwargs = {"additional_headers": {"Authorization": "Bearer " + str(server.token)},
              "origin": server.base_url, "open_timeout": 5, "close_timeout": 2,
              "max_size": 300 * 1024}
    if "proxy" in inspect.signature(connect).parameters:
        kwargs["proxy"] = None
    tokens: list[int] = []
    positions: list[int] = []
    text_parts: list[str] = []
    started = time.monotonic()
    with connect(str(server.base_url).replace("http:", "ws:") + "/ws", **kwargs) as connection:
        require(json.loads(connection.recv(timeout=5)).get("type") == "hello", "WebSocket greeting missing")
        connection.send(json.dumps({"type": "prompt", "id": "gui-smoke-fixture", "text": PROMPT}))
        deadline = time.monotonic() + 30
        finished = None
        while time.monotonic() < deadline:
            message = json.loads(connection.recv(timeout=max(0.1, deadline - time.monotonic())))
            require(message.get("type") != "error", "GUI WebSocket reported a generation error")
            event = message.get("event", {})
            if event.get("type") == "token_generated":
                tokens.append(event["token_id"])
                positions.append(event["position"])
            if event.get("type") == "token":
                require(isinstance(event.get("text"), str), "WebSocket text event malformed")
                text_parts.append(event["text"])
            if event.get("type") == "generation_done":
                finished = event
                break
    require(finished is not None, "distributed generation did not terminate")
    require(finished["reason"] == "MaxTokens" and finished["tokens_generated"] == 8,
            "distributed generation reported an incorrect terminal count/reason")
    require(tokens == reference_ids and positions == list(range(8)),
            "distributed GUI token IDs or sequence positions differ from the reference")
    require("".join(text_parts) == reference_text,
            "distributed decoded text differs from the local API output")
    return {"status": "passed", "token_ids": tokens, "positions": positions,
            "text_matches_local_api": True, "tokens_generated": 8, "reason": "MaxTokens",
            "elapsed_seconds": time.monotonic() - started}


def check_worker_session(host: GuiServer, worker: GuiServer, fixture: Path, reference_text: str,
                         required_ws: bool) -> dict[str, Any]:
    started = host.json("POST", "/api/session/host", {"model": str(fixture), "max_tokens": 8,
        "listen": "127.0.0.1:0", "memory_mib": 64, "raw_prompt": True, "coordinator_only": True})
    code = started["pairing_code"]
    require(isinstance(code, str) and bool(code), "host did not issue a pairing secret")
    host.secrets.add(code)
    worker.secrets.add(code)
    host_peer = started["session"]["peer_id"]
    status = host.wait_status(lambda value: bool((value.get("node") or {}).get("listen_addresses")), "host listener")
    addresses = status["node"]["listen_addresses"]
    address = next((value for value in addresses if value.startswith("/ip4/127.0.0.1/") and "/quic-v1" in value), None)
    require(address is not None, "host did not expose its actual loopback QUIC listener")
    if "/p2p/" not in address:
        address += "/p2p/" + host_peer
    joined = worker.json("POST", "/api/session/join", {"model": str(fixture), "code": code,
        "master_addr": address, "listen": "127.0.0.1:0", "memory_mib": 64, "duration_minutes": 10})
    require(joined["peer_id"] != host_peer, "GUI processes reused a device identity")
    host.wait_status(lambda value: (value.get("node") or {}).get("authenticated_count", 0) == 1,
                     "mutual worker authentication")
    transitions = []
    for action, expected in [("pause", "paused"), ("resume", "available"), ("drain", "paused"), ("resume", "available")]:
        ack = worker.json("POST", "/api/worker/control", {"action": action})
        require(ack["state"] == expected and ack["active_jobs"] == 0, f"{action} acknowledgement did not match idle worker state")
        worker.wait_status(lambda value: ((value.get("node") or {}).get("worker_state") or {}).get("state") == expected,
                           f"live {expected} worker state")
        transitions.append({"action": action, "acknowledged_state": expected, "active_jobs": 0})
    reference = json.loads((fixture / "reference.json").read_text())
    reference_ids = reference["greedy_ids"][len(reference["input_ids"]):]
    require(len(reference_ids) == 8, "fixture must provide exactly eight reference output tokens")
    generation = distributed_generation(host, reference_ids, reference_text, required_ws)
    if generation["status"] == "passed":
        worker.wait_status(lambda value: ((value.get("node") or {}).get("worker_state") or {}).get("active_jobs") == 0,
                           "released worker lease")
        host_log = host.log_text()
        worker_log = worker.log_text()
        require("master routing prefill hidden state to worker" in host_log,
                "distributed generation has no actual prefill routing evidence")
        require(worker_log.count("worker decode inference complete") >= 7,
                "worker did not record seven actual decode completions")
        generation["worker_decode_completions"] = worker_log.count("worker decode inference complete")
        generation["prefill_routed"] = True
    ack = worker.json("POST", "/api/worker/control", {"action": "stop"})
    require(ack["state"] == "stopped" and ack["active_jobs"] == 0, "worker stop was not acknowledged after cleanup")
    worker.wait_status(lambda value: not value["session"]["running"], "worker task exit")
    worker.json("POST", "/api/session/stop", {})
    require(worker.json("GET", "/api/status")["session"]["role"] == "idle", "stopped session did not become idle")
    host.json("POST", "/api/session/stop", {})
    return {"distinct_identities": True, "mutually_authenticated_workers": 1,
            "controls_tested_with": "idle worker; active-drain semantics have separate unit tests",
            "transitions": transitions, "stop_acknowledged": True, "session_cleanup_idle": True,
            "distributed_generation": generation}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True, help="explicit already-built synapse-gui binary")
    parser.add_argument("--fixture", type=Path, default=Path(__file__).resolve().parents[1] / "crates/synapse-core/tests/fixtures/tiny-llama")
    parser.add_argument("--out", type=Path, required=True, help="new directory for redacted evidence")
    parser.add_argument("--source-revision", default=None, help="source revision of the supplied binary, if known")
    parser.add_argument("--require-websocket", action="store_true")
    args = parser.parse_args()
    binary, fixture, output = args.binary.resolve(), args.fixture.resolve(), args.out.resolve()
    require(binary.is_file() and os.access(binary, os.X_OK), "--binary must name an executable GUI binary")
    require((fixture / "reference.json").is_file() and (fixture / "manifest.json").is_file(), "fixture inputs are missing")
    output.mkdir(parents=True, exist_ok=False)
    result: dict[str, Any] = {"schema_version": 1, "status": "failed", "scope": "two GUI processes on one loopback host",
        "binary_sha256": digest(binary), "source_revision": args.source_revision,
        "fixture_manifest_sha256": digest(fixture / "manifest.json"), "physical_wan_verified": False}
    servers: list[GuiServer] = []
    try:
        with tempfile.TemporaryDirectory(prefix="synapse-gui-smoke-") as temporary:
            try:
                root = Path(temporary)
                for label in ("host", "worker"):
                    server = GuiServer(label, binary, fixture, root)
                    servers.append(server)
                    server.ready()
                host, worker = servers
                result["authentication_and_configuration"] = check_auth_and_configuration(host)
                result["compatible_api"] = check_compatible_api(host)
                reference_text = result["compatible_api"].pop("completion_text")
                result["worker_session"] = check_worker_session(host, worker, fixture, reference_text,
                                                                args.require_websocket)
                result["status"] = "passed"
            finally:
                for server in reversed(servers):
                    server.stop()
                secrets = set().union(*(server.secrets for server in servers))
                leaked = any([server.save_redacted_log(output, secrets) for server in servers])
                result["cleanup"] = {server.label: server.cleanup_result for server in servers}
                result["secret_log_leak_detected"] = leaked
                require(not leaked, "a credential appeared in server logs; saved evidence was redacted")
                require(all(server.cleanup_result == "exited" for server in servers), "a smoke server failed to exit")
    except Exception as error:
        result["status"] = "failed"
        # Exception text is restricted to static harness assertions and local
        # transport errors. Never serialize response bodies or request headers.
        message = str(error)
        for server in servers:
            for secret in server.secrets:
                if secret:
                    message = message.replace(secret, "[REDACTED]")
        result["failure"] = {"type": type(error).__name__, "message": message}
    (output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "evidence": str(output / "summary.json")}, indent=2))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
