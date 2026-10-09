"""Shared fixtures for gufo provider tests.

Provides a FAKE gufo server: a stdlib-only Python HTTP server written
to a temp script (shebang = this interpreter) wrapped in a /bin/sh
launcher that injects a state directory, so the driver can spawn it as
a subprocess exactly like a real ``gufo serve llm``. It serves:

  GET  /health              200 {"status":"ok"}
  GET  /metrics             Prometheus text with gufo rate gauges
  GET  /v1/models           fixed OpenAI model list (multi-model)
  POST /v1/responses        native OpenResponses SSE (created, deltas,
                           completed with usage)
  POST /v1/chat/completions SSE chunks + data: [DONE]

It records the last request body per endpoint into the state directory
(multi-model `model` forwarding assertions) and lifecycle markers
(stream completed / cancelled) like the llama-cpp fake.
"""

import stat
import sys
import textwrap
from pathlib import Path
from typing import Any

import pytest
from provider_lib.config import ProviderSettings

FAKE_SCRIPT = textwrap.dedent(
    """\
    import json
    import os
    import sys
    import time
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    STATE_DIR = sys.argv[1]
    DELAY = float(os.environ.get("FAKE_SSE_DELAY", "0.02"))
    N_DELTAS = int(os.environ.get("FAKE_N_DELTAS", "8"))

    METRICS_TEXT = \"\"\"# HELP llamacpp:prompt_tokens_total Total prompt tokens processed
    # TYPE llamacpp:prompt_tokens_total counter
    llamacpp:prompt_tokens_total 7920
    # HELP llamacpp:tokens_predicted_total Total tokens generated
    # TYPE llamacpp:tokens_predicted_total counter
    llamacpp:tokens_predicted_total 120
    # HELP llamacpp:prompt_tokens_seconds Prompt processing speed in tokens per second
    # TYPE llamacpp:prompt_tokens_seconds gauge
    llamacpp:prompt_tokens_seconds 1412.21
    # HELP llamacpp:predicted_tokens_seconds Generation speed in tokens per second
    # TYPE llamacpp:predicted_tokens_seconds gauge
    llamacpp:predicted_tokens_seconds 36.2534
    # HELP llamacpp:kv_cache_usage_ratio KV cache usage ratio
    # TYPE llamacpp:kv_cache_usage_ratio gauge
    llamacpp:kv_cache_usage_ratio 0.0
    \"\"\"


    def mark(name):
        with open(os.path.join(STATE_DIR, name), "w") as fh:
            fh.write(str(time.time()))


    def get_arg(flag, default=None):
        argv = sys.argv
        if flag in argv:
            return argv[argv.index(flag) + 1]
        return default


    def sse(obj, event_type=None):
        et = event_type or obj.get("type", "message")
        return ("event: %s\\ndata: %s\\n\\n" % (et, json.dumps(obj))).encode()


    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args):
            pass

        def _send(self, code, body, ctype="application/json"):
            data = json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path == "/health":
                self._send(200, {"status": "ok"})
            elif self.path == "/metrics":
                data = METRICS_TEXT.encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/plain; version=0.0.4")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
            elif self.path == "/v1/models":
                # gufo is multi-model: several served names on one instance.
                self._send(200, {"object": "list", "data": [
                    {"id": "served-alpha", "object": "model", "owned_by": "gufo"},
                    {"id": "served-beta", "object": "model", "owned_by": "gufo"},
                ]})
            else:
                self._send(404, {"error": "not found"})

        def _stream(self, frames, cancel_marker):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            try:
                for frame in frames():
                    self.wfile.write(frame)
                    self.wfile.flush()
                    time.sleep(DELAY)
                mark("stream_completed")
            except (BrokenPipeError, ConnectionResetError, OSError):
                mark(cancel_marker)
            self.close_connection = True

        def do_POST(self):
            length = int(self.headers.get("Content-Length", 0) or 0)
            raw = self.rfile.read(length) if length else b"{}"
            try:
                request = json.loads(raw or b"{}")
            except ValueError:
                request = {}
            if self.path in ("/v1/responses", "/v1/chat/completions"):
                # Record the forwarded body (multi-model assertions).
                with open(os.path.join(STATE_DIR, "last_request.json"), "w") as fh:
                    json.dump({"path": self.path, "body": request}, fh)
            if self.path == "/v1/responses":
                def frames():
                    yield sse({"type": "response.created", "response": {
                        "id": "resp_gufo_1", "object": "response",
                        "status": "in_progress", "output": []}})
                    for i in range(N_DELTAS):
                        yield sse({"type": "response.output_text.delta",
                                   "item_id": "msg_1", "output_index": 0,
                                   "content_index": 0, "delta": "tok%d" % i})
                    term_usage = {"input_tokens": 3,
                                  "output_tokens": N_DELTAS,
                                  "total_tokens": 3 + N_DELTAS,
                                  "input_tokens_details":
                                      {"cached_tokens": 1}}
                    term_resp = {"id": "resp_gufo_1", "object": "response",
                                 "status": "completed", "output": [],
                                 "usage": term_usage}
                    term_timings = {"prompt_ms": 120.5, "predicted_ms": 3400.0}
                    if request.get("__timings_in_usage__"):
                        term_usage["timings"] = term_timings
                    else:
                        term_resp["timings"] = term_timings
                    yield sse({"type": "response.completed",
                               "response": term_resp})
                self._stream(frames, "responses_cancelled")
            elif self.path == "/v1/chat/completions":
                def frames():
                    for i in range(N_DELTAS):
                        yield sse({"id": "chatcmpl_gufo",
                                   "object": "chat.completion.chunk",
                                   "choices": [{"index": 0, "delta": {
                                       "content": "tok%d" % i},
                                       "finish_reason": None}]})
                    yield sse({"id": "chatcmpl_gufo",
                               "object": "chat.completion.chunk",
                               "choices": [{"index": 0, "delta": {},
                                           "finish_reason": "stop"}]})
                    yield b"data: [DONE]\\n\\n"
                self._stream(frames, "chat_cancelled")
            else:
                self._send(404, {"error": "not found"})

    port = int(get_arg("--port", "0"))
    assert port > 0, "fake gufo requires --port"
    mark("server_started")
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    server.serve_forever()
    """
)

# Script that prints to stderr and exits immediately (early-exit case).
EARLY_EXIT_SCRIPT = textwrap.dedent(
    """\
    import sys
    print("fake-gufo: loading model", file=sys.stderr)
    print("fake-gufo: fatal error: failed to initialize device", file=sys.stderr)
    sys.exit(3)
    """
)


def _write_script(path: Path, content: str) -> Path:
    shebang = f"#!{sys.executable}\n"
    path.write_text(shebang + content)
    path.chmod(path.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return path


@pytest.fixture
def fake_gufo_binary(tmp_path: Path) -> str:
    """Executable that behaves like `gufo serve llm`, with state dir."""
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    script = _write_script(tmp_path / "fake_gufo_server.py", FAKE_SCRIPT)
    wrapper = tmp_path / "fake_gufo"
    wrapper.write_text(f'#!/bin/sh\nexec {script} {state_dir} "$@"\n')
    wrapper.chmod(wrapper.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return str(wrapper)


@pytest.fixture
def state_dir(tmp_path: Path) -> Path:
    return tmp_path / "state"


@pytest.fixture
def early_exit_binary(tmp_path: Path) -> str:
    return str(_write_script(tmp_path / "early_exit.py", EARLY_EXIT_SCRIPT))


def make_settings(
    tmp_path: Path, binary: str, provider_port: int = 8181, **kw: Any
) -> ProviderSettings:
    base: dict[str, Any] = {
        "MACHINE_UID": "gufo-test",
        "MACHINE_SECRET": "tok",
        "AGENT_ID": "test-agent",
        "ADMIN_BASE_URL": "http://localhost:9999",
        "CACHE_DIR": tmp_path / "cache",
        "MODELS_DIR": tmp_path / "models",
        "PROVIDER_PORT": provider_port,
        "GUFO_SERVER_PATH": binary,
        "SERVER_START_HEALTH_TIMEOUT": 15,
        "MACHINE_METRICS_INTERVAL": 0.05,
    }
    base.update(kw)
    Path(base["CACHE_DIR"]).mkdir(parents=True, exist_ok=True)
    Path(base["MODELS_DIR"]).mkdir(parents=True, exist_ok=True)
    return ProviderSettings(**base)  # type: ignore[arg-type]


@pytest.fixture
def local_model_file(tmp_path: Path) -> str:
    p = tmp_path / "models" / "fake.gguf"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"gguf-fake")
    return str(p)
