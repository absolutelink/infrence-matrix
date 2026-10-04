"""Shared fixtures for halogen provider tests.

Provides a FAKE halogen server: a stdlib-only Python HTTP server
written to a temp script (shebang = this interpreter), wrapped in a
/bin/sh launcher that injects a state directory. Unlike the llama-cpp /
gufo fakes, this one takes its port from the HALOGEN_API_PORT **env
var** (halogen is env-configured, not argv-configured) and records the
full environment it was launched with so tests can assert the
HALOGEN_* mapping end to end.

It serves:

  GET  /health              200 {"status":"ok"}
  GET  /v1/models           fixed OpenAI model list
  POST /v1/responses        OpenResponses SSE (created, deltas, completed)
  POST /v1/chat/completions SSE chunks + data: [DONE]

Markers (stream completed / cancelled) are written to the state dir.
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

    # Record the environment we were launched with (HALOGEN_* assertions).
    with open(os.path.join(STATE_DIR, "env.json"), "w") as fh:
        json.dump(
            {k: v for k, v in os.environ.items() if k.startswith("HALOGEN_")},
            fh,
            indent=1,
        )

    # The entrypoint word must be "all" (legacy contract).
    if "all" not in sys.argv[2:]:
        print("fake-halogen: expected entrypoint arg 'all'", file=sys.stderr)
        sys.exit(2)


    def mark(name):
        with open(os.path.join(STATE_DIR, name), "w") as fh:
            fh.write(str(time.time()))


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
            elif self.path == "/v1/models":
                self._send(200, {"object": "list", "data": [
                    {"id": "halogen-qwen3.8-27b", "object": "model",
                     "owned_by": "fake-halogen"}
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
            if self.path == "/v1/responses":
                def frames():
                    yield sse({"type": "response.created", "response": {
                        "id": "resp_halogen_1", "object": "response",
                        "status": "in_progress", "output": []}})
                    for i in range(N_DELTAS):
                        yield sse({"type": "response.output_text.delta",
                                   "item_id": "msg_1", "output_index": 0,
                                   "content_index": 0, "delta": "tok%d" % i})
                    yield sse({"type": "response.completed", "response": {
                        "id": "resp_halogen_1", "object": "response",
                        "status": "completed", "output": [],
                        "usage": {"input_tokens": 5,
                                  "output_tokens": N_DELTAS,
                                  "total_tokens": 5 + N_DELTAS}}})
                self._stream(frames, "responses_cancelled")
            elif self.path == "/v1/chat/completions":
                def frames():
                    for i in range(N_DELTAS):
                        yield sse({"id": "chatcmpl_halogen",
                                   "object": "chat.completion.chunk",
                                   "choices": [{"index": 0, "delta": {
                                       "content": "tok%d" % i},
                                       "finish_reason": None}]})
                    yield sse({"id": "chatcmpl_halogen",
                               "object": "chat.completion.chunk",
                               "choices": [{"index": 0, "delta": {},
                                           "finish_reason": "stop"}]})
                    yield b"data: [DONE]\\n\\n"
                self._stream(frames, "chat_cancelled")
            else:
                self._send(404, {"error": "not found"})

    port = int(os.environ.get("HALOGEN_API_PORT", "0"))
    assert port > 0, "fake halogen requires HALOGEN_API_PORT"
    mark("server_started")
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    server.serve_forever()
    """
)

EARLY_EXIT_SCRIPT = textwrap.dedent(
    """\
    import sys
    print("fake-halogen: loading checkpoint", file=sys.stderr)
    print("fake-halogen: fatal error: rocm init failed", file=sys.stderr)
    sys.exit(4)
    """
)


def _write_script(path: Path, content: str) -> Path:
    shebang = f"#!{sys.executable}\n"
    path.write_text(shebang + content)
    path.chmod(path.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return path


@pytest.fixture
def fake_halogen_binary(tmp_path: Path) -> str:
    """Executable that behaves like the halogen entrypoint, env-driven."""
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    script = _write_script(tmp_path / "fake_halogen_server.py", FAKE_SCRIPT)
    wrapper = tmp_path / "fake_halogen"
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
        "MACHINE_UID": "halogen-test",
        "PROVIDER_REGISTRATION_TOKEN": "tok",
        "ADMIN_BASE_URL": "http://localhost:9999",
        "CACHE_DIR": tmp_path / "cache",
        "MODELS_DIR": tmp_path / "models",
        "PROVIDER_PORT": provider_port,
        "HALOGEN_SERVER_PATH": binary,
        "SERVER_START_HEALTH_TIMEOUT": 15,
        "MACHINE_METRICS_INTERVAL": 0.05,
    }
    base.update(kw)
    Path(base["CACHE_DIR"]).mkdir(parents=True, exist_ok=True)
    Path(base["MODELS_DIR"]).mkdir(parents=True, exist_ok=True)
    return ProviderSettings(**base)  # type: ignore[arg-type]


@pytest.fixture
def local_artifacts(tmp_path: Path) -> dict[str, str]:
    """A .hgn checkpoint file and a tokenizer directory."""
    ckpt = tmp_path / "models" / "qwen3.8-27b-p1w4d-d2.hgn"
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    ckpt.write_bytes(b"hgn-fake")
    tok = tmp_path / "models" / "tokenizer"
    tok.mkdir(parents=True, exist_ok=True)
    (tok / "tokenizer.json").write_text("{}")
    return {"checkpoint": str(ckpt), "tokenizer": str(tok)}
