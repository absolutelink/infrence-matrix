"""Shared fixtures for llama-cpp provider tests.

Provides a FAKE llama-server: a stdlib-only Python HTTP server written to
a temp script (shebang = this interpreter) wrapped in a /bin/sh launcher
that injects a state directory, so the driver can spawn it as a
subprocess exactly like a real llama-server. It serves:

  GET  /health              200 {"status":"ok"}
  GET  /v1/models           fixed OpenAI model list
  POST /v1/responses        SSE: created, N deltas, completed (usage)
  POST /v1/chat/completions SSE: chunks + data: [DONE]
  POST /v1/embeddings       fixed CreateEmbeddingResponse (500 on __fail__)

It records lifecycle markers into the state directory so tests can assert
on server-side observations such as mid-stream client cancellation
(BrokenPipe/ConnectionReset) and full-stream completion.
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
            elif self.path == "/v1/models":
                self._send(200, {"object": "list", "data": [
                    {"id": "fake-llama-model", "object": "model",
                     "owned_by": "fake"}
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
            # SSE ends at connection close; with HTTP/1.1 keep-alive the
            # client would otherwise wait forever for a response body
            # terminator.
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
                        "id": "resp_fake_1", "object": "response",
                        "status": "in_progress", "output": []}})
                    for i in range(N_DELTAS):
                        yield sse({"type": "response.output_text.delta",
                                   "item_id": "msg_1", "output_index": 0,
                                   "content_index": 0, "delta": "tok%d" % i})
                    yield sse({"type": "response.completed", "response": {
                        "id": "resp_fake_1", "object": "response",
                        "status": "completed", "output": [],
                        "usage": {"input_tokens": 3,
                                  "output_tokens": N_DELTAS,
                                  "total_tokens": 3 + N_DELTAS}}})
                self._stream(frames, "responses_cancelled")
            elif self.path == "/v1/chat/completions":
                def frames():
                    for i in range(N_DELTAS):
                        yield sse({"id": "chatcmpl_fake",
                                   "object": "chat.completion.chunk",
                                   "choices": [{"index": 0, "delta": {
                                       "content": "tok%d" % i},
                                       "finish_reason": None}]})
                    yield sse({"id": "chatcmpl_fake",
                               "object": "chat.completion.chunk",
                               "choices": [{"index": 0, "delta": {},
                                           "finish_reason": "stop"}]})
                    yield b"data: [DONE]\\n\\n"
                self._stream(frames, "chat_cancelled")
            elif self.path == "/v1/embeddings":
                if request.get("__fail__"):
                    self._send(500, {"error": "embedding boom"})
                else:
                    inp = request.get("input", "")
                    n = len(inp) if isinstance(inp, list) else 1
                    self._send(200, {
                        "object": "list",
                        "data": [
                            {"object": "embedding", "index": i,
                             "embedding": [0.1, 0.2, 0.3]}
                            for i in range(n)
                        ],
                        "model": "fake-llama-model",
                        "usage": {"prompt_tokens": 3, "total_tokens": 3},
                    })
            else:
                self._send(404, {"error": "not found"})

    port = int(get_arg("--port", "0"))
    assert port > 0, "fake llama-server requires --port"
    mark("server_started")
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    server.serve_forever()
    """
)

# Script that prints to stderr and exits immediately (early-exit case).
EARLY_EXIT_SCRIPT = textwrap.dedent(
    """\
    import sys
    print("fake: loading model", file=sys.stderr)
    print("fake: fatal error: failed to initialize backend", file=sys.stderr)
    sys.exit(3)
    """
)


def _write_script(path: Path, content: str) -> Path:
    shebang = f"#!{sys.executable}\n"
    path.write_text(shebang + content)
    path.chmod(path.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return path


@pytest.fixture
def fake_llama_binary(tmp_path: Path) -> str:
    """Executable that behaves like llama-server, with a state dir injected."""
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    script = _write_script(tmp_path / "fake_llama_server.py", FAKE_SCRIPT)
    wrapper = tmp_path / "fake_llama_server"
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
        "MACHINE_UID": "llama-test",
        "MACHINE_SECRET": "tok",
        "AGENT_ID": "test-agent",
        "ADMIN_BASE_URL": "http://localhost:9999",
        "CACHE_DIR": tmp_path / "cache",
        "MODELS_DIR": tmp_path / "models",
        "PROVIDER_PORT": provider_port,
        "LLAMA_SERVER_PATH": binary,
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
