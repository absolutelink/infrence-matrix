"""Shared fixtures for halogen-flash provider tests.

Provides a FAKE halogen-flash server: a stdlib-only Python HTTP server
written to a temp script (shebang = this interpreter), wrapped in a
/bin/sh launcher that injects a state directory. The fake is
env-driven (HALOGEN_API_PORT) like the real engine, records its
HALOGEN_* environment, and serves **native OpenResponses SSE** on
/v1/responses with deliberately NON-SPEC usage shapes so the driver's
`calculate_usage` override is exercised end to end.

  GET  /health              200 {"status":"ok"}
  GET  /v1/models           fixed model list
  POST /v1/responses        native OpenResponses SSE; the terminal
                           response.completed carries a chat-style
                           usage blob (prompt_tokens/completion_tokens
                           + timings) instead of a spec usage object.
  POST /v1/chat/completions OpenAI chat.completion.chunk SSE + [DONE]
                           (the real API port speaks chat natively; the
                           driver relays these chunks).
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

    with open(os.path.join(STATE_DIR, "env.json"), "w") as fh:
        json.dump(
            {k: v for k, v in os.environ.items() if k.startswith("HALOGEN_")},
            fh,
            indent=1,
        )

    if "all" not in sys.argv[2:]:
        print("fake-flash: expected entrypoint arg 'all'", file=sys.stderr)
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
                    {"id": "halogen-qwen3.8-flash", "object": "model",
                     "owned_by": "fake-flash"}
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
                with open(os.path.join(STATE_DIR, "last_request.json"), "w") as fh:
                    json.dump(request, fh)

                def frames():
                    yield sse({"type": "response.created", "response": {
                        "id": "resp_flash_1", "object": "response",
                        "status": "in_progress", "output": []}})
                    for i in range(N_DELTAS):
                        yield sse({"type": "response.output_text.delta",
                                   "item_id": "msg_1", "output_index": 0,
                                   "content_index": 0, "delta": "tok%d" % i})
                    # NON-SPEC terminal usage: chat-style counts + native
                    # timings, no input_tokens/output_tokens. The driver's
                    # calculate_usage override must normalize it.
                    yield sse({"type": "response.completed", "response": {
                        "id": "resp_flash_1", "object": "response",
                        "status": "completed", "output": [],
                        "usage": {
                            "prompt_tokens": 100,
                            "completion_tokens": N_DELTAS,
                            "timings": {"cache_n": 80, "ttft": 0.4}}}})
                self._stream(frames, "responses_cancelled")
            elif self.path == "/v1/chat/completions":
                with open(os.path.join(STATE_DIR, "last_chat_request.json"), "w") as fh:
                    json.dump(request, fh)
                if os.environ.get("FAKE_CHAT_STATUS"):
                    code = int(os.environ["FAKE_CHAT_STATUS"])
                    self._send(code, {"error": {"message": "chat unavailable"}})
                    return

                def frames():
                    yield sse({"id": "chatcmpl_flash", "object": "chat.completion.chunk",
                               "created": 1, "model": "halogen-qwen3.8-flash",
                               "choices": [{"index": 0,
                                            "delta": {"role": "assistant"}}]},
                              event_type="message")
                    for i in range(N_DELTAS):
                        yield sse({"id": "chatcmpl_flash",
                                   "object": "chat.completion.chunk",
                                   "created": 1, "model": "halogen-qwen3.8-flash",
                                   "choices": [{"index": 0,
                                                "delta": {"content": "tok%d" % i},
                                                "finish_reason": None}]})
                    yield sse({"id": "chatcmpl_flash",
                               "object": "chat.completion.chunk",
                               "created": 1, "model": "halogen-qwen3.8-flash",
                               "choices": [{"index": 0, "delta": {},
                                           "finish_reason": "stop"}]})
                    # OpenAI streaming-shape final usage chunk: choices
                    # empty, usage present (what include_usage produces).
                    yield sse({"id": "chatcmpl_flash",
                               "object": "chat.completion.chunk",
                               "created": 1, "model": "halogen-qwen3.8-flash",
                               "choices": [],
                               "usage": {"prompt_tokens": 100,
                                         "completion_tokens": N_DELTAS,
                                         "total_tokens": 100 + N_DELTAS}})
                    yield b"data: [DONE]\\n\\n"
                self._stream(frames, "chat_cancelled")
            else:
                self._send(404, {"error": "not found"})

    port = int(os.environ.get("HALOGEN_API_PORT", "0"))
    assert port > 0, "fake flash requires HALOGEN_API_PORT"
    mark("server_started")
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    server.serve_forever()
    """
)

EARLY_EXIT_SCRIPT = textwrap.dedent(
    """\
    import sys
    print("fake-flash: loading w4b checkpoint", file=sys.stderr)
    print("fake-flash: fatal error: npu attach failed", file=sys.stderr)
    sys.exit(5)
    """
)


def _write_script(path: Path, content: str) -> Path:
    shebang = f"#!{sys.executable}\n"
    path.write_text(shebang + content)
    path.chmod(path.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return path


@pytest.fixture
def fake_flash_binary(tmp_path: Path) -> str:
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    script = _write_script(tmp_path / "fake_flash_server.py", FAKE_SCRIPT)
    wrapper = tmp_path / "fake_flash"
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
        "MACHINE_UID": "flash-test",
        "PROVIDER_REGISTRATION_TOKEN": "tok",
        "ADMIN_BASE_URL": "http://localhost:9999",
        "CACHE_DIR": tmp_path / "cache",
        "MODELS_DIR": tmp_path / "models",
        "PROVIDER_PORT": provider_port,
        "HALOGEN_FLASH_SERVER_PATH": binary,
        "SERVER_START_HEALTH_TIMEOUT": 15,
        # Boot budget is a separate (huge) knob; keep tests bounded.
        "ENGINE_BOOT_TIMEOUT": 15,
        "MACHINE_METRICS_INTERVAL": 0.05,
    }
    base.update(kw)
    Path(base["CACHE_DIR"]).mkdir(parents=True, exist_ok=True)
    Path(base["MODELS_DIR"]).mkdir(parents=True, exist_ok=True)
    return ProviderSettings(**base)  # type: ignore[arg-type]


@pytest.fixture
def local_artifacts(tmp_path: Path) -> dict[str, str]:
    ckpt = tmp_path / "models" / "qwen38-flash-next-w4b.hgn"
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    ckpt.write_bytes(b"hgn-fake")
    tok = tmp_path / "models" / "tokenizer"
    tok.mkdir(parents=True, exist_ok=True)
    (tok / "tokenizer.json").write_text("{}")
    return {"checkpoint": str(ckpt), "tokenizer": str(tok)}


# NPU probe stubs: fake available / unavailable results.
NPU_AVAILABLE = {
    "available": True,
    "checks": {"device": True, "xrt": True, "engine": True},
    "fabric_clock_held": True,
    "fabric_clock_detail": "held",
    "reasons": [],
}
NPU_UNAVAILABLE = {
    "available": False,
    "checks": {"device": False, "xrt": False, "engine": False},
    "fabric_clock_held": None,
    "fabric_clock_detail": "no fabric clock control readable",
    "reasons": ["the NPU driver (amdxdna) is not loaded on this host"],
}
