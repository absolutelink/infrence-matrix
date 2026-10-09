"""Shared fixtures for talkies provider tests.

Two stubs, mirroring how gufo/llama-cpp stub their engines:

1. **Fake talkies python** (subprocess lifecycle tests): an executable
   standing in for ``TALKIES_PYTHON``. The driver invokes it as
   ``<python> -m uvicorn talkies.server:app --host 127.0.0.1 --port P``;
   the fake parses ``--port``, dumps the uvicorn argv + the TALKIES_* env
   it received into a state dir, and serves ``/healthz`` + ``/api/ps``
   (stdlib http.server) with the pinned slug present unless
   ``FAKE_TALKIES_PS_EMPTY=1``.

2. **Stub talkies engine** (audio-proxy tests): a real FastAPI app served
   by uvicorn on an ephemeral port implementing the slice of the talkies
   surface the driver proxies — ``/healthz``, ``/api/ps``, speech (wav +
   chunked PCM + error paths), multipart transcriptions, the voice
   catalog, and the live-ASR WS echo — with bearer auth like the real
   server (everything except /healthz).
"""

import json
import stat
import sys
import textwrap
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import pytest
from provider_lib.config import ProviderSettings

TTS_SLUG = "qwen3-tts-1.7b-custom"
ASR_SLUG = "whisper-large-v3-turbo"

REGISTRY: dict[str, Any] = {
    "models": {
        TTS_SLUG: {
            "repo": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
            "revision": "deadbeef",
            "executor": "qwen3_tts",
        },
        ASR_SLUG: {
            "repo": "deepdml/faster-whisper-large-v3-turbo-ct2",
            "executor": "whisper",
            "dependencies": ["some/dep-repo"],
        },
        "sherpa-zipformer-en-left-64": {
            "repo": "csukuangfj/sherpa-onnx-streaming-zipformer-en-2023-06-26",
            "executor": "sherpa",
            "download_patterns": ["tokens.txt", "encoder*.onnx"],
        },
    }
}


@pytest.fixture
def talkies_registry(tmp_path: Path) -> Path:
    """A bundled-registry stand-in at TALKIES_MODELS_FILE."""
    path = tmp_path / "models.json"
    path.write_text(json.dumps(REGISTRY))
    return path


@pytest.fixture
def model_dir(tmp_path: Path) -> Path:
    """A pre-populated prefetch target ($DATA/models/<slug>) so no network
    is ever touched. The default data root is MODELS_DIR (tmp/models)."""
    d = tmp_path / "models" / "models" / TTS_SLUG
    d.mkdir(parents=True)
    (d / "model.safetensors").write_bytes(b"weights")
    return d


def make_settings(
    tmp_path: Path,
    python: str = "/opt/venv/bin/python",
    *,
    models_file: Path | None = None,
    data_dir: Path | None = None,
    **kw: Any,
) -> ProviderSettings:
    base: dict[str, Any] = {
        "MACHINE_UID": "talkies-test",
        "MACHINE_SECRET": "tok",
        "AGENT_ID": "test-agent",
        "ADMIN_BASE_URL": "http://localhost:9999",
        "CACHE_DIR": tmp_path / "cache",
        "MODELS_DIR": tmp_path / "models",
        "PROVIDER_PORT": 8181,
        "TALKIES_PYTHON": python,
        "TALKIES_MODELS_FILE": str(models_file or (tmp_path / "models.json")),
        "TALKIES_DATA_DIR": str(data_dir) if data_dir else "",
        "TALKIES_BOOT_TIMEOUT": 15,
        "MACHINE_METRICS_INTERVAL": 0.05,
    }
    base.update(kw)
    Path(base["CACHE_DIR"]).mkdir(parents=True, exist_ok=True)
    Path(base["MODELS_DIR"]).mkdir(parents=True, exist_ok=True)
    return ProviderSettings(**base)  # type: ignore[arg-type]


# ----------------------------------------------------------------------
# 1. Fake talkies python (spawned by the driver as a subprocess)
# ----------------------------------------------------------------------

FAKE_ENGINE_SCRIPT = textwrap.dedent(
    """\
    import json
    import os
    import sys
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    STATE_DIR = os.environ["FAKE_TALKIES_STATE"]

    argv = sys.argv
    port = int(argv[argv.index("--port") + 1]) if "--port" in argv else 0
    assert port > 0, "fake talkies requires --port"

    def _record():
        env = {k: v for k, v in os.environ.items() if k.startswith("TALKIES_")}
        env["HF_HUB_OFFLINE"] = os.environ.get("HF_HUB_OFFLINE", "")
        env["HF_HOME"] = os.environ.get("HF_HOME", "")
        with open(os.path.join(STATE_DIR, "spawn.json"), "w") as fh:
            json.dump({"argv": argv, "env": env}, fh)
    _record()

    PRELOAD = [s for s in os.environ.get("TALKIES_PRELOAD", "").split(",") if s]
    PS_EMPTY = os.environ.get("FAKE_TALKIES_PS_EMPTY") == "1"
    TOKEN = os.environ.get("TALKIES_AUTH_TOKEN", "")


    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args):
            pass

        def _send(self, code, body):
            data = json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _authed(self):
            return not TOKEN or self.headers.get("Authorization") == f"Bearer {TOKEN}"

        def do_GET(self):
            if self.path == "/healthz":
                self._send(200, {"ok": True, "models": PRELOAD})
            elif self.path == "/api/ps":
                if not self._authed():
                    self._send(401, {"detail": "unauthorized"})
                    return
                models = [] if PS_EMPTY else [{"id": s} for s in PRELOAD]
                self._send(200, {"models": models})
            else:
                self._send(404, {"detail": "not found"})

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    server.serve_forever()
    """
)

EARLY_EXIT_SCRIPT = textwrap.dedent(
    """\
    import sys
    print("talkies: loading model", file=sys.stderr)
    print("talkies: fatal error: CUDA out of memory", file=sys.stderr)
    sys.exit(3)
    """
)


def _write_script(path: Path, content: str) -> Path:
    shebang = f"#!{sys.executable}\n"
    path.write_text(shebang + content)
    path.chmod(path.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return path


@pytest.fixture
def fake_talkies_python(tmp_path: Path) -> str:
    """Executable standing in for $TALKIES_PYTHON (uvicorn argv compatible)."""
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    script = _write_script(tmp_path / "fake_talkies_engine.py", FAKE_ENGINE_SCRIPT)
    wrapper = tmp_path / "fake_python"
    wrapper.write_text(
        f'#!/bin/sh\nexport FAKE_TALKIES_STATE="{state_dir}"\n'
        f'exec {sys.executable} {script} "$@"\n'
    )
    wrapper.chmod(wrapper.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return str(wrapper)


@pytest.fixture
def early_exit_python(tmp_path: Path) -> str:
    return str(_write_script(tmp_path / "early_exit.py", EARLY_EXIT_SCRIPT))


@pytest.fixture
def spawn_state(tmp_path: Path) -> Path:
    return tmp_path / "state"


# ----------------------------------------------------------------------
# 2. Stub talkies engine (real FastAPI app, in-process uvicorn)
# ----------------------------------------------------------------------


def make_stub_engine(slug: str = TTS_SLUG, token: str = "stub-token") -> Any:
    """FastAPI app mimicking the talkies server surface the driver proxies."""
    from fastapi import FastAPI, Form, Header, HTTPException, UploadFile, WebSocket
    from fastapi.responses import JSONResponse, Response, StreamingResponse

    stub: dict[str, Any] = {
        "last_speech": None,
        "last_transcription": None,
        "last_voices_query": None,
        "ps_models": [slug],
    }

    app = FastAPI()

    def _check(authorization: str | None) -> None:
        if token and authorization != f"Bearer {token}":
            raise HTTPException(status_code=401, detail="unauthorized")

    @app.get("/healthz")
    def healthz() -> dict[str, Any]:
        return {"ok": True, "device": "cpu", "models": [slug]}

    @app.get("/api/ps")
    def ps(authorization: str | None = Header(default=None)) -> dict[str, Any]:
        _check(authorization)
        return {"models": [{"id": s} for s in stub["ps_models"]]}

    @app.post("/v1/audio/speech")
    async def speech(body: dict[str, Any]) -> Response:
        stub["last_speech"] = body
        if body.get("model") != slug:
            raise HTTPException(status_code=404, detail="unknown model")
        if str(body.get("input") or "") == "boom":
            raise HTTPException(status_code=400, detail="input rejected")
        fmt = str(body.get("response_format") or "mp3").lower()
        if fmt == "pcm":

            async def chunks():
                for _ in range(4):
                    yield b"\x00\x01" * 100

            return StreamingResponse(
                chunks(),
                media_type="application/octet-stream",
                headers={"X-Sample-Rate": "24000"},
            )
        ctype = {
            "wav": "audio/wav",
            "mp3": "audio/mpeg",
            "opus": "audio/ogg",
            "aac": "audio/aac",
            "flac": "audio/flac",
        }.get(fmt, "application/octet-stream")
        return Response(content=b"RIFFfake-audio", media_type=ctype)

    @app.post("/v1/audio/transcriptions")
    async def transcriptions(
        authorization: str | None = Header(default=None),
        file: UploadFile | None = None,
        model: str = Form(default=""),
        language: str | None = Form(default=None),
        response_format: str | None = Form(default=None),
    ) -> JSONResponse:
        _check(authorization)
        body = await file.read() if file is not None else b""
        stub["last_transcription"] = {
            "model": model,
            "language": language,
            "response_format": response_format,
            "filename": file.filename if file else None,
            "bytes": body,
        }
        if model != slug:
            raise HTTPException(status_code=404, detail="unknown model")
        return JSONResponse({"text": f"stub transcript for {model}"})

    @app.get("/v1/audio/voices")
    async def voices(
        authorization: str | None = Header(default=None),
        model: str | None = None,
    ) -> dict[str, Any]:
        _check(authorization)
        stub["last_voices_query"] = model
        return {
            "voices": [
                {
                    "voice": "Vivian",
                    "model": slug,
                    "default": True,
                    "origin": "builtin",
                },
                {
                    "voice": "clone1",
                    "model": slug,
                    "default": False,
                    "origin": "custom",
                },
            ]
        }

    @app.websocket("/v1/audio/transcriptions/stream")
    async def stream(ws: WebSocket) -> None:
        # talkies enforces the bearer on the WS upgrade too; rejecting
        # pre-accept surfaces as an HTTP 403 to the bridge client — this
        # locks the driver's additional_headers usage end to end.
        if token and ws.headers.get("authorization") != f"Bearer {token}":
            await ws.close(code=1008)
            return
        await ws.accept()
        start = json.loads(await ws.receive_text())
        await ws.send_text(
            json.dumps(
                {
                    "type": "ready",
                    "model": start.get("model"),
                    "encoding": "pcm_s16le",
                    "sample_rate": 16000,
                    "channels": 1,
                }
            )
        )
        try:
            while True:
                message = await ws.receive()
                if message["type"] == "websocket.disconnect":
                    return
                if message.get("text") is not None:
                    control = json.loads(message["text"])
                    if control.get("type") == "end":
                        await ws.send_text(
                            json.dumps({"type": "final", "text": "done"})
                        )
                        await ws.send_text(
                            json.dumps({"type": "stats", "canceled": False})
                        )
                        await ws.close(code=1000)
                        return
                    await ws.send_text(json.dumps({"type": "echo-text", **control}))
                elif message.get("bytes") is not None:
                    await ws.send_text(
                        json.dumps(
                            {
                                "type": "partial",
                                "bytes_seen": len(message["bytes"]),
                                "is_final": False,
                            }
                        )
                    )
        except Exception:  # noqa: BLE001
            return

    return app, stub


@asynccontextmanager
async def serve_stub(engine: Any) -> AsyncIterator[tuple[str, int]]:
    """Run the stub engine on an ephemeral loopback port."""
    import asyncio

    import uvicorn

    config = uvicorn.Config(engine, host="127.0.0.1", port=0, log_level="warning")
    server = uvicorn.Server(config)
    task = asyncio.create_task(server.serve())
    while not server.started:
        await asyncio.sleep(0.02)
    port = int(server.servers[0].sockets[0].getsockname()[1])
    try:
        yield f"http://127.0.0.1:{port}", port
    finally:
        server.should_exit = True
        await task
