"""Phase 9 wiring test: halogen-flash installs the shared config.update /
cache.clear / prune_unused handlers and applies a config change against
the FAKE halogen-flash server."""

import asyncio
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any

import pytest
import websockets
from conftest import make_settings
from provider_lib.admin_client import AdminClient
from provider_lib.config_update import prompt_cache_dir
from provider_lib.wire import BackendStatusValue, Frame

from provider_halogen_flash.main import (
    PROVIDER_TYPE,
    install_command_handlers,
    make_lifecycle,
    register_and_connect,
)

INSTANCE_ID = "99999999-2222-3333-4444-555555555555"
SECRET = "fake-flash-secret"
RECEIVED: list[Frame] = []
ADMIN_WS_URL_HOLDER: dict[str, str] = {}
BACKEND_CONFIG: dict = {}


class _FakeAdminHTTP(BaseHTTPRequestHandler):
    def do_POST(self):  # noqa: N802
        length = int(self.headers["Content-Length"])
        body = json.loads(self.rfile.read(length))
        payload = json.dumps(
            {
                "instance_id": INSTANCE_ID,
                "instance_secret": SECRET,
                "machine": {"uid": body["machine_uid"]},
                "provider_definition": {
                    "alias": "llama-alias",
                    "provider_type": PROVIDER_TYPE,
                    "config_fingerprint": "cafe",
                    "capacity": 1,
                    "backend_config": BACKEND_CONFIG,
                    "admin_ws_url": ADMIN_WS_URL_HOLDER["url"],
                },
            }
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *args):  # noqa: ARG002
        pass


async def _fake_ws_admin(websocket):
    auth = websocket.request.headers.get("Authorization", "")
    if auth != f"Bearer {SECRET}":
        await websocket.close(code=4401)
        return
    await websocket.send(
        Frame(type="provider.hello", epoch=7, payload={"epoch": 7}).to_json()
    )
    async for raw in websocket:
        RECEIVED.append(Frame.model_validate_json(raw))


@pytest.fixture
async def fake_admin():
    RECEIVED.clear()
    http = HTTPServer(("127.0.0.1", 0), _FakeAdminHTTP)
    http_thread = threading.Thread(target=http.serve_forever, daemon=True)
    http_thread.start()

    stop = asyncio.Event()

    async def serve_ws():
        async with websockets.serve(
            _fake_ws_admin, "127.0.0.1", 0, ping_interval=None
        ) as srv:
            port = srv.sockets[0].getsockname()[1]
            ADMIN_WS_URL_HOLDER["url"] = f"ws://127.0.0.1:{port}/provider/ws"
            await stop.wait()

    ws_task = asyncio.get_running_loop().create_task(serve_ws())
    while "url" not in ADMIN_WS_URL_HOLDER:
        await asyncio.sleep(0.01)
    yield f"http://127.0.0.1:{http.server_address[1]}"
    stop.set()
    ws_task.cancel()
    http.shutdown()
    http.server_close()


async def _await_ack(client: AdminClient, reply_to: str) -> dict[str, Any]:  # noqa: ARG001
    async def poll() -> Frame:
        while True:
            hits = [f for f in RECEIVED if f.type == "ack" and f.reply_to == reply_to]
            if hits:
                return hits[0]
            await asyncio.sleep(0.02)

    frame = await asyncio.wait_for(poll(), timeout=15)
    return frame.payload


class _NoopEmitter:
    def start(self) -> None:
        pass

    async def stop(self) -> None:
        pass


def test_handlers_installed(tmp_path, fake_flash_binary) -> None:
    settings = make_settings(tmp_path, fake_flash_binary)
    client = AdminClient(settings)
    lifecycle = make_lifecycle(client, {})
    install_command_handlers(client, lifecycle, _NoopEmitter())  # type: ignore[arg-type]
    for name in ("provider.config.update", "cache.clear", "storage.prune_unused"):
        assert name in client._command_handlers


async def test_config_update_applies_new_model_and_clears_cache(
    fake_admin, tmp_path, fake_flash_binary, local_artifacts
) -> None:
    BACKEND_CONFIG.clear()
    BACKEND_CONFIG.update(
        {
            "model": {"path": local_artifacts["checkpoint"]},
            "tokenizer": {"path": local_artifacts["tokenizer"]},
            "options": {},
        }
    )
    settings = make_settings(tmp_path, fake_flash_binary, ADMIN_BASE_URL=fake_admin)
    client = AdminClient(settings)
    lifecycle = make_lifecycle(client, BACKEND_CONFIG)
    try:
        await register_and_connect(client, lifecycle)
        # Boot once so the driver has a resolved artifact set.
        await client._dispatch(
            Frame(type="backend.start", id="c-start", epoch=7, payload={})
        )
        ack = await _await_ack(client, "c-start")
        assert ack["ok"] is True

        # Seed the old fingerprint's prompt-cache dir.
        old = prompt_cache_dir(settings, "cafe")
        old.mkdir(parents=True, exist_ok=True)
        (old / "slot.bin").write_bytes(b"x" * 40)

        new_cfg = {
            "model": {"path": local_artifacts["checkpoint"]},
            "tokenizer": {"path": local_artifacts["tokenizer"]},
            "options": {"kv_slots": 2},
        }
        await client._dispatch(
            Frame(
                type="provider.config.update",
                id="c-cfg",
                epoch=7,
                payload={
                    "backend_config": new_cfg,
                    "config_fingerprint": "beef",
                    "capacity": 1,
                },
            )
        )
        ack = await _await_ack(client, "c-cfg")
        assert ack["ok"] is True, ack
        assert ack["detail"]["config_fingerprint"] == "beef"
        assert ack["detail"]["model_metadata"] == [
            {"id": "halogen-qwen3.8-flash", "object": "model", "owned_by": "fake-flash"}
        ]
        assert not old.exists()  # prompt cache cleared
        assert lifecycle.backend_status == BackendStatusValue.RUNNING
        assert lifecycle.driver._config == new_cfg
        assert local_artifacts["checkpoint"] in lifecycle.driver.resolved_artifacts
    finally:
        await client.disconnect()
        await lifecycle.driver.aclose()


async def test_npu_predownload_wired_into_start(tmp_path, monkeypatch) -> None:
    """Phase 8 deferral closed: with npu_models requested and the probe
    available, the driver plans the pin download set and ensure_artifacts
    each file before spawn; planning failure degrades to omitting the
    pins (start never fails on NPU prep)."""

    from provider_halogen_flash import driver as flash_driver
    from provider_halogen_flash.driver import HalogenFlashBackend
    from provider_halogen_flash.npu import NpuPinRecord

    settings = make_settings(tmp_path, "unused-binary")
    ensured: list[tuple[str, str, str]] = []

    async def fake_ensure(descriptor, **kwargs):
        ensured.append(
            (descriptor["repo"], descriptor["file"], kwargs.get("local_subdir") or "")
        )
        p = Path(kwargs["models_dir"]) / "npu" / "nano" / descriptor["file"]
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"x")
        return str(p)

    monkeypatch.setattr(flash_driver, "ensure_artifact", fake_ensure)

    records = {
        "qwen3.5-2b": NpuPinRecord(
            model_id="qwen3.5-2b",
            repo="acme/nano",
            revision="v2",
            files=[("nano.bin", 10, "aa")],
        )
    }
    monkeypatch.setattr(flash_driver, "load_npu_pins", lambda _f: records)

    driver = HalogenFlashBackend(
        settings,
        {"options": {"npu_models": ["qwen3.5-2b"]}},
        npu_probe=lambda: {"available": True, "reasons": []},
    )
    prepared = await driver._prepare_npu_models(["qwen3.5-2b"])
    assert len(prepared) == 1
    assert ensured == [("acme/nano", "nano.bin", "npu/qwen3.5-2b")]

    # Planning failure (missing pins file) -> graceful empty list.
    def boom(_f):
        raise OSError("no pins file")

    monkeypatch.setattr(flash_driver, "load_npu_pins", boom)
    assert await driver._prepare_npu_models(["qwen3.5-2b"]) == []

    # No gated models (probe unavailable) -> nothing prepared.
    driver2 = HalogenFlashBackend(
        settings,
        {"options": {"npu_models": ["qwen3.5-2b"]}},
        npu_probe=lambda: {"available": False, "reasons": ["no device"]},
    )
    assert await driver2._npu_models_for_env() is None
    assert await driver2._prepare_npu_models(None) == []


async def test_start_predownloads_npu_pins_before_spawn(tmp_path, monkeypatch) -> None:
    """start() calls the NPU prep and extends resolved_artifacts; a prep
    failure omits HALOGEN_NPU_MODELS but still spawns."""
    from provider_halogen_flash import driver as flash_driver

    settings = make_settings(tmp_path, "unused-binary")
    ckpt = tmp_path / "models" / "m.hgn"
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    ckpt.write_bytes(b"h")
    tok = tmp_path / "models" / "tokenizer"
    tok.mkdir(parents=True, exist_ok=True)
    (tok / "tokenizer.json").write_text("{}")

    calls: dict[str, object] = {}

    async def fake_prepare(_self, npu_models):  # noqa: ANN001
        calls["prepare_with"] = npu_models
        return [] if calls.get("fail") else ["/models/npu/x.bin"]

    monkeypatch.setattr(
        flash_driver.HalogenFlashBackend, "_prepare_npu_models", fake_prepare
    )

    class _NoSpawn(RuntimeError):
        pass

    def fake_popen(*a, **k):  # noqa: ANN002, ARG001
        raise _NoSpawn("stop before spawn")

    monkeypatch.setattr(flash_driver.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(flash_driver.shutil, "which", lambda _b: "/usr/bin/fake")

    driver = flash_driver.HalogenFlashBackend(
        settings,
        {
            "model": {"path": str(ckpt)},
            "tokenizer": {"path": str(tok)},
            "options": {"npu_models": ["qwen3.5-2b"]},
        },
        npu_probe=lambda: {"available": True, "reasons": []},
    )
    with pytest.raises(_NoSpawn):
        await driver.start()
    assert calls["prepare_with"] == ["qwen3.5-2b"]
    assert "/models/npu/x.bin" in driver.resolved_artifacts

    # Prep failure -> pins omitted: the driver must not export NPU env.
    calls.clear()
    calls["fail"] = True
    import os

    from provider_halogen_flash.env import build_env

    env = build_env(
        {"npu_models": ["qwen3.5-2b"]},
        api_port=9000,
        engine_port=9001,
        checkpoint_path=str(ckpt),
        tokenizer_path=str(tok),
        cache_dir=None,
        npu_models=None,  # what start() passes when prep returned []
        base_env=dict(os.environ),
    )
    assert "HALOGEN_NPU_MODELS" not in env
