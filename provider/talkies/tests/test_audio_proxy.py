"""Phase 24 audio-proxy tests against the stub talkies engine.

The driver's engine hooks (speech / transcribe / voices / voice enrollment
/ live-ASR WS bridge) run against a real in-process uvicorn stub of the
talkies surface (conftest.make_stub_engine), with the alias -> slug
rewrite asserted at the engine boundary. The WS bridge is exercised
end-to-end through the agent's own /v1 surface (registry-routed).
"""

import asyncio
import json
from typing import Any

import pytest
import websockets
from conftest import TTS_SLUG, make_settings, make_stub_engine, serve_stub
from fastapi import HTTPException
from provider_lib.app_factory import BackendOverrides, create_provider_app
from provider_lib.backend import BackendLifecycle
from provider_lib.config_update import ConfigState
from provider_lib.registry import BackendHandle, BackendRegistry

from provider_talkies.driver import TalkiesBackend


def _connected(
    tmp_path: Any, port: int, token: str, slug: str = TTS_SLUG
) -> TalkiesBackend:
    """A driver wired to the stub engine without spawning a process."""
    driver = TalkiesBackend(make_settings(tmp_path), {"model": {"slug": slug}})
    driver.apply_config({"model": {"slug": slug}})
    driver.backend_port = port
    driver.slug = slug
    driver._auth_token = token
    driver._data_dir = tmp_path / "data"
    return driver


async def _collect(stream: Any) -> bytes:
    return b"".join([chunk async for chunk in stream])


# ---------------------------------------------------------------------------
# speech
# ---------------------------------------------------------------------------


async def test_speech_rewrites_alias_to_slug_and_relays_wav(tmp_path) -> None:
    engine, stub = make_stub_engine()
    async with serve_stub(engine) as (_url, port):
        driver = _connected(tmp_path, port, "stub-token")
        try:
            stream = driver.speech(
                {"model": "my-tts-alias", "input": "hello", "response_format": "wav"}
            )
            data = await _collect(stream.chunks)
            assert stream.headers == {"Content-Type": "audio/wav"}
            assert data == b"RIFFfake-audio"
            # The engine saw the slug, never the alias.
            assert stub["last_speech"]["model"] == TTS_SLUG
            assert stub["last_speech"]["input"] == "hello"
        finally:
            await driver.aclose()


async def test_speech_pcm_headers_and_incremental_chunks(tmp_path) -> None:
    engine, _stub = make_stub_engine()
    async with serve_stub(engine) as (_url, port):
        driver = _connected(tmp_path, port, "stub-token")
        try:
            stream = driver.speech(
                {"model": "my-tts-alias", "input": "hello", "response_format": "pcm"}
            )
            assert stream.headers == {
                "Content-Type": "application/octet-stream",
                "X-Sample-Rate": "24000",
            }
            chunks = [c async for c in stream.chunks]
            assert len(chunks) > 1  # incremental PCM
            assert all(isinstance(c, bytes) and c for c in chunks)
        finally:
            await driver.aclose()


async def test_speech_applies_recorded_defaults(tmp_path) -> None:
    engine, stub = make_stub_engine()
    async with serve_stub(engine) as (_url, port):
        driver = TalkiesBackend(
            make_settings(tmp_path),
            {
                "model": {"slug": TTS_SLUG},
                "defaults": {"voice": "Vivian", "response_format": "wav", "speed": 1.1},
            },
        )
        driver.backend_port = port
        driver.slug = TTS_SLUG
        driver._auth_token = "stub-token"
        try:
            stream = driver.speech({"model": "alias", "input": "hi"})
            await _collect(stream.chunks)
            body = stub["last_speech"]
            assert body["voice"] == "Vivian"
            assert body["response_format"] == "wav"
            assert body["speed"] == 1.1
            assert stream.headers["Content-Type"] == "audio/wav"
        finally:
            await driver.aclose()


async def test_speech_upstream_error_raises_mapped_httpexception(tmp_path) -> None:
    """Truth of the S1 contract: the route commits HTTP 200 before the pump
    runs the generator, so an upstream non-2xx cannot be remapped to a real
    status — the driver raises HTTPException (upstream status + detail) from
    the generator, which ABORTS the stream (and releases the slot, next
    test) rather than relaying an error body as audio bytes."""
    engine, _stub = make_stub_engine()
    async with serve_stub(engine) as (_url, port):
        driver = _connected(tmp_path, port, "stub-token")
        try:
            stream = driver.speech({"model": "alias", "input": "boom"})
            with pytest.raises(HTTPException) as exc_info:
                await _collect(stream.chunks)
            assert exc_info.value.status_code == 400
            assert "input rejected" in str(exc_info.value.detail)
        finally:
            await driver.aclose()


async def test_speech_error_releases_lifecycle_slot(tmp_path) -> None:
    """The pump's release-on-upstream-close holds for the error path too."""
    engine, _stub = make_stub_engine()
    async with serve_stub(engine) as (_url, port):
        driver = _connected(tmp_path, port, "stub-token")

        async def fake_start() -> None:
            return None

        async def fake_health() -> bool:
            return True

        driver.start = fake_start  # type: ignore[method-assign]
        driver.health = fake_health  # type: ignore[method-assign]
        lifecycle = BackendLifecycle(driver, capacity=1)
        try:
            await lifecycle.start()
            stream = await lifecycle.speech({"model": TTS_SLUG, "input": "boom"})
            with pytest.raises(HTTPException):
                await _collect(stream.chunks)
            for _ in range(100):
                if lifecycle.in_flight == 0:
                    break
                await asyncio.sleep(0.02)
            assert lifecycle.in_flight == 0
        finally:
            await driver.aclose()


# ---------------------------------------------------------------------------
# transcribe
# ---------------------------------------------------------------------------


async def test_transcribe_relays_multipart_with_slug(tmp_path) -> None:
    engine, stub = make_stub_engine(slug=TTS_SLUG)
    async with serve_stub(engine) as (_url, port):
        driver = _connected(tmp_path, port, "stub-token")
        try:
            status, headers, body = await driver.transcribe(
                {"model": "my-asr-alias", "language": "en", "response_format": "json"},
                [("file", "clip.wav", b"RIFFfake", "audio/wav")],
            )
            assert status == 200
            assert headers["content-type"].startswith("application/json")
            assert json.loads(body)["text"] == f"stub transcript for {TTS_SLUG}"
            rec = stub["last_transcription"]
            assert rec["model"] == TTS_SLUG  # alias rewritten
            assert rec["language"] == "en"
            assert rec["filename"] == "clip.wav"
            assert rec["bytes"] == b"RIFFfake"
        finally:
            await driver.aclose()


async def test_transcribe_relay_returns_upstream_triple(tmp_path) -> None:
    engine, _stub = make_stub_engine()
    async with serve_stub(engine) as (_url, port):
        driver = _connected(tmp_path, port, "stub-token")
        try:
            # Whatever the engine answers relays as the raw triple.
            status, _headers, body = await driver.transcribe(
                {"model": "alias"}, [("file", "x.wav", b"RIFF", "audio/wav")]
            )
            assert status == 200  # driver always rewrites to its own slug
            assert b"stub transcript" in body
        finally:
            await driver.aclose()


# ---------------------------------------------------------------------------
# voices catalog + enrollment
# ---------------------------------------------------------------------------


async def test_voices_passthrough_with_slug_query(tmp_path) -> None:
    engine, stub = make_stub_engine()
    async with serve_stub(engine) as (_url, port):
        driver = _connected(tmp_path, port, "stub-token")
        try:
            catalog = await driver.voices()
            assert stub["last_voices_query"] == TTS_SLUG
            names = {v["voice"] for v in catalog["voices"]}
            assert {"Vivian", "clone1"} <= names
        finally:
            await driver.aclose()


async def test_voice_put_writes_wav_and_sidecars(tmp_path) -> None:
    engine, _stub = make_stub_engine()
    async with serve_stub(engine) as (_url, port):
        driver = _connected(tmp_path, port, "stub-token")
        try:
            summary = await driver.voice_put("myclone", b"RIFFwav", "ref text", "en")
            voices = tmp_path / "data" / "custom-voices"
            assert (voices / "myclone.wav").read_bytes() == b"RIFFwav"
            assert (voices / "myclone.txt").read_text() == "ref text"
            assert (voices / "myclone.lang").read_text() == "en"
            assert summary["voice_id"] == "myclone"
            assert summary["size_bytes"] == len(b"RIFFwav")
            assert summary["model"] == TTS_SLUG

            # Sidecars are optional and cleared when omitted on re-enroll.
            await driver.voice_put("myclone", b"RIFFwav2", None, None)
            assert not (voices / "myclone.txt").exists()
            assert not (voices / "myclone.lang").exists()
        finally:
            await driver.aclose()


async def test_voice_put_rejects_traversal_names(tmp_path) -> None:
    engine, _stub = make_stub_engine()
    async with serve_stub(engine) as (_url, port):
        driver = _connected(tmp_path, port, "stub-token")
        try:
            for bad in ("../evil", "a/b", "a\\b", "..", ".", "x" * 65, ""):
                with pytest.raises(HTTPException) as exc_info:
                    await driver.voice_put(bad, b"RIFF", None, None)
                assert exc_info.value.status_code == 400
            # Nothing escaped into the parent dir.
            assert not (tmp_path / "data" / "evil.wav").exists()
        finally:
            await driver.aclose()


async def test_voice_delete_removes_files(tmp_path) -> None:
    engine, _stub = make_stub_engine()
    async with serve_stub(engine) as (_url, port):
        driver = _connected(tmp_path, port, "stub-token")
        try:
            await driver.voice_put("gone", b"RIFF", "text", "en")
            result = await driver.voice_delete("gone")
            assert result["deleted"] is True
            voices = tmp_path / "data" / "custom-voices"
            assert not (voices / "gone.wav").exists()
            assert not (voices / "gone.txt").exists()
            assert not (voices / "gone.lang").exists()
            assert (await driver.voice_delete("gone"))["deleted"] is False
        finally:
            await driver.aclose()


# ---------------------------------------------------------------------------
# live ASR WS bridge (end-to-end through the agent /v1 surface)
# ---------------------------------------------------------------------------


async def test_transcribe_stream_bridge_rewrites_and_relays(tmp_path) -> None:
    import uvicorn

    engine, _stub = make_stub_engine()
    driver = _connected(tmp_path, 0, "stub-token")

    async def fake_start() -> None:
        return None

    async def fake_health() -> bool:
        return True

    driver.start = fake_start  # type: ignore[method-assign]
    driver.health = fake_health  # type: ignore[method-assign]

    async with serve_stub(engine) as (_url, engine_port):
        driver.backend_port = engine_port
        lifecycle = BackendLifecycle(driver, capacity=1, instance_id="inst-1")
        await lifecycle.start()
        registry = BackendRegistry()
        registry.add(
            BackendHandle("inst-1", lifecycle, ConfigState(), alias="live-asr")
        )
        settings = make_settings(tmp_path)
        app = create_provider_app(
            settings,
            BackendOverrides(
                provider_type="talkies", version="test", registry=registry
            ),
        )
        config = uvicorn.Config(app, host="127.0.0.1", port=0, log_level="warning")
        server = uvicorn.Server(config)
        task = asyncio.create_task(server.serve())
        while not server.started:
            await asyncio.sleep(0.02)
        agent_port = int(server.servers[0].sockets[0].getsockname()[1])
        try:
            async with websockets.connect(
                f"ws://127.0.0.1:{agent_port}/v1/audio/transcriptions/stream"
                "?model=live-asr"
            ) as ws:
                # The start frame carries the client alias; the bridge must
                # rewrite it to the engine slug before the engine accepts.
                await ws.send(
                    json.dumps(
                        {
                            "type": "start",
                            "model": "live-asr",
                            "encoding": "pcm_s16le",
                            "sample_rate": 16000,
                            "channels": 1,
                        }
                    )
                )
                ready = json.loads(await asyncio.wait_for(ws.recv(), 5))
                assert ready["type"] == "ready"
                assert ready["model"] == TTS_SLUG  # rewritten

                await ws.send(b"\x00\x01" * 160)
                partial = json.loads(await asyncio.wait_for(ws.recv(), 5))
                assert partial["type"] == "partial"
                assert partial["bytes_seen"] == 320

                await ws.send(json.dumps({"type": "end"}))
                finals = [
                    json.loads(await asyncio.wait_for(ws.recv(), 5)) for _ in range(2)
                ]
                assert [m["type"] for m in finals] == ["final", "stats"]
        finally:
            server.should_exit = True
            await task
            await lifecycle.stop()
            await driver.aclose()
        # Slot released when the bridge returns on disconnect.
        assert lifecycle.in_flight == 0


# ---------------------------------------------------------------------------
# review-fix regressions (S2 round 1)
# ---------------------------------------------------------------------------


async def test_transcribe_does_not_leak_speech_defaults(tmp_path) -> None:
    """L1: speech-only recorded defaults (voice / response_format=pcm /
    speed / instructions) must never be injected into a transcription form;
    only the ASR-meaningful `language` merges."""
    engine, stub = make_stub_engine()
    async with serve_stub(engine) as (_url, port):
        driver = TalkiesBackend(
            make_settings(tmp_path),
            {
                "model": {"slug": TTS_SLUG},
                "defaults": {
                    "language": "en",
                    "voice": "Vivian",
                    "response_format": "pcm",
                    "speed": 1.5,
                },
            },
        )
        driver.backend_port = port
        driver.slug = TTS_SLUG
        driver._auth_token = "stub-token"
        try:
            status, _headers, _body = await driver.transcribe(
                {"model": "alias"}, [("file", "c.wav", b"RIFF", "audio/wav")]
            )
            assert status == 200
            rec = stub["last_transcription"]
            assert rec["language"] == "en"  # ASR-relevant default applied
            assert rec["response_format"] is None  # speech-only: not leaked
        finally:
            await driver.aclose()


async def test_transcribe_relay_is_bounded(tmp_path) -> None:
    """L2: the relay uses a finite httpx timeout (TALKIES_BOOT_TIMEOUT),
    never None — a hung engine must not pin the inference slot forever."""
    engine, _stub = make_stub_engine()
    async with serve_stub(engine) as (_url, port):
        driver = _connected(tmp_path, port, "stub-token")
        driver._settings.TALKIES_BOOT_TIMEOUT = 7
        try:
            seen: dict[str, Any] = {}
            real_post = driver._http_client().post

            def spy_post(*args: Any, **kwargs: Any):
                seen["timeout"] = kwargs.get("timeout")
                return real_post(*args, **kwargs)

            driver._client.post = spy_post  # type: ignore[method-assign]
            await driver.transcribe(
                {"model": "alias"}, [("file", "c.wav", b"RIFF", "audio/wav")]
            )
            assert seen["timeout"] == 7.0
        finally:
            await driver.aclose()


class _NeverUsedWS:
    """Agent-side socket the bridge must not touch before the engine
    connection succeeds."""

    async def receive(self) -> dict[str, Any]:  # pragma: no cover
        raise AssertionError("engine connect must fail first")

    async def send_text(self, _t: str) -> None:  # pragma: no cover
        raise AssertionError("engine connect must fail first")

    async def send_bytes(self, _b: bytes) -> None:  # pragma: no cover
        raise AssertionError("engine connect must fail first")


async def test_transcribe_stream_requires_engine_auth(tmp_path) -> None:
    """N1: the bridge sends the bearer on the WS upgrade; the stub rejects a
    wrong token pre-accept (HTTP 403), so the bridge raises instead of
    silently serving an unauthenticated engine."""
    engine, _stub = make_stub_engine(token="right-token")
    async with serve_stub(engine) as (_url, port):
        driver = _connected(tmp_path, port, "wrong-token")
        try:
            with pytest.raises(Exception) as exc_info:
                await driver.transcribe_stream(_NeverUsedWS())  # type: ignore[arg-type]
            assert (
                "403" in str(exc_info.value) or "status" in str(exc_info.value).lower()
            )
        finally:
            await driver.aclose()
