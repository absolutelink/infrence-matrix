"""Phase 24 (audio) provider data-plane: speech / transcription / voices.

Covers the slot-admission + release-on-upstream-close contract of the new
``/v1/audio/*`` routes and their ``BackendLifecycle`` wrappers, mirroring the
embeddings + stream-close tests. All drivers are fakes (no GPU).
"""

import asyncio
import json
from collections.abc import AsyncIterator
from typing import Any

import httpx
import pytest

from provider_lib import app_factory as app_factory_module
from provider_lib.app_factory import BackendOverrides, create_provider_app
from provider_lib.backend import BackendDriver, BackendLifecycle, SpeechStream
from provider_lib.config import ProviderSettings
from provider_lib.registry import BackendHandle, BackendRegistry


def _settings() -> ProviderSettings:
    return ProviderSettings(  # type: ignore[call-arg]
        MACHINE_UID="m-audio",
        MACHINE_SECRET="tok",
        AGENT_ID="m-audio-agent",
        ADMIN_BASE_URL="http://admin:8000",
        CACHE_DIR="/tmp/prov_test_cache",
        MODELS_DIR="/tmp/prov_test_models",
    )


class AudioDriver(BackendDriver):
    """Fake driver implementing the full audio surface with close tracking."""

    def __init__(self, *, tag: str = "A", infinite_speech: bool = False) -> None:
        self.tag = tag
        self.infinite_speech = infinite_speech
        self.speech_open = 0
        self.speech_close = 0
        self.transcribe_calls = 0
        self.voices_calls = 0
        self.voice_put_calls = 0
        self.voice_delete_calls = 0
        self.transcribe_stream_calls = 0
        # Phase 25: the served name the last voice op received (assertable).
        self.last_voice_model: str | None = None

    async def start(self) -> None:
        pass

    async def stop(self) -> None:
        pass

    async def health(self) -> bool:
        return True

    async def list_models(self) -> list[dict[str, Any]]:
        return [{"id": self.tag, "object": "model"}]

    def stream_responses(
        self, request: dict[str, Any]
    ) -> AsyncIterator[dict[str, Any]]:
        async def gen() -> AsyncIterator[dict[str, Any]]:
            yield {"type": "response.completed", "response": {"status": "completed"}}

        return gen()

    def speech(self, request: dict[str, Any]) -> SpeechStream:
        return SpeechStream(
            headers={"Content-Type": "audio/wav", "X-Tag": self.tag},
            chunks=self._speech_chunks(),
        )

    async def _speech_chunks(self) -> AsyncIterator[bytes]:
        self.speech_open += 1
        try:
            i = 0
            while self.infinite_speech or i < 3:
                yield b"chunk-%d" % i
                i += 1
                await asyncio.sleep(0.01)
        finally:
            self.speech_close += 1

    async def transcribe(
        self,
        form: dict[str, str],
        files: list[tuple[str, str, bytes, str]],
    ) -> tuple[int, dict[str, str], bytes]:
        self.transcribe_calls += 1
        body = json.dumps({"text": form.get("prompt", "x"), "tag": self.tag}).encode()
        return 200, {"content-type": "application/json"}, body

    async def voices(self, model: str | None = None) -> dict[str, Any]:
        self.voices_calls += 1
        self.last_voice_model = model
        return {"object": "list", "data": [{"voice_id": "alloy"}]}

    async def voice_put(
        self,
        name: str,
        wav: bytes,
        ref_text: str | None,
        language: str | None,
        model: str | None = None,
    ) -> dict[str, Any]:
        self.voice_put_calls += 1
        self.last_voice_model = model
        return {"object": "voice", "voice_id": name, "size_bytes": len(wav)}

    async def voice_delete(self, name: str, model: str | None = None) -> dict[str, Any]:
        self.voice_delete_calls += 1
        self.last_voice_model = model
        return {"object": "voice", "voice_id": name, "deleted": True}

    async def transcribe_stream(self, websocket: Any) -> None:
        self.transcribe_stream_calls += 1


class PlainDriver(BackendDriver):
    """Driver with NO audio surface (all defaults -> NotImplementedError)."""

    async def start(self) -> None:
        pass

    async def stop(self) -> None:
        pass

    async def health(self) -> bool:
        return True

    async def list_models(self) -> list[dict[str, Any]]:
        return [{"id": "plain", "object": "model"}]

    def stream_responses(
        self, request: dict[str, Any]
    ) -> AsyncIterator[dict[str, Any]]:
        async def gen() -> AsyncIterator[dict[str, Any]]:
            yield {"type": "response.completed", "response": {"status": "completed"}}

        return gen()


async def _make(
    driver: BackendDriver, *, started: bool = True, capacity: int = 1
) -> tuple[BackendLifecycle, httpx.AsyncClient]:
    lifecycle = BackendLifecycle(driver, capacity=capacity)
    if started:
        await lifecycle.start()
    app = create_provider_app(
        _settings(),
        BackendOverrides(provider_type="test", version="v", lifecycle=lifecycle),
    )
    client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://provider"
    )
    return lifecycle, client


# ---------------------------------------------------------------------------
# speech
# ---------------------------------------------------------------------------
async def test_speech_streams_bytes_and_releases() -> None:
    driver = AudioDriver()
    lifecycle, client = await _make(driver)
    async with client:
        resp = await client.post("/v1/audio/speech", json={"model": "m"})
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "audio/wav"
        assert resp.headers.get("x-tag") == "A"
        assert resp.content == b"chunk-0chunk-1chunk-2"
        assert lifecycle.in_flight == 0


async def test_speech_429_at_capacity() -> None:
    lifecycle, client = await _make(AudioDriver(), capacity=1)
    async with client:
        await lifecycle.acquire_slot()
        resp = await client.post("/v1/audio/speech", json={"model": "m"})
        assert resp.status_code == 429
        assert lifecycle.in_flight == 1
        await lifecycle.release_slot()


async def test_speech_503_when_not_started() -> None:
    lifecycle, client = await _make(AudioDriver(), started=False)
    async with client:
        resp = await client.post("/v1/audio/speech", json={"model": "m"})
        assert resp.status_code == 503
        assert lifecycle.in_flight == 0


async def test_speech_501_when_hooks_unset() -> None:
    lifecycle, client = await _make(PlainDriver())
    async with client:
        resp = await client.post("/v1/audio/speech", json={"model": "m"})
        assert resp.status_code == 501
        assert lifecycle.in_flight == 0


async def test_speech_release_on_abandoned_stream() -> None:
    """Consumer pulls one chunk then abandons: upstream closes, slot frees."""
    driver = AudioDriver(infinite_speech=True)
    lifecycle = BackendLifecycle(driver, capacity=1)
    await lifecycle.start()

    stream = await lifecycle.speech({})
    assert lifecycle.in_flight == 1

    it = stream.chunks.__aiter__()
    first = await it.__anext__()
    assert first == b"chunk-0"
    assert driver.speech_close == 0  # infinite upstream still open
    await stream.chunks.aclose()

    assert driver.speech_close == 1
    assert lifecycle.in_flight == 0


# ---------------------------------------------------------------------------
# transcriptions (multipart relay)
# ---------------------------------------------------------------------------
async def test_transcriptions_relay_ok() -> None:
    driver = AudioDriver()
    lifecycle, client = await _make(driver)
    async with client:
        resp = await client.post(
            "/v1/audio/transcriptions",
            data={"model": "m", "prompt": "hello"},
            files={"file": ("a.wav", b"RIFF....", "audio/wav")},
        )
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "application/json"
        assert resp.json() == {"text": "hello", "tag": "A"}
        assert driver.transcribe_calls == 1
        assert lifecycle.in_flight == 0


async def test_transcriptions_429_at_capacity() -> None:
    lifecycle, client = await _make(AudioDriver(), capacity=1)
    async with client:
        await lifecycle.acquire_slot()
        resp = await client.post(
            "/v1/audio/transcriptions",
            data={"model": "m"},
            files={"file": ("a.wav", b"x", "audio/wav")},
        )
        assert resp.status_code == 429
        assert lifecycle.in_flight == 1
        await lifecycle.release_slot()


async def test_transcriptions_503_when_not_started() -> None:
    lifecycle, client = await _make(AudioDriver(), started=False)
    async with client:
        resp = await client.post(
            "/v1/audio/transcriptions",
            data={"model": "m"},
            files={"file": ("a.wav", b"x", "audio/wav")},
        )
        assert resp.status_code == 503
        assert lifecycle.in_flight == 0


async def test_transcriptions_501_when_hooks_unset() -> None:
    lifecycle, client = await _make(PlainDriver())
    async with client:
        resp = await client.post(
            "/v1/audio/transcriptions",
            data={"model": "m"},
            files={"file": ("a.wav", b"x", "audio/wav")},
        )
        assert resp.status_code == 501
        assert lifecycle.in_flight == 0


# ---------------------------------------------------------------------------
# voices (no slot)
# ---------------------------------------------------------------------------
async def test_voices_routes_do_not_consume_slots() -> None:
    driver = AudioDriver()
    lifecycle, client = await _make(driver, capacity=1)
    async with client:
        # Occupy the only slot: catalog/enrollment ops must still succeed.
        await lifecycle.acquire_slot()
        assert lifecycle.in_flight == 1

        got = await client.get("/v1/audio/voices", params={"model": "m"})
        assert got.status_code == 200
        assert got.json()["data"][0]["voice_id"] == "alloy"

        put = await client.put(
            "/v1/audio/voices/myvoice",
            params={"model": "m", "ref_text": "hi", "language": "en"},
            content=b"RIFF....",
        )
        assert put.status_code == 200
        assert put.json()["voice_id"] == "myvoice"

        dele = await client.delete("/v1/audio/voices/myvoice", params={"model": "m"})
        assert dele.status_code == 200
        assert dele.json()["deleted"] is True

        # Still exactly the one manually-held slot; nothing leaked.
        assert lifecycle.in_flight == 1
        assert driver.voices_calls == 1
        assert driver.voice_put_calls == 1
        assert driver.voice_delete_calls == 1
        # Phase 25: the requested served name is threaded to the driver on
        # every voice op (catalog + enrollment), not dropped at the seam.
        assert driver.last_voice_model == "m"
        await lifecycle.release_slot()


async def test_voice_ops_thread_requested_model_name() -> None:
    """Phase 25: the voices GET/PUT/DELETE routes pass the request's ``model``
    (query param) down to the driver hook so a multi-model driver can resolve
    name -> engine model."""
    driver = AudioDriver()
    lifecycle, client = await _make(driver)
    async with client:
        await client.get("/v1/audio/voices", params={"model": "alpha"})
        assert driver.last_voice_model == "alpha"
        await client.put(
            "/v1/audio/voices/v1", params={"model": "beta"}, content=b"RIFF"
        )
        assert driver.last_voice_model == "beta"
        await client.delete("/v1/audio/voices/v1", params={"model": "gamma"})
        assert driver.last_voice_model == "gamma"
        # No model query param -> None (legacy single-model callers).
        await client.get("/v1/audio/voices")
        assert driver.last_voice_model is None


async def test_voices_503_when_not_started() -> None:
    _lifecycle, client = await _make(AudioDriver(), started=False)
    async with client:
        assert (
            await client.get("/v1/audio/voices", params={"model": "m"})
        ).status_code == 503


async def test_voices_501_when_hooks_unset() -> None:
    _lifecycle, client = await _make(PlainDriver())
    async with client:
        assert (
            await client.get("/v1/audio/voices", params={"model": "m"})
        ).status_code == 501


# ---------------------------------------------------------------------------
# unknown-model 404 routing (registry path)
# ---------------------------------------------------------------------------
async def _registry_client(
    handles: list[BackendHandle],
) -> httpx.AsyncClient:
    reg = BackendRegistry()
    for h in handles:
        reg.add(h)
    app = create_provider_app(
        _settings(),
        BackendOverrides(provider_type="test", version="v", registry=reg),
    )
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://provider"
    )


async def test_audio_routes_unknown_model_is_404() -> None:
    lifecycle = BackendLifecycle(AudioDriver(tag="A"), instance_id="i1")
    await lifecycle.start()
    handle = BackendHandle("i1", lifecycle, None, alias="alpha")
    client = await _registry_client([handle])
    async with client:
        assert (
            await client.post("/v1/audio/speech", json={"model": "nope"})
        ).status_code == 404
        assert (
            await client.post(
                "/v1/audio/transcriptions",
                data={"model": "nope"},
                files={"file": ("a.wav", b"x", "audio/wav")},
            )
        ).status_code == 404
        assert (
            await client.get("/v1/audio/voices", params={"model": "nope"})
        ).status_code == 404
        assert (
            await client.put(
                "/v1/audio/voices/v", params={"model": "nope"}, content=b"x"
            )
        ).status_code == 404
        assert (
            await client.delete("/v1/audio/voices/v", params={"model": "nope"})
        ).status_code == 404


# ---------------------------------------------------------------------------
# MEDIUM-1: hop-by-hop header stripping on the transcription relay.
# ---------------------------------------------------------------------------
class RelayHeaderDriver(AudioDriver):
    """AudioDriver whose transcribe() returns hop-by-hop + keep headers."""

    async def transcribe(
        self,
        form: dict[str, str],
        files: list[tuple[str, str, bytes, str]],
    ) -> tuple[int, dict[str, str], bytes]:
        headers = {
            "content-type": "application/json",
            "content-encoding": "gzip",  # httpx already decoded -> must drop
            "transfer-encoding": "chunked",  # hop-by-hop -> must drop
            "content-length": "999",  # starlette recomputes -> must drop
            "connection": "keep-alive",  # hop-by-hop -> must drop
            "x-upstream": "keep-me",  # end-to-end -> must survive
        }
        return 200, headers, b'{"text":"ok"}'


async def test_transcriptions_relay_strips_hop_by_hop() -> None:
    _lifecycle, client = await _make(RelayHeaderDriver())
    async with client:
        resp = await client.post(
            "/v1/audio/transcriptions",
            data={"model": "m"},
            files={"file": ("a.wav", b"x", "audio/wav")},
        )
        assert resp.status_code == 200
        lowered = {k.lower() for k in resp.headers.keys()}
        for bad in ("content-encoding", "transfer-encoding", "connection"):
            assert bad not in lowered, bad
        # content-length is recomputed by starlette to the real body length.
        assert resp.headers.get("content-length") == str(len(b'{"text":"ok"}'))
        # The end-to-end custom header survives.
        assert resp.headers.get("x-upstream") == "keep-me"


# ---------------------------------------------------------------------------
# MEDIUM-2: voice name validation (traversal / unsafe names -> 400, no driver).
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "encoded_name",
    ["%2e%2e", "a%2Fb", "%252e%252e", "%2e"],  # "..", "a/b", "%2e%2e", "."
)
async def test_voice_name_validation_rejects_unsafe(encoded_name: str) -> None:
    driver = AudioDriver()
    _lifecycle, client = await _make(driver)
    async with client:
        put = await client.put(
            f"/v1/audio/voices/{encoded_name}",
            params={"model": "m"},
            content=b"RIFF",
        )
        assert put.status_code == 400, encoded_name
        dele = await client.delete(
            f"/v1/audio/voices/{encoded_name}", params={"model": "m"}
        )
        assert dele.status_code == 400, encoded_name
    # The driver was never reached for any rejected name.
    assert driver.voice_put_calls == 0
    assert driver.voice_delete_calls == 0


async def test_voice_name_validation_accepts_safe() -> None:
    driver = AudioDriver()
    _lifecycle, client = await _make(driver)
    async with client:
        ok = await client.put(
            "/v1/audio/voices/my.voice-1_A", params={"model": "m"}, content=b"RIFF"
        )
        assert ok.status_code == 200
    assert driver.voice_put_calls == 1


# ---------------------------------------------------------------------------
# MEDIUM-3: oversize voice body -> 413 (limit is a module constant).
# ---------------------------------------------------------------------------
async def test_voice_put_oversize_body_is_413(monkeypatch: pytest.MonkeyPatch) -> None:
    driver = AudioDriver()
    monkeypatch.setattr(app_factory_module, "MAX_VOICE_BYTES", 8)
    _lifecycle, client = await _make(driver)
    async with client:
        resp = await client.put(
            "/v1/audio/voices/toolarge", params={"model": "m"}, content=b"x" * 32
        )
        assert resp.status_code == 413
    # Body rejected before the driver ran.
    assert driver.voice_put_calls == 0


async def test_transcriptions_oversize_upload_is_413(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    driver = AudioDriver()
    monkeypatch.setattr(app_factory_module, "MAX_TRANSCRIPTION_BYTES", 8)
    _lifecycle, client = await _make(driver)
    async with client:
        resp = await client.post(
            "/v1/audio/transcriptions",
            data={"model": "m"},
            files={"file": ("big.wav", b"x" * 32, "audio/wav")},
        )
        assert resp.status_code == 413
    # Upload rejected before the driver ran.
    assert driver.transcribe_calls == 0
