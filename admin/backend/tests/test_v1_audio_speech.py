"""Route integration for POST /v1/audio/speech (Phase 24) with the agent
stubbed via respx.

Mirrors ``test_v1_embeddings.py``: a connected, running ``tts`` provider
instance is seeded in the UTF8 test DB so the real ``InferenceScheduler``
admits against it (base_url ``http://127.0.0.1:8081``), and the agent's
``/v1/audio/speech`` is mocked with respx (no litellm — audio bypasses it).
Asserts the admin-side speech contract (ARCHITECTURE.md §7 Audio specifics):

- 2xx relays the audio body through as a stream, preserving ``Content-Type``
  and ``X-Sample-Rate``;
- an ``AudioUsageSample`` (chars + WAV-sniffed seconds) is persisted and NO
  ``ResponseRecord`` is written;
- non-2xx relays status + JSON error body;
- 404 unknown / disabled / non-tts alias; 400 missing model / empty input;
  503 no provider;
- the scheduler slot is released on success, upstream error, and an abandoned
  (client-disconnect) stream.
"""

import io
import math
import struct
import uuid
import wave
from typing import Any

import httpx
import pytest
import respx
from fastapi.testclient import TestClient
from sqlmodel import Session

from app.api.v1 import audio_speech as audio_speech_module
from app.models import AudioUsageSample, ProviderInstance, ResponseRecord
from tests.helpers import (
    get_or_create_machine,
    make_agent,
    make_definition,
    make_instance,
)

AGENT = "http://127.0.0.1:8081"


@pytest.fixture
def client() -> TestClient:
    from app.main import app

    with TestClient(app) as c:
        yield c


def seed_audio_instance(
    session: Session,
    *,
    alias: str,
    machine_uid: str,
    modality: str = "tts",
    enabled: bool = True,
    websocket_connected: bool = True,
) -> ProviderInstance:
    machine = get_or_create_machine(session, uid=machine_uid, host="127.0.0.1")
    agent = make_agent(session, machine, connected=websocket_connected)
    definition = make_definition(
        session,
        alias=alias,
        provider_type="mock",
        modality=modality,
        enabled=enabled,
        backend_config={"model": {"file": "m.gguf"}},
    )
    return make_instance(session, agent, definition, backend_status="running")


def make_wav(sample_rate: int = 24000, duration: float = 0.5) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        n = int(sample_rate * duration)
        w.writeframes(
            b"".join(
                struct.pack(
                    "<h",
                    int(0.3 * 32767 * math.sin(2 * math.pi * 440 * i / sample_rate)),
                )
                for i in range(n)
            )
        )
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Happy path (buffered WAV)
# ---------------------------------------------------------------------------
@respx.mock
def test_speech_relays_wav_and_persists_usage(
    client: TestClient, session: Session
) -> None:
    instance = seed_audio_instance(session, alias="tts-a", machine_uid="tts-a-m")
    wav = make_wav(24000, 0.5)
    respx.post(f"{AGENT}/v1/audio/speech").mock(
        return_value=httpx.Response(
            200, content=wav, headers={"content-type": "audio/wav"}
        )
    )

    resp = client.post(
        "/v1/audio/speech", json={"model": "tts-a", "input": "hello world"}
    )
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "audio/wav"
    assert resp.content == wav

    sample = (
        session.query(AudioUsageSample)
        .filter(AudioUsageSample.provider_instance_id == instance.id)
        .first()
    )
    assert sample is not None
    assert sample.modality == "tts"
    assert sample.chars == len("hello world")
    # 0.5s WAV (24000 * 0.5 frames * 2 bytes / 48000 byte-rate).
    assert sample.audio_seconds is not None
    assert abs(sample.audio_seconds - 0.5) < 0.01
    assert sample.provider_definition_id == instance.provider_definition_id
    # No conversation turn for audio.
    assert session.query(ResponseRecord).count() == 0

    assert client.app.state.scheduler.active_count("tts-a") == 0


# ---------------------------------------------------------------------------
# Streaming PCM preserves X-Sample-Rate
# ---------------------------------------------------------------------------
@respx.mock
def test_speech_stream_preserves_sample_rate_header(
    client: TestClient, session: Session
) -> None:
    seed_audio_instance(session, alias="tts-pcm", machine_uid="tts-pcm-m")
    respx.post(f"{AGENT}/v1/audio/speech").mock(
        return_value=httpx.Response(
            200,
            content=b"\x00\x01" * 100,
            headers={
                "content-type": "application/octet-stream",
                "X-Sample-Rate": "24000",
            },
        )
    )

    resp = client.post(
        "/v1/audio/speech",
        json={"model": "tts-pcm", "input": "stream me", "response_format": "pcm"},
    )
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "application/octet-stream"
    assert resp.headers["x-sample-rate"] == "24000"

    sample = session.query(AudioUsageSample).first()
    assert sample is not None
    # Raw PCM has no WAV header -> duration not derivable.
    assert sample.audio_seconds is None


# ---------------------------------------------------------------------------
# Non-2xx relay
# ---------------------------------------------------------------------------
@respx.mock
def test_speech_upstream_429_relays_status_and_releases(
    client: TestClient, session: Session
) -> None:
    seed_audio_instance(session, alias="tts-429", machine_uid="tts-429-m")
    respx.post(f"{AGENT}/v1/audio/speech").mock(
        return_value=httpx.Response(429, json={"detail": "backend busy"})
    )

    scheduler = client.app.state.scheduler
    release_calls: list[tuple[str, str]] = []
    real_release = scheduler.release

    async def spy_release(alias: str, request_id: str):
        release_calls.append((alias, request_id))
        return await real_release(alias, request_id)

    scheduler.release = spy_release  # type: ignore[method-assign]

    resp = client.post("/v1/audio/speech", json={"model": "tts-429", "input": "x"})
    assert resp.status_code == 429
    assert resp.json()["detail"] == "backend busy"
    assert [a for a, _ in release_calls] == ["tts-429"]
    assert scheduler.active_count("tts-429") == 0
    # No usage sample on a failed call.
    assert session.query(AudioUsageSample).count() == 0


# ---------------------------------------------------------------------------
# Upstream connect failure -> 502 + release
# ---------------------------------------------------------------------------
@respx.mock
def test_speech_connect_error_502_and_release(
    client: TestClient, session: Session
) -> None:
    seed_audio_instance(session, alias="tts-err", machine_uid="tts-err-m")
    respx.post(f"{AGENT}/v1/audio/speech").mock(
        side_effect=httpx.ConnectError("boom", request=httpx.Request("POST", AGENT))
    )

    scheduler = client.app.state.scheduler
    release_calls: list[str] = []
    real_release = scheduler.release

    async def spy_release(alias: str, request_id: str):
        release_calls.append(alias)
        return await real_release(alias, request_id)

    scheduler.release = spy_release  # type: ignore[method-assign]

    resp = client.post("/v1/audio/speech", json={"model": "tts-err", "input": "x"})
    assert resp.status_code == 502
    assert resp.json()["error"]["code"] == "upstream_failed"
    assert release_calls == ["tts-err"]
    assert scheduler.active_count("tts-err") == 0


# ---------------------------------------------------------------------------
# Abandoned stream releases the slot (client-disconnect path)
# ---------------------------------------------------------------------------
@respx.mock
@pytest.mark.asyncio
async def test_speech_abandoned_stream_releases(session: Session) -> None:
    seed_audio_instance(session, alias="tts-ab", machine_uid="tts-ab-m")
    respx.post(f"{AGENT}/v1/audio/speech").mock(
        return_value=httpx.Response(
            200, content=b"x" * 1000, headers={"content-type": "audio/wav"}
        )
    )

    released: list[str] = []

    class FakeScheduler:
        async def release(self, alias: str, request_id: str) -> None:
            released.append(alias)

    client = httpx.AsyncClient()
    req = client.build_request("POST", f"{AGENT}/v1/audio/speech", json={})
    upstream = await client.send(req, stream=True)

    gen = audio_speech_module._relay_speech(
        scheduler=FakeScheduler(),  # type: ignore[arg-type]
        alias="tts-ab",
        request_id="r1",
        upstream=upstream,
        client=client,
        definition_id=None,
        instance_id=uuid.uuid4(),
        chars=3,
    )
    # Pull one chunk then abandon (GeneratorExit into the finally).
    await gen.__anext__()
    await gen.aclose()

    assert released == ["tts-ab"]
    # LOW-2 guard: an abandoned stream never reaches clean EOF, so ``completed``
    # stays False and NO usage sample is persisted. Removing the ``if
    # completed:`` guard in ``_relay_speech`` fails this assertion.
    assert session.query(AudioUsageSample).count() == 0


# ---------------------------------------------------------------------------
# Mid-stream upstream error releases the slot and persists no usage
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_speech_midstream_error_releases_without_usage(
    session: Session,
) -> None:
    released: list[str] = []

    class FakeScheduler:
        async def release(self, alias: str, request_id: str) -> None:
            released.append(alias)

    class FailingStream:
        # Yields one chunk (2xx body started) then raises mid-stream.
        async def aiter_raw(self):
            yield b"chunk-1"
            raise httpx.ReadError("boom", request=httpx.Request("POST", AGENT))

        async def aclose(self) -> None:
            return None

    client = httpx.AsyncClient()
    gen = audio_speech_module._relay_speech(
        scheduler=FakeScheduler(),  # type: ignore[arg-type]
        alias="tts-mid",
        request_id="r1",
        upstream=FailingStream(),  # type: ignore[arg-type]
        client=client,
        definition_id=None,
        instance_id=uuid.uuid4(),
        chars=3,
    )
    await gen.__anext__()  # first chunk delivered
    with pytest.raises(httpx.ReadError):
        await gen.__anext__()  # upstream raises mid-stream

    assert released == ["tts-mid"]
    # LOW-2 guard: a mid-stream error skips the ``completed = True`` line, so no
    # usage sample is written even though the slot is released.
    assert session.query(AudioUsageSample).count() == 0


# ---------------------------------------------------------------------------
# Validation / admission errors
# ---------------------------------------------------------------------------
def test_speech_unknown_alias_404(client: TestClient) -> None:
    resp = client.post("/v1/audio/speech", json={"model": "nope", "input": "x"})
    assert resp.status_code == 404


def test_speech_llm_alias_rejected_404(client: TestClient, session: Session) -> None:
    seed_audio_instance(
        session, alias="tts-llm", machine_uid="tts-llm-m", modality="llm"
    )
    resp = client.post("/v1/audio/speech", json={"model": "tts-llm", "input": "x"})
    assert resp.status_code == 404
    assert "not a speech" in resp.json()["detail"]


def test_speech_asr_alias_rejected_404(client: TestClient, session: Session) -> None:
    seed_audio_instance(
        session, alias="tts-asr", machine_uid="tts-asr-m", modality="asr"
    )
    resp = client.post("/v1/audio/speech", json={"model": "tts-asr", "input": "x"})
    assert resp.status_code == 404


def test_speech_disabled_alias_404(client: TestClient, session: Session) -> None:
    seed_audio_instance(
        session, alias="tts-off", machine_uid="tts-off-m", enabled=False
    )
    resp = client.post("/v1/audio/speech", json={"model": "tts-off", "input": "x"})
    assert resp.status_code == 404
    assert "disabled" in resp.json()["detail"]


@pytest.mark.parametrize(
    "payload",
    [
        {"input": "x"},  # no model
        {"model": "tts-v"},  # no input
        {"model": "tts-v", "input": ""},  # empty input
        {"model": "tts-v", "input": 5},  # wrong type
    ],
)
def test_speech_bad_request_400(client: TestClient, payload: dict[str, Any]) -> None:
    resp = client.post("/v1/audio/speech", json=payload)
    assert resp.status_code == 400


def test_speech_no_provider_503(client: TestClient, session: Session) -> None:
    seed_audio_instance(
        session,
        alias="tts-np",
        machine_uid="tts-np-m",
        websocket_connected=False,
    )
    resp = client.post("/v1/audio/speech", json={"model": "tts-np", "input": "x"})
    assert resp.status_code == 503


# ---------------------------------------------------------------------------
# WAV header sniffing unit tests
# ---------------------------------------------------------------------------
def test_wav_duration_parse() -> None:
    from app.services import audio as audio_svc

    assert audio_svc.wav_duration_seconds(make_wav(24000, 1.0)) == pytest.approx(1.0)
    assert audio_svc.wav_duration_seconds(make_wav(16000, 0.25)) == pytest.approx(0.25)
    # Non-WAV / truncated / garbage -> None.
    assert audio_svc.wav_duration_seconds(b"ID3notwav") is None
    assert audio_svc.wav_duration_seconds(b"") is None
    assert audio_svc.wav_duration_seconds(b"RIFFxxxxWAVE") is None


def test_wav_duration_streamed_placeholder() -> None:
    """Streamed WAVs carry a 0xFFFFFFFF data-size placeholder; the actual
    delivered byte total must drive the duration instead."""
    import struct as _struct

    from app.services import audio as audio_svc

    byte_rate = 24000 * 2  # 24 kHz 16-bit mono
    header = b"RIFF" + _struct.pack("<I", 0xFFFFFFFE) + b"WAVE"
    header += b"fmt " + _struct.pack("<IHHIIHH", 16, 1, 1, 24000, byte_rate, 2, 16)
    header += b"data" + _struct.pack("<I", 0xFFFFFFFF)
    payload = 48000  # exactly 1.0 s of audio actually delivered
    total = len(header) + payload

    # Placeholder header alone -> no trustworthy size -> None.
    assert audio_svc.wav_duration_seconds(header) is None
    # With the delivered total: (total - data_start) / byte_rate == 1.0 s.
    assert (
        audio_svc.wav_duration_seconds(header, total_bytes=total)
        == pytest.approx(1.0)
    )
    # A lying (too-large but non-placeholder) size also falls back to reality.
    lying = header[: len(header) - 4] + _struct.pack("<I", 10**9)
    assert (
        audio_svc.wav_duration_seconds(lying, total_bytes=total) == pytest.approx(1.0)
    )
