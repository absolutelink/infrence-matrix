"""Phase 24 (audio) mock provider: real-socket uvicorn end-to-end.

Proves the inherited ``/v1/audio/*`` routes reach ``MockBackend``'s fakes over
a live socket: valid WAV speech, incremental PCM streaming, multipart
transcription relay, the voice catalog roundtrip, and the live-ASR WS bridge.
"""

import asyncio
import io
import json
import wave
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import httpx
import websockets
from provider_lib.backend import BackendLifecycle
from provider_lib.config import ProviderSettings

from provider_mock.backend import MockBackend
from provider_mock.main import build_app


def _settings() -> ProviderSettings:
    return ProviderSettings(  # type: ignore[call-arg]
        MACHINE_UID="mock-audio",
        MACHINE_SECRET="tok",
        AGENT_ID="mock-audio-agent",
        ADMIN_BASE_URL="http://127.0.0.1:1",
        PROVIDER_PORT=8081,
        CACHE_DIR="/tmp/mock_audio_test_cache",
        MODELS_DIR="/tmp/mock_audio_test_models",
    )


@asynccontextmanager
async def _serve(
    driver: MockBackend,
) -> AsyncIterator[tuple[str, str, BackendLifecycle]]:
    import uvicorn

    lifecycle = BackendLifecycle(driver, capacity=1)
    await lifecycle.start()
    config = uvicorn.Config(
        build_app(lifecycle, _settings()),
        host="127.0.0.1",
        port=0,
        log_level="warning",
    )
    server = uvicorn.Server(config)
    task = asyncio.create_task(server.serve())
    while not server.started:
        await asyncio.sleep(0.02)
    port = server.servers[0].sockets[0].getsockname()[1]
    try:
        yield f"http://127.0.0.1:{port}", f"ws://127.0.0.1:{port}", lifecycle
    finally:
        server.should_exit = True
        await task


async def test_speech_returns_valid_wav() -> None:
    driver = MockBackend(model="mock-model")
    async with _serve(driver) as (http_url, _ws_url, lifecycle):
        async with httpx.AsyncClient(base_url=http_url) as client:
            resp = await client.post(
                "/v1/audio/speech",
                json={
                    "model": "mock-model",
                    "input": "hello",
                    "response_format": "wav",
                },
            )
            assert resp.status_code == 200
            assert resp.headers["content-type"] == "audio/wav"
    data = resp.content
    assert data[:4] == b"RIFF" and data[8:12] == b"WAVE"
    with wave.open(io.BytesIO(data), "rb") as w:
        assert w.getnchannels() == 1
        assert w.getframerate() == driver.audio_sample_rate
        assert w.getsampwidth() == 2
        assert w.getnframes() > 0
    assert driver.speech_calls == 1
    assert lifecycle.in_flight == 0


async def test_speech_pcm_streams_multiple_chunks() -> None:
    driver = MockBackend(model="mock-model")
    driver.apply_config({"audio": {"pcm_chunk_count": 6, "pcm_chunk_delay": 0.01}})
    async with _serve(driver) as (http_url, _ws_url, lifecycle):
        chunks: list[bytes] = []
        async with httpx.AsyncClient(base_url=http_url) as client:
            async with client.stream(
                "POST",
                "/v1/audio/speech",
                json={"model": "mock-model", "response_format": "pcm"},
            ) as resp:
                assert resp.status_code == 200
                assert resp.headers["content-type"] == "application/octet-stream"
                assert resp.headers.get("x-sample-rate") == str(
                    driver.audio_sample_rate
                )
                async for piece in resp.aiter_bytes():
                    if piece:
                        chunks.append(piece)
        assert len(chunks) > 1
        # PCM is raw 16-bit frames (no WAV header).
        assert b"".join(chunks)[:4] != b"RIFF"
    assert lifecycle.in_flight == 0


async def test_transcriptions_multipart_relay() -> None:
    driver = MockBackend(model="mock-model")
    async with _serve(driver) as (http_url, _ws_url, lifecycle):
        async with httpx.AsyncClient(base_url=http_url) as client:
            resp = await client.post(
                "/v1/audio/transcriptions",
                data={"model": "mock-model", "prompt": "the quick brown fox"},
                files={"file": ("clip.wav", b"RIFFfakeaudio", "audio/wav")},
            )
            assert resp.status_code == 200
            assert resp.headers["content-type"] == "application/json"
            assert resp.json() == {"text": "the quick brown fox"}
    assert driver.transcribe_calls == 1
    assert lifecycle.in_flight == 0


async def test_voices_roundtrip() -> None:
    driver = MockBackend(model="mock-model")
    async with _serve(driver) as (http_url, _ws_url, lifecycle):
        async with httpx.AsyncClient(base_url=http_url) as client:
            listing = await client.get(
                "/v1/audio/voices", params={"model": "mock-model"}
            )
            assert listing.status_code == 200
            names = {v["voice_id"] for v in listing.json()["data"]}
            assert {"alloy", "echo"} <= names

            put = await client.put(
                "/v1/audio/voices/clone1",
                params={
                    "model": "mock-model",
                    "ref_text": "hello there",
                    "language": "en",
                },
                content=b"RIFFclipped",
            )
            assert put.status_code == 200
            assert put.json()["voice_id"] == "clone1"
            assert put.json()["size_bytes"] == len(b"RIFFclipped")

            after = await client.get("/v1/audio/voices", params={"model": "mock-model"})
            assert "clone1" in {v["voice_id"] for v in after.json()["data"]}

            dele = await client.delete(
                "/v1/audio/voices/clone1", params={"model": "mock-model"}
            )
            assert dele.status_code == 200
            assert dele.json()["deleted"] is True

            final = await client.get("/v1/audio/voices", params={"model": "mock-model"})
            assert "clone1" not in {v["voice_id"] for v in final.json()["data"]}
    # Voice ops never consume a slot.
    assert lifecycle.in_flight == 0


async def test_ws_transcriptions_stream_bridge() -> None:
    driver = MockBackend(model="mock-model")
    async with _serve(driver) as (_http_url, ws_url, lifecycle):
        async with websockets.connect(
            f"{ws_url}/v1/audio/transcriptions/stream?model=mock-model"
        ) as ws:
            raw = await asyncio.wait_for(ws.recv(), 5)
            msg: dict[str, Any] = json.loads(raw)
            assert msg["type"] == "transcript"
            assert msg["text"] == "mock live transcript"
            await ws.close()
    # Slot released once the bridge returns on client disconnect.
    for _ in range(100):
        if lifecycle.in_flight == 0:
            break
        await asyncio.sleep(0.02)
    assert lifecycle.in_flight == 0
    assert driver.transcribe_stream_calls == 1
