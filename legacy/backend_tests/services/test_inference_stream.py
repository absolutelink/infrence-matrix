import asyncio
from types import SimpleNamespace

import pytest

from app.services import inference_stream


@pytest.mark.asyncio
async def test_first_frame_deadline_is_not_reset_by_keepalives(monkeypatch):
    monkeypatch.setattr(inference_stream, "FIRST_FRAME_TIMEOUT_SECONDS", 0.06)
    monkeypatch.setattr(inference_stream, "KEEPALIVE_SECONDS", 0.01)

    async def agent_lines():
        while True:
            await asyncio.sleep(0.005)
            yield ": agent still waiting"

    async def guard(awaitable):
        return await awaitable

    lease = SimpleNamespace(
        guard=guard, request_id="test", server=SimpleNamespace(id="server")
    )
    stream = inference_stream.lines_with_keepalive(agent_lines(), lease)
    with pytest.raises(TimeoutError, match="dispatch deadline"):
        async for _line in stream:
            pass


@pytest.mark.asyncio
async def test_first_frame_keepalives_stop_when_data_arrives(monkeypatch):
    monkeypatch.setattr(inference_stream, "FIRST_FRAME_TIMEOUT_SECONDS", 0.02)
    monkeypatch.setattr(inference_stream, "KEEPALIVE_SECONDS", 0.005)

    async def agent_lines():
        await asyncio.sleep(0.01)
        yield 'data: {"choices": []}'
        await asyncio.sleep(0.03)
        yield "data: [DONE]"

    async def guard(awaitable):
        return await awaitable

    lease = SimpleNamespace(
        guard=guard, request_id="test", server=SimpleNamespace(id="server")
    )
    frames = [
        line
        async for line in inference_stream.lines_with_keepalive(agent_lines(), lease)
    ]
    assert None in frames
    assert frames[-1] == "data: [DONE]"


@pytest.mark.asyncio
async def test_header_wait_emits_keepalives_and_times_out(monkeypatch):
    monkeypatch.setattr(inference_stream, "FIRST_FRAME_TIMEOUT_SECONDS", 0.06)
    monkeypatch.setattr(inference_stream, "KEEPALIVE_SECONDS", 0.01)
    interrupted = asyncio.Event()

    class SlowResponse:
        async def __aenter__(self):
            try:
                await asyncio.Event().wait()
            finally:
                interrupted.set()

        async def __aexit__(self, *_args):
            pass

    class SlowClient:
        def stream(self, *_args, **_kwargs):
            return SlowResponse()

    async def guard(awaitable):
        return await awaitable

    lease = SimpleNamespace(
        guard=guard, dispatch_headers=lambda: {"X-Inference-Request-ID": "test"}
    )
    frames = []
    with pytest.raises(TimeoutError, match="dispatch deadline"):
        async for frame in inference_stream.upstream_with_keepalive(
            SlowClient(), "http://agent", {}, lease
        ):
            frames.append(frame)
    assert frames and all(frame is None for frame in frames)
    assert interrupted.is_set()
