"""Phase 13 wiring: the driver's shared ring feeds batched backend.logs.

Replaces the old per-line `send_event("backend.logs", {stream, line})`
path: the driver appends to the provider_lib CursorLogRing and the
LogStreamer drains it into batched frames with the
`{"lines": [{ts, stream, text}], "dropped": N}` shape.
"""

from conftest import make_settings
from provider_lib.log_ring import CursorLogRing
from provider_lib.log_stream import install_log_streaming
from provider_lib.wire import Frame

from provider_halogen_flash.main import make_lifecycle


class FakeClient:
    """Duck-typed AdminClient: records events + command handlers."""

    def __init__(self, settings) -> None:
        self.settings = settings
        self.events: list[tuple[str, dict]] = []
        self.handlers: dict = {}

    def on_command(self, name, handler) -> None:
        self.handlers[name] = handler

    async def send_event(self, type_, payload) -> None:
        self.events.append((type_, payload))


async def test_batched_backend_logs_wiring(
    tmp_path, fake_flash_binary, local_artifacts
) -> None:
    settings = make_settings(tmp_path, fake_flash_binary)
    client = FakeClient(settings)
    lifecycle = make_lifecycle(
        client,
        {
            "artifacts": {
                "model": {"path": local_artifacts["checkpoint"]},
                "tokenizer": {"path": local_artifacts["tokenizer"]},
            }
        },
    )
    driver = lifecycle.driver
    try:
        assert isinstance(driver.log_ring, CursorLogRing)
        bundle = install_log_streaming(client, lifecycle, attach_provider_handler=False)

        driver._append_log("stdout", "hello")
        driver._append_log("stderr", "bad news")
        await bundle.streamer.flush_once()

        logs = [p for t, p in client.events if t == "backend.logs"]
        assert len(logs) == 1
        assert set(logs[0]) == {"lines", "dropped"}
        assert [e["text"] for e in logs[0]["lines"]] == ["hello", "bad news"]
        assert all({"ts", "stream", "text"} == set(e) for e in logs[0]["lines"])
        # No legacy per-line frames left behind.
        assert all("line" not in p for _, p in client.events)

        # backend.logs.get catch-up handler is registered and reads the
        # same ring with seq resume.
        assert "backend.logs.get" in client.handlers
        ack = await bundle.handle_logs_get(
            Frame(
                type="backend.logs.get",
                id="c1",
                payload={"kind": "backend", "since": 1},
            )
        )
        assert ack["ok"] is True
        assert [e["text"] for e in ack["detail"]["lines"]] == ["bad news"]
        assert ack["detail"]["seq"] == 2
    finally:
        await driver.aclose()
