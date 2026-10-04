"""Shared fakes for the Phase 9 provider-side handler tests.

A duck-typed ``RecordingClient`` (stands in for ``AdminClient``) and a
``TrackDriver`` fake backend driver with ``apply_config`` +
``resolved_artifacts`` so the shared ``provider_lib.config_update``
handlers can be exercised without a real admin or subprocess.
"""

from collections.abc import AsyncIterator
from typing import Any

from provider_lib.backend import BackendDriver
from provider_lib.config import ProviderSettings
from provider_lib.wire import Frame


class RecordingClient:
    """Duck-typed AdminClient: records commands + emitted events."""

    def __init__(self, settings: ProviderSettings) -> None:
        self.settings = settings
        self.handlers: dict[str, Any] = {}
        self.events: list[tuple[str, dict]] = []

    def on_command(self, name: str, handler: Any) -> None:
        self.handlers[name] = handler

    async def send_event(self, type_: str, payload: dict) -> None:
        self.events.append((type_, payload))

    async def dispatch(self, type_: str, payload: dict) -> dict:
        frame = Frame(type=type_, id="cmd-1", epoch=1, payload=payload)
        return await self.handlers[type_](frame)


class TrackDriver(BackendDriver):
    """Fake driver with apply_config + start-failure knob."""

    def __init__(self, *, start_raises: bool = False) -> None:
        self.start_raises = start_raises
        self.applied: list[dict] = []
        self.started = False
        self.stop_calls = 0
        self.resolved_artifacts: list[str] = ["/models/kept.gguf"]

    async def start(self) -> None:
        if self.start_raises:
            raise RuntimeError("spawn failed")
        self.started = True

    async def stop(self) -> None:
        self.stop_calls += 1
        self.started = False

    async def health(self) -> bool:
        return self.started

    async def list_models(self) -> list[dict[str, Any]]:
        return [{"id": "track-model", "object": "model"}]

    def stream_responses(
        self, request: dict[str, Any]
    ) -> AsyncIterator[dict[str, Any]]:
        async def gen() -> AsyncIterator[dict[str, Any]]:
            yield {"type": "response.completed"}

        return gen()

    def apply_config(self, backend_config: dict[str, Any]) -> None:
        self.applied.append(dict(backend_config))
