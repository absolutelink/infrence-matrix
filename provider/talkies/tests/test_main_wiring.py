"""Main wiring tests: the talkies agent hosts N drivable backends behind one
/v1 surface (gufo-shaped registry wiring), stamping alias + modality +
backend_config from the registration response / assignment entries."""

from typing import Any

from conftest import ASR_SLUG, TTS_SLUG, make_settings
from provider_lib.admin_client import RegistrationResult
from provider_lib.config_update import ConfigState

from provider_talkies.main import (
    PROVIDER_TYPE,
    VERSION,
    apply_registration,
    build_registry,
    make_handle,
    make_lifecycle,
)

INSTANCE_A = "aaaaaaaa-0000-0000-0000-000000000001"
INSTANCE_B = "bbbbbbbb-0000-0000-0000-000000000002"


class _StubClient:
    def __init__(self, settings: Any) -> None:
        self.settings = settings
        self.is_registered = False
        self.sent: list[tuple[str, dict[str, Any]]] = []

    async def send_event(self, kind: str, payload: dict[str, Any]) -> None:
        self.sent.append((kind, payload))


def _reg_backends() -> list[dict[str, Any]]:
    return [
        {
            "instance_id": INSTANCE_A,
            "definition": {
                "alias": "tts-main",
                "modality": "tts",
                "capacity": 1,
                "config_fingerprint": "fp-a",
                "backend_config": {"model": {"slug": TTS_SLUG}},
            },
        },
        {
            "instance_id": INSTANCE_B,
            "definition": {
                "alias": "asr-main",
                "modality": "asr",
                "capacity": 2,
                "config_fingerprint": "fp-b",
                "backend_config": {"model": {"slug": ASR_SLUG}},
            },
        },
    ]


def test_constants() -> None:
    assert PROVIDER_TYPE == "talkies"
    assert VERSION == "dev"


def test_build_registry_hosts_one_backend_per_placed_definition(tmp_path) -> None:
    client = _StubClient(make_settings(tmp_path))
    result = RegistrationResult(
        {
            "agent_id": "agent-uuid",
            "agent_secret": "s",
            "machine": {"uid": "u"},
            "backends": _reg_backends(),
        }
    )
    registry = build_registry(client, result)
    assert len(registry) == 2
    by_alias = {h.alias: h for h in registry.handles()}
    assert set(by_alias) == {"tts-main", "asr-main"}

    tts = by_alias["tts-main"]
    assert tts.lifecycle.capacity == 1
    driver = tts.lifecycle.driver
    assert driver.slug is None  # resolved at start, not at wiring time
    assert driver.alias == "tts-main"
    assert driver.modality == "tts"
    assert driver._config == {"model": {"slug": TTS_SLUG}}
    assert tts.config_state.applied_fingerprint == "fp-a"

    asr = by_alias["asr-main"]
    assert asr.lifecycle.capacity == 2
    assert asr.lifecycle.driver.modality == "asr"
    assert asr.config_state.applied_fingerprint == "fp-b"


def test_make_handle_for_live_assignment(tmp_path) -> None:
    client = _StubClient(make_settings(tmp_path))
    entry = {
        "instance_id": INSTANCE_A,
        "alias": "tts-live",
        "modality": "tts",
        "capacity": 1,
        "config_fingerprint": "fp-x",
        "backend_config": {"model": {"slug": TTS_SLUG}},
    }
    handle = make_handle(client, entry)
    assert handle.alias == "tts-live"
    driver = handle.lifecycle.driver
    assert driver.alias == "tts-live"
    assert driver.modality == "tts"
    assert handle.config_state.applied_fingerprint == "fp-x"


def test_apply_registration_single_backend(tmp_path) -> None:
    client = _StubClient(make_settings(tmp_path))
    lifecycle = make_lifecycle(client, {})
    result = RegistrationResult(
        {
            "agent_id": "agent-uuid",
            "agent_secret": "s",
            "machine": {"uid": "u"},
            "backends": _reg_backends()[:1],
        }
    )
    state = ConfigState()
    apply_registration(lifecycle, result, state)
    assert lifecycle.instance_id == INSTANCE_A
    driver = lifecycle.driver
    assert driver.alias == "tts-main"
    assert driver.modality == "tts"
    assert driver._config == {"model": {"slug": TTS_SLUG}}
    assert state.applied_fingerprint == "fp-a"
