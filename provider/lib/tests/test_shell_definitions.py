"""Phase 14 — shell definitions: provider-side handling.

A shell definition (no authored backend_config) registers with
`backend_config: null` / `config_fingerprint: null` in the definition
echo. The provider must persist/load that state honestly (never coerce
to `{}` or a hash of `{}`), and `backend.start` before any config is
applied must NAK `no_config` (the shared fence).
"""

import json
from typing import Any

from provider_lib.admin_client import RegistrationResult
from provider_lib.config import ProviderSettings
from provider_lib.config_update import ConfigState, install_config_handlers


def _definition_echo(**overrides: Any) -> dict[str, Any]:
    d = {
        "id": "d-1",
        "alias": "shell-model",
        "provider_type": "mock",
        "backend_config": None,
        "config_fingerprint": None,
        "idle_timeout_seconds": 300,
        "capacity": 1,
        "vram_required_bytes": 0,
        "model_metadata": {},
    }
    d.update(overrides)
    return d


def _registration_body(**overrides: Any) -> dict[str, Any]:
    b = {
        "instance_id": "i-1",
        "instance_secret": "s",
        "machine": {},
        "provider_definition": _definition_echo(
            **{
                k[len("def_") :]: v
                for k, v in overrides.items()
                if k.startswith("def_")
            }
        ),
    }
    b.update({k: v for k, v in overrides.items() if not k.startswith("def_")})
    return b


def test_registration_result_type_adopted_flag() -> None:
    r = RegistrationResult(_registration_body(type_adopted=True))
    assert r.type_adopted is True
    r2 = RegistrationResult(_registration_body())
    assert r2.type_adopted is False


def test_persist_shell_config_keeps_null_fingerprint(tmp_path) -> None:
    """provider_config.json carries the null fingerprint — never the hash
    of {}."""
    from provider_lib.admin_client import AdminClient

    settings = ProviderSettings(
        MACHINE_UID="m-test",
        PROVIDER_REGISTRATION_TOKEN="tok",
        ADMIN_BASE_URL="http://admin:8000",
        CACHE_DIR=str(tmp_path / "cache"),
        MODELS_DIR=str(tmp_path / "models"),
    )
    result = RegistrationResult(_registration_body(type_adopted=True))
    real = AdminClient.__new__(AdminClient)
    real.settings = settings
    real._persist_config(result)
    stored = json.loads((settings.provider_config_path).read_text())
    assert stored["config_fingerprint"] is None
    assert stored["provider_definition"]["backend_config"] is None
    # Round-trips through load_config.
    loaded = real.load_config()
    assert loaded["config_fingerprint"] is None


def test_no_config_fence_naks_before_first_config() -> None:
    """install_config_handlers exposes client.no_config_nak; with
    applied_fingerprint None it returns the no_config NAK payload, and
    after a fingerprint is applied it returns None (proceed)."""
    import asyncio

    from provider_lib.admin_client import AdminClient
    from provider_lib.wire import Frame

    state = ConfigState()  # applied_fingerprint None

    class _Lifecycle:
        capacity = 1

    real_client = AdminClient.__new__(AdminClient)
    # __new__ skips __init__; seed the attribute on_command needs.
    real_client._command_handlers = {}
    install_config_handlers(
        real_client,
        _Lifecycle(),
        state,
        ProviderSettings(
            MACHINE_UID="m-test",
            PROVIDER_REGISTRATION_TOKEN="tok",
            ADMIN_BASE_URL="http://admin:8000",
            CACHE_DIR="/tmp/p14",
            MODELS_DIR="/tmp/p14",
        ),
    )
    fence = getattr(real_client, "no_config_nak", None)
    assert fence is not None
    nak = asyncio.run(fence(Frame(type="backend.start")))
    assert nak == {"ok": False, "error": "no_config", "detail": {"step": "validate"}}

    state.applied_fingerprint = "fp-after-config"
    assert asyncio.run(fence(Frame(type="backend.start"))) is None
