"""Command + env construction for the talkies engine subprocess.

talkies reads its entire configuration from ``TALKIES_*`` environment
variables at server import (docs: psyb0t/docker-talkies
``docs/configuration.md``), and ``python -m talkies`` hardcodes
``0.0.0.0:8000`` — unusable for a private per-backend port. The driver
therefore launches the ASGI app directly:

    <TALKIES_PYTHON> -m uvicorn talkies.server:app --host 127.0.0.1 --port <p>

with the env built here. One process per backend definition: the enabled
set is exactly that definition's single slug (the registry is read only
at import, so a config change requires a restart — which is exactly the
Phase 9 config-update flow).
"""

from __future__ import annotations

import os
import secrets
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from provider_lib.config import ProviderSettings

# talkies response_format -> the Content-Type its encoder answers with
# (src/talkies/tts.py _FORMATS). The SpeechStream contract fixes response
# headers synchronously (before any upstream byte is observed), so the
# driver derives them from the request instead of the upstream response.
CONTENT_TYPES: dict[str, str] = {
    "mp3": "audio/mpeg",
    "opus": "audio/ogg",
    "aac": "audio/aac",
    "flac": "audio/flac",
    "wav": "audio/wav",
    "pcm": "application/octet-stream",
}

# Every talkies TTS backend produces 24 kHz mono PCM (Qwen3-TTS and
# Chatterbox Turbo alike); the PCM streaming path signals it verbatim.
TALKIES_PCM_SAMPLE_RATE = "24000"


def resolve_slug(config: dict[str, Any]) -> str:
    """The definition's single enabled slug (``model.slug``).

    Raises ``RuntimeError`` when absent/blank — the schema requires it, but
    the driver must fail loudly rather than boot an all-models engine.
    """
    model = config.get("model")
    slug = model.get("slug") if isinstance(model, dict) else None
    if not isinstance(slug, str) or not slug.strip():
        raise RuntimeError("talkies backend_config requires model.slug")
    return slug.strip()


def resolve_revision(config: dict[str, Any]) -> str | None:
    """Optional prefetch revision (blank/None = registry default)."""
    model = config.get("model")
    revision = model.get("revision") if isinstance(model, dict) else None
    if isinstance(revision, str) and revision.strip():
        return revision.strip()
    return None


def resolve_data_dir(settings: ProviderSettings) -> Path:
    """talkies data root (models/ + files/ + custom-voices/ subdirs).

    ``TALKIES_DATA_DIR`` env wins; otherwise the agent's ``MODELS_DIR`` —
    model weights belong on the models volume.
    """
    if settings.TALKIES_DATA_DIR:
        return Path(settings.TALKIES_DATA_DIR)
    return Path(settings.MODELS_DIR)


def resolve_auth_token(config: dict[str, Any]) -> str:
    """Bearer token for the engine; empty config => random per start."""
    security = config.get("security")
    token = security.get("auth_token") if isinstance(security, dict) else None
    if isinstance(token, str) and token:
        return token
    return secrets.token_urlsafe(32)


def resolve_device(config: dict[str, Any]) -> str:
    engine = config.get("engine")
    device = engine.get("device") if isinstance(engine, dict) else None
    if isinstance(device, str) and device.strip():
        return device.strip()
    return "cuda"


def resolve_model_concurrency(config: dict[str, Any]) -> int:
    limits = config.get("limits")
    n = limits.get("model_concurrency") if isinstance(limits, dict) else None
    return n if isinstance(n, int) and n >= 1 else 1


def resolve_model_ttl(config: dict[str, Any]) -> int:
    limits = config.get("limits")
    ttl = limits.get("model_ttl") if isinstance(limits, dict) else None
    return ttl if isinstance(ttl, int) and ttl >= 0 else 0


def resolve_max_upload_bytes(config: dict[str, Any]) -> int | None:
    limits = config.get("limits")
    value = limits.get("max_upload_bytes") if isinstance(limits, dict) else None
    return value if isinstance(value, int) and value >= 0 else None


def resolve_defaults(config: dict[str, Any]) -> dict[str, Any]:
    """Recorded request-fallback fields (only non-null keys)."""
    defaults = config.get("defaults")
    if not isinstance(defaults, dict):
        return {}
    return {k: v for k, v in defaults.items() if v is not None}


def build_command(python: str, port: int) -> list[str]:
    """uvicorn argv for the engine (private loopback bind; talkies' own
    ``python -m talkies`` hardcodes 0.0.0.0:8000 and is never used)."""
    return [
        python,
        "-m",
        "uvicorn",
        "talkies.server:app",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
    ]


def build_env(
    config: dict[str, Any],
    *,
    settings: ProviderSettings,
    data_dir: Path,
    auth_token: str,
) -> dict[str, str]:
    """Full subprocess environment: the agent env plus the TALKIES_* set.

    The registry is read only at talkies import, so every knob here is
    fixed for the process lifetime (a config change = a restart).
    """
    slug = resolve_slug(config)
    env = dict(os.environ)
    env.update(
        {
            # Exactly this definition's slug is enabled and preloaded, so
            # `backend running` <=> the model is resident (TTL 0 pins it).
            "TALKIES_MODELS_FILE": str(settings.TALKIES_MODELS_FILE),
            "TALKIES_ENABLED_MODELS": slug,
            "TALKIES_PRELOAD": slug,
            "TALKIES_MODEL_TTL": str(resolve_model_ttl(config)),
            "TALKIES_DATA_DIR": str(data_dir),
            "TALKIES_DEVICE": resolve_device(config),
            "TALKIES_AUTH_TOKEN": auth_token,
            "TALKIES_MODEL_CONCURRENCY": f"{slug}={resolve_model_concurrency(config)}",
            # The server must never reach for the Hub at request time; the
            # driver's prefetch pass (offline flag cleared in-process only)
            # is the single point of network egress.
            "HF_HUB_OFFLINE": "1",
            # The base image bakes HF_HOME=/data/hf (an unmounted path once
            # TALKIES_DATA_DIR is blanked); dependency-repo snapshots are
            # consumed from the HF cache at request time under offline mode,
            # so the cache must live on the same mounted volume the driver
            # prefetched into. HF_HUB_CACHE is pinned EXPLICITLY (not left to
            # the HF_HOME/$HOME default): it must equal the cache_dir the
            # prefetch pass hands snapshot_download for dependency repos —
            # one level, not HF_HOME's implicit <HOME>/hub.
            "HF_HOME": str(data_dir / "hf"),
            "HF_HUB_CACHE": str(data_dir / "hf"),
        }
    )
    max_upload = resolve_max_upload_bytes(config)
    if max_upload is not None:
        env["TALKIES_MAX_UPLOAD_BYTES"] = str(max_upload)
    return env
