"""Command + env construction for the talkies engine subprocess.

talkies reads its entire configuration from ``TALKIES_*`` environment
variables at server import (docs: psyb0t/docker-talkies
``docs/configuration.md``), and ``python -m talkies`` hardcodes
``0.0.0.0:8000`` — unusable for a private per-backend port. The driver
therefore launches the ASGI app directly:

    <TALKIES_PYTHON> -m uvicorn talkies.server:app --host 127.0.0.1 --port <p>

with the env built here. One process per backend definition: the enabled
set is exactly that definition's served slugs (Phase 25 multi-model: one
process serves several names, each mapped to a registry slug; a legacy
single-slug definition serves exactly one). The registry is read only at
import, so a config change requires a restart — which is exactly the
Phase 9 config-update flow.
"""

from __future__ import annotations

import os
import secrets
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from provider_lib.config import ProviderSettings
    from provider_lib.models import ModelSpec

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


# ---------------------------------------------------------------------------
# Phase 25 multi-model: per-served-model extraction.
#
# A served model's ``backend_config`` is the per-model object declared by the
# type's ``x-served-model-config-schema`` (talkies: ``{slug, revision?,
# defaults?, limits?}``). A LEGACY single-model definition instead synthesizes
# one spec whose ``backend_config`` is the FULL sectioned config (``model.slug``
# nested), so every extractor below reads BOTH shapes: the flat per-model key
# first, then the legacy nested ``model`` section. This keeps the multi-slug
# path and the byte-identical legacy path on one code path.
# ---------------------------------------------------------------------------


def model_slug(backend_config: Any) -> str | None:
    """The talkies slug of one served model (per-model or legacy shape)."""
    if not isinstance(backend_config, dict):
        return None
    slug = backend_config.get("slug")
    if isinstance(slug, str) and slug.strip():
        return slug.strip()
    model = backend_config.get("model")
    if isinstance(model, dict):
        slug = model.get("slug")
        if isinstance(slug, str) and slug.strip():
            return slug.strip()
    return None


def model_revision(backend_config: Any) -> str | None:
    """The optional prefetch revision of one served model (either shape)."""
    if not isinstance(backend_config, dict):
        return None
    revision = backend_config.get("revision")
    if isinstance(revision, str) and revision.strip():
        return revision.strip()
    model = backend_config.get("model")
    if isinstance(model, dict):
        revision = model.get("revision")
        if isinstance(revision, str) and revision.strip():
            return revision.strip()
    return None


def model_concurrency(backend_config: Any) -> int | None:
    """The per-model ``limits.model_concurrency`` (either shape), else ``None``."""
    if not isinstance(backend_config, dict):
        return None
    limits = backend_config.get("limits")
    n = limits.get("model_concurrency") if isinstance(limits, dict) else None
    return n if isinstance(n, int) and n >= 1 else None


def resolve_slugs(config: dict[str, Any], models: list[ModelSpec] | None) -> list[str]:
    """Ordered-unique slugs across the enabled served models.

    With no served-model list (legacy path / ``set_models`` not yet called)
    this is exactly the single ``model.slug`` from the shared config — so a
    legacy definition boots one slug, byte-identical to Phase 24.

    A multi-model definition (non-empty ``models``) whose enabled served model
    lacks a ``backend_config.slug`` is a config error and raises with an
    accurate message (NOT the misleading legacy ``requires model.slug``).
    """
    if not models:
        return [resolve_slug(config)]
    slugs: list[str] = []
    seen: set[str] = set()
    for spec in models:
        if not getattr(spec, "enabled", True):
            continue
        slug = model_slug(getattr(spec, "backend_config", None))
        if not slug:
            raise RuntimeError(
                f"talkies served model {getattr(spec, 'name', '?')!r} is "
                "missing backend_config.slug"
            )
        if slug not in seen:
            seen.add(slug)
            slugs.append(slug)
    if not slugs:
        raise RuntimeError(
            "talkies multi-model definition has no enabled served models"
        )
    return slugs


def resolve_revisions(
    config: dict[str, Any], models: list[ModelSpec] | None
) -> dict[str, str]:
    """``slug -> revision`` for prefetch (only slugs with a revision).

    Legacy (no models): the single ``model.revision`` mapped onto its slug.
    """
    if not models:
        revision = resolve_revision(config)
        return {resolve_slug(config): revision} if revision else {}
    out: dict[str, str] = {}
    for spec in models:
        if not getattr(spec, "enabled", True):
            continue
        bc = getattr(spec, "backend_config", None)
        slug = model_slug(bc)
        revision = model_revision(bc)
        if slug and revision:
            out[slug] = revision
    return out


def resolve_model_concurrency_map(
    config: dict[str, Any], models: list[ModelSpec] | None
) -> dict[str, int]:
    """``slug -> model_concurrency`` merged across served models.

    Each slug takes its own ``limits.model_concurrency`` when set, else the
    shared top-level default (``resolve_model_concurrency(config)``). When
    SEVERAL served names map to the SAME slug (one shared engine pool) the
    maximum across them wins — never last-write-wins. Legacy (no models)
    collapses to ``{slug: default}`` — the Phase 24 single-slug value.
    """
    default = resolve_model_concurrency(config)
    if not models:
        return {resolve_slug(config): default}
    out: dict[str, int] = {}
    for spec in models:
        if not getattr(spec, "enabled", True):
            continue
        bc = getattr(spec, "backend_config", None)
        slug = model_slug(bc)
        if not slug:
            continue
        n = model_concurrency(bc) or default
        # Same slug from two names => one engine pool: keep the widest.
        out[slug] = max(out.get(slug, 0), n)
    return out or {resolve_slug(config): default}


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
    models: list[ModelSpec] | None = None,
) -> dict[str, str]:
    """Full subprocess environment: the agent env plus the TALKIES_* set.

    The registry is read only at talkies import, so every knob here is
    fixed for the process lifetime (a config change = a restart).

    ``models`` is the served-model list (Phase 25 multi-model): every enabled
    slug is enabled + preloaded (comma-joined) and gets its own
    ``TALKIES_MODEL_CONCURRENCY`` entry (``slug1=N,slug2=M``). When omitted
    (legacy single-slug definition) this collapses to exactly the Phase 24
    single-slug env.
    """
    slugs = resolve_slugs(config, models)
    enabled = ",".join(slugs)
    concurrency_map = resolve_model_concurrency_map(config, models)
    concurrency = ",".join(f"{slug}={n}" for slug, n in concurrency_map.items())
    env = dict(os.environ)
    env.update(
        {
            # Every served slug is enabled and preloaded, so `backend running`
            # <=> all models resident (TTL 0 pins them).
            "TALKIES_MODELS_FILE": str(settings.TALKIES_MODELS_FILE),
            "TALKIES_ENABLED_MODELS": enabled,
            "TALKIES_PRELOAD": enabled,
            "TALKIES_MODEL_TTL": str(resolve_model_ttl(config)),
            "TALKIES_DATA_DIR": str(data_dir),
            "TALKIES_DEVICE": resolve_device(config),
            "TALKIES_AUTH_TOKEN": auth_token,
            "TALKIES_MODEL_CONCURRENCY": concurrency,
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
