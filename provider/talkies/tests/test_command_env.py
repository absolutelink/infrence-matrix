"""Command + env construction tests (pure, no subprocess).

The talkies engine is launched as
``<TALKIES_PYTHON> -m uvicorn talkies.server:app --host 127.0.0.1 --port <p>``
with the full TALKIES_* env: single-slug ENABLED + PRELOAD (the
per-definition-process rule), TTL 0 pinning by default, per-model
concurrency, device, data dir, and the bearer token.
"""

from pathlib import Path
from typing import Any

import pytest

from provider_talkies import env as tenv

CANONICAL: dict[str, Any] = {
    "model": {"slug": "qwen3-tts-1.7b-custom", "revision": "abc123"},
    "engine": {"device": "cuda"},
    "defaults": {"voice": "Vivian", "response_format": "wav"},
    "limits": {"model_concurrency": 2, "max_upload_bytes": 52428800, "model_ttl": 0},
    "security": {"auth_token": "s3cret"},
}


def test_build_command_uvicorn_private_loopback() -> None:
    cmd = tenv.build_command("/opt/venv/bin/python", 45123)
    assert cmd == [
        "/opt/venv/bin/python",
        "-m",
        "uvicorn",
        "talkies.server:app",
        "--host",
        "127.0.0.1",
        "--port",
        "45123",
    ]
    # NEVER `python -m talkies` (hardcodes 0.0.0.0:8000).
    assert "talkies" not in " ".join(cmd[:3]) or cmd[1:3] == ["-m", "uvicorn"]


def test_build_env_full_talkies_set(tmp_path) -> None:
    from conftest import make_settings

    settings = make_settings(tmp_path, models_file=tmp_path / "models.json")
    env = tenv.build_env(
        CANONICAL,
        settings=settings,
        data_dir=tmp_path / "data",
        auth_token="s3cret",
    )
    assert env["TALKIES_MODELS_FILE"] == str(tmp_path / "models.json")
    # One process per definition: exactly this slug enabled + preloaded.
    assert env["TALKIES_ENABLED_MODELS"] == "qwen3-tts-1.7b-custom"
    assert env["TALKIES_PRELOAD"] == "qwen3-tts-1.7b-custom"
    # TTL 0 = pin (VRAM exactness).
    assert env["TALKIES_MODEL_TTL"] == "0"
    assert env["TALKIES_MODEL_CONCURRENCY"] == "qwen3-tts-1.7b-custom=2"
    assert env["TALKIES_DEVICE"] == "cuda"
    assert env["TALKIES_DATA_DIR"] == str(tmp_path / "data")
    assert env["TALKIES_AUTH_TOKEN"] == "s3cret"
    assert env["TALKIES_MAX_UPLOAD_BYTES"] == "52428800"
    # The server never reaches for the Hub at request time.
    assert env["HF_HUB_OFFLINE"] == "1"
    # HF_HOME is re-pointed onto the resolved data dir (the base image bakes
    # /data/hf, which the matrix compose never mounts).
    assert env["HF_HOME"] == str(tmp_path / "data" / "hf")
    # NEW-1: the hub cache is pinned explicitly to the SAME directory the
    # prefetch pass uses as snapshot_download(cache_dir=...) — not HF_HOME's
    # implicit <HOME>/hub.
    assert env["HF_HUB_CACHE"] == str(tmp_path / "data" / "hf")


def test_build_env_defaults_when_sections_absent(tmp_path) -> None:
    from conftest import make_settings

    settings = make_settings(tmp_path)
    env = tenv.build_env(
        {"model": {"slug": "whisper-large-v3-turbo"}},
        settings=settings,
        data_dir=Path(settings.MODELS_DIR),
        auth_token="tok",
    )
    assert env["TALKIES_ENABLED_MODELS"] == "whisper-large-v3-turbo"
    assert env["TALKIES_PRELOAD"] == "whisper-large-v3-turbo"
    assert env["TALKIES_MODEL_TTL"] == "0"
    assert env["TALKIES_MODEL_CONCURRENCY"] == "whisper-large-v3-turbo=1"
    assert env["TALKIES_DEVICE"] == "cuda"
    assert "TALKIES_MAX_UPLOAD_BYTES" not in env


def test_resolve_slug_requires_model_section() -> None:
    with pytest.raises(RuntimeError, match="model.slug"):
        tenv.resolve_slug({})
    with pytest.raises(RuntimeError, match="model.slug"):
        tenv.resolve_slug({"model": {"slug": "  "}})
    assert tenv.resolve_slug({"model": {"slug": " kokoro-82m "}}) == "kokoro-82m"


def test_resolve_auth_token_random_when_empty() -> None:
    fixed = tenv.resolve_auth_token({"security": {"auth_token": "keep"}})
    assert fixed == "keep"
    a = tenv.resolve_auth_token({"security": {"auth_token": ""}})
    b = tenv.resolve_auth_token({})
    assert a and b and a != b  # random per start


def test_resolve_data_dir_prefers_env_then_models_dir(tmp_path) -> None:
    from conftest import make_settings

    derived = make_settings(tmp_path)
    assert tenv.resolve_data_dir(derived) == Path(derived.MODELS_DIR)
    pinned = make_settings(tmp_path, data_dir=tmp_path / "data")
    assert tenv.resolve_data_dir(pinned) == tmp_path / "data"


def test_resolve_revision_blank_is_none() -> None:
    assert tenv.resolve_revision({"model": {"slug": "s", "revision": "r1"}}) == "r1"
    assert tenv.resolve_revision({"model": {"slug": "s", "revision": "  "}}) is None
    assert tenv.resolve_revision({"model": {"slug": "s"}}) is None


def test_resolve_defaults_drops_nulls() -> None:
    defaults = tenv.resolve_defaults(
        {"defaults": {"voice": "af_heart", "language": None, "speed": 1.2}}
    )
    assert defaults == {"voice": "af_heart", "speed": 1.2}
