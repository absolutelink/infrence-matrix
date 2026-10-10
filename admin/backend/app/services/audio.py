"""Shared Phase 24 (audio) helpers for the /v1 and /admin/api audio routes.

Two concerns live here so the speech / transcription / voices routes and the
admin enrollment router stay thin:

* **Agent address resolution** — the audio data plane bypasses litellm and
  dials the agent's env-port ``/v1`` surface directly (ARCHITECTURE.md §7
  Audio specifics). The scheduler already builds that URL per admission
  (``http://{machine.reachable}:{agent.base_port}``); the no-slot paths
  (client ``GET /v1/audio/voices`` and admin enrollment) resolve the same
  address from a *connected* agent hosting the definition, preferring one
  whose backend is loaded (running/in_use) so the agent answers instead of
  503-ing a stopped backend.

* **Cheap clip-duration sniffing** — ``AudioUsageSample.audio_seconds`` is
  derived only when it is free: a canonical WAV header (RIFF/WAVE ``fmt `` +
  ``data`` chunk sizes) parsed from the first bytes of a streamed response.
  Compressed containers (mp3/opus/aac) are never decoded — the field stays
  ``None``.
"""

import struct
import uuid
from typing import Any

from sqlmodel import Session, col, select

from app.models import (
    ProviderAgent,
    ProviderDefinition,
    ProviderInstance,
)
from app.services.scheduler import RUNNING_BACKEND_STATUSES

# Bytes of a streamed response retained to sniff a WAV header. A canonical
# 44-byte header plus room for a couple of optional RIFF chunks (LIST/fact)
# before the ``data`` chunk; anything past this is not needed for the parse.
_WAV_SNIFF_BYTES = 1024


def agent_base_url_for_definition(
    session: Session, definition: ProviderDefinition
) -> str | None:
    """``http://{machine}:{agent.base_port}`` for a connected agent hosting
    ``definition`` (the audio data-plane address), or ``None`` when no agent
    of the definition is connected.

    Prefers an agent whose backend for this definition is loaded
    (running/in_use) so the dialed agent answers rather than 503-ing a
    stopped backend; falls back to any connected agent (the agent's own
    route returns 503 if that backend is not ready — passed through).
    """
    rows = session.exec(
        select(ProviderInstance, ProviderAgent)
        .join(ProviderAgent, col(ProviderInstance.agent_id) == col(ProviderAgent.id))
        .where(
            ProviderInstance.provider_definition_id == definition.id,
            col(ProviderAgent.websocket_connected) == True,  # noqa: E712
        )
    ).all()
    fallback: str | None = None
    for inst, agent in rows:
        machine = agent.machine
        address = machine.reachable_address() if machine else None
        if not address:
            continue
        url = f"http://{address}:{agent.base_port}"
        if inst.backend_status in RUNNING_BACKEND_STATUSES:
            return url
        if fallback is None:
            fallback = url
    return fallback


def wav_duration_seconds(prefix: bytes, total_bytes: int | None = None) -> float | None:
    """Duration (seconds) of a canonical PCM WAV from its leading bytes.

    Returns ``None`` unless the prefix holds a ``RIFF``/``WAVE`` header with a
    parseable ``fmt `` byte-rate and a ``data`` chunk size. Never raises — a
    truncated or non-WAV prefix simply yields ``None`` (the caller leaves
    ``audio_seconds`` null).

    Streamed WAVs (e.g. talkies' Qwen3 output) write a ``0xFFFFFFFF``
    placeholder for the ``data`` size because the total is unknown when the
    header is emitted. When ``total_bytes`` (the actual delivered payload
    size) is given and the header size is a placeholder or exceeds reality,
    the real byte count is used instead.
    """
    if len(prefix) < 12 or prefix[:4] != b"RIFF" or prefix[8:12] != b"WAVE":
        return None
    byte_rate: int | None = None
    data_size: int | None = None
    # Walk the chunk list starting after the 12-byte RIFF/WAVE header.
    pos = 12
    n = len(prefix)
    while pos + 8 <= n:
        chunk_id = prefix[pos : pos + 4]
        try:
            (chunk_size,) = struct.unpack_from("<I", prefix, pos + 4)
        except struct.error:
            return None
        body = pos + 8
        if chunk_id == b"fmt " and body + 16 <= n:
            # fmt chunk: audioformat(2) channels(2) samplerate(4) byterate(4) ...
            try:
                (_fmt, _ch, _rate, byte_rate, _align, _bits) = struct.unpack_from(
                    "<HHIIHH", prefix, body
                )
            except struct.error:
                return None
        elif chunk_id == b"data":
            data_size = chunk_size
            data_start = body
            break
        # Chunks are word-aligned (padded to an even byte count).
        pos = body + chunk_size + (chunk_size & 1)
    if not byte_rate or data_size is None or byte_rate <= 0:
        return None
    # Placeholder or impossible header size (streamed WAV): fall back to the
    # actual delivered payload length when known.
    if total_bytes is not None and (data_size >= 0xFFFFFFFE or data_size > total_bytes):
        data_size = max(0, total_bytes - data_start)
    if data_size >= 0xFFFFFFFE:
        return None
    return data_size / byte_rate


def sniff_limit() -> int:
    """How many leading response bytes to retain for WAV header sniffing."""
    return _WAV_SNIFF_BYTES


def audio_usage_kwargs(
    *,
    definition_id: uuid.UUID | None,
    instance_id: uuid.UUID,
    modality: str,
    chars: int,
    audio_seconds: float | None,
) -> dict[str, Any]:
    """Keyword args for an ``AudioUsageSample`` (kept in one place so the
    speech/transcription persist helpers stay consistent)."""
    return {
        "provider_definition_id": definition_id,
        "provider_instance_id": instance_id,
        "modality": modality,
        "chars": chars,
        "audio_seconds": audio_seconds,
    }
