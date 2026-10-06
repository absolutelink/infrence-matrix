"""Phase 13 log ingest + read store (Redis-only).

Consumes batched ``backend.logs`` / ``provider.logs`` events from the
provider WS and appends them to capped Redis lists:

  im:logs:backend:{instance_id}   List of JSON entries, newest LEFT
  im:logs:provider:{instance_id}  same shape
  im:logs:seq:{instance_id}       shared monotonic ingest counter (INCR)
  im:logs:dropped:{kind}:{id}     latest provider-reported drop counter

Entry shape (stored JSON): ``{"seq": int, "ts": iso, "stream": str,
"text": str}``. ``seq`` is assigned at ingest via ``INCR
im:logs:seq:{instance_id}`` — shared across both kinds — so a single
cursor orders ``kind=all`` merges correctly across LPUSH lists (list
index would not survive interleaving between the two lists).

Cap/TTL: LTRIM to ``LOGS_CAP`` (2000) newest entries, EXPIRE 1h —
logs are ephemeral ops telemetry, never Postgres.

Read cursor scheme (``GET /admin/api/instances/{id}/logs``): entries
are returned **newest first**; ``since`` means "only entries with
``seq > since``" and ``cursor`` in the response is the max ``seq``
returned (echoed ``since`` when nothing matched). The UI polls
``since=cursor`` for the live tail. If more than ``limit`` unseen
entries exist, the newest ``limit`` are returned (older unseen entries
are skipped — tailing prioritizes freshness over paging completeness,
and the 2000-line cap bounds the loss window).

Everything here is best-effort: ``ingest_log_batch`` never raises on a
malformed payload (logs must never break WS handling), and normalizes
the legacy per-line ``{"stream", "line"}`` shape to a one-element
batch for transition safety.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from app.core.redis import get_redis_from_app
from app.services import redis_keys

logger = logging.getLogger("admin.log_store")

KIND_BACKEND = "backend"
KIND_PROVIDER = "provider"
KIND_ALL = "all"


def normalize_log_batch(payload: Any) -> tuple[list[dict[str, Any]], int]:
    """Extract ``[{"ts","stream","text"}...]`` + dropped from an event.

    Accepts the Phase 13 batched shape ``{"lines": [...], "dropped": N}``
    and the legacy per-line shape ``{"stream": s, "line": l}`` (a
    pre-batching provider still mid-upgrade). Anything unusable yields
    an empty list — never an exception.
    """
    if not isinstance(payload, dict):
        return [], 0
    dropped_raw = payload.get("dropped")
    try:
        dropped = int(dropped_raw) if dropped_raw is not None else 0
    except (TypeError, ValueError):  # fmt: skip
        dropped = 0

    lines = payload.get("lines")
    normalized: list[dict[str, Any]] = []
    if isinstance(lines, list):
        for item in lines:
            if not isinstance(item, dict):
                continue
            text = item.get("text")
            if text is None:
                # Legacy per-line entry shape inside a batch.
                text = item.get("line")
            if text is None:
                continue
            normalized.append(
                {
                    "ts": str(item.get("ts") or ""),
                    "stream": str(item.get("stream") or "stdout"),
                    "text": str(text),
                }
            )
    elif "line" in payload:
        # Legacy single-line event.
        normalized.append(
            {
                "ts": str(payload.get("ts") or ""),
                "stream": str(payload.get("stream") or "stdout"),
                "text": str(payload.get("line", "")),
            }
        )
    return [e for e in normalized if e["text"]], dropped


async def ingest_log_batch(app: Any, instance_id: str, kind: str, payload: Any) -> None:
    """Append one batched log event to the instance's Redis list.

    ``kind`` is ``"backend"`` or ``"provider"``. Best-effort: Redis
    errors are logged (at the admin's own logger, never re-ingested)
    and swallowed.
    """
    entries, dropped = normalize_log_batch(payload)
    if kind not in (KIND_BACKEND, KIND_PROVIDER):
        return
    try:
        redis_client = get_redis_from_app(app)
        list_key = (
            redis_keys.logs_backend_key(instance_id)
            if kind == KIND_BACKEND
            else redis_keys.logs_provider_key(instance_id)
        )
        if entries:
            seq_key = redis_keys.logs_seq_key(instance_id)
            start = await redis_client.incrby(seq_key, len(entries))
            # 1-based seq: `since=0` (the default) means "everything".
            base = start - len(entries)
            pipe = redis_client.pipeline()
            for offset, entry in enumerate(entries):
                stored = {"seq": base + offset + 1, **entry}
                pipe.lpush(list_key, json.dumps(stored))
            pipe.ltrim(list_key, 0, redis_keys.LOGS_CAP - 1)
            pipe.expire(list_key, redis_keys.LOGS_TTL_SECONDS)
            if dropped:
                pipe.set(
                    redis_keys.logs_dropped_key(instance_id, kind),
                    dropped,
                    ex=redis_keys.LOGS_TTL_SECONDS,
                )
            # Keep the seq counter alive at least as long as the tails.
            pipe.expire(seq_key, redis_keys.LOGS_TTL_SECONDS)
            await pipe.execute()
        elif dropped:
            await redis_client.set(
                redis_keys.logs_dropped_key(instance_id, kind),
                dropped,
                ex=redis_keys.LOGS_TTL_SECONDS,
            )
    except Exception:  # noqa: BLE001 - logs are best-effort
        logger.warning(
            "log ingest failed (kind=%s instance=%s)", kind, instance_id, exc_info=True
        )


async def read_logs(
    redis_client: Any,
    instance_id: str,
    *,
    kind: str = KIND_BACKEND,
    since: int = 0,
    limit: int = 500,
) -> dict[str, Any]:
    """Read the tail for a kind with the monotonic-seq cursor.

    Returns ``{"entries": [ {seq, ts, stream, text} ... ] (newest
    first), "cursor": int, "dropped": int, "gap": bool, "oldest_seq":
    int}``. Missing keys read as empty, not an error.

    ``gap`` is True when ``since`` points before the oldest entry still
    retained in the Redis list(s) (``since + 1 < oldest_seq``): entries
    between were LTRIM'd away and the UI can show "N entries lost".
    ``oldest_seq`` is the smallest retained seq (0 when nothing is
    retained).

    NOTE (M2): the ``seq``/``cursor`` here belongs to the ADMIN's ingest
    space (``im:logs:seq:{instance_id}``, assigned at LPUSH time). It
    is UNRELATED to the provider-side ring seq in the ``backend.logs.get``
    ack (assigned by the provider's shared ``SeqCounter``). Clients must
    never mix the two cursors.
    """
    list_keys: list[str] = []
    dropped_kinds: list[str] = []
    if kind == KIND_ALL:
        list_keys = [
            redis_keys.logs_backend_key(instance_id),
            redis_keys.logs_provider_key(instance_id),
        ]
        dropped_kinds = [KIND_BACKEND, KIND_PROVIDER]
    elif kind == KIND_BACKEND:
        list_keys = [redis_keys.logs_backend_key(instance_id)]
        dropped_kinds = [KIND_BACKEND]
    elif kind == KIND_PROVIDER:
        list_keys = [redis_keys.logs_provider_key(instance_id)]
        dropped_kinds = [KIND_PROVIDER]
    else:
        raise ValueError(f"unknown log kind: {kind!r}")

    # Over-fetch: with two merged lists each may hold the newest entries;
    # 2*limit keeps the merge window honest before filtering by seq.
    fetch = limit * len(list_keys)
    pipe = redis_client.pipeline()
    for key in list_keys:
        pipe.lrange(key, 0, fetch - 1)
        # Tail of the list = oldest retained entry (for gap detection).
        pipe.lrange(key, -1, -1)
    for dk in dropped_kinds:
        pipe.get(redis_keys.logs_dropped_key(instance_id, dk))
    results = await pipe.execute()

    raw_lists = results[0 : len(list_keys) * 2 : 2]
    oldest_raws = results[1 : len(list_keys) * 2 : 2]
    dropped_values = results[len(list_keys) * 2 :]
    dropped = sum(int(v) for v in dropped_values if v is not None)

    oldest_seqs: list[int] = []
    entries: list[dict[str, Any]] = []
    for raw, oldest_raw in zip(raw_lists, oldest_raws, strict=True):
        if oldest_raw:
            try:
                oldest = json.loads(oldest_raw[0])
                oldest_seqs.append(int(oldest.get("seq", 0)))
            except (TypeError, ValueError, IndexError, KeyError):  # fmt: skip
                pass
        for item in raw:
            try:
                entry = json.loads(item)
            except (TypeError, ValueError):  # fmt: skip
                continue
            if not isinstance(entry, dict):
                continue
            try:
                seq = int(entry.get("seq", 0))
            except (TypeError, ValueError):  # fmt: skip
                seq = 0
            if seq > since:
                entries.append(
                    {
                        "seq": seq,
                        "ts": entry.get("ts", ""),
                        "stream": entry.get("stream", "stdout"),
                        "text": entry.get("text", ""),
                    }
                )
    entries.sort(key=lambda e: e["seq"], reverse=True)
    entries = entries[:limit]
    cursor = entries[0]["seq"] if entries else since
    oldest_seq = min(oldest_seqs) if oldest_seqs else 0
    gap = bool(oldest_seqs) and (since + 1 < oldest_seq)
    return {
        "entries": entries,
        "cursor": cursor,
        "dropped": dropped,
        "gap": gap,
        "oldest_seq": oldest_seq,
    }
