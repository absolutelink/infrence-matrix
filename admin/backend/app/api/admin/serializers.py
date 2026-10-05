"""Shared serialization helpers for admin read dicts.

Several timestamp columns (``created_at`` / ``updated_at`` /
``completed_at``) are plain ``DateTime`` without timezone, so
``.isoformat()`` emits no UTC offset and browsers in non-UTC locales
misparse the value as local time. The stored values are always UTC
(written via ``get_datetime_utc()``); the offset is restored here at
the serialization layer instead of a schema migration. Aware columns
(``last_seen``, ``last_request_at``, ``TokenUsageSample.created_at``)
pass through untouched — no double-offsetting.
"""

from datetime import UTC, datetime


def iso_utc(dt: datetime | None) -> str | None:
    """ISO-8601 with an explicit UTC offset; None passthrough."""
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.isoformat()
