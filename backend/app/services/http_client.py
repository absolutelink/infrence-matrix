"""Shared bounded httpx client for backend -> agent HTTP calls."""

import httpx

_shared_client: httpx.AsyncClient | None = None


def get_http_client() -> httpx.AsyncClient:
    """Return the process-wide shared client, creating it on first use.

    Lazy creation keeps import order simple and lets tests patch the accessor.
    A generous connection cap with bounded keepalive reuses connections across
    requests instead of a fresh TCP handshake per call.
    """
    global _shared_client
    if _shared_client is None or _shared_client.is_closed:
        _shared_client = httpx.AsyncClient(
            timeout=300.0,
            limits=httpx.Limits(
                max_connections=500,
                max_keepalive_connections=100,
                keepalive_expiry=30.0,
            ),
        )
    return _shared_client


async def aclose_http_client() -> None:
    """Close the shared client and reset the singleton (called at shutdown)."""
    global _shared_client
    if _shared_client is not None and not _shared_client.is_closed:
        await _shared_client.aclose()
    _shared_client = None
