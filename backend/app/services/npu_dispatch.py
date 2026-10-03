"""Shared lease + agent-proxy dispatch for halogen-flash NPU routes.

The rerank, decisions and moderations endpoints all resolve an NPU virtual
alias to a flash instance, acquire an inference lease on that instance's
model, proxy a POST to the agent, and record telemetry. This factors that
dance so each route only supplies its upstream path, body, and how to turn
the raw dict into an HTTPException on failure.
"""

import asyncio
import logging
import time
import uuid

from fastapi import HTTPException
from sqlmodel import Session

from app.services.agent_manager import agent_manager
from app.services.inference_scheduler import InferenceLeaseHandle, inference_scheduler
from app.services.inference_target import InferenceTarget, resolve_inference_target
from app.services.token_stats import record_request_telemetry

logger = logging.getLogger(__name__)


def resolve_npu_target(
    db: Session, model_ref: str, agent_id: str | None, capability: str
) -> InferenceTarget:
    """Resolve ``<alias>-<suffix>`` and require the NPU capability.

    ``capability`` is the suffix expected (``rerank`` / ``decide`` /
    ``guard``); a name that resolves but not to that suffix is a 400.
    """
    try:
        target = resolve_inference_target(db, model_ref, agent_id)
    except (LookupError, ValueError) as e:
        raise HTTPException(404, str(e)) from e
    if target.server is None or target.npu_model is None:
        raise HTTPException(
            400,
            f"Model '{model_ref}' does not support {capability}",
        )
    from app.services.npu_aliases import suffix_for

    if suffix_for(target.npu_model) != capability:
        raise HTTPException(
            400,
            f"Model '{model_ref}' does not support {capability}",
        )
    return target


async def dispatch_npu(
    target: InferenceTarget,
    path: str,
    body: dict,
    *,
    request_id_prefix: str,
    is_disconnected=None,
    timeout: float = 1800.0,
) -> dict:
    """Acquire a lease on the target's flash instance and proxy a POST.

    Returns the upstream JSON dict. Always releases the lease. Raises
    HTTPException on proxy/parse failure with telemetry recorded.
    """
    server = target.server
    model = target.model
    lease: InferenceLeaseHandle | None = None
    started = time.monotonic()
    terminal_outcome = "completed"
    try:
        lease = await inference_scheduler.acquire(
            model.id,
            f"{request_id_prefix}-{uuid.uuid4()}",
            preferred_server_id=target.preferred_server_id,
            required_agent_id=target.required_agent_id,
            is_cancelled=is_disconnected,
        )
        server = lease.server
        response = await lease.guard(
            agent_manager.send_to_agent(
                str(server.agent_id),
                "POST",
                f"/proxy/{server.id}/{path}",
                body,
                timeout=timeout,
                headers={**lease.dispatch_headers()},
            )
        )
        success_latency_ms = (time.monotonic() - started) * 1000.0
        if not isinstance(response, dict):
            raise HTTPException(500, f"Invalid {path} response from server")
        usage = response.get("usage") or {}
        total_tokens = usage.get("prompt_tokens", usage.get("total_tokens", 0))
        record_request_telemetry(
            server,
            {"prompt_tokens": total_tokens},
            response.get("timings"),
            latency_ms=success_latency_ms,
            status="success",
        )
        return response
    except HTTPException:
        terminal_outcome = "failed"
        record_request_telemetry(
            server,
            None,
            None,
            latency_ms=(time.monotonic() - started) * 1000.0,
            status="error",
        )
        raise
    except asyncio.CancelledError:
        terminal_outcome = "cancelled"
        record_request_telemetry(
            server,
            None,
            None,
            latency_ms=(time.monotonic() - started) * 1000.0,
            status="cancelled",
        )
        raise
    except TimeoutError as e:
        terminal_outcome = "failed"
        record_request_telemetry(
            server,
            None,
            None,
            latency_ms=(time.monotonic() - started) * 1000.0,
            status="error",
        )
        raise HTTPException(503, str(e)) from e
    except Exception as e:
        terminal_outcome = "failed"
        record_request_telemetry(
            server,
            None,
            None,
            latency_ms=(time.monotonic() - started) * 1000.0,
            status="error",
        )
        logger.error(f"{path} error: {e}")
        raise HTTPException(500, f"Failed {path} request: {e}") from e
    finally:
        if lease is not None:
            await lease.release(terminal_outcome)
