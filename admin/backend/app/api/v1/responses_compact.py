"""Public OpenResponses endpoint: ``POST /v1/responses/compact`` (Phase 20).

Model-driven conversation compaction. The accumulated input items (plus the
``previous_response_id`` chain, ``instructions``, and ``prompt_cache_key``)
are sent through the same scheduler admission + ``litellm.aresponses``
non-stream path as ``/v1/responses`` (Phase 6), but with a fixed compression
prompt, and the model's summary is re-framed as a spec ``CompactResource``:
admin-owned ``cmp_<uuid>`` id, ``object: "response.compaction"``, one
``compaction`` output item whose ``encrypted_content`` is base64 JSON of the
summary, and concrete ``usage`` from the provider response. The turn is
persisted as a ``ResponseRecord`` with ``parameters.api_format="compaction"``.

Non-stream only: the spec's compact body has no ``stream`` field, so every
error is a real HTTP JSON response (400/404 pre-stream, 503/504 admission,
502 upstream failure) — mirroring the ``/v1/responses`` non-stream contract.

Flow:

1. Parse the body: ``model`` (alias) required; at least one of ``input`` or
   ``previous_response_id`` required; optional ``instructions`` and
   ``prompt_cache_key``.
2. Resolve the enabled ``ProviderDefinition`` by alias (``modality == "llm"``)
   -> else 404; ``previous_response_id`` not found -> 404.
3. Mint ``client_response_id = "cmp_" + uuid4().hex`` (the admin owns the
   client-facing id; litellm's wrapped id is stored in parameters).
4. Build the litellm ``input`` with ``build_litellm_input`` (on continuation
   the prior record's items are prepended — the admin owns the chain and never
   passes ``previous_response_id`` to litellm).
5. ``ensure_registered(alias)`` then ``scheduler.acquire(alias, request_id)``
   -> ``Admission``. ``NoProviderAvailable``/``QueueCleared`` -> 503,
   ``QueueTimeout`` -> 504 (JSON, since there is no stream).
6. ``litellm.aresponses`` non-stream with the fixed compression prompt as
   ``instructions`` (request instructions, if any, are prepended). Upstream
   failure -> failed ``ResponseRecord`` + 502 JSON error envelope.
7. Normalize the terminal response, extract the summary text from its
   ``output_text`` parts, wrap it as one ``compaction`` item, persist, release
   the slot (shielded, idempotent), return the ``CompactResource`` JSON.

Auth model: trusted LAN — this public route is unauthenticated (see
``app/core/config.py``).
"""

import asyncio
import base64
import contextlib
import copy
import json
import logging
import uuid
from datetime import UTC, datetime
from typing import Any

import litellm
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
from sqlmodel import Session, select

from app.api.v1.responses import (
    _as_item_list,
    _clean_output_items,
    _normalize_response,
    build_litellm_input,
    get_scheduler,
    map_exception_to_error,
    persist_turn_logged,
)
from app.core.db import engine
from app.models import ProviderDefinition, ResponseRecord
from app.services.alias_registry import ensure_registered
from app.services.scheduler import NoProviderAvailable, QueueCleared, QueueTimeout
from app.services.sse import to_dict

logger = logging.getLogger("admin.v1.responses.compact")

router = APIRouter(tags=["responses"])

# Fixed compression prompt sent as ``instructions`` to the model. A request's
# own ``instructions`` (if any) are prepended so operator guidance survives.
COMPACT_INSTRUCTIONS = (
    "Compress the conversation into terse notes that preserve every fact, "
    "decision, constraint, and open question. Output plain text only."
)

# Zeroed spec usage shape, synthesized when the provider omits usage entirely.
_ZEROED_USAGE: dict[str, Any] = {
    "input_tokens": 0,
    "output_tokens": 0,
    "total_tokens": 0,
    "input_tokens_details": {"cached_tokens": 0},
    "output_tokens_details": {"reasoning_tokens": 0},
}


def _extract_summary_text(output_items: list[Any]) -> str:
    """Concatenate every ``output_text`` content part across the output items.

    Works on the cleaned terminal output (post ``_clean_output_items``) so the
    summary reflects exactly what the spec response carries.
    """
    parts: list[str] = []
    for item in output_items:
        if not isinstance(item, dict):
            continue
        content = item.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if isinstance(part, dict) and part.get("type") == "output_text":
                text = part.get("text")
                if isinstance(text, str):
                    parts.append(text)
    return "".join(parts)


@router.post("/v1/responses/compact")
async def compact_response(request: Request) -> Any:
    body = await request.json()
    if not isinstance(body, dict):
        raise HTTPException(
            status_code=400, detail="request body must be a JSON object"
        )

    alias = body.get("model")
    if not alias:
        raise HTTPException(status_code=400, detail="'model' is required")

    input_value = body.get("input")
    previous_response_id = body.get("previous_response_id")
    if input_value is None and not previous_response_id:
        raise HTTPException(
            status_code=400,
            detail="either 'input' or 'previous_response_id' is required",
        )

    with Session(engine) as session:
        definition = session.exec(
            select(ProviderDefinition).where(ProviderDefinition.alias == alias)
        ).first()
        if definition is None:
            raise HTTPException(
                status_code=404, detail=f"unknown model alias '{alias}'"
            )
        if not definition.enabled:
            raise HTTPException(
                status_code=404, detail=f"model alias '{alias}' is disabled"
            )
        if definition.modality != "llm":
            raise HTTPException(
                status_code=404,
                detail=f"model alias '{alias}' is not a chat model",
            )
        definition_id = definition.id
        provider_type = definition.provider_type

        previous: ResponseRecord | None = None
        if previous_response_id:
            previous = session.exec(
                select(ResponseRecord).where(
                    ResponseRecord.response_id == previous_response_id
                )
            ).first()
            if previous is None:
                raise HTTPException(
                    status_code=404,
                    detail=f"previous_response_id '{previous_response_id}' not found",
                )

    # Compaction records are always stored (the body schema has no ``store``).
    client_response_id = f"cmp_{uuid.uuid4().hex}"
    request_id = client_response_id

    ensure_registered(alias)
    litellm_input = build_litellm_input(body, previous)
    input_items_for_record = _as_item_list(litellm_input)

    # The request's own instructions (if any) are prepended to the fixed
    # compression prompt so operator guidance is preserved through compaction.
    body_instructions = body.get("instructions")
    if isinstance(body_instructions, str) and body_instructions:
        instructions = f"{body_instructions}\n\n{COMPACT_INSTRUCTIONS}"
    else:
        instructions = COMPACT_INSTRUCTIONS

    scheduler = get_scheduler(request)
    try:
        admission = await scheduler.acquire(alias, request_id)
    except NoProviderAvailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except QueueCleared as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except QueueTimeout as exc:
        raise HTTPException(status_code=504, detail=str(exc)) from exc

    base_params: dict[str, Any] = {
        "model": alias,
        "provider_type": provider_type,
        "api_format": "compaction",
        "stream": False,
    }

    try:
        try:
            litellm_kwargs: dict[str, Any] = {
                "model": alias,
                "custom_llm_provider": "openai",
                "api_base": f"{admission.base_url}/v1",
                "stream": False,
                "input": litellm_input,
                "instructions": instructions,
            }
            if "prompt_cache_key" in body:
                litellm_kwargs["prompt_cache_key"] = body["prompt_cache_key"]
            result = await litellm.aresponses(**litellm_kwargs)
        except Exception as exc:  # noqa: BLE001 - upstream litellm failure
            error = map_exception_to_error(exc)
            logger.warning(
                "litellm compaction failed for %s: %s", client_response_id, exc
            )
            persist_turn_logged(
                client_response_id=client_response_id,
                previous_response_id=previous_response_id,
                input_items=input_items_for_record,
                output_items=[],
                definition_id=definition_id,
                instance_id=uuid.UUID(admission.instance_id),
                parameters={
                    **base_params,
                    "litellm_response_id": None,
                    "api_base": admission.base_url,
                },
                status="failed",
                usage=None,
                error=error,
                store=True,
            )
            return JSONResponse(
                status_code=502,
                content={
                    "error": {
                        "type": error["type"],
                        "code": error["code"],
                        "message": error["message"],
                    }
                },
            )

        # Copy: to_dict passes plain dicts through — don't mutate litellm's
        # accumulated response object.
        data = dict(to_dict(result))
        litellm_wrapped_id = data.get("id")
        output_items = _clean_output_items(_as_item_list(data.get("output")))
        data["output"] = output_items
        _normalize_response(data, store=True, requested={"model": alias})

        summary_text = _extract_summary_text(data["output"])
        usage = data.get("usage") or copy.deepcopy(_ZEROED_USAGE)

        compaction_item: dict[str, Any] = {
            "type": "compaction",
            "id": f"cmpitem_{uuid.uuid4().hex}",
            "encrypted_content": base64.b64encode(
                json.dumps(
                    {
                        "summary": summary_text,
                        "source_count": len(input_items_for_record),
                    }
                ).encode()
            ).decode(),
        }

        persist_turn_logged(
            client_response_id=client_response_id,
            previous_response_id=previous_response_id,
            input_items=input_items_for_record,
            output_items=[compaction_item],
            definition_id=definition_id,
            instance_id=uuid.UUID(admission.instance_id),
            parameters={
                **base_params,
                "litellm_response_id": litellm_wrapped_id,
                "api_base": admission.base_url,
            },
            status=data.get("status", "completed"),
            usage=usage,
            error=None,
            store=True,
        )
        return JSONResponse(
            content={
                "id": client_response_id,
                "object": "response.compaction",
                "created_at": int(datetime.now(UTC).timestamp()),
                "output": [compaction_item],
                "usage": usage,
            }
        )
    finally:
        # Guaranteed slot release on EVERY exit path — including
        # asyncio.CancelledError (client disconnect during the await), which
        # `except Exception` does not catch. Release is idempotent.
        with contextlib.suppress(Exception):
            await asyncio.shield(scheduler.release(alias, request_id))
