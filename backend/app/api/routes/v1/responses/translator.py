"""Translation between OpenResponses items and llama.cpp chat format.

Request direction: input items -> llama.cpp chat messages + tools payload.
Response direction: llama.cpp (streaming chunks or final message) ->
output items + semantic streaming events.

llama.cpp facts baked in here:
- Tool calling needs --jinja on llama-server; tools use the nested
  {"type":"function","function":{...}} OpenAI shape.
- Non-streaming tool calls come back as message.tool_calls[] with
  finish_reason="tool_calls"; streaming uses delta.tool_calls[] fragments.
- Thinking models expose reasoning via reasoning_content deltas.
"""

from __future__ import annotations

import logging
from typing import Any

from app.api.routes.v1.responses import events as ev
from app.api.routes.v1.responses.schemas import (
    CreateResponseBody,
    ResponseResource,
    ResponseText,
    ResponseTextFormat,
    new_id,
)

logger = logging.getLogger(__name__)


class TranslationError(Exception):
    """Raised when input items cannot be represented in llama.cpp chat format."""


def _part_text(part: Any) -> str | None:
    """Extract text from a content part; non-text parts have no llama form."""
    if isinstance(part, str):
        return part
    ptype = part.get("type") if isinstance(part, dict) else getattr(part, "type", None)
    if ptype in ("input_text", "output_text", "reasoning_text", "summary_text"):
        text = part.get("text") if isinstance(part, dict) else part.text
        return str(text) if text is not None else None
    if ptype == "refusal":
        refusal = part.get("refusal") if isinstance(part, dict) else part.refusal
        return str(refusal) if refusal is not None else None
    return None


def _part_is_image(part: Any) -> str | None:
    """Image URL from an input_image part, else None."""
    if isinstance(part, str):
        return None
    ptype = part.get("type") if isinstance(part, dict) else getattr(part, "type", None)
    if ptype == "input_image":
        url = (
            part.get("image_url")
            if isinstance(part, dict)
            else getattr(part, "image_url", None)
        )
        return str(url) if url else None
    return None


def _content_to_llama(content: Any) -> str | list[dict[str, Any]]:
    """Flatten content (string or parts) to a llama.cpp message content.

    Text parts merge into a string; image parts become llama.cpp's
    {"type": "image_url", "image_url": {"url": ...}} entries interleaved in
    order. The server rejects images when no mmproj projector is loaded
    (surfaced as a model error), so no client-side gating is needed.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    parts: list[dict[str, Any]] = []
    text_buf: list[str] = []

    def flush_text() -> None:
        if text_buf:
            parts.append({"type": "text", "text": "".join(text_buf)})
            text_buf.clear()

    for part in content:
        image_url = _part_is_image(part)
        if image_url is not None:
            flush_text()
            parts.append({"type": "image_url", "image_url": {"url": image_url}})
            continue
        text = _part_text(part)
        if text is None:
            ptype = (
                part.get("type")
                if isinstance(part, dict)
                else getattr(part, "type", "unknown")
            )
            if ptype in ("input_file", "input_video"):
                raise TranslationError(
                    f"{ptype} content is not supported by llama.cpp chat; "
                    "use input_text or input_image parts."
                )
            raise TranslationError(f"Unsupported content part type: {ptype}")
        text_buf.append(text)
    flush_text()
    if not parts:
        return ""
    if len(parts) == 1 and parts[0]["type"] == "text":
        return parts[0]["text"]
    return parts


def tools_to_llama(request: CreateResponseBody) -> dict[str, Any]:
    """Translate Responses flat tools/tool_choice to llama.cpp nested shapes."""
    extra: dict[str, Any] = {}
    if request.tools:
        extra["tools"] = [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    **({"description": t.description} if t.description else {}),
                    **({"parameters": t.parameters} if t.parameters else {}),
                    **({"strict": t.strict} if t.strict is not None else {}),
                },
            }
            for t in request.tools
        ]

    choice = request.tool_choice
    if isinstance(choice, str):
        extra["tool_choice"] = choice
    elif isinstance(choice, dict) or hasattr(choice, "type"):
        ctype = choice.type if hasattr(choice, "type") else choice.get("type")
        if ctype == "function":
            cname = (
                choice.name if hasattr(choice, "name") else choice.get("name")  # type: ignore[union-attr]
            )
            extra["tool_choice"] = {"type": "function", "function": {"name": cname}}
        elif ctype == "allowed_tools":
            # llama.cpp has no allowed_tools; we pass through choice mode and
            # enforce the subset ourselves after sampling.
            extra["tool_choice"] = "auto"
    return extra


def allowed_tool_names(request: CreateResponseBody) -> set[str] | None:
    """Names permitted by an allowed_tools tool_choice, else None."""
    choice = request.tool_choice
    if isinstance(choice, dict) and choice.get("type") == "allowed_tools":
        return {t.get("name") for t in choice.get("tools", [])}
    if not isinstance(choice, dict) and hasattr(choice, "type"):
        try:
            if choice.type == "allowed_tools":  # type: ignore[union-attr]
                return {t.name for t in choice.tools}  # type: ignore[union-attr]
        except AttributeError:
            pass
    return None


def input_items_to_llama_messages(
    request: CreateResponseBody, history_items: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Build llama.cpp chat messages from history + this request's input.

    history_items are prior stored items (previous_response_id chaining):
    previous.input + previous.output were already merged by the caller into
    this list in semantic order. Message content may be a string or llama.cpp
    multimodal parts (text + image_url entries).
    """
    messages: list[dict[str, Any]] = []
    if request.instructions:
        messages.append({"role": "system", "content": request.instructions})

    all_items: list[Any] = []
    for raw in history_items:
        all_items.append(raw)
    if isinstance(request.input, str):
        all_items.append({"type": "message", "role": "user", "content": request.input})
    else:
        all_items.extend(request.input)

    for item in all_items:
        itype = (
            item.get("type") if isinstance(item, dict) else getattr(item, "type", None)
        )
        if itype == "item_reference":
            raise TranslationError(
                "item_reference is not supported (no stored item registry)"
            )
        if itype == "compaction":
            # Compacted summary (our compact endpoint stores the summary text
            # as encrypted_content): replay as assistant context so chains can
            # continue from a compaction item (spec recovery flow).
            content = (
                item.get("encrypted_content")
                if isinstance(item, dict)
                else item.encrypted_content
            )
            messages.append({"role": "assistant", "content": content or ""})
            continue
        if itype == "reasoning":
            continue  # reasoning items are model-internal; not replayed
        if itype == "function_call":
            # Assistant tool call: describe it in text (llama.cpp receives
            # tool calls through the template; replaying as text keeps the
            # exchange visible for templates without native tool replay).
            name = item.get("name") if isinstance(item, dict) else item.name
            arguments = (
                item.get("arguments") if isinstance(item, dict) else item.arguments
            )
            messages.append(
                {
                    "role": "assistant",
                    "content": f"[Called tool {name} with arguments: {arguments}]",
                }
            )
        elif itype == "function_call_output":
            call_id = item.get("call_id") if isinstance(item, dict) else item.call_id
            output = item.get("output") if isinstance(item, dict) else item.output
            text = _content_to_llama(output)
            if not isinstance(text, str):
                text = "".join(
                    p.get("text", "") for p in text if p.get("type") == "text"
                )
            messages.append(
                {"role": "user", "content": f"[Tool result ({call_id})]: {text}"}
            )
        elif itype == "message":
            role = item.get("role") if isinstance(item, dict) else item.role
            content = item.get("content") if isinstance(item, dict) else item.content
            phase = (
                item.get("phase")
                if isinstance(item, dict)
                else getattr(item, "phase", None)
            )
            llama_content = _content_to_llama(content)
            if role == "assistant" and phase and isinstance(llama_content, str):
                # Phase labels (commentary/final_answer) are preserved as a
                # bracketed cue so the model's continuation keeps context.
                llama_content = f"[assistant phase: {phase}] {llama_content}".strip()
            messages.append({"role": role or "user", "content": llama_content})
        else:
            raise TranslationError(f"Unknown input item type: {itype}")
    return messages


# ============================================================================
# Responses -> gufo native /v1/responses (request builder)
# ============================================================================


def _field(obj: Any, key: str) -> Any:
    """Read a field from a dict or a pydantic model uniformly."""
    if isinstance(obj, dict):
        return obj.get(key)
    return getattr(obj, key, None)


def _native_content_part(part: Any) -> dict[str, Any]:
    """Map one broker content part to gufo's native Responses input part.

    gufo accepts input_text / input_image (HTTPS or data URL) for user turns
    and output_text for assistant turns. Unsupported part types raise a
    TranslationError so the request fails fast with a 400 instead of being
    silently dropped.
    """
    if isinstance(part, str):
        return {"type": "input_text", "text": part}
    ptype = _field(part, "type")
    if ptype in ("input_text", "output_text"):
        return {"type": ptype, "text": str(_field(part, "text") or "")}
    if ptype == "input_image":
        url = _field(part, "image_url")
        if not url:
            raise TranslationError("input_image requires an image_url")
        out: dict[str, Any] = {"type": "input_image", "image_url": str(url)}
        detail = _field(part, "detail")
        if detail:
            out["detail"] = detail
        return out
    if ptype in ("input_file", "input_video"):
        raise TranslationError(
            f"{ptype} content is not supported by gufo's native /v1/responses; "
            "use input_text or input_image parts."
        )
    raise TranslationError(f"Unsupported content part type: {ptype}")


def _native_message_content(content: Any, role: str) -> list[dict[str, Any]]:
    if content is None:
        return []
    text_type = "output_text" if role == "assistant" else "input_text"
    if isinstance(content, str):
        return [{"type": text_type, "text": content}]
    parts: list[dict[str, Any]] = []
    for part in content:
        mapped = _native_content_part(part)
        if mapped["type"] in ("input_text", "output_text"):
            mapped = {"type": text_type, "text": mapped.get("text", "")}
        parts.append(mapped)
    return parts


def _summary_parts(raw: Any) -> list[dict[str, Any]]:
    """Coerce a list of reasoning content parts into gufo summary_text parts."""
    out: list[dict[str, Any]] = []
    for part in raw or []:
        text = _field(part, "text")
        if text:
            out.append({"type": "summary_text", "text": str(text)})
    return out


def _native_reasoning_item(item: Any) -> dict[str, Any]:
    """Replay a prior reasoning item in gufo's accepted shape.

    gufo requires ``summary`` (array of summary_text) and rejects a ``content``
    key or a non-null ``encrypted_content``. Broker reasoning items may carry
    the raw text in ``content`` (reasoning_text) with an empty ``summary``, so
    fold any reasoning_text into summary parts.
    """
    native_summary = _summary_parts(_field(item, "summary"))
    if not native_summary:
        native_summary = _summary_parts(_field(item, "content"))
    return {"type": "reasoning", "summary": native_summary}


def _native_function_call_item(item: Any) -> dict[str, Any]:
    call_id = _field(item, "call_id")
    if not call_id:
        raise TranslationError("function_call items require a call_id")
    return {
        "type": "function_call",
        "call_id": str(call_id),
        "name": str(_field(item, "name") or "unknown"),
        "arguments": _field(item, "arguments") or "{}",
    }


def _native_function_output_item(item: Any) -> dict[str, Any]:
    call_id = _field(item, "call_id")
    if not call_id:
        raise TranslationError("function_call_output items require a call_id")
    output = _field(item, "output")
    if isinstance(output, str) or output is None:
        native_output: Any = output or ""
    else:
        native_output = [_native_content_part(part) for part in output]
    return {
        "type": "function_call_output",
        "call_id": str(call_id),
        "output": native_output,
    }


def input_items_to_gufo_native(
    request: CreateResponseBody, history_items: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Build gufo's native Responses ``input`` array from history + this turn.

    History is materialized into the request because gufo has no
    ``previous_response_id``. Prior assistant reasoning, function calls and
    function outputs are replayed verbatim (gufo owns tool/reasoning
    semantics), unlike the llama.cpp text-substitution path.
    """
    all_items: list[Any] = list(history_items)
    if isinstance(request.input, str):
        all_items.append({"type": "message", "role": "user", "content": request.input})
    else:
        all_items.extend(request.input)

    native: list[dict[str, Any]] = []
    for item in all_items:
        itype = _field(item, "type")
        if itype == "item_reference":
            raise TranslationError(
                "item_reference is not supported (no stored item registry)"
            )
        if itype == "reasoning":
            native.append(_native_reasoning_item(item))
        elif itype == "function_call":
            native.append(_native_function_call_item(item))
        elif itype == "function_call_output":
            native.append(_native_function_output_item(item))
        elif itype == "compaction":
            content = _field(item, "encrypted_content")
            native.append(
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": content or ""}],
                }
            )
        elif itype == "message":
            role = _field(item, "role") or "user"
            content = _field(item, "content")
            native.append(
                {
                    "type": "message",
                    "role": role,
                    "content": _native_message_content(content, role),
                }
            )
        else:
            raise TranslationError(f"Unknown input item type: {itype}")
    return native


def _native_tools(request: CreateResponseBody) -> list[dict[str, Any]]:
    tools: list[dict[str, Any]] = []
    for t in request.tools:
        entry: dict[str, Any] = {"type": "function", "name": t.name}
        if t.description:
            entry["description"] = t.description
        if t.parameters is not None:
            entry["parameters"] = t.parameters
        if t.strict is not None:
            entry["strict"] = t.strict
        tools.append(entry)
    return tools


def _native_tool_choice(request: CreateResponseBody) -> Any:
    """Map the broker tool_choice to gufo's accepted value.

    allowed_tools has no gufo equivalent: hard-error rather than silently
    widening to auto (parity decision: no fallback).
    """
    choice = request.tool_choice
    if isinstance(choice, str):
        return choice
    ctype = getattr(choice, "type", None) or (
        choice.get("type") if isinstance(choice, dict) else None
    )
    if ctype == "allowed_tools":
        raise TranslationError(
            "tool_choice='allowed_tools' is not supported by gufo's native "
            "/v1/responses"
        )
    if ctype == "function":
        name = getattr(choice, "name", None) or (
            choice.get("name") if isinstance(choice, dict) else None
        )
        return {"type": "function", "name": name}
    return choice


def _native_text(request: CreateResponseBody) -> dict[str, Any] | None:
    if not (request.text and request.text.format):
        return None
    fmt = request.text.format
    if fmt.type == "json_object":
        return {"format": {"type": "json_object"}}
    if fmt.type == "json_schema" and getattr(fmt, "schema_", None):
        spec: dict[str, Any] = {
            "type": "json_schema",
            "name": getattr(fmt, "name", None) or "Reply",
            "schema": fmt.schema_,
        }
        if getattr(fmt, "strict", None) is not None:
            spec["strict"] = fmt.strict
        if getattr(fmt, "description", None):
            spec["description"] = fmt.description
        return {"format": spec}
    return None


def gufo_native_payload(
    request: CreateResponseBody,
    history: list[dict[str, Any]],
    stream: bool,
    reasoning_effort: str | None = None,
) -> dict[str, Any]:
    """Assemble the gufo native ``POST /v1/responses`` body.

    Only the field subset gufo accepts is emitted; broker-only fields
    (metadata, include, max_tool_calls, prompt_cache_key, service_tier,
    truncation, top_logprobs) are intentionally dropped. ``store`` is forced
    false because gufo has no server-side conversation storage — the broker
    persists the turn in its own DB.
    """
    payload: dict[str, Any] = {
        "input": input_items_to_gufo_native(request, history),
        "stream": stream,
        "store": False,
    }
    if request.instructions is not None:
        payload["instructions"] = request.instructions
    if request.max_output_tokens is not None:
        payload["max_output_tokens"] = request.max_output_tokens
    if request.temperature is not None:
        payload["temperature"] = request.temperature
    if request.top_p is not None:
        payload["top_p"] = request.top_p
    if request.presence_penalty is not None:
        payload["presence_penalty"] = request.presence_penalty
    if request.frequency_penalty is not None:
        payload["frequency_penalty"] = request.frequency_penalty
    if request.parallel_tool_calls is not None:
        payload["parallel_tool_calls"] = request.parallel_tool_calls
    if reasoning_effort is not None:
        payload["reasoning"] = {"effort": reasoning_effort}
    if request.tools:
        payload["tools"] = _native_tools(request)
        payload["tool_choice"] = _native_tool_choice(request)
    text = _native_text(request)
    if text is not None:
        payload["text"] = text
    return payload


def gufo_usage_to_spec(usage: dict[str, Any]) -> dict[str, Any]:
    """Map gufo's Responses usage object to the broker's spec usage dict.

    gufo already reports ``input_tokens``/``output_tokens``/``total_tokens``
    with ``input_tokens_details.cached_tokens`` and
    ``output_tokens_details.reasoning_tokens`` (openai_chat.cpp:2011-2020).
    """
    input_tokens = int(usage.get("input_tokens") or 0)
    output_tokens = int(usage.get("output_tokens") or 0)
    cached = int((usage.get("input_tokens_details") or {}).get("cached_tokens") or 0)
    reasoning = int(
        (usage.get("output_tokens_details") or {}).get("reasoning_tokens") or 0
    )
    return ev.usage_from_llama(
        input_tokens, output_tokens, reasoning, cached
    ).model_dump(exclude_none=True)


def gufo_buffered_to_output_items(data: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract broker output items from a buffered gufo /v1/responses object.

    gufo builds the same item shapes as the spec (message / reasoning /
    function_call), so items pass through with unset optional keys removed —
    both on the item and inside its content/summary parts — matching the
    conformance harness's "null optionals must be absent" rule. The caller
    coerces the result into typed OutputItem models.
    """
    items: list[dict[str, Any]] = []
    for item in data.get("output") or []:
        if not isinstance(item, dict):
            continue
        cleaned = {k: v for k, v in item.items() if v is not None}
        for key in ("content", "summary"):
            parts = cleaned.get(key)
            if isinstance(parts, list):
                cleaned[key] = [
                    {pk: pv for pk, pv in part.items() if pv is not None}
                    if isinstance(part, dict)
                    else part
                    for part in parts
                ]
        items.append(cleaned)
    return items


def decode_gufo_stream_event(
    state: StreamState,
    event: dict[str, Any],
    gufo_calls: dict[str, dict[str, Any]],
) -> tuple[bool, dict[str, Any] | None, bool]:
    """Feed one gufo native Responses SSE event into a StreamState.

    gufo emits typed OpenResponses events (response.output_text.delta,
    response.reasoning_summary_text.delta, response.output_item.added,
    response.function_call_arguments.delta, and a terminal
    response.completed/incomplete/failed). Deltas drive the same
    StreamState used by the chat path so broker item IDs and sequence
    numbers stay our own.

    Returns ``(terminal, usage_data, incomplete)``. ``terminal`` is True on
    response.completed / response.incomplete (the caller stops reading);
    ``usage_data`` carries gufo's usage + timings for the terminal event.
    response.failed raises RuntimeError.
    """
    etype = event.get("type")
    if etype == "response.output_text.delta":
        delta = event.get("delta") or ""
        if delta:
            state.add_text_delta(delta)
        return False, None, False
    if etype == "response.reasoning_summary_text.delta":
        delta = event.get("delta") or ""
        if delta:
            # add_reasoning_delta opens the item and emits the broker's
            # canonical response.reasoning.delta (what clients like OpenWebUI
            # render as the live thinking stream), with correct
            # item_id/output_index.
            state.add_reasoning_delta(delta)
            # Also emit the gufo-native reasoning_summary_text.delta twin so
            # OpenAI-SDK summary consumers keep working. Emitted after
            # add_reasoning_delta so the reasoning item already exists.
            ev.reasoning_summary_text_delta(
                state.seq,
                state._reasoning_id or "",
                state.output_index,
                0,
                0,
                delta,
            )
        return False, None, False
    if etype == "response.output_item.added":
        item = event.get("item") or {}
        if item.get("type") == "function_call":
            gid = item.get("id") or new_id("fc")
            gufo_calls[gid] = {
                "index": len(gufo_calls),
                "call_id": item.get("call_id"),
                "name": item.get("name"),
            }
        return False, None, False
    if etype == "response.function_call_arguments.delta":
        gid = event.get("item_id")
        meta = gufo_calls.get(gid)
        if meta is None:
            meta = {
                "index": len(gufo_calls),
                "call_id": None,
                "name": event.get("name"),
            }
            gufo_calls[gid] = meta
        state.add_tool_call_delta(
            meta["index"],
            {
                "id": meta["call_id"],
                "function": {
                    "name": meta["name"],
                    "arguments": event.get("delta") or "",
                },
            },
        )
        return False, None, False
    if etype in ("response.completed", "response.incomplete"):
        resp = event.get("response") or {}
        usage = resp.get("usage")
        usage_data = (
            {**usage, "timings": resp.get("timings") or {}}
            if isinstance(usage, dict) and usage
            else None
        )
        return True, usage_data, etype == "response.incomplete"
    if etype == "response.failed":
        err = (event.get("response") or {}).get("error") or {}
        raise RuntimeError(f"gufo response error: {err.get('message') or 'failed'}")
    return False, None, False


# ============================================================================
# llama.cpp -> Responses (streaming)
# ============================================================================


class StreamState:
    """Accumulates llama.cpp stream deltas into Responses items + events.

    Emits, per message item: output_item.added, content_part.added,
    output_text.delta*, output_text.done, content_part.done,
    output_item.done. Function calls follow the same pattern with
    function_call_arguments.* events. Reasoning deltas map to
    response.reasoning_text.* events on a leading reasoning item.
    """

    def __init__(self, seq: ev.SSEmitter, model: str) -> None:
        self.seq = seq
        self.model = model
        self.output_items: list[dict[str, Any]] = []
        self.output_index = 0
        # current open item builders
        self._msg_id: str | None = None
        self._msg_text = ""
        self._msg_open = False
        self._msg_phase: str | None = None
        self._reasoning_id: str | None = None
        self._reasoning_text = ""
        self._reasoning_open = False
        # function calls: llama delta index -> (item_id, name, args)
        self._calls: dict[int, dict[str, Any]] = {}
        self._call_order: list[int] = []
        self.reasoning_tokens = 0
        # Phase label echoed on the assistant output message (from the
        # request's final assistant message phase, if any)
        self.assistant_phase: str | None = None

    # -- message item -----------------------------------------------------

    def _open_message(self) -> str:
        self._msg_id = new_id("msg")
        item = {
            "type": "message",
            "id": self._msg_id,
            "status": "in_progress",
            "role": "assistant",
            "content": [],
        }
        if self.assistant_phase:
            item["phase"] = self.assistant_phase
        ev.output_item_added(self.seq, self.output_index, item)
        part = {"type": "output_text", "text": "", "annotations": []}
        ev.content_part_added(self.seq, self._msg_id, self.output_index, 0, part)
        self._msg_open = True
        return self._msg_id

    def add_text_delta(self, delta: str) -> None:
        if not delta:
            return
        if not self._msg_open:
            # llama.cpp gives no explicit end-of-reasoning marker mid-stream;
            # content starting means thinking is over — close it first.
            self.finish_reasoning()
            self._open_message()
        assert self._msg_id is not None
        self._msg_text += delta
        ev.output_text_delta(self.seq, self._msg_id, self.output_index, 0, delta)

    # -- reasoning item ---------------------------------------------------

    def add_reasoning_delta(self, delta: str) -> None:
        if not delta:
            return
        if not self._reasoning_open:
            self._reasoning_id = new_id("rs")
            item = {
                "type": "reasoning",
                "id": self._reasoning_id,
                "status": "in_progress",
                "summary": [],
                "content": [],
            }
            ev.output_item_added(self.seq, self.output_index, item)
            self._reasoning_open = True
        assert self._reasoning_id is not None
        self._reasoning_text += delta
        self.reasoning_tokens += max(1, len(delta) // 4)
        ev.reasoning_delta(self.seq, self._reasoning_id, self.output_index, 0, delta)

    # -- function calls ---------------------------------------------------

    def add_tool_call_delta(self, index: int, tc: dict[str, Any]) -> None:
        # A tool call after reasoning also ends the thinking item
        self.finish_reasoning()
        call = self._calls.get(index)
        if call is None:
            call = {
                "item_id": new_id("fc"),
                "call_id": tc.get("id") or new_id("call"),
                "name": "",
                "arguments": "",
                "open": False,
            }
            self._calls[index] = call
            self._call_order.append(index)
        fn = tc.get("function") or {}
        name_delta = fn.get("name")
        args_delta = fn.get("arguments")
        if name_delta and not call["name"]:
            call["name"] = name_delta
        if (name_delta or args_delta is not None) and not call["open"]:
            item = {
                "type": "function_call",
                "id": call["item_id"],
                "call_id": call["call_id"],
                "name": call["name"] or "unknown",
                "arguments": "",
                "status": "in_progress",
            }
            ev.output_item_added(self.seq, self.output_index, item)
            call["open"] = True
        if args_delta:
            call["arguments"] += args_delta
            ev.function_call_arguments_delta(
                self.seq, call["item_id"], self.output_index, args_delta
            )

    # -- finish -----------------------------------------------------------

    def finish_item(self, item_type: str) -> None:
        """Close the currently open item(s) of a kind; llama.cpp signals
        finish_reason per choice, we close reasoning before message."""

    def finish_reasoning(self) -> None:
        if not self._reasoning_open:
            return
        assert self._reasoning_id is not None
        if self._reasoning_text:
            ev.reasoning_done(
                self.seq, self._reasoning_id, self.output_index, 0, self._reasoning_text
            )
        item = {
            "type": "reasoning",
            "id": self._reasoning_id,
            "status": "completed",
            "summary": [],
            "content": [{"type": "reasoning_text", "text": self._reasoning_text}],
        }
        ev.output_item_done(self.seq, self.output_index, item)
        self.output_items.append(item)
        self.output_index += 1
        self._reasoning_open = False
        self._reasoning_id = None
        self._reasoning_text = ""
        # reasoning opens before the message; next item gets its own index
        self._pending_call_index_shift = getattr(self, "_pending_call_index_shift", 0)

    def finish_message(self) -> None:
        if not self._msg_open:
            return
        assert self._msg_id is not None
        text = self._msg_text
        ev.output_text_done(self.seq, self._msg_id, self.output_index, 0, text)
        part = {"type": "output_text", "text": text, "annotations": []}
        ev.content_part_done(self.seq, self._msg_id, self.output_index, 0, part)
        item = {
            "type": "message",
            "id": self._msg_id,
            "status": "completed",
            "role": "assistant",
            "content": [{"type": "output_text", "text": text, "annotations": []}],
        }
        if self.assistant_phase:
            item["phase"] = self.assistant_phase
        ev.output_item_done(self.seq, self.output_index, item)
        self.output_items.append(item)
        self.output_index += 1
        self._msg_open = False
        self._msg_id = None
        self._msg_text = ""

    def finish_calls(self) -> None:
        for index in self._call_order:
            call = self._calls[index]
            if not call["open"]:
                continue
            ev.function_call_arguments_done(
                self.seq, call["item_id"], self.output_index, call["arguments"]
            )
            item = {
                "type": "function_call",
                "id": call["item_id"],
                "call_id": call["call_id"],
                "name": call["name"] or "unknown",
                "arguments": call["arguments"] or "{}",
                "status": "completed",
            }
            ev.output_item_done(self.seq, self.output_index, item)
            self.output_items.append(item)
            self.output_index += 1
        self._calls.clear()
        self._call_order.clear()

    def apply_allowed_tools(self, allowed: set[str] | None) -> bool:
        """Suppress function_call items not in allowed set.

        Returns True if all calls were suppressed; suppressed items are
        removed from output and a synthetic assistant message notes it so
        clients (e.g. OpenWebUI handling tools themselves) keep a coherent
        transcript.
        """
        if allowed is None or not self._call_order:
            return False
        disallowed = [
            i
            for i in self._call_order
            if (self._calls[i]["name"] or "unknown") not in allowed
        ]
        if not disallowed:
            return False
        for index in disallowed:
            call = self._calls.pop(index)
            self._call_order.remove(index)
            logger.info(
                "Suppressed disallowed tool call %s (allowed_tools)", call["name"]
            )
        note = "Tool call suppressed: not in allowed_tools for this request."
        self.add_text_delta(note)
        return not self._call_order

    @property
    def has_calls(self) -> bool:
        return bool(self._call_order)


def build_response_resource(
    request: CreateResponseBody,
    response_id: str,
    created_at: int,
    *,
    status: str = "in_progress",
    output: list[dict[str, Any]] | None = None,
    usage: dict[str, Any] | None = None,
    previous_response_id: str | None = None,
) -> ResponseResource:
    """Echo the request into a ResponseResource snapshot."""
    response = ResponseResource(
        id=response_id,
        created_at=created_at,
        status=status,  # type: ignore[arg-type]
        model=request.model,
        previous_response_id=previous_response_id,
        instructions=request.instructions,
        output=output or [],
        tools=[
            # Spec: strict defaults true; description/parameters optional
            {
                "type": "function",
                "name": t.name,
                **({"description": t.description} if t.description else {}),
                **({"parameters": t.parameters} if t.parameters else {}),
                "strict": t.strict if t.strict is not None else True,
            }
            for t in request.tools
        ],  # type: ignore[arg-type]
        tool_choice=request.tool_choice,  # type: ignore[arg-type]
        truncation=request.truncation,
        parallel_tool_calls=request.parallel_tool_calls,
        top_p=request.top_p,
        presence_penalty=request.presence_penalty,
        frequency_penalty=request.frequency_penalty,
        top_logprobs=request.top_logprobs,
        temperature=request.temperature,
        reasoning=request.reasoning,
        usage=usage,  # type: ignore[arg-type]
        max_output_tokens=request.max_output_tokens,
        max_tool_calls=request.max_tool_calls,
        store=request.store,
        background=request.background,
        service_tier="default",
        metadata=request.metadata,
        safety_identifier=request.safety_identifier,
        prompt_cache_key=request.prompt_cache_key,
    )
    if request.text is not None:
        fmt = request.text.format
        if fmt is not None and fmt.type == "json_schema":
            response.text = ResponseText(
                format=ResponseTextFormat(
                    type="json_schema",
                    name=getattr(fmt, "name", None),
                    description=getattr(fmt, "description", None),
                    schema_=getattr(fmt, "schema_", None),
                    strict=getattr(fmt, "strict", None)
                    if getattr(fmt, "strict", None) is not None
                    else True,
                ),
                verbosity=request.text.verbosity,
            )
        elif fmt is not None and fmt.type == "json_object":
            response.text = ResponseText(
                format=ResponseTextFormat(type="json_object"),
                verbosity=request.text.verbosity,
            )
        else:
            # type "text" (or no format): echo only what was set so the
            # serialized shape matches textResponseFormatSchema {type}.
            response.text = ResponseText(
                format=ResponseTextFormat(type="text"),
                verbosity=request.text.verbosity,
            )
    return response
