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


def _content_to_text(content: Any) -> str:
    """Flatten content (string or parts) to plain text."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    texts = [_part_text(p) for p in content]
    if any(t is None for t in texts):
        raise TranslationError(
            "Non-text input content (image/file/video) is not supported by local "
            "GGUF models without vision; use input_text parts."
        )
    return "".join(t or "" for t in texts)


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
) -> list[dict[str, str]]:
    """Build llama.cpp chat messages from history + this request's input.

    history_items are prior stored items (previous_response_id chaining):
    previous.input + previous.output were already merged by the caller into
    this list in semantic order.
    """
    messages: list[dict[str, str]] = []
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
            raise TranslationError(
                "compaction items are not supported in this implementation"
            )
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
            text = _content_to_text(output)
            messages.append(
                {"role": "user", "content": f"[Tool result ({call_id})]: {text}"}
            )
        elif itype == "message":
            role = item.get("role") if isinstance(item, dict) else item.role
            content = item.get("content") if isinstance(item, dict) else item.content
            messages.append(
                {"role": role or "user", "content": _content_to_text(content)}
            )
        else:
            raise TranslationError(f"Unknown input item type: {itype}")
    return messages


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
        self._reasoning_id: str | None = None
        self._reasoning_text = ""
        self._reasoning_open = False
        # function calls: llama delta index -> (item_id, name, args)
        self._calls: dict[int, dict[str, Any]] = {}
        self._call_order: list[int] = []
        self.reasoning_tokens = 0

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
        ev.output_item_added(self.seq, self.output_index, item)
        part = {"type": "output_text", "text": "", "annotations": []}
        ev.content_part_added(self.seq, self._msg_id, self.output_index, 0, part)
        self._msg_open = True
        return self._msg_id

    def add_text_delta(self, delta: str) -> None:
        if not delta:
            return
        if not self._msg_open:
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
        tools=[t.model_dump(exclude_none=True) for t in request.tools],  # type: ignore[arg-type]
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
        response.text = ResponseText(
            format=ResponseTextFormat(
                type=fmt.type,
                name=getattr(fmt, "name", None),
                description=getattr(fmt, "description", None),
                schema_=getattr(fmt, "schema_", None),
                strict=getattr(fmt, "strict", None),
            ),
            verbosity=request.text.verbosity,
        )
    return response
