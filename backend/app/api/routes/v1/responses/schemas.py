"""OpenResponses schemas (spec v2026-04-24).

Pydantic models mirroring the Open Responses specification:
https://www.openresponses.org/specification

Request surface (CreateResponseBody), response surface (ResponseResource),
items, content parts, tools, and shared enums.
"""

import uuid
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field, StringConstraints


def new_id(prefix: str) -> str:
    """Spec-style opaque id: prefix + 32 hex chars."""
    return f"{prefix}_{uuid.uuid4().hex}"


def serialize(model: BaseModel, **kwargs: Any) -> dict[str, Any]:
    """Spec-serialization: omit None-valued optional fields and use aliases.

    The Open Responses conformance schemas mark several fields strictly
    non-nullable (temperature, top_p, parallel_tool_calls, ...) and use
    optional() (key absent, not null) for optional keys like phase,
    logprobs, encrypted_content. The `schema` key on json_schema formats
    is a serialized alias, so dumps must use by_alias.
    """
    return model.model_dump(exclude_none=True, by_alias=True, **kwargs)


# ============================================================================
# Enums
# ============================================================================

MessageRole = Literal["user", "assistant", "system", "developer"]
ItemStatus = Literal["in_progress", "completed", "incomplete"]
FunctionCallStatus = Literal["in_progress", "completed", "incomplete"]
ImageDetail = Literal["low", "high", "auto"]
ReasoningEffort = Literal["none", "low", "medium", "high", "xhigh"]
ReasoningSummary = Literal["concise", "detailed", "auto"]
ServiceTier = Literal["auto", "default", "flex", "priority"]
TruncationMode = Literal["auto", "disabled"]
Verbosity = Literal["low", "medium", "high"]
TextVerbosity = Verbosity
ToolChoiceValue = Literal["none", "auto", "required"]
AssistantPhase = Literal["commentary", "final_answer"]


# ============================================================================
# Content parts
# ============================================================================


class InputTextContent(BaseModel):
    """A text input to the model."""

    type: Literal["input_text"] = "input_text"
    text: str


class InputImageContent(BaseModel):
    """An image input to the model."""

    type: Literal["input_image"] = "input_image"
    image_url: str
    detail: ImageDetail = "auto"


class InputFileContent(BaseModel):
    """A file input to the model."""

    type: Literal["input_file"] = "input_file"
    filename: str | None = None
    file_data: str | None = None
    file_url: str | None = None


class InputVideoContent(BaseModel):
    """A video input to the model."""

    type: Literal["input_video"] = "input_video"
    video_url: str


class OutputTextContent(BaseModel):
    """Model-generated text output."""

    type: Literal["output_text"] = "output_text"
    text: str
    annotations: list[dict[str, Any]] = Field(default_factory=list)
    logprobs: list[dict[str, Any]] | None = None


class RefusalContent(BaseModel):
    """Model refusal output."""

    type: Literal["refusal"] = "refusal"
    refusal: str


class ReasoningTextContent(BaseModel):
    """Raw reasoning text content."""

    type: Literal["reasoning_text"] = "reasoning_text"
    text: str


class SummaryTextContent(BaseModel):
    """Reasoning summary text content."""

    type: Literal["summary_text"] = "summary_text"
    text: str


# Union of user-provided content parts
UserContentPart = Annotated[
    InputTextContent | InputImageContent | InputFileContent | InputVideoContent,
    Field(discriminator="type"),
]
# Union of model-generated content parts
ModelContentPart = Annotated[
    OutputTextContent | RefusalContent,
    Field(discriminator="type"),
]
ReasoningContentPart = Annotated[
    ReasoningTextContent | SummaryTextContent,
    Field(discriminator="type"),
]


# ============================================================================
# Item params (input)
# ============================================================================


class UserMessageItemParam(BaseModel):
    type: Literal["message"] = "message"
    role: MessageRole = "user"
    content: str | list[UserContentPart | OutputTextContentParam | RefusalContentParam]
    id: str | None = None
    status: ItemStatus | None = None
    phase: AssistantPhase | None = None


class OutputTextContentParam(BaseModel):
    type: Literal["output_text"] = "output_text"
    text: str
    annotations: list[dict[str, Any]] = Field(default_factory=list)


class RefusalContentParam(BaseModel):
    type: Literal["refusal"] = "refusal"
    refusal: str


class ReasoningSummaryContentParam(BaseModel):
    type: Literal["summary_text"] = "summary_text"
    text: str


class ReasoningItemParam(BaseModel):
    type: Literal["reasoning"] = "reasoning"
    summary: list[ReasoningSummaryContentParam] = Field(default_factory=list)
    id: str | None = None
    content: list[ReasoningTextContent] | None = None
    encrypted_content: str | None = None
    status: ItemStatus | None = None


class FunctionCallItemParam(BaseModel):
    type: Literal["function_call"] = "function_call"
    call_id: str
    name: str
    arguments: str
    id: str | None = None
    status: FunctionCallStatus | None = None


class FunctionCallOutputItemParam(BaseModel):
    type: Literal["function_call_output"] = "function_call_output"
    call_id: str
    output: str | list[UserContentPart]
    id: str | None = None
    status: FunctionCallStatus | None = None


class ItemReferenceParam(BaseModel):
    type: Literal["item_reference"] = "item_reference"
    id: str


class CompactionSummaryItemParam(BaseModel):
    type: Literal["compaction"] = "compaction"
    id: str
    encrypted_content: str


class CompactRequestBody(BaseModel):
    """POST /v1/responses/compact request body (spec 2026-04-24)."""

    model: str
    input: str | list[InputItem]
    previous_response_id: str | None = None
    instructions: str | None = None
    prompt_cache_key: str | None = None


class CompactResource(BaseModel):
    """Response of the compaction endpoint."""

    id: str = Field(default_factory=lambda: new_id("cmp"))
    object: Literal["response.compaction"] = "response.compaction"
    output: list[CompactionItem] = Field(default_factory=list)
    created_at: int
    usage: Usage


InputItem = Annotated[
    UserMessageItemParam
    | ReasoningItemParam
    | FunctionCallItemParam
    | FunctionCallOutputItemParam
    | ItemReferenceParam
    | CompactionSummaryItemParam,
    Field(discriminator="type"),
]


# ============================================================================
# Tools
# ============================================================================


class FunctionToolParam(BaseModel):
    """Flat function tool definition (Responses API shape)."""

    type: Literal["function"] = "function"
    name: str
    description: str | None = None
    parameters: dict[str, Any] | None = None
    strict: bool | None = None


ToolParam = FunctionToolParam


class SpecificFunctionChoice(BaseModel):
    type: Literal["function"] = "function"
    name: str


class AllowedToolsChoice(BaseModel):
    type: Literal["allowed_tools"] = "allowed_tools"
    tools: list[SpecificFunctionChoice]
    mode: ToolChoiceValue = "auto"


ToolChoice = ToolChoiceValue | SpecificFunctionChoice | AllowedToolsChoice


# ============================================================================
# Request body
# ============================================================================


class ReasoningParam(BaseModel):
    effort: ReasoningEffort | None = None
    summary: ReasoningSummary | None = None


class TextResponseFormat(BaseModel):
    type: Literal["text"] = "text"


class JsonObjectResponseFormat(BaseModel):
    type: Literal["json_object"] = "json_object"


class JsonSchemaResponseFormat(BaseModel):
    type: Literal["json_schema"] = "json_schema"
    name: str | None = None
    description: str | None = None
    schema_: dict[str, Any] | None = Field(default=None, alias="schema")
    strict: bool | None = None

    model_config = {"populate_by_name": True}


TextFormat = Annotated[
    TextResponseFormat | JsonObjectResponseFormat | JsonSchemaResponseFormat,
    Field(discriminator="type"),
]


class TextParam(BaseModel):
    format: TextFormat | None = None
    verbosity: TextVerbosity | None = None


class StreamOptionsParam(BaseModel):
    include_obfuscation: bool = True


class ResponseTextFormat(BaseModel):
    """Resolved text format echoed on the response."""

    type: Literal["text", "json_object", "json_schema"] = "text"
    name: str | None = None
    description: str | None = None
    schema_: dict[str, Any] | None = Field(default=None, alias="schema")
    strict: bool | None = None
    verbosity: TextVerbosity | None = None

    model_config = {"populate_by_name": True}


class ResponseText(BaseModel):
    format: ResponseTextFormat = Field(default_factory=ResponseTextFormat)
    verbosity: TextVerbosity | None = None


class CreateResponseBody(BaseModel):
    """POST /v1/responses request body."""

    model: str
    input: str | list[InputItem]
    previous_response_id: str | None = None
    include: list[
        Literal["reasoning.encrypted_content", "message.output_text.logprobs"]
    ] = Field(default_factory=list)
    tools: list[ToolParam] = Field(default_factory=list)
    tool_choice: ToolChoice = "auto"
    metadata: dict[str, str] = Field(default_factory=dict)
    text: TextParam | None = None
    temperature: float | None = Field(default=None, ge=0, le=2)
    top_p: float | None = Field(default=None, ge=0, le=1)
    presence_penalty: float | None = Field(default=None, ge=-2, le=2)
    frequency_penalty: float | None = Field(default=None, ge=-2, le=2)
    parallel_tool_calls: bool | None = None
    stream: bool = False
    stream_options: StreamOptionsParam | None = None
    background: bool = False
    max_output_tokens: int | None = None
    max_tool_calls: int | None = None
    reasoning: ReasoningParam | None = None
    safety_identifier: str | None = None
    prompt_cache_key: str | None = None
    truncation: TruncationMode = "auto"
    instructions: str | None = None
    store: bool = True
    service_tier: ServiceTier = "auto"
    top_logprobs: int | None = None


# ============================================================================
# Response objects
# ============================================================================


class Error(BaseModel):
    code: str
    message: str


class IncompleteDetails(BaseModel):
    reason: str


class Usage(BaseModel):
    input_tokens: int
    output_tokens: int
    total_tokens: int
    input_tokens_details: dict[str, int] = Field(
        default_factory=lambda: {"cached_tokens": 0}
    )
    output_tokens_details: dict[str, int] = Field(
        default_factory=lambda: {"reasoning_tokens": 0}
    )


# Output items (ItemField union)


class AssistantMessageItem(BaseModel):
    type: Literal["message"] = "message"
    id: str = Field(default_factory=lambda: new_id("msg"))
    role: Literal["assistant"] = "assistant"
    status: ItemStatus = "completed"
    content: list[OutputTextContent | RefusalContent] = Field(default_factory=list)
    phase: AssistantPhase | None = None


class ReasoningItem(BaseModel):
    type: Literal["reasoning"] = "reasoning"
    id: str = Field(default_factory=lambda: new_id("rs"))
    status: ItemStatus = "completed"
    summary: list[SummaryTextContent] = Field(default_factory=list)
    content: list[ReasoningTextContent] | None = None
    encrypted_content: str | None = None


class FunctionCallItem(BaseModel):
    type: Literal["function_call"] = "function_call"
    id: str = Field(default_factory=lambda: new_id("fc"))
    call_id: str = Field(default_factory=lambda: new_id("call"))
    name: str
    arguments: str = "{}"
    status: FunctionCallStatus = "completed"


class CompactionItem(BaseModel):
    type: Literal["compaction"] = "compaction"
    id: str = Field(default_factory=lambda: new_id("cmp"))
    encrypted_content: str
    created_by: str | None = None
    status: ItemStatus = "completed"


OutputItem = Annotated[
    AssistantMessageItem | ReasoningItem | FunctionCallItem | CompactionItem,
    Field(discriminator="type"),
]


class FunctionTool(BaseModel):
    """Function tool echoed on the response (flat spec shape)."""

    type: Literal["function"] = "function"
    name: str
    description: str | None = None
    parameters: dict[str, Any] | None = None
    strict: bool | None = None


class ResponseFunctionToolChoice(BaseModel):
    type: Literal["function"] = "function"
    name: str | None = None


class ResponseAllowedToolChoice(BaseModel):
    type: Literal["allowed_tools"] = "allowed_tools"
    tools: list[ResponseFunctionToolChoice]
    mode: ToolChoiceValue = "auto"


ResponseToolChoice = (
    ToolChoiceValue | ResponseFunctionToolChoice | ResponseAllowedToolChoice
)


class ResponseResource(BaseModel):
    """Full response resource."""

    id: str = Field(default_factory=lambda: new_id("resp"))
    object: Literal["response"] = "response"
    created_at: int
    completed_at: int | None = None
    status: Literal["queued", "in_progress", "completed", "incomplete", "failed"]
    incomplete_details: IncompleteDetails | None = None
    model: str
    previous_response_id: str | None = None
    instructions: str | None = None
    output: list[OutputItem] = Field(default_factory=list)
    error: Error | None = None
    tools: list[FunctionTool] = Field(default_factory=list)
    tool_choice: ResponseToolChoice = "auto"
    truncation: TruncationMode = "auto"
    parallel_tool_calls: bool | None = None
    text: ResponseText = Field(default_factory=ResponseText)
    top_p: float | None = None
    presence_penalty: float | None = None
    frequency_penalty: float | None = None
    top_logprobs: int | None = None
    temperature: float | None = None
    reasoning: ReasoningParam | None = None
    usage: Usage | None = None
    max_output_tokens: int | None = None
    max_tool_calls: int | None = None
    store: bool = True
    background: bool = False
    service_tier: ServiceTier = "default"
    metadata: dict[str, str] = Field(default_factory=dict)
    safety_identifier: str | None = None
    prompt_cache_key: str | None = None
    # Input items echoed for chaining/retrieval (internal convenience)
    input: list[InputItem] = Field(default_factory=list)


# ============================================================================
# Spec serialization (conformance schemas)
# ============================================================================


# Nullable ResponseResource keys the conformance schemas require PRESENT
# (null-allowed): their absence (undefined) fails validation.
_RESPONSE_NULLABLE_FIELDS: tuple[str, ...] = (
    "completed_at",
    "incomplete_details",
    "previous_response_id",
    "instructions",
    "error",
    "reasoning",
    "usage",
    "max_output_tokens",
    "max_tool_calls",
    "safety_identifier",
    "prompt_cache_key",
)

# Echo params the conformance schemas mark strictly non-nullable — both null
# and absent fail. Defaults mirror the OpenAI Responses API echo.
_RESPONSE_ECHO_DEFAULTS: dict[str, Any] = {
    "temperature": 1.0,
    "top_p": 1.0,
    "presence_penalty": 0.0,
    "frequency_penalty": 0.0,
    "top_logprobs": 0,
    "parallel_tool_calls": True,
}


def serialize_spec(model: BaseModel, **kwargs: Any) -> dict[str, Any]:
    """Serialize a ResponseResource/CompactResource per the conformance schemas.

    Extends serialize() (None-stripped, aliased) with the shape the harness's
    Zod schemas require:
    - Nullable top-level response keys re-added as explicit null (key must be
      present); echo params defaulted to concrete values (null/absent fail).
    - Tool echoes get description/parameters back as null (required-but-
      nullable there).
    - reasoning echoes get effort/summary back as null.
    Output *items* keep serialize()'s key-absent behavior for their optional
    fields (phase, logprobs, encrypted_content, content) — null fails the
    item schemas' optional() semantics.
    """
    data = serialize(model, **kwargs)
    if isinstance(model, ResponseResource):
        for key in _RESPONSE_NULLABLE_FIELDS:
            data.setdefault(key, None)
        for key, value in _RESPONSE_ECHO_DEFAULTS.items():
            data.setdefault(key, value)
        for tool in data.get("tools") or []:
            if isinstance(tool, dict):
                tool.setdefault("description", None)
                tool.setdefault("parameters", None)
        reasoning = data.get("reasoning")
        if isinstance(reasoning, dict):
            reasoning.setdefault("effort", None)
            reasoning.setdefault("summary", None)
    return data


# ============================================================================
# Error envelope (JSON error responses)
# ============================================================================


class ErrorBody(BaseModel):
    message: str
    type: Literal[
        "server_error",
        "invalid_request",
        "not_found",
        "model_error",
        "too_many_requests",
    ]
    param: str | None = None
    code: str | None = None


class ErrorResponse(BaseModel):
    error: ErrorBody


# ============================================================================
# Helpers
# ============================================================================

MetadataKey = Annotated[str, StringConstraints(max_length=64)]
MetadataValue = Annotated[str, StringConstraints(max_length=512)]
