"""Validated, typed options supported by the server settings UI."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ServerOptions(BaseModel):
    """Core llama-server options exposed by the UI.

    None means that llama.cpp should use its own default and the agent omits
    the corresponding command-line flag.
    """

    model_config = ConfigDict(extra="forbid")

    threads: int | None = Field(default=None, ge=-1)
    threads_batch: int | None = Field(default=None, ge=-1)
    batch_size: int | None = Field(default=None, ge=1)
    ubatch_size: int | None = Field(default=None, ge=1)
    keep: int | None = Field(default=None, ge=-1)
    predict: int | None = Field(default=None, ge=-1)
    swa_full: bool | None = None
    cache_type_k: (
        Literal[
            "f32",
            "f16",
            "bf16",
            "q8_0",
            "q4_0",
            "q4_1",
            "iq4_nl",
            "q5_0",
            "q5_1",
            "turbo4",
        ]
        | None
    ) = None
    cache_type_v: (
        Literal[
            "f32",
            "f16",
            "bf16",
            "q8_0",
            "q4_0",
            "q4_1",
            "iq4_nl",
            "q5_0",
            "q5_1",
            "turbo4",
        ]
        | None
    ) = None
    kv_offload: bool | None = None
    cache_prompt: bool | None = None
    cache_reuse: int | None = Field(default=None, ge=0)
    ctx_checkpoints: int | None = Field(default=None, ge=0)
    checkpoint_every: int | None = Field(default=None, ge=0)
    cache_ram: int | None = Field(default=None, ge=0)
    slot_save_path: str | None = None
    gpu_layers: int | str | None = Field(default=None, pattern=r"^(auto|all)$")
    device: str | None = None
    split_mode: Literal["none", "layer", "row", "tensor"] | None = None
    tensor_split: str | None = None
    main_gpu: int | None = Field(default=None, ge=0)
    fit: Literal["on", "off"] | None = None
    fit_target: str | None = None
    fit_ctx: int | None = Field(default=None, ge=1)
    temperature: float | None = Field(default=None, ge=0)
    top_k: int | None = Field(default=None, ge=0)
    top_p: float | None = Field(default=None, ge=0, le=1)
    min_p: float | None = Field(default=None, ge=0, le=1)
    repeat_penalty: float | None = Field(default=None, ge=0)
    presence_penalty: float | None = None
    frequency_penalty: float | None = None
    seed: int | None = None
    parallel: int | None = Field(default=None, ge=1)
    reasoning: Literal["on", "off", "auto"] | None = None
    reasoning_budget: int | None = Field(default=None, ge=0)
    spec_draft_p_min: float | None = Field(default=None, ge=0, le=1)
    strict_mtp_qwen: bool | None = None
    kv_unified: bool | None = None
    no_mmap: bool | None = None
    no_cache_idle_slots: bool | None = None
    cont_batching: bool | None = None
    warmup: bool | None = None
    context_shift: bool | None = None
    jinja: bool | None = None

    @model_validator(mode="after")
    def validate_dependencies(self) -> ServerOptions:
        if self.cache_reuse is not None and self.cache_prompt is False:
            raise ValueError("cache_reuse requires cache_prompt to be enabled")
        if self.fit_target is not None and self.fit != "on":
            raise ValueError("fit_target requires fit to be enabled")
        if self.fit_ctx is not None and self.fit != "on":
            raise ValueError("fit_ctx requires fit to be enabled")
        if self.tensor_split is not None and self.split_mode not in ("row", "tensor"):
            raise ValueError("tensor_split requires row or tensor split mode")
        return self


def validate_server_options(options: dict | ServerOptions) -> dict:
    """Validate and normalize JSON options before persistence or dispatch."""
    if isinstance(options, ServerOptions):
        return options.model_dump(exclude_none=True)
    return ServerOptions.model_validate(options).model_dump(exclude_none=True)
