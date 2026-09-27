"""Token usage statistics for the UI status bar."""

from fastapi import APIRouter
from pydantic import BaseModel, Field

from app.services.token_stats import token_stats_snapshot

router = APIRouter(prefix="/stats", tags=["stats"])


class LiveRates(BaseModel):
    window_seconds: int
    decode_tokens_per_second: float | None = None
    prefill_tokens_per_second: float | None = None


class TokenTotals(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class GlobalTokenStats(BaseModel):
    live: LiveRates
    last_24h: TokenTotals
    last_7d: TokenTotals
    last_30d: TokenTotals


class ServerTokenStats(BaseModel):
    id: str
    alias: str
    decode_tokens_per_second: float | None = None
    prefill_tokens_per_second: float | None = None
    last_7d: TokenTotals
    last_30d: TokenTotals


class TokenStatsResponse(BaseModel):
    global_: GlobalTokenStats = Field(alias="global")
    servers: list[ServerTokenStats]


@router.get("/tokens", response_model=TokenStatsResponse)
async def get_token_stats() -> TokenStatsResponse:
    """Return live and historical token usage, per server and globally."""
    return TokenStatsResponse.model_validate(await token_stats_snapshot())
