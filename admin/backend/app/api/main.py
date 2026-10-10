from fastapi import APIRouter

from app.api.admin.agents import router as admin_agents_router
from app.api.admin.definitions import router as admin_definitions_router
from app.api.admin.health import router as admin_health_router
from app.api.admin.huggingface import router as admin_huggingface_router
from app.api.admin.instances import router as admin_instances_router
from app.api.admin.machines import router as admin_machines_router
from app.api.admin.provider_types import router as admin_provider_types_router
from app.api.admin.providers import router as admin_providers_router
from app.api.admin.responses import router as admin_responses_router
from app.api.admin.voices import router as admin_voices_router
from app.api.v1.audio_speech import router as v1_audio_speech_router
from app.api.v1.audio_transcriptions import router as v1_audio_transcriptions_router
from app.api.v1.audio_transcriptions_ws import (
    router as v1_audio_transcriptions_ws_router,
)
from app.api.v1.audio_voices import router as v1_audio_voices_router
from app.api.v1.chat_completions import router as v1_chat_router
from app.api.v1.embeddings import router as v1_embeddings_router
from app.api.v1.models import router as v1_models_router
from app.api.v1.responses import router as v1_responses_router
from app.api.v1.responses_compact import router as v1_compact_router
from app.api.v1.responses_ws import router as v1_responses_ws_router
from app.api.v1.stubs import router as v1_stubs_router
from app.api.ws import router as provider_ws_router

api_router = APIRouter()

api_router.include_router(admin_health_router)
api_router.include_router(admin_providers_router)
# Phase 9 operator CRUD + instance actions (trusted LAN, like the rest of
# /admin/api).
api_router.include_router(admin_machines_router)
api_router.include_router(admin_definitions_router)
api_router.include_router(admin_instances_router)
# Phase 12: ProviderType registry reads + schema-consensus overrides.
api_router.include_router(admin_provider_types_router)
# Phase 12: HF proxy for the hf-file picker widget (trusted LAN, read-only).
api_router.include_router(admin_huggingface_router)
# Phase 10 UI reads: response log, usage stats, dashboard overview.
api_router.include_router(admin_responses_router)
# Phase 16: provider-agent reads (containers bound to machine+type).
api_router.include_router(admin_agents_router)
# Phase 24: saved-voice enrollment (UI catalog; proxies the agent HTTP surface).
api_router.include_router(admin_voices_router)
# Public OpenAI-compatible inference API lives at /v1 (NOT under /admin/api).
api_router.include_router(v1_responses_router)
# Phase 21: Responses-over-WebSocket transport (same /v1/responses path).
api_router.include_router(v1_responses_ws_router)
# Phase 20: response compaction (non-stream only).
api_router.include_router(v1_compact_router)
api_router.include_router(v1_chat_router)
api_router.include_router(v1_embeddings_router)
# Phase 24: audio (tts/asr) — direct httpx passthrough, litellm bypassed.
api_router.include_router(v1_audio_speech_router)
api_router.include_router(v1_audio_transcriptions_router)
# Phase 24 S4: live-ASR WebSocket relay (WS /v1/audio/transcriptions/stream).
api_router.include_router(v1_audio_transcriptions_ws_router)
api_router.include_router(v1_audio_voices_router)
api_router.include_router(v1_models_router)
# 501 stubs for dropped endpoints (registered last; explicit paths only).
api_router.include_router(v1_stubs_router)
# Provider WebSocket lives at the app root (/provider/ws), not under /admin/api.
api_router.include_router(provider_ws_router)
