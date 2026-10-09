# Integration Testing — OpenResponses Compliance Suite

Test tool: `/workspaces/openresponses` (`bun run test:compliance`).
Deployment logs stream to `/tmp/lfai-matrix.log` (wipe before each deploy
cycle). Skill: `.agents/skills/integration-testing/SKILL.md`.

```bash
# full run (deployment)
bun run test:compliance --base-url https://matrix.thelink.family/v1 \
  --api-key none --model <alias>

# single test
bun run test:compliance --base-url https://matrix.thelink.family/v1 \
  --api-key none --model <alias> --filter <test-id>

# fast local run against the mock provider
docker compose up -d   # admin + postgres + redis + provider-mock (see development.md § Seeding)
bun run test:compliance --base-url http://localhost:8000/v1 \
  --api-key none --model mock-model
```

## Overhaul reset

This tracker was **reset for the `litellm-architecture-overhaul`**
(Phase 6 in progress). The Responses API is now litellm-driven
(`admin/backend/app/api/v1/responses.py` + `app/services/sse.py`). The
**Responses-over-WebSocket transport was removed** during the overhaul
(then not part of the spec) and **re-added in Phase 21** as a WS upgrade on
`/v1/responses` (`admin/backend/app/api/v1/responses_ws.py`) once the spec
added the transport — the 7 `websocket-*` tests are now applicable again
(see the WS table below). All pre-overhaul run history (old leases,
`server_instances`, `inference_slot_protocol`, WS-transport clusters)
lives in git history and is intentionally not carried forward.

**Applicable tests (HTTP/SSE only):** run 2026-10-04 against a local
uvicorn admin + mock provider (branch `litellm-architecture-overhaul`,
commit `8d2d9b8` + this phase's fixes):

| Test ID | Name | Status |
|---|---|---|
| `basic-response` | Basic Text Response | ✅ pass |
| `assistant-phase` | Assistant Message Phase | ✅ pass |
| `response-output-phase-schema` | Response Output Phase Schema | ✅ pass (local schema fixture) |
| `streaming-response` | Streaming Response | ✅ pass |
| `system-prompt` | System Prompt | ✅ pass |
| `tool-calling` | Tool Calling | ✅ pass (mock emits a canned `function_call` when `tools` present) |
| `image-input` | Image Input | ✅ pass (mock echoes text; validates acceptance + schema, not vision) |
| `multi-turn` | Multi-turn Conversation | ✅ pass |
| `compact-response` | Compaction Endpoint | ✅ pass (Phase 20 `POST /v1/responses/compact`; local mock run 2026-10-09 — `prompt_cache_key` forwarded through litellm `aresponses` `**kwargs` and accepted) |
| `compact-missing-model` | Compaction Missing Required Model | ✅ pass (route returns 400 on missing `model`; local mock run 2026-10-09) |

**Applicable tests (Responses-over-WebSocket, Phase 21):** run 2026-10-09
against a local uvicorn admin + mock provider (branch
`litellm-architecture-overhaul`, commit `d3f721b` — WS transport on
`/v1/responses`). All 7 pass on the first run with **no transport fixes
required**; the admin log shows clean WS accept/open/close per turn with
no errors.

| Test ID | Name | Status |
|---|---|---|
| `websocket-response` | WebSocket Response | ✅ pass (local mock run 2026-10-09) |
| `websocket-sequential-responses` | WebSocket Sequential Responses | ✅ pass (local mock run 2026-10-09) |
| `websocket-continuation` | WebSocket Continuation | ✅ pass (store:false continuation via the per-connection `_TurnCache`; local mock run 2026-10-09) |
| `websocket-reconnect-store-false-recovery` | WebSocket Store False Reconnect Recovery | ✅ pass (store:false id not resurrected across sockets → `previous_response_not_found`, then clean recovery; local mock run 2026-10-09) |
| `websocket-previous-response-not-found` | WebSocket Missing Previous Response | ✅ pass (`previous_response_not_found` error envelope, socket stays open; local mock run 2026-10-09) |
| `websocket-failed-continuation-evicts-cache` | WebSocket Failed Continuation Evicts Cache | ✅ pass (WS-only `function_call_output.call_id` validation fails the turn + evicts the referenced id; local mock run 2026-10-09) |
| `websocket-compact-new-chain` | WebSocket Compact New Chain | ✅ pass (`/responses/compact` output fed back as WS input, no `previous_response_id`; local mock run 2026-10-09) |

### Run 2026-10-09 — deployment (`matrix.thelink.family`, alias `rocinante-tiny`, llama-cpp provider)

1 passed / 16 failed (7 of the failures are N/A `websocket-*`). The
whole HTTP+SSE cluster (basic, system-prompt, assistant-phase, streaming,
tool-calling, image-input, multi-turn) failed identically on **null /
missing terminal response fields** — llama.cpp's `/v1/responses`
terminal response omits `completed_at`, `background`, `service_tier`,
`presence_penalty`, `frequency_penalty`, `top_logprobs`,
`max_tool_calls`, `safety_identifier`, `prompt_cache_key`, and sends
`tools`/`tool_choice`/`truncation`/`text`/`top_p`/`temperature`/`store`/
`parallel_tool_calls` as `null`, plus `usage.output_tokens_details:
null`. The mock passed because it emits a spec-complete skeleton.

**Fix (admin-side, pending deploy):** `_normalize_response()` in
`admin/backend/app/api/v1/responses.py` fills spec defaults on the
terminal response in both transports (non-stream JSON +
`response.completed`/`response.incomplete` frames), mirroring the mock's
`_base_response` defaults. Covered by
`tests/test_v1_responses.py::test_non_stream_normalizes_sparse_provider_response`
and `::test_stream_terminal_frame_is_normalized`.

`compact-response` / `compact-missing-model` still 404 (no
`/responses/compact` route — open scope decision).

**Rerun after first deploy (same day): 6 passed / 11 failed** (7 WS N/A +
2 compaction + 2 real). Remaining real failures:

- `streaming-response` — the suite validates **every** lifecycle/delta
  event, not just the terminal frame: llama.cpp's `response.created`/
  `response.in_progress` are bare `{id, object, status}` skeletons, and
  its `output_item.*`/`content_part.*`/`output_text.*` events omit
  `output_index`/`content_index`/`item_id`. Fixed admin-side: lifecycle
  skeletons now go through `_normalize_response` (status
  `in_progress`/`queued`, `completed_at` null), and `SSEEmitter` tracks
  item/content position and fills missing required indices (never
  overwriting provider values).
- `tool-calling` — echoed request tools lacked the required-nullable
  `strict`/`description`/`parameters` keys; `_normalize_response` now
  fills them per entry (on copies — the list may be the client's own).

New tests: `test_stream_lifecycle_frames_are_normalized`,
`test_non_stream_echoed_tools_get_required_keys`,
`test_position_fields_filled_when_provider_omits_them`,
`test_position_fields_never_overwrite_provider_values`.

**Rerun after second deploy: 7 passed / 10 failed** (7 WS N/A + 2
compaction + 1 real). Tool-calling went green; `streaming-response` now
fails on a single event: llama.cpp's `response.content_part.added` part
lacks the required `annotations` array (its `.done` carries it). Fixed:
`SSEEmitter` fills `annotations: []` on output_text parts when absent
(fill-only, on a copy).

**WebSocket tests (were N/A at that deployment run):** all `websocket-*`
tests — `websocket-response`, `websocket-sequential-responses`,
`websocket-continuation`, `websocket-reconnect-store-false-recovery`,
`websocket-previous-response-not-found`,
`websocket-failed-continuation-evicts-cache`,
`websocket-compact-new-chain` — were recorded **N/A** in that run because
the WS Responses transport had been dropped in the overhaul. **Phase 21
re-added the transport** (`admin/backend/app/api/v1/responses_ws.py`,
commit `d3f721b`); all 7 now pass against the local mock (see the WS
applicable-tests table above). The Phase 21 S4 deployment run against
`rocinante-tiny` (llama-cpp) is also **17/17 green** (2026-10-09).

All six KEY tests (basic/streaming/system/assistant-phase/output-phase/
multi-turn) pass, plus tool-calling and image-input. The two compaction
failures are the absent `/responses/compact` route — the endpoint is not
in the overhaul's §7 scope; decide Phase 7/8 whether to implement or
formally drop it from the applicable set.

### Fixes this run required (were failing, now green)

- **Spec default `stream=false`**: the admin route defaulted to
  streaming; clients that omit `stream` got SSE instead of JSON.
  (`admin/backend/app/api/v1/responses.py`)
- **Provider non-stream path**: the provider `/v1/responses` always
  SSE'd; it now drains the driver stream and returns the terminal
  response as JSON when `stream` is false.
  (`provider/lib/provider_lib/app_factory.py`)
- **Spec-complete response objects**: the mock's terminal response lacked
  required fields (`created_at`, `tools`, `temperature`, usage details,
  ...) and `response.output_item.added` items lacked `status`.
  (`provider/mock/provider_mock/backend.py`)
- **Null-default injection**: litellm's `exclude_none=False` dump re-adds
  `phase: null` / `logprobs: null` which the Zod schema rejects
  (optional but non-nullable); the admin strips them from output items.
  (`admin/backend/app/api/v1/responses.py` `_clean_output_items`)
- **Tool calls**: the mock now emits a canned `function_call` turn when
  the request carries `tools`.
  (`provider/mock/provider_mock/backend.py`)
- **Presence keepalive**: the provider client now sends `ping` frames
  every 20s (`AdminClient.PING_INTERVAL_SECONDS`); previously nothing
  did, so idle connections were swept `disconnected` after 60s.

## Where failures live now

| Symptom | Likely location |
| --- | --- |
| Stream framing / id / `sequence_number` | `admin/backend/app/services/sse.py` |
| Missing native streaming (fake-stream APIError) | `admin/backend/app/services/alias_registry.py` (ensure_registered) |
| Chain reconstruction / persistence | `admin/backend/app/api/v1/responses.py` (`build_litellm_input`, `persist_turn`) |
| Admission / boot / VRAM | `admin/backend/app/services/scheduler.py` |
| Provider output not spec-clean | `provider/lib` + `provider/<type>` translation |

Fidelity ground truth: `spike/litellm-fidelity/FINDINGS.md`.

## History

| Date | Commit | Passed | Failed | N/A | Notes |
|---|---|---|---|---|---|
| (overhaul reset) | — | — | — | 7 WS | Tracker reset for litellm overhaul; awaiting Phase 6 conformance run |
| 2026-10-04 | 8d2d9b8 + docs-phase fixes | 8 | 2 | 7 WS | First post-overhaul run vs local uvicorn admin + mock. All 6 KEY tests pass; failures are `compact-response` / `compact-missing-model` (no `/responses/compact` route). Fixes listed above landed in this run. |
| 2026-10-09 | `_normalize_response` fix (pre-deploy) | 1 | 9 | 7 WS | Full run vs deployment, `rocinante-tiny` (llama-cpp). All 7 HTTP+SSE tests failed on cluster 1 (null/missing terminal response fields from llama.cpp) + 2 compaction 404s. Fix landed admin-side; re-run after deploy. |
| 2026-10-09 | terminal-normalize deploy (pre stream-frame fix) | 6 | 4 | 7 WS | Rerun after first deploy: cluster 1 cleared (basic/system/assistant/multi-turn/image now pass). Remaining: `streaming-response` (non-terminal lifecycle frames + delta events lack required fields — fixed via lifecycle normalization + emitter position tracking), `tool-calling` (`tools[].strict` — fixed), 2 compaction 404s. |
| 2026-10-09 | lifecycle/emitter + annotations deploys | 8 | 2 | 7 WS | **All applicable HTTP+SSE tests green** (`streaming-response` cleared by the lifecycle-frame normalization + emitter position/annotations fills). Only the 2 compaction 404s remain — Phase 20 (`/v1/responses/compact`) planned; see IMPLEMENTATION_STATUS.md. |
| 2026-10-09 | Phase 20 deploy (`ba2a799`) | 10 | 0 | 7 WS | **FULL SUITE GREEN on deployment** (`rocinante-tiny`, llama-cpp): all 10 applicable HTTP+SSE tests pass, incl. both compaction tests via the new `/v1/responses/compact`. The 7 `websocket-*` remain N/A (transport dropped). Phase 6 conformance gate: closed. |
| 2026-10-04 | Phase 7 (chat completions + models + stubs) | 4 | 0 | — | KEY-only regression re-run (`--filter basic-response,streaming-response,system-prompt,multi-turn`) vs local uvicorn admin (`p7-machine`/`p7-model` mock): **4/4 pass**. Proves the `alias_registry` mode change (`responses` → `chat` union registration, required for native chat `acompletion`) did not regress the Phase 6 responses path. Live chat spot-check also green: stream (data-only SSE, admin `chatcmpl-` ids, usage chunk, `[DONE]`), non-stream JSON, `/v1/models`, 501 stub envelope. |
| 2026-10-09 | Phase 20 S2 (`b891664` compact route) | 10 | 0 | 7 WS | **Phase 20 verification vs local uvicorn admin + mock (`mock-model`): all applicable HTTP+SSE tests green, including both compaction tests.** `compact-response` (with `prompt_cache_key: "openresponses-compact-test"`) and `compact-missing-model` (400 on missing `model`) pass. The known `prompt_cache_key` risk did NOT materialize — litellm 1.103.2 `aresponses` accepts it via `**kwargs` and the mock tolerates it, so no route fix was needed. Full run: 10 passed / 0 failed / 7 N/A (`websocket-*`, transport dropped). No admin code or test changes. |
| 2026-10-09 | Phase 21 S3 (`d3f721b` WS transport) | 17 | 0 | 0 | **FULL SUITE GREEN incl. WebSocket vs local uvicorn admin + mock (`mock-model`): 17 passed / 0 failed / 0 N/A.** The 7 `websocket-*` tests (re-enabled by the Phase 21 WS transport on `/v1/responses`) all pass on the first run — **no transport fixes required**. Verified: single WS turn, sequential turns on one socket, store:false continuation via the per-connection `_TurnCache`, cross-socket store:false miss → `previous_response_not_found` + clean recovery, WS-only `function_call_output.call_id` validation + evict-on-failed-continuation, and `/responses/compact` output fed back as WS input. Admin log shows clean WS accept/open/close per turn, no errors. No admin/provider code or test changes. |
| 2026-10-09 | Phase 21 S4 deploy (`dea38bc`) | 17 | 0 | 0 | **FULL SUITE GREEN ON DEPLOYMENT** (`rocinante-tiny`, llama-cpp, via `wss://` through Traefik): 17/17 including all 7 `websocket-*`. Admin redeployed from `:develop`; provider agents re-registered cleanly (both report version `dev` — no provider recreate needed). Phase 21 conformance gate: closed. |
| 2026-10-09 | Phase 21 post-ship CI fix (`8750d35`) | 17 | 0 | 0 | The close-out push's CI run caught a WS race (client disconnect after the terminal frame cancelled the committed turn, dropping the store=true record — CI-timing only, local runs won). Fixed via `terminal_sent` guard (committed turns are awaited, never cancelled) + deterministic regression test; admin suite 496 green, WS 10x stable; CI green; admin redeployed and **17/17 re-verified on the deployment**. |
| 2026-10-09 | Phase 21 addendum (`615b1ae`, halogen-flash redeployed) | 16 | 1 | 0 | **`rocinante` (halogen-flash)** full suite: all reasoning-event/item conformance failures fixed (A-S1 rename + A-S4 item sanitize) — 16/17. Sole failure `image-input` → 502: the Flash backend is not vision-capable (capability gap, not a protocol bug; use `rocinante-tiny` for image tests). |
| 2026-10-09 | Phase 21 addendum (`615b1ae`, llama-cpp) | 17 | 0 | 0 | **`rocinante-tiny` full suite 17/17 warm.** Note: WS turns against a COLD (stopped) backend exceed the compliance client's 30s per-turn timeout during model load — cold-boot WS timeouts are expected; warm the alias first (HTTP request or `instances` boot) before a WS compliance run. |
| 2026-10-09 | Phase 22 (`75a0f10`, llm-comply openai-chat) | 8+1sk | 0 | 1 | Second toolchain: `llm-comply --format openai-chat` (vendored OpenAI spec) — non-stream chat bodies normalized (`_normalize_chat_response`). **8/9 + 1 name-heuristic skip on both `rocinante-tiny` and `rocinante`** on the deployment; bun open-responses regression on tiny still 17/17. First rocinante attempt hit a crashed Flash engine (environment, fixed by agent restart). |
