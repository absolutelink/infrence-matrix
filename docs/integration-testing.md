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
(`admin/backend/app/api/v1/responses.py` + `app/services/sse.py`), and
the **Responses-over-WebSocket transport was removed** (not part of the
OpenResponses spec). All pre-overhaul run history (old leases,
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
| `compact-response` | Compaction Endpoint | ❌ 404 — `POST /v1/responses/compact` not implemented (Phase 7+ scope decision; not a Phase 6 blocker) |
| `compact-missing-model` | Compaction Missing Required Model | ❌ 404 — same missing route |

**Not applicable (transport removed):** all `websocket-*` tests —
`websocket-response`, `websocket-sequential-responses`,
`websocket-continuation`, `websocket-reconnect-store-false-recovery`,
`websocket-previous-response-not-found`,
`websocket-failed-continuation-evicts-cache`,
`websocket-compact-new-chain`. Record as **N/A — WS Responses transport
dropped in the overhaul.** Do not chase.

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
| 2026-10-04 | Phase 7 (chat completions + models + stubs) | 4 | 0 | — | KEY-only regression re-run (`--filter basic-response,streaming-response,system-prompt,multi-turn`) vs local uvicorn admin (`p7-machine`/`p7-model` mock): **4/4 pass**. Proves the `alias_registry` mode change (`responses` → `chat` union registration, required for native chat `acompletion`) did not regress the Phase 6 responses path. Live chat spot-check also green: stream (data-only SSE, admin `chatcmpl-` ids, usage chunk, `[DONE]`), non-stream JSON, `/v1/models`, 501 stub envelope. |
