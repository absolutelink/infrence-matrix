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

**Applicable tests (HTTP/SSE only):**

| Test ID | Name | Status |
|---|---|---|
| `basic-response` | Basic Text Response | ⬜ not yet run (post-overhaul) |
| `assistant-phase` | Assistant Message Phase | ⬜ |
| `response-output-phase-schema` | Response Output Phase Schema | ⬜ |
| `streaming-response` | Streaming Response | ⬜ |
| `system-prompt` | System Prompt | ⬜ |
| `tool-calling` | Tool Calling | ⬜ |
| `image-input` | Image Input | ⬜ (needs a vision-capable provider; mock may not cover) |
| `multi-turn` | Multi-turn Conversation | ⬜ |
| `compact-response` | Compaction Endpoint | ⬜ (compaction route status TBD — Phase 6/7) |
| `compact-missing-model` | Compaction Missing Required Model | ⬜ |

**Not applicable (transport removed):** all `websocket-*` tests —
`websocket-response`, `websocket-sequential-responses`,
`websocket-continuation`, `websocket-reconnect-store-false-recovery`,
`websocket-previous-response-not-found`,
`websocket-failed-continuation-evicts-cache`,
`websocket-compact-new-chain`. Record as **N/A — WS Responses transport
dropped in the overhaul.** Do not chase.

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
