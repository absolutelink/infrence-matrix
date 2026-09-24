# Integration Testing — OpenResponses Compliance Suite

Test tool: `/workspaces/openresponses` (`bun run test:compliance`).
Deployment logs stream to `/tmp/lfai-matrix.log` (wipe before each deploy cycle).

```bash
# full run
bun run test:compliance --base-url https://matrix.thelink.family/v1 --api-key none --model voyager

# single test
bun run test:compliance --base-url https://matrix.thelink.family/v1 --api-key none --model voyager --filter <test-id>
```

## Status

Baseline run: 2026-09-24 — **3 passed / 14 failed / 17 total**
Run 2 (after clusters 1–3 + 5 fix): 2026-09-24 — **10 passed / 7 failed / 17 total**

| Test ID | Name | Status | Notes |
|---|---|---|---|
| `basic-response` | Basic Text Response | ✅ PASS | |
| `assistant-phase` | Assistant Message Phase | ✅ PASS | |
| `response-output-phase-schema` | Response Output Phase Schema | ✅ PASS | local schema fixture, no HTTP |
| `streaming-response` | Streaming Response | ✅ PASS | |
| `websocket-response` | WebSocket Response | ❌ FAIL | WS framing: SSE-style `event:`+`data:` frames over WS |
| `websocket-sequential-responses` | WebSocket Sequential Responses | ❌ FAIL | same WS framing issue |
| `websocket-continuation` | WebSocket Continuation | ❌ FAIL | same WS framing issue |
| `websocket-reconnect-store-false-recovery` | WebSocket Store False Reconnect Recovery | ❌ FAIL | same WS framing issue |
| `websocket-previous-response-not-found` | WebSocket Missing Previous Response | ✅ PASS | |
| `websocket-failed-continuation-evicts-cache` | WebSocket Failed Continuation Evicts Cache | ❌ FAIL | same WS framing issue |
| `websocket-compact-new-chain` | WebSocket Compact New Chain | ❌ FAIL | WS framing + server rejects compact output items (`compaction items are not supported in this implementation`) |
| `system-prompt` | System Prompt | ✅ PASS | |
| `tool-calling` | Tool Calling | ✅ PASS | |
| `image-input` | Image Input | ❌ FAIL | llama-server has no mmproj → upstream 500 swallowed into `completed` w/ empty output (should be `response.failed` or surfaced error) |
| `multi-turn` | Multi-turn Conversation | ✅ PASS | |
| `compact-response` | Compaction Endpoint | ✅ PASS | |
| `compact-missing-model` | Compaction Missing Required Model | ✅ PASS | |

## Failure clusters

1. ~~**Response schema: null vs required fields**~~ — FIXED (serialize_spec)
2. ~~**`output.0: Invalid input`**~~ — FIXED (key-absent item optionals via serialize_spec)
3. ~~**Streaming final response incomplete**~~ — FIXED (serialize_spec + completed_at)
4. **WebSocket framing** — server sends SSE-style `event:`/`data:` text frames over the
   WebSocket; clients expect raw JSON per frame. Causes parse failures and terminal-event
   timeouts. Affects: all websocket-* tests (except previous-response-not-found which tolerates it).
   Fix: `ws.py` `_stream_to_ws` should send raw JSON per WS message (drop SSE framing).
5. ~~**HTTP 500 on assistant `phase` labels and multi-turn**~~ — FIXED (was translation error)
6. **Compaction items rejected as input** — `input_items_to_llama_messages` raises on
   `compaction` type. Affects: websocket-compact-new-chain (after WS framing fix).
   Fix: accept compaction items as context (their summary is the content).
7. **image-input: upstream errors swallowed** — llama-server w/o mmproj returns 500 for
   image parts; `_complete` produced `completed` + empty output instead of failing.
   Ops fix: select an mmproj projector for the voyager instance (agent supports it).
   Code fix: surface upstream error as `response.failed`/HTTP error.

## History

| Date | Commit | Passed | Failed | Notes |
|---|---|---|---|---|
| 2026-09-24 | (baseline) | 3 | 14 | Initial full run |
| 2026-09-24 | serialize_spec fix (pushed) | 10 | 7 | Clusters 1–3 + 5 fixed: spec serializer (`serialize_spec`), `completed_at` at finalize, dropped `reasoning_text.*` event twins. Unblocked: basic-response, system-prompt, image-input-schema, tool-calling, streaming-response, assistant-phase, multi-turn, compact-response. Remaining: WS framing (6 tests), compaction-as-input, image-input (no mmproj on voyager + swallowed upstream 500) |