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

| Test ID | Name | Status | Notes |
|---|---|---|---|
| `basic-response` | Basic Text Response | ❌ FAIL | `output.0` schema invalid; `parallel_tool_calls`/`top_p`/`presence_penalty`/`frequency_penalty`/`top_logprobs`/`temperature` sent as `null` (expected numbers/bool); `text.verbosity` invalid |
| `assistant-phase` | Assistant Message Phase | ❌ FAIL | HTTP 500 |
| `response-output-phase-schema` | Response Output Phase Schema | ✅ PASS | local schema fixture, no HTTP |
| `streaming-response` | Streaming Response | ❌ FAIL | 369 events; final `response.completed` object missing `completed_at`, `incomplete_details`, `previous_response_id`, `instructions`, `error`, `reasoning`, `max_output_tokens`, `max_tool_calls` (undefined), null numerics |
| `websocket-response` | WebSocket Response | ❌ FAIL | server sends `event:`+`data:` frames over WS (must be raw JSON only); timed out waiting terminal event |
| `websocket-sequential-responses` | WebSocket Sequential Responses | ❌ FAIL | same WS framing issue |
| `websocket-continuation` | WebSocket Continuation | ❌ FAIL | same WS framing issue |
| `websocket-reconnect-store-false-recovery` | WebSocket Store False Reconnect Recovery | ❌ FAIL | same WS framing issue |
| `websocket-previous-response-not-found` | WebSocket Missing Previous Response | ✅ PASS | |
| `websocket-failed-continuation-evicts-cache` | WebSocket Failed Continuation Evicts Cache | ❌ FAIL | same WS framing issue |
| `websocket-compact-new-chain` | WebSocket Compact New Chain | ❌ FAIL | WS framing + server rejects compact output items (`compaction items are not supported in this implementation`) |
| `system-prompt` | System Prompt | ❌ FAIL | same schema issues as basic-response |
| `tool-calling` | Tool Calling | ❌ FAIL | `output.0` invalid + null numeric fields |
| `image-input` | Image Input | ❌ FAIL | null numeric fields (output apparently OK) |
| `multi-turn` | Multi-turn Conversation | ❌ FAIL | HTTP 500 |
| `compact-response` | Compaction Endpoint | ❌ FAIL | `output.0` invalid |
| `compact-missing-model` | Compaction Missing Required Model | ✅ PASS | |

## Failure clusters

1. **Response schema: null vs required fields** — backend sends explicit `null` for
   `parallel_tool_calls`, `top_p`, `temperature`, `presence_penalty`, `frequency_penalty`,
   `top_logprobs`, `text.verbosity` (missing/invalid) — schema requires concrete values.
   Affects: basic-response, system-prompt, image-input, tool-calling.
2. **`output.0: Invalid input`** — output item shape doesn't match schema (likely missing
   `annotations`/`status`/`content` typing). Affects: basic-response, system-prompt, tool-calling, compact-response.
3. **Streaming final response incomplete** — `response.completed` payload lacks required fields
   (`completed_at`, `instructions`, `previous_response_id`, `error`, `reasoning`,
   `max_output_tokens`, `max_tool_calls`). Affects: streaming-response.
4. **WebSocket framing** — server sends SSE-style `event:`/`data:` text frames over the
   WebSocket; clients expect raw JSON per frame. Causes parse failures and terminal-event
   timeouts. Affects: all websocket-* tests (except previous-response-not-found which tolerates it).
5. **HTTP 500 on assistant `phase` labels and multi-turn** — affects assistant-phase, multi-turn.
6. **Compaction output items not accepted as input** — affects websocket-compact-new-chain,
   possibly compact-response.

## History

| Date | Commit | Passed | Failed | Notes |
|---|---|---|---|---|
| 2026-09-24 | (baseline) | 3 | 14 | Initial full run |
| 2026-09-24 | pending deploy | ? | ? | Clusters 1–3 fix: spec serializer (`serialize_spec`) — concrete echo params, nullable keys present-as-null, key-absent item optionals, tool echo nulls, `completed_at` at finalize; dropped `reasoning_text.*` event twins (harness union rejects them). Expected to unblock: basic-response, system-prompt, image-input, tool-calling, streaming-response; reduces noise for compact-response |