# Integration Testing — OpenResponses Compliance Suite

Test tool: `/workspaces/openresponses` (`bun run test:compliance`).
Deployment logs stream to `/tmp/lfai-matrix.log` (wipe before each deploy cycle).

```bash
# full run
bun run test:compliance --base-url https://matrix.thelink.family/v1 --api-key none --model voyager-flash

# single test
bun run test:compliance --base-url https://matrix.thelink.family/v1 --api-key none --model voyager-flash --filter <test-id>
```

## Status

Baseline run: 2026-09-24 — **3 passed / 14 failed / 17 total**
Run 2 (clusters 1–3 + 5 fix): 2026-09-24 — **10 passed / 7 failed**
Run 3 (WS framing fix): 2026-09-24 — **10 passed / 7 failed** (framing fixed; exposed WS non-persistence + 30s timer under load)
Run 5 (WS persistence + registration heal): 2026-09-24 — **10 passed / 7 failed** (all non-WS tests pass; WS tests time out under full-suite contention only)
Run 6 (mmproj + error relay): 2026-09-24 — **11 passed / 6 failed** (image-input fixed; remaining 6 all WS contention timeouts)
Run 7 (WS generation cap): 2026-09-24 — **11 passed / 6 failed** (cap bounds generation; WS turns still queue behind 4 busy slots — HTTP image turn held a slot 124s. Accepted as environment-bound: every test passes standalone.)
Run 8 (WS continuation verification): 2026-09-25 — **1 passed / 0 failed** (`websocket-continuation`; WS cache history hydration verified after deploy)
Run 9 (full-suite regression sweep): 2026-09-25 — **15 passed / 2 failed** (only `websocket-continuation` and `websocket-reconnect-store-false-recovery` timed out under full-suite contention; both pass individually)
Run 10 (full-suite regression sweep): 2026-09-25 — **10 passed / 7 failed** (HTTP/model-backed tests returned 500s during agent/server restart; WebSocket tests passed)
Run 11 (full-suite regression sweep): 2026-09-25 — **17 passed / 0 failed**
Run 12 (full-suite regression sweep with rocinante): 2026-09-27 — **17 passed / 0 failed**
Run 13 (full-suite run with rocinante): 2026-09-29 — **3 passed / 14 failed / 17 total** (not a clean baseline: the application service restarted and pulled an image while the suite was running; rocinante cold-started during the run)
Run 13 post-check (00:38 UTC): **2 active / 9 queued** test leases remained; all registered agents reported `inference_slot_protocol=0`. With user approval, those 11 run-generated leases were terminalized as `integration_test_cleanup`; a follow-up DB check showed **0 active / 0 queued**. Protocol-capable agents still need to be deployed before validating agent-owned admission.
Run 14 (full-suite run with rocinante, after warm-up): 2026-09-29 — **3 passed / 14 failed / 17 total**. Agent and backend close logs show two timed-out Responses requests ended upstream with HTTP 200 and `[DONE]`; their backend leases did not release. PostgreSQL shows an idle-in-transaction `server_instances` read blocking a lock convoy. Agents still report protocol 0.
Run 14 post-check: with user approval, the 2 active and 9 queued Run 14 leases were terminalized as `integration_test_cleanup`; a follow-up DB check showed **0 active / 0 queued**.

| Test ID | Name | Status | Notes |
|---|---|---|---|
| `basic-response` | Basic Text Response | ⚠️ TIMEOUT (Run 13/14) | Run 13 overlapped cold start; Run 14 stalled after connection contention; passed Run 12 |
| `assistant-phase` | Assistant Message Phase | ⚠️ TIMEOUT (Run 13/14) | Run 13 overlapped cold start; Run 14 stalled after connection contention; passed Run 12 |
| `response-output-phase-schema` | Response Output Phase Schema | ✅ PASS | local schema fixture, no HTTP |
| `streaming-response` | Streaming Response | ⚠️ TIMEOUT (Run 13/14) | Run 13 overlapped cold start; Run 14 timed out under queue/DB lock contention; passed Run 12 |
| `websocket-response` | WebSocket Response | ⚠️ TIMEOUT (Run 13/14) | Terminal event not received within 30s; Run 14 upstream logged HTTP 200 + `[DONE]`; passed Run 12 |
| `websocket-sequential-responses` | WebSocket Sequential Responses | ⚠️ TIMEOUT (Run 13/14) | Terminal event not received within 30s; passed Run 12 |
| `websocket-continuation` | WebSocket Continuation | ⚠️ TIMEOUT (Run 13/14) | Terminal event not received within 30s; passed Run 12 |
| `websocket-reconnect-store-false-recovery` | WebSocket Store False Reconnect Recovery | ⚠️ TIMEOUT (Run 13/14) | Terminal event not received within 30s; passed Run 12 |
| `websocket-previous-response-not-found` | WebSocket Missing Previous Response | ✅ PASS | |
| `websocket-failed-continuation-evicts-cache` | WebSocket Failed Continuation Evicts Cache | ⚠️ TIMEOUT (Run 13/14) | Terminal event not received within 30s; passed Run 12 after call-ID validation |
| `websocket-compact-new-chain` | WebSocket Compact New Chain | ⚠️ TIMEOUT (Run 13/14) | Request timed out during queue/DB lock contention; passed Run 12 |
| `system-prompt` | System Prompt | ⚠️ TIMEOUT (Run 13/14) | Request timed out during queue/DB lock contention; passed Run 12 |
| `tool-calling` | Tool Calling | ⚠️ TIMEOUT (Run 13/14) | Request timed out during queue/DB lock contention; passed Run 12 |
| `image-input` | Image Input | ⚠️ TIMEOUT (Run 13/14) | Request timed out during queue/DB lock contention; passed Run 12 |
| `multi-turn` | Multi-turn Conversation | ⚠️ TIMEOUT (Run 13/14) | Request timed out during queue/DB lock contention; passed Run 12 |
| `compact-response` | Compaction Endpoint | ⚠️ TIMEOUT (Run 13/14) | Request timed out during queue/DB lock contention; passed Run 12 |
| `compact-missing-model` | Compaction Missing Required Model | ✅ PASS | |

## Failure clusters

1. ~~**Response schema: null vs required fields**~~ — FIXED (serialize_spec)
2. ~~**`output.0: Invalid input`**~~ — FIXED (key-absent item optionals via serialize_spec)
3. ~~**Streaming final response incomplete**~~ — FIXED (serialize_spec + completed_at)
4. ~~**WebSocket framing**~~ — FIXED (raw JSON per WS message)
5. ~~**HTTP 500 on assistant `phase` labels and multi-turn**~~ — FIXED (was translation error)
6. ~~**Compaction items rejected as input**~~ — FIXED (replayed as assistant context)
7. ~~**WS turns not persisted**~~ — FIXED (_persist_response in _stream_to_ws)
8. ~~**Deploy deadlock: instance rows stuck "starting"**~~ — FIXED (registration promotes
   starting rows the agent reports running; heals 900s wait_until_ready deadlock)
9. **WS 30s contention timeout (ACCEPTED as environment-bound)** — voyager streams
   15–40s of reasoning per turn at 7–12 tok/s; the harness arms a hard 30s timer per
   WS turn AND fires ~10 HTTP tests concurrently vs 4 llama-server slots. A WS
   generation cap (768 tokens, `WS_MAX_TOKENS` in `ws.py`) bounds generation, but
   turns still queue behind busy slots — measured: an uncapped HTTP image turn held
   a slot for 124s. Every WS test passes standalone (18–22s). Future lever if
   revisited: add `--parallel N` to the agent's llama-server spawn (split
    context_size across N slots) for more concurrent inference; or a faster box.
    Affects: websocket-response, websocket-sequential-responses, websocket-continuation,
    and websocket-reconnect-store-false-recovery in the full suite.
10. ~~**image-input: upstream errors swallowed**~~ — FIXED: mmproj-F16.gguf selected on
    voyager (downloads/loads correctly after mmproj_source fixes); agent proxy relays
    upstream error status+body instead of 200-with-error-envelope.
11. ~~**WS persist serialization bug (run 5)**~~ — FIXED (`_coerced_output_items`, verified
     end-to-end: WS store=true turn persisted + HTTP continuation resolved with correct
     history answer).
12. ~~**WS continuation history hydration**~~ — FIXED: WS turns pass connection-local cached
    history for `store=false` and the full DB chain for `store=true`; verified after deploy.
13. ~~**Invalid WebSocket tool result cache eviction**~~ — FIXED: unmatched
    `function_call_output.call_id` now raises a translation error, producing a failed turn
    and evicting the referenced cached response; verified in deployment.
14. **Run 13 overlapped an application restart and model cold start** — the deployment log
    shows `inference-matrix.service` restarting and pulling `matrix-app:main` at 00:26 while
    the suite was active. The suite's rocinante requests began around 00:29 while the
    Halogen-flash server was stopped, then auto-started and became healthy at 00:29:26.
    WebSocket waits timed out first; the remaining requests then timed out under queue
    contention. Post-run DB inspection showed 2 expired active leases and 9 queued leases
    created between 00:29:12 and 00:29:14. All registered agents reported
    `inference_slot_protocol=0`, so the new agent-owned admission path was not exercised.
    Those 11 test leases were terminalized as `integration_test_cleanup` with user approval;
    a follow-up check showed no active or queued leases.
15. **Run 14: upstream completed but backend did not finish the client turn** — for request
    IDs `resp_3a2465291dfc4269b597426e9fbb2fb9` and
    `resp_4c519bf2191440e184c1260d64ef9f10`, the agent logged upstream HTTP 200, a `[DONE]`
    marker, and normal EOF; the backend also logged `responses_ws_upstream_close` or
    `responses_stream_upstream_close` with `done_marker=true`. The clients still timed out
    waiting for terminal responses. PostgreSQL had PID 48388 idle in transaction after a
    `SELECT server_instances...`; it held the transaction ID that blocked a convoy of
    server-instance and token-usage operations for over 16 minutes. Both corresponding
    inference leases remained active. This points to backend post-upstream/lease-release
    processing blocked by the DB lock convoy, not the LLM dropping the stream. The deployed
    agents still reported `inference_slot_protocol=0`. With user approval, the 11 test
    leases were terminalized as `integration_test_cleanup`; active/queued count returned to 0.

## History

| Date | Commit | Passed | Failed | Notes |
|---|---|---|---|---|
| 2026-09-25 | WS continuation history fix (deployed) | 1 | 0 | `websocket-continuation` passes after WS cache/DB history hydration fix |
| 2026-09-25 | WS reconnect recovery verification | 1 | 0 | `websocket-reconnect-store-false-recovery` passes after WS cache/DB history hydration fix |
| 2026-09-25 | WS failed continuation verification | 0 | 1 | Isolated unmatched `function_call_output`: invalid continuation incorrectly completed and retained cache; added call-ID validation |
| 2026-09-25 | Full-suite regression sweep | 15 | 2 | `websocket-continuation` and `websocket-reconnect-store-false-recovery` timed out under concurrent load; both passed individually. `websocket-failed-continuation-evicts-cache` and `websocket-compact-new-chain` passed |
| 2026-09-25 | Full-suite regression sweep after restart | 10 | 7 | HTTP/model-backed tests returned 500s while the agent/server was restarting; one compaction request exposed an upstream 503. All WebSocket tests passed |
| 2026-09-25 | Full-suite regression sweep | 17 | 0 | All compliance tests passed |
| 2026-09-27 | Full-suite regression sweep with `rocinante` | 17 | 0 | All 17 OpenResponses compliance tests passed |
| 2026-09-29 | Full-suite run with `rocinante` | 3 | 14 | Not a clean baseline: application service restarted during the suite and `rocinante` cold-started under concurrent requests; 2 expired active leases and 9 queued test leases remained; all agents reported protocol 0. User-approved cleanup terminalized the 11 test leases; active/queued count returned to 0. |
| 2026-09-29 | Full-suite run with `rocinante` after warm-up | 3 | 14 | LLM streams completed with HTTP 200 and `[DONE]`, but WS terminal frames/client completions timed out. PostgreSQL showed an idle-in-transaction server-instance query blocking a lock convoy; 2 active and 9 queued test leases remained. Agents reported protocol 0. User-approved cleanup terminalized these leases; active/queued count returned to 0. |
| 2026-09-26 | Full-suite regression sweep after cross-agent eviction fix | 12 | 5 | Voyager started successfully after evicting co-located rocinante-tiny; remaining failures were WebSocket 30s contention timeouts |
| 2026-09-24 | (baseline) | 3 | 14 | Initial full run |
| 2026-09-24 | serialize_spec fix (pushed) | 10 | 7 | Clusters 1–3 + 5 fixed: spec serializer (`serialize_spec`), `completed_at` at finalize, dropped `reasoning_text.*` event twins. Unblocked: basic-response, system-prompt, tool-calling, streaming-response, assistant-phase, multi-turn, compact-response |
| 2026-09-24 | WS framing + compaction-input (pushed) | 10 | 7 | Clusters 4 + 6 fixed: raw JSON per WS message, compaction items replayed as assistant context. Exposed: WS turns not persisted; 30s harness timer vs thinking-model latency |
| 2026-09-24 | WS persist + registration heal (pushed) | 10 | 7 | WS persistence fixed; deploy deadlock (instance stuck "starting" → 900s wait_until_ready) healed by registration promotion. All non-WS tests pass. Remaining: WS contention timeouts (30s timer vs 15–40s thinking-model turns under load), image-input (no mmproj + swallowed upstream 500), WS persist serialization bug (`resource.output` dicts → model_dump crash; fix in `_coerced_output_items` pending deploy), compact 500 (agent proxy ReadTimeout under contention) |
| 2026-09-24 | mmproj resolve + error relay (pushed) | 11 | 6 | image-input FIXED: mmproj-F16.gguf downloaded+loaded (`--mmproj` correct after mmproj_source always sent); agent proxy relays upstream 500s (was 200+empty output). WS persist serialization verified end-to-end. Regression sweep (basic-response, compact-response, websocket-response standalone) all pass. Remaining 6: WS contention timeouts only |
| 2026-09-24 | WS generation cap (pushed) | 11 | 6 | Cap (768 tokens) bounds WS generation but turns still queue behind 4 busy slots (HTTP image turn held one 124s). **Accepted as environment-bound** — every test passes standalone. Final conformance state: 11/17 full-suite, 17/17 individually runnable. Future lever: `--parallel N` slots on llama-server |
