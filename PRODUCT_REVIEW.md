# Inference Matrix Product Review

## Scope

This review covers the backend, agent service, frontend, persistence, deployment, security, reliability, testing, and compatibility with:

- OpenResponses specification version 2026-04-24
- OpenAI-compatible `/v1` endpoint behavior

The references below are based on the current repository source. Findings are prioritized as follows:

- **P0:** Release blocker or severe security/protocol defect
- **P1:** High-impact compatibility, reliability, or product defect
- **P2:** Important quality, usability, or operational improvement

## P0 Release Blockers

### P0.1 Invalid Python exception syntax

Files contain Python syntax such as:

```python
except TypeError, ValueError:
```

This syntax is invalid. It appears in:

- `backend/app/services/llama_server.py:181,205`
- `backend/app/services/gpu.py:113,145,165,205`
- `backend/app/services/agent_manager.py:214,508`
- `backend/app/services/inference_scheduler.py:69`

Use tuple syntax instead:

```python
except (TypeError, ValueError):
```

Add application import and syntax validation to CI.

### P0.2 Responses schema import failures

`backend/app/api/routes/v1/responses/schemas.py` references `InputItem` before it is defined at line 204 and `Usage` before it is defined at line 217.

The module does not use postponed annotations, so importing the module can fail before the application starts.

Move the referenced declarations above their use or add postponed annotations and verify Pydantic model rebuilding. Add a dedicated import smoke test.

### P0.3 No authentication or authorization

There is no centralized authentication dependency for:

- `/v1/*` inference endpoints
- Agent registration and management
- Model download and deletion
- Server start and stop
- Agent WebSockets
- UI event WebSockets
- Metrics
- File uploads
- Benchmarks

Relevant files:

- `backend/app/api/deps.py`
- `backend/app/main.py:86-113`
- `agent/app/main.py:21-34`

This allows any reachable caller to control inference processes, delete models, access telemetry, and submit requests.

Recommended design:

- Separate inference-client credentials from operator credentials.
- Use a dedicated per-agent secret or mTLS for backend-agent communication.
- Authenticate WebSocket handshakes.
- Add resource-level authorization.
- Require authentication for metrics and management operations.
- Document the intentionally unauthenticated local-development mode, if retained.

### P0.4 Agent registration creates an SSRF pivot

`backend/app/api/routes/agents.py:13-23` accepts arbitrary agent host and port values. The backend subsequently connects to those values from `agent_manager`.

An attacker can use this to target loopback services, cloud metadata endpoints, databases, container services, or other private hosts.

Recommended changes:

- Replace public caller-controlled registration with authenticated enrollment.
- Bind the agent identity to an enrollment credential.
- Allowlist permitted network ranges.
- Reject loopback, link-local, metadata, multicast, and unexpected private addresses unless explicitly configured.
- Revalidate DNS and redirects.
- Replace arbitrary `method` and `path` forwarding with a fixed command enum.

### P0.5 Agent filesystem path traversal

`agent/app/api/routes/servers.py:111-138` accepts arbitrary absolute model paths. Similar path handling affects:

- Model existence checks
- Model metadata
- Model deletion
- Main model paths
- Projector paths
- Draft model paths
- `--model` arguments passed to llama-server

Recommended changes:

- Accept registered model IDs instead of arbitrary paths.
- Resolve paths with `Path.resolve()`.
- Enforce containment under the configured model root.
- Reject symlink escapes.
- Apply the same validation to every model operation.

## OpenResponses Compliance

### P0.6 Incorrect first streaming event

OpenResponses requires the first streaming event to be `response.output_item.added`.

The HTTP stream currently emits lifecycle events first:

- `backend/app/api/routes/v1/responses/router.py:437-442`

The WebSocket path has the same issue:

- `backend/app/api/routes/v1/responses/ws.py:166-168`

The existing unit test at `backend/tests/api/routes/test_responses_unit.py:350-371` codifies the incorrect ordering and must be changed.

Add conformance tests for the entire event state machine, not just individual event shapes.

### P0.7 Streaming errors omit the required `error` event

`router.py:581-587` emits `response.failed` but does not emit an `error` event first.

The helper exists at `backend/app/api/routes/v1/responses/events.py:347-349` but is not used.

Streaming failures must emit a correctly shaped `error` event followed by `response.failed`, then the terminal `[DONE]` marker where applicable.

### P0.8 `allowed_tools` is enforced too late

HTTP streaming emits tool-call events before applying the allowed-tool policy at `router.py:518`.

This means clients can receive a disallowed call before it is removed from the final response. WebSocket transport does not apply the same policy.

Recommended behavior:

- Enforce the policy before emitting any tool-call event.
- Reject the response with a model or invalid-request error, or translate the call into a documented safe fallback.
- Share the enforcement path between HTTP and WebSocket transports.

### P1.1 Input item schemas are too permissive

`UserMessageItemParam` at `schemas.py:138-145` accepts every message role and accepts output/refusal content types in user input.

Define separate discriminated schemas for:

- User messages
- System messages
- Developer messages
- Assistant messages
- Function calls
- Function call outputs
- Reasoning items
- Item references

### P1.2 Unsupported fields are silently accepted

The Responses request model accepts fields that are ignored or only echoed:

- `parallel_tool_calls`
- `max_tool_calls`
- `reasoning.effort`
- `reasoning.summary`
- `include`
- `prompt_cache_key`
- `safety_identifier`
- `service_tier`
- `truncation`
- `top_logprobs`
- `text.verbosity`
- `stream_options.include_obfuscation`

Implement each field or reject unsupported values with a structured `invalid_request` response. Silent acceptance creates false compatibility.

### P1.3 `truncation="disabled"` is not enforced

The server does not reliably detect context overflow and return an error when truncation is disabled.

Implement model-context accounting or reject the option until it can be honored.

### P1.4 Metadata constraints are missing

`MetadataKey` and `MetadataValue` are defined at `schemas.py:581-582` but are not used. The implementation does not enforce metadata entry count, key length, or value length limits.

Use a constrained metadata model and enforce the maximum number of entries.

### P1.5 Unsupported multimodal inputs fail late

`input_file` and `input_video` are accepted by the schema but rejected later by the translator. `input_file` also permits all source fields to be absent.

Validate source alternatives during request parsing and return a structured client error. Advertise model capabilities explicitly.

### P1.6 Item references are always rejected

`translator.py:185-188` rejects `item_reference`. Either implement an item registry or return an explicit documented unsupported-feature error.

### P1.7 Tool-call replay is semantically lossy

`translator.py:203-225` converts previous function calls and outputs into bracketed ordinary text.

This changes model behavior and prevents lossless tool-call continuation. Use native assistant/tool message shapes where supported and test the templates used by each model.

### P1.8 Refusal, logprob, and reasoning output is incomplete

The implementation does not fully translate:

- Model refusals
- Output logprobs
- Requested reasoning summaries
- Encrypted reasoning content
- `include`-controlled reasoning output

Implement these features or reject requests that require them.

### P1.9 Response serialization leaks internal fields

`ResponseResource.input` at `schemas.py:487-488` is serialized by `serialize_spec()` even though it is an internal convenience field.

Exclude it from public serialization or mark it as a vendor-prefixed extension.

### P1.10 Response storage lifecycle is incomplete

Stored responses can be used for chaining, but there are no public retrieval, deletion, or listing endpoints.

Decide whether storage is an internal implementation detail or implement response lifecycle operations with authorization and pagination.

### P1.11 WebSocket connection limit is only checked on receive

`responses/ws.py:289-306` checks the one-hour deadline only after a new client message arrives.

Use a background deadline task that emits `websocket_connection_limit_reached` and closes the socket at the actual limit.

### P1.12 HTTP and WebSocket validation differs

WebSocket tool-output validation exists at `responses/ws.py:105-123`, but HTTP requests do not consistently validate that function-call output IDs match prior calls.

Use a shared validation path for both transports.

## OpenAI Endpoint Compatibility

### P1.13 OpenAI error envelopes are inconsistent

Chat Completions and legacy Completions use ordinary `HTTPException`, producing FastAPI `{"detail": ...}` responses.

Relevant locations:

- `backend/app/api/routes/v1/v1_chat_completions.py:623-624`
- `backend/app/api/routes/v1/v1_chat_completions.py:752-754`
- `backend/app/api/routes/v1/v1_completions.py:407-409`

Create a shared error handler returning:

```json
{
  "error": {
    "message": "...",
    "type": "invalid_request_error",
    "param": "model",
    "code": "model_not_found"
  }
}
```

Apply it to validation errors, model errors, capacity errors, and unexpected failures.

### P1.14 Chat Completions request surface is incomplete

`v1_chat_completions.py:67-86` does not fully support common OpenAI fields, including:

- `n`
- `response_format`
- `logprobs`
- `top_logprobs`
- `seed`
- `user`
- `modalities`
- Audio fields
- Prediction fields
- Full multimodal message content

Reject unsupported fields rather than silently ignoring them.

### P1.15 Chat streaming usage chunks are fragile

The endpoint unconditionally converts upstream chunks into a one-choice chunk and then emits its own usage chunk.

Preserve upstream usage-only chunks with `choices: []`, avoid duplicates, and test OpenAI-compatible and llama.cpp-native stream shapes.

### P1.16 Chat streaming errors do not reliably terminate

`v1_chat_completions.py:431-435` emits an error object but does not guarantee the terminal `[DONE]` marker.

Define and test terminal behavior for validation errors, upstream failures, disconnects, and cancellation.

### P1.17 Chat usage incorrectly replaces zero values

`v1_chat_completions.py:718-724` uses truthiness fallback. Valid zero token counts are treated as missing.

Use explicit `None` or key-presence checks.

### P1.18 Legacy Completions is only partially implemented

`v1_completions.py` accepts but does not faithfully implement:

- Token-array prompts
- `n`
- `logprobs`
- `echo`
- `suffix`
- Multiple choices
- Scheduler leases for streaming
- Reliable stream errors

Either complete the endpoint, document its reduced contract, or remove it from compatibility claims.

### P1.19 Embeddings base64 format is incorrect

`v1_embeddings.py:20-27` accepts `encoding_format="base64"`, but `EmbeddingData.embedding` is always `list[float]`.

Implement the actual base64 representation or reject that format.

### P1.20 Audio endpoints are placeholders

The audio routes currently return placeholder text and empty speech files:

- `v1_audio.py:154`
- `v1_audio.py:253`
- `v1_audio.py:307`

Do not advertise these routes as compatible until they perform actual inference and return the required formats.

### P1.21 Audio uploads lack limits

Audio files are read fully into memory at `v1_audio.py:117-119` and `217-219`.

Add content-length checks, streaming writes, maximum-size enforcement, cleanup, and duration validation.

### P1.22 File and batch identifiers are inconsistent

File responses can return raw UUIDs rather than consistently formatted `file-*` IDs. Batch responses similarly return raw UUIDs even though the implementation creates `batch_*` identifiers.

Use stable OpenAI-compatible IDs throughout storage, responses, URLs, and tests.

### P1.23 Batch processing is incomplete

`v1_batches.py` creates batch records but no complete processing worker is evident. Statuses can remain in `validating` or `cancelling` without actual execution.

Implement a durable worker, input validation, result files, retry behavior, cancellation, expiration, and recovery after restart.

## Runtime and Reliability

### P1.24 Resource exhaustion controls are insufficient

There are no strong controls for:

- Concurrent model downloads
- Disk consumption
- Server process count
- VRAM admission
- Prompt size
- Tool schema size
- Upload size
- Request rate
- Event and log payload size

Enforce limits at the reverse proxy, API, scheduler, agent, and storage layers.

### P1.25 Scheduler can over-admit when telemetry fails

`inference_scheduler.py:60-85` treats unavailable slot telemetry as unknown but continues to claim capacity using fallback values.

Use a conservative fallback or fail closed, and expose deterministic `429` or `503` overload responses.

### P1.26 Queued inference leases can become orphaned

If acquisition is cancelled after `_queue()`, cleanup is not guaranteed. Add cancellation-safe cleanup and a periodic lease reaper.

### P1.27 Reconnect task ownership is fragile

`agent_manager.py:222-261` can leave reconnect task entries behind or return without scheduling future retries.

Use one supervised retry loop with exponential backoff, jitter, state transitions, and `finally` cleanup.

### P1.28 Process shutdown cleanup is incomplete

`agent/app/main.py:72-78` stops monitoring but does not clearly stop all active llama-server processes. Background log forwarding and related tasks may continue.

Use process groups, explicit shutdown timeouts, task supervision, and bounded logs.

### P1.29 Startup failures are swallowed

The agent logs startup failures but remains healthy at `agent/app/main.py:40-70`.

Separate liveness from readiness. Readiness should fail when required registration, model runtime, or monitoring services are unavailable.

### P1.30 Configuration validation is too permissive

`ServerSpec.options` is an unrestricted dictionary and ports, context sizes, batch sizes, GPU layers, and parallelism are not sufficiently bounded.

Use strict typed options, numeric bounds, model-specific validation, port-range validation, and atomic port reservation.

### P1.31 Sensitive data is over-logged

Full request payloads, model paths, commands, and exception strings can be logged or returned to clients.

Redact credentials, authorization headers, prompt content, filesystem paths, and internal command details. Return stable public error codes.

## Frontend

### P0.32 Generated client response wrappers are mishandled

The generated client returns an Axios response wrapper whose payload is under `.data`.

The following pages treat the wrapper as the payload:

- `frontend/src/routes/_layout/responses/index.tsx:320-364`
- `frontend/src/routes/_layout/embeddings/index.tsx:47-64`
- `frontend/src/routes/_layout/audio/index.tsx:53-69`
- `frontend/src/routes/_layout/completions/index.tsx:108-118`

Use consistent destructuring:

```ts
const { data } = await serviceCall(...)
```

Add integration tests for every non-streaming endpoint.

### P0.33 `VITE_API_URL` is not wired into the generated client

`frontend/.env` defines `VITE_API_URL`, but the generated client has no base URL configuration.

Centralize API URL resolution for generated REST calls, raw `fetch`, and WebSocket URLs. Regenerate the client rather than editing generated files manually where possible.

### P1.34 SSE parsing is duplicated and fragile

Chat, Completions, and Responses each implement their own parser. They:

- Swallow JSON and protocol errors
- Assume `\n\n` framing only
- Do not robustly handle CRLF
- Drop the final unterminated buffer
- Do not reliably verify `[DONE]`
- Do not consistently surface structured server errors

Create one tested SSE parser and one shared streaming request abstraction.

### P1.35 Streaming requests survive navigation

Abort controllers and timers are not consistently cleaned up when routes unmount.

Add effect cleanup and ensure abandoned requests release backend capacity.

### P1.36 Chat mutates React state

`chat/index.tsx:204-217` mutates the existing assistant message object.

Copy the message object before updating content or reasoning.

### P1.37 Model selection is inconsistent

Chat, Responses, and Completions use server aliases. Embeddings and Audio use registered model names.

Provide a shared selector that exposes model ID, server alias, endpoint capability, readiness, and cold-start status.

### P1.38 Audio UI does not match API support

The UI only supports basic transcription while generated APIs expose translation and speech. Missing UI includes:

- Translation
- Text-to-speech
- Prompt and temperature
- SRT and verbose JSON formats
- Audio playback and download
- Verbose result metadata

Either implement these features or clearly label the page as experimental.

### P1.39 Responses UI state can become inconsistent

The Responses page can retain an old `previous_response_id` when storage is disabled and later re-enabled. Failed responses can also leave partial assistant and error entries separately.

Clear chain state when storage or model changes, provide an unconditional new-conversation action, and represent request status directly on each entry.

### P1.40 Mobile layout is incomplete

Several pages use fixed widths and two-column layouts:

- `responses/index.tsx`
- `completions/index.tsx`
- `embeddings/index.tsx`
- `audio/index.tsx`

Collapse to one column on small screens, move the Responses inspector into a drawer, and enable mobile Playwright projects.

### P2.41 Accessibility gaps

Add:

- Accessible names for icon-only buttons
- `aria-live` status regions for streaming
- `role="alert"` for errors
- Proper label/control associations
- Focus management on errors and completed responses

### P2.42 Clipboard errors are unhandled

`embeddings/index.tsx:158-161` calls `navigator.clipboard.writeText()` without handling rejection.

Use the existing clipboard hook and report failures to the user.

### P2.43 Frontend test coverage is insufficient

Add unit, integration, and Playwright coverage for:

- Client response unwrapping
- API URL configuration
- SSE parsing
- Abort and navigation races
- Responses chaining
- Empty and unavailable model states
- Audio validation
- Mobile layouts
- Accessibility

### P2.44 Mutating lint command is unsuitable for CI

`frontend/package.json:9` defines lint as:

```text
biome check --write --unsafe
```

Separate non-mutating CI checks from explicit formatting/fix commands.

## Deployment and Operations

### P0.45 Compose topology is inconsistent

`compose.yml` defines `postgres`, `frontend`, and `agent`. `compose.deploy.yml` references `proxy`, `db`, `adminer`, and `backend`, which do not exist in the base file.

Consolidate on one topology and validate overlays with:

```bash
docker compose -f compose.yml -f compose.deploy.yml config
```

### P0.46 Health checks target nonexistent endpoints

Compose checks `/api/health`, but the backend does not define that route in `backend/app/main.py`. The agent exposes `/health`, not `/api/health`.

Add and document:

- `/health/live` for process liveness
- `/health/ready` for dependency readiness

Update Compose, reverse proxy configuration, monitoring, and smoke tests.

### P0.47 Production defaults are unsafe

Current defaults include:

- `CORS_ALLOW_ALL_ORIGINS=True`
- Empty database passwords permitted
- `changethis` database password in Compose
- Public agent port exposure
- Unauthenticated metrics
- No API authentication

Fail closed in production and require explicit secrets and origins.

### P1.48 Agent image may not contain llama.cpp

The base agent image does not clearly install llama.cpp while configuration expects `/usr/local/bin/llama-server`.

Select a concrete llama.cpp recipe in deployment or fail readiness with a clear binary/runtime diagnostic.

### P1.49 Migrations lack operational safeguards

Migrations run automatically without clearly defined:

- Database readiness retries
- Multi-worker migration locking
- Backup gates
- Rollback policy
- Destructive migration runbooks

Run migrations as a controlled release step and document irreversible changes.

### P1.50 Backups are not operationally complete

Define and test:

- Scheduled database backups
- Encrypted off-host storage
- Retention policy
- Restore verification
- RPO and RTO
- Secret exclusion or secret-manager integration

### P1.51 Observability documentation overstates implementation

Documented metrics and alert labels do not consistently exist in `backend/app/api/routes/metrics.py`.

Add metric contract tests, clear stale gauges, avoid high-cardinality labels, protect metrics, and publish only verified alert rules.

### P1.52 Deployment documentation is stale

Several documents describe service names, ports, workflows, and frontend behavior that do not match current source.

Treat deployment documentation as versioned product behavior and validate it in CI against actual Compose and application routes.

### P1.53 No meaningful end-to-end deployment test

Add an automated environment that verifies:

- Clean PostgreSQL startup
- Migrations
- Backend readiness
- Agent registration
- Server startup
- Model resolution
- Chat and Responses inference
- WebSocket events
- Restart recovery
- Shutdown cleanup

Use a deterministic mock llama server for CI and reserve GPU tests for dedicated runners.

## Recommended Implementation Order

1. Fix invalid Python syntax and Responses schema import failures.
2. Add authentication and authorization before public deployment.
3. Lock down agent enrollment, SSRF, filesystem paths, and arbitrary commands.
4. Correct OpenResponses event ordering, error events, and tool enforcement.
5. Add official OpenResponses conformance tests for HTTP/SSE and WebSocket.
6. Create a shared OpenAI error and validation layer.
7. Decide which endpoints are supported, experimental, or unavailable.
8. Fix generated-client response unwrapping and API URL configuration.
9. Add upload, download, prompt, scheduler, process, and storage limits.
10. Consolidate Compose topology and health/readiness contracts.
11. Add backend-agent contract tests and frontend Playwright coverage.
12. Update documentation only after automated checks validate its claims.

## Definition of Done

The product should not claim full OpenResponses/OpenAI compatibility until:

- The complete backend imports successfully.
- Authentication is enforced.
- Official OpenResponses conformance tests pass for HTTP/SSE and WebSocket.
- Streaming event ordering matches the specification.
- Streaming errors include the required error lifecycle.
- Unsupported request fields are rejected or implemented.
- OpenAI error envelopes are consistent across all `/v1` routes.
- Audio, Files, Batches, Embeddings, and Completions behavior matches the advertised support level.
- Compose validation, health checks, migrations, and smoke tests pass in CI.
- Frontend non-streaming, streaming, mobile, and accessibility tests pass.
