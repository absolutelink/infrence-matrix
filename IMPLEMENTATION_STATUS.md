# Inference Matrix — Implementation Status

**Overhaul branch:** `litellm-architecture-overhaul`
**Last updated:** 2026-10-07 (Phase 14 ✅ shell definitions — `provider_type`/`backend_config` nullable at create, type adopted at first registration (`type_adopted`), new admin-owned `awaiting_config` pre-state excludes shells from scheduling//v1/models/pushes; `NullableJSON` (none_as_null) for the JSON null-vs-SQL-NULL trap; `seed_provider_type` dev ceremony removed. Phase 13 ✅ log views. Prior: Phase 12 schema-driven backend config, Phase 10 admin UI rework.)

This file tracks the litellm-based architecture overhaul (see
[ARCHITECTURE.md](ARCHITECTURE.md)). Each phase lists its features with
enough detail that a fresh session can pick up any single feature. When
starting a feature, read the linked protocol/doc first.

## Phase Overview

| Phase | Scope | Status |
| --- | --- | --- |
| 0 | litellm Responses streaming fidelity spike | ✅ Complete |
| 1 | Repo restructure: `admin/` + `provider/` workspaces, Redis in compose, CI/Dockerfiles | ✅ Complete |
| 2 | Squashed schema (5 tables) + Redis key layout | ✅ Complete |
| 3 | Provider registration + WS auth + epoch fencing + connection manager | ✅ Complete |
| 4 | Provider lib: backend lifecycle, `/v1` translation surface, slot admission | ✅ Complete |
| 5 | llama-cpp provider + model downloader + machine metrics + ownership | ✅ Complete |
| 6 | Scheduler + `/v1/responses` via litellm + SSE emitter + persistence + eviction + idle reaper + cold-boot keepalive | ✅ Complete |
| 7 | `/v1/chat/completions` + `/v1/models` + 501 stubs | ✅ Complete |
| 8 | gufo / halogen / halogen-flash provider ports | ✅ Complete |
| 9 | `provider.config.update` / fingerprint / cache-clear flow | ✅ Complete |
| 10 | Admin UI rework | ✅ Complete |
| 11 | Docs consolidation + full E2E validation | ✅ Complete |
| 12 | Schema-driven backend config: `ProviderType` registry + JSON Schema consensus + HF picker + rjsf form | ✅ Complete |
| 13 | Server + backend log capture, Redis tails, UI log views | ✅ Complete |
| 14 | Shell definitions: deferred typing + `awaiting_config` pre-state | ✅ Complete |

Legend: ✅ complete · 🟡 in progress · ⬜ pending

---

## Phase 0 — litellm Fidelity Spike ✅

**Decision doc:** `spike/litellm-fidelity/FINDINGS.md`

Verified `litellm.aresponses(stream=True)` against a toy upstream emitting
the full 26-event OpenResponses set. **litellm is suitable.** Constraints
carried into Phase 6:

1. Admin owns the client-facing `resp_<uuid>` and the emitter; litellm's
   affinity-wrapped id is stored for continuation but never surfaced.
2. Every alias must be registered with litellm
   (`supports_native_streaming: True`, `mode: "responses"`,
   `litellm_provider: "openai"`) or native streaming isn't selected.
3. `response.failed` arrives as `MidStreamFallbackError`; admin must map
   exceptions to spec terminal frames.
4. Admin owns the chain: reconstruct full `input` from DB; never pass
   `previous_response_id` to litellm.
5. Provider-side translation layer is still required — litellm faithfully
   transports whatever the provider emits; it does not fix non-compliance.

Reproduce: `cd spike/litellm-fidelity && uv run python spike2.py`.

---

## Phase 1 — Restructure ✅

- uv workspace members: `admin/backend`, `provider/lib`, `provider/mock`,
  `provider/llama-cpp`, `spike/litellm-fidelity` (see root
  `pyproject.toml`).
- `compose.yml` runs **postgres + redis + admin + provider-mock** with
  healthchecks and `REDIS_URL`/`POSTGRES_*` wiring. **Full local stack
  works with no hardware.**
- Root `Dockerfile` builds frontend (bun) → admin image (python 3.14, uv,
  `--no-install-workspace --package matrix-admin`).
- CI `build-and-push.yml` starts postgres 16 + redis 7 services and tests
  the admin. It also builds `matrix-app` and (added post-overhaul) the
  two **llama-cpp provider images** — Vulkan (official
  `full-vulkan` base) and CUDA 12 (`server-cuda12` base) — via
  `provider/llama-cpp/{Dockerfile,Dockerfile.cuda12}`; `provider/llama-cpp`
  tests run in the `test-provider` job. No `:latest` tag is produced.
- The pre-overhaul code was removed in the final cleanup; it lives in git
  history only — never restore patterns from it.

---

## Phase 2 — Schema + Redis Keys ✅

**Models:** `admin/backend/app/models.py`
**Redis doc:** `admin/backend/docs/redis-keys.md`
**Migration:** `admin/backend/alembic/versions/0140d6ad9f48_initial_squashed_schema.py`

Six tables: `Machine`, `ProviderDefinition`, `ProviderInstance`,
`ResponseRecord`, `TokenUsageSample` (fields in ARCHITECTURE.md §4).
Phase 12 added `ProviderType` (the committed backend_config schema +
consensus state) in a follow-up migration — see §Phase 12.
`Machine.reachable_address()` = `dns or host or ip`. Uniqueness:
`ProviderInstance` unique on `(machine_id, provider_definition_id)`.

Redis namespaces: `im:ws:*` (secret/epoch/owner/presence),
`im:metrics:*` (owner/machine/cats), `im:sched:*` (queue/wait/active/
lock), `im:vram:used:{machine_uid}`. All TTL-bounded so a crashed admin
self-heals.

---

## Phase 3 — Registration + WS + Auth ✅

**Admin:** `app/api/admin/providers.py`, `app/api/ws.py`,
`app/services/connection_manager.py`, `app/services/presence_sweep.py`,
`app/services/wire.py`.
**Provider:** `provider_lib/admin_client.py`, `provider_lib/wire.py`.
**Protocol:** `docs/ws-protocol.md`.

- `POST /admin/api/providers/register`: validates token→definition→
  provider_type match, exact version match (409), machine pre-exists
  (404). Merges hardware, upserts instance, issues per-instance secret to
  Redis, returns config + fingerprint.
- Provider dials `/provider/ws` with `Bearer <secret>`; auth is
  constant-time vs Redis. On accept: bump `im:ws:epoch`, send
  `provider.hello`, claim `im:ws:owner`.
- `ConnectionManager.send_command(instance_id, type, payload, timeout=30)`
  awaits a matching `ack`. Frame envelope `{v,type,id,reply_to,epoch,ts,
  payload}`.
- Dead socket → instance `disconnected` (instant live path + presence
  sweep safety net). Reconnect backoff 1s→30s; old socket closed 4409.
- `ping`/`pong` keep presence alive.

Tests: `test_registration.py`, `test_ws_connection.py`.

---

## Phase 4 — Provider Lib Lifecycle + `/v1` Surface ✅

**Docs:** `provider/README.md`.
**Core:** `provider_lib/backend.py` (`BackendDriver` ABC +
`BackendLifecycle`), `provider_lib/app_factory.py`
(`create_provider_app`, `BackendOverrides`).

- `BackendDriver`: `start/stop/health/list_models/stream_responses`
  (+ optional `stream_chat_completions`, `aclose`).
- `BackendLifecycle`: state machine (STOPPED→STARTING→RUNNING→STOPPING),
  `acquire_slot`/`release_slot` tied to inbound connection lifecycle,
  emits `backend.status` per transition, `_owned_stream` releases slot on
  stream close/cancel/error.
- Provider app serves `/health`, `/v1/models`, `/v1/responses`,
  `/v1/chat/completions` on `PROVIDER_PORT`; translation layer normalizes
  the backend output to spec.
- Mock provider (`provider/mock`) implements the full contract with fake
  streaming.

Tests: `provider/lib/tests`, `provider/mock/tests` (lifecycle, admission,
stream release).

---

## Phase 5 — llama-cpp + Downloader + Machine Metrics ✅

**Provider:** `provider/llama-cpp/` (`LlamaCppBackend` driver,
`build_llama_command`, `CursorLogRing`, `main.py`).
**Lib:** `provider_lib/downloader.py`, `provider_lib/metrics.py`.
**Admin:** `app/services/metrics_service.py`.

- llama.cpp driver: subprocess management, log ring with cursor, health
  wait, artifact resolution (main/mmproj/draft), `--flash-attn on|off`.
- Model downloader emits `download.progress` (percent/bytes/speed).
- Machine metrics collectors: GPU (NVML/rocm-sysfs), os_ram/cpu (psutil),
  storage. Emitted only when the admin assigns ownership.
- Metrics ownership: `im:metrics:owner:{machine_uid}` lease (TTL 30s),
  assignment on connect, failover on expiry. `im:metrics:cats:{id}` holds
  declared categories.

Tests: `provider/llama-cpp/tests`, `test_metrics_ownership.py`.

---

## Phase 6 — Scheduler + `/v1/responses` ✅ COMPLETE

**Admin:** `app/services/scheduler.py`, `app/services/sse.py`,
`app/services/alias_registry.py`, `app/api/v1/responses.py`.
**Design:** ARCHITECTURE.md §6, §7; FINDINGS.md.

> **⚠ VRAM ledger semantic change (Phase 6 close-out).** VRAM is now
> accounted **per booted instance**, not per request slot. The previous
> ledger had `release` clear the `im:vram:used` entry, so a
> running-but-idle backend was counted as holding **0 VRAM** — physically
> false (a booted llama-server keeps its weights resident) and it made
> idle-backend eviction impossible (nothing recorded to free). The
> authoritative ledger is `self._booted: {instance_id: (machine_uid,
> vram_bytes)}`, merged with the Redis mirror and the DB's
> running/in_use×`vram_required_bytes` state. `release` now frees only
> the capacity slot; the hold persists until the backend is actually
> stopped by the idle reaper or an eviction. **Any tooling or mental
> model that read `im:vram:used` as "in-flight requests" must be
> updated: it is "loaded backends".**

### Done (tests green)
- `InferenceScheduler` — in-process per-alias FIFO + Redis mirror.
  `acquire`/`release`; `NoProviderAvailable`→503, `QueueTimeout`→504.
  Slot = capacity (+ machine free VRAM only when a boot is needed).
  Prefers already-running instances. Boots via `backend.start` under
  `im:sched:lock`. Remembers `_booted` to avoid redundant boots.
  `release` cancellation-safe (shielded) and does **not** clear the
  booted hold.
- **VRAM eviction (was `TODO(phase6-eviction)`)** — when a needed boot
  doesn't fit, **idle different-alias** backends on the same machine are
  stopped (`backend.stop`) **LRU-first** by `last_request_at` (null =
  oldest). Victims must have zero active in-process slots and a positive
  hold; the requesting alias's own instances are never victims. NAK
  (drain race) → next candidate; if nothing frees enough the request
  stays queued (no force-kill). Evict+boot share one `im:sched:lock`
  acquisition plus an in-process `_evicting` set so concurrent acquires
  can't double-target a victim. Logged at WARNING.
- **Idle-timeout reaper (was `TODO(phase6-idle-reaper)`)** — real
  periodic task (`IDLE_REAPER_INTERVAL_SECONDS`, default 15.0,
  injectable for tests), started by `start_background`, stopped by
  `stop`. Stops connected, loaded instances with zero active slots whose
  `last_request_at` (fallback `created_at`) is older than the
  definition's `idle_timeout_seconds`; `idle_timeout_seconds == 0` means
  never idle-stop. Clears ledger + mirror + DB status on success; a
  NAK/exception is retried next tick; each tick is try/except-guarded.
  Each tick also refreshes the `im:vram:used` TTL for this process's
  booted holds.
- **Cold-boot SSE keepalive** — the streaming `POST /v1/responses`
  response starts immediately and emits `: keep-alive` SSE comment lines
  every `KEEPALIVE_INTERVAL_SECONDS` (default 10) while
  `scheduler.acquire` blocks on a cold boot / FIFO wait, so Traefik
  doesn't kill long cold starts. **Error contract (API-visible):** the
  fast pre-stream checks stay HTTP for both transports (unknown/disabled
  alias → 404, missing fields → 400) and the cheap zero-candidates check
  stays a pre-stream **503** (never open a stream that can't respond);
  scheduler errors arriving *inside* the stream (`QueueTimeout`, or
  `NoProviderAvailable` after a race) surface as spec `response.failed`
  + `error` frames + `[DONE]` and a failed `ResponseRecord` — no HTTP
  status is possible once bytes are sent. The non-stream path keeps
  503/504 JSON unchanged. Client disconnect during the keepalive phase
  cancels the pending acquire (waiter dropped) and releases cleanly — no
  leaked slot or waiter.
- `SSEEmitter` — re-frames litellm events: replaces `response.id` with
  admin `resp_<uuid>` on lifecycle frames, reassigns `sequence_number`
  monotonically, passes usage/output[]/non-canonical events through,
  synthesizes `response.failed`+`error` on exception.
- `alias_registry.ensure_registered(alias)` — registers litellm native
  streaming alias (idempotent, process-cached).
- `POST /v1/responses` route — parse body, resolve definition, mint id,
  build litellm input from DB chain (prepend prior input+output items),
  ensure_registered, acquire, stream through emitter, persist
  ResponseRecord + TokenUsageSample, release in finally.
- Error taxonomy: `map_exception_to_error` → NotFound=invalid_request/
  model_not_found, else server_error/upstream_failed.
- Lifespan wires scheduler + presence sweep.
- Tests: `test_scheduler.py`, `test_sse_emitter.py`, `test_v1_responses.py`
  (admin suite green; Phase 6 close-out added eviction, reaper, ledger,
  and keepalive/error-contract coverage).

### Remaining for Phase 6
- [x] ~~**VRAM eviction**~~ — see **Done** above.
- [x] ~~**Idle-timeout reaper**~~ — see **Done** above.
- [x] ~~**Keepalive during cold boot**~~ — see **Done** above.
- [x] **Tool forwarding** — client `tools`/`tool_choice` are forwarded to
      litellm (`PASSTHROUGH_FIELDS`); tool *execution* is provider-side
      (the backend emits `function_call` items; the mock emits a canned
      one). An admin-side multi-turn tool-execution loop is a Phase 7
      concern, not Phase 6.
- [x] **Conformance gate** — ran 2026-10-04 against a local uvicorn
      admin + mock provider: **all 6 KEY tests pass**
      (`basic-response`, `streaming-response`, `system-prompt`,
      `assistant-phase`, `response-output-phase-schema`, `multi-turn`)
      plus `tool-calling` and `image-input`. The only applicable failures
      are `compact-response` / `compact-missing-model` — no
      `/v1/responses/compact` route exists (endpoint scope decision for
      Phase 7/8; not a Phase 6 blocker). Full table + fixes:
      `docs/integration-testing.md`.
- [x] **Non-stream path** — verified: spec default (`stream` absent or
      false) returns JSON with the admin-owned `resp_<uuid>` and persists
      identically to the stream path.

### Fixes landed during the docs/conformance pass
- **Provider reconnect wired (was a documented gap):** both
  `provider/mock` and `provider/llama-cpp` entrypoints now drive the
  admin socket with `AdminClient.run_forever()` alongside uvicorn
  (registration split into `register_provider` + persistent WS),
  re-emitting `provider.status` on every (re)connect; llama-cpp stops
  its metrics emitter on disconnect. Verified empirically: admin restart
  → provider reconnects with a bumped epoch. Proven by
  `provider/mock/tests/test_provider_reconnect.py`.
- **Presence keepalive pings:** `AdminClient` now sends a `ping` frame
  every 20s (`PING_INTERVAL_SECONDS`). Nothing sent them before, so idle
  provider connections were swept `disconnected` after the 60s presence
  TTL despite the protocol promising otherwise.
- **Spec default `stream=false`** on `/v1/responses` (admin route
  defaulted to true); provider `/v1/responses` honors it too (drains the
  driver stream, returns the terminal response as JSON).
- **Spec-clean output items:** admin strips litellm-injected
  `phase: null` / `logprobs: null` that the Zod schema rejects; the mock
  emits spec-complete response objects (`created_at`, `completed_at`,
  `tools`, sampling params, usage details) and `status` on added items.

Phase 6 is complete: eviction, the idle reaper, and cold-boot SSE
keepalive all landed (with the per-booted-instance VRAM ledger change
they depend on). The three previously-open items are now Done above.

### Phase 6 close-out review fixes
- **B1 — admission respects `_evicting`:** the eviction path holds the
  *requesting* alias's `im:sched:lock`, never the victim's, so a
  concurrent acquire for the victim's own alias could previously admit
  onto a backend mid-stop. `_try_admit` now skips any candidate instance
  present in `self._evicting` (both already-booted and needs-boot paths).
  Tests: `test_try_admit_skips_instance_in_evicting`,
  `test_concurrent_acquire_not_admitted_onto_eviction_victim`.
- **S1 — call-time `aresponses` errors are terminal:** the awaited
  `litellm.aresponses(...)` in `_stream_response` moved inside the
  inner try, so a connection-refused/APIError raised before the first
  event emits `response.failed` + `error` + `[DONE]` instead of
  truncating the SSE stream after keepalives (still released + persisted
  failed in the `finally`; not swallowed by the `SchedulerError`
  branch). Test: `test_stream_aresponses_call_time_error_is_terminal`.
- **S2 — out-of-band stops prune the VRAM hold:** a `backend.status` /
  `provider.status` frame reporting `stopped`/`error` now calls
  `scheduler.note_backend_stopped` from `connection_manager`
  (event-driven, idempotent), dropping the stale `_booted` hold + Redis
  `im:vram:used` field so the machine isn't permanently over-counted.
  `stopping` is excluded (weights still resident). Reached via
  `app.state.scheduler` — no import cycle. Tests:
  `test_external_stop_prunes_booted_hold_and_mirror`,
  `test_running_status_does_not_prune_booted_hold`.
- **N2 — eviction victims include `_booted` regardless of DB status:**
  `_eviction_candidates` no longer filters on DB `running/in_use`; the
  merged-ledger positive-hold requirement is what makes an instance
  evictable, so a stale-"stopped" `_booted` hold can be reclaimed.
  Test: `test_eviction_can_reclaim_stopped_status_booted_instance`.
- **N1:** removed the stale unchecked "Keepalive during cold boot" TODO
  (already Done above).

---

## Phase 7 — Chat Completions + Models + Stubs ✅

**Admin:** `app/api/v1/chat_completions.py`, `app/api/v1/models.py`,
`app/api/v1/stubs.py`, `app/services/alias_registry.py` (mode change).
**Provider:** `provider_lib/app_factory.py` (chat non-stream aggregation),
`provider/mock` (final usage chunk on chat stream).
**Design:** ARCHITECTURE.md §7 (+ "Chat completions specifics").

### Done
- [x] `POST /v1/chat/completions` via `litellm.acompletion` against the
      same scheduler admission + provider port. OpenAI chat body:
      `model`/`messages` required, `stream` default false; passthrough:
      temperature, top_p, max_tokens, tools, tool_choice, stop,
      presence_penalty, frequency_penalty, seed, n, user,
      response_format, logprobs, top_logprobs.
- [x] **Native chat streaming registration (the key litellm finding).**
      `ensure_registered` now registers the union entry
      `mode: "chat"` + `supports_native_streaming: True` (was
      `mode: "responses"`). Source evidence (litellm 1.103.x):
      `aresponses` decides fake-vs-native streaming *only* via
      `supports_native_streaming` (`OpenAIResponsesAPIConfig.should_fake_stream`
      → `litellm.utils.supports_native_streaming`); `mode` is never read
      on the responses dispatch path. But `acompletion` **bridges to the
      Responses API whenever `model_cost[alias]["mode"] == "responses"`**
      (`main.py` `responses_api_bridge_check` + bridge dispatch) — the
      Phase 6 registration would have sent chat traffic to
      `/v1/responses`. Probed empirically against a toy dual-endpoint
      server: with `mode: "chat"`, aresponses stream+non-stream still hit
      the provider's `/v1/responses` natively (no Phase 6 regression)
      and acompletion hits `/v1/chat/completions` with real chunks +
      usage. Chat route additionally passes `_skip_responses_api_bridge=True`
      so litellm's gpt-5 conditional bridge (reasoning+tools on
      OpenAI-looking endpoints) can never reroute our calls.
- [x] **Chat id ownership**: admin mints `chatcmpl-<uuid>` and replaces
      `id` on every streamed chunk and the non-stream response.
      litellm passes the raw upstream chat id through un-wrapped (unlike
      the base64 responses id); captured as `parameters.litellm_id`.
- [x] **SSE pass-through**: data-only frames (no `event:` lines per chat
      spec), `data: [DONE]` terminator. Admin forces
      `stream_options.include_usage` so the stream carries a final usage
      chunk for token persistence.
- [x] **Persistence**: `ResponseRecord` with
      `parameters={"api_format": "chat_completions", "litellm_id": ...,
      "request_parameters": {echo}}`; request `messages` → `input_items`;
      aggregated assistant message(s) (content + tool-call deltas merged
      by index) → `output_items`; `previous_response_id` always NULL
      (chat is stateless). `TokenUsageSample` from usage
      (`prompt_tokens_details.cached_tokens` mapped onto the
      responses-shaped `input_tokens_details` that `persist_turn` reads).
      Cancellation-safe finally mirrors Phase 6 exactly: client
      disconnect before the terminal chunk → slot released +
      `status="failed"`, `error.code="client_disconnected"`.
- [x] **Error mapping**: pre-stream → HTTP (400 bad body/messages,
      404 unknown/disabled alias, 503/504 scheduler); non-stream upstream
      failure → 502 JSON; mid-stream → chat-style
      `data: {"error": {...}}` + `[DONE]` (not `response.failed`).
- [x] `GET /v1/models` — enabled `ProviderDefinition`s, alias asc;
      `id`=alias, `owned_by`=provider_type, `created`=int epoch from
      `created_at`, `model_metadata` merged over the rest (core fields
      authoritative). No scheduler involvement.
- [x] 501 stubs (`app/api/v1/stubs.py`) with the OpenAI error envelope
      (`type: not_supported_error`, `code: endpoint_not_supported`,
      `param`: path) for POST `/v1/embeddings`, `/v1/completions`,
      `/v1/rerank`, `/v1/moderations`, `/v1/decisions`,
      `/v1/audio/{speech,transcriptions,translations}`, and
      GET/POST/DELETE(+content/cancel) item routes for `/v1/files` and
      `/v1/batches`. Hidden from the OpenAPI schema.
- [x] Routers wired in `app/api/main.py` (public /v1).
- [x] Provider-port chat non-stream: `app_factory.py` now drains the
      driver chunk stream and returns an aggregated `chat.completion`
      JSON when `stream` is falsy (same fix class as Phase 6's responses
      non-stream; litellm's non-stream parse otherwise fails against an
      always-SSE upstream). Mock emits a final `usage` chunk on its chat
      stream (honors `include_usage`).
- [x] Tests: `test_v1_chat_completions.py` (stream happy path + tool
      delta merge, non-stream + default, mid-stream error framing,
      generator-aclose disconnect, 404/400/503, acquire/release-once
      spy, native-streaming registration probe), `test_v1_models.py`,
      `test_v1_stubs.py`. Admin suite **108 passed** (72 baseline + 36
      new, no Phase 6 regression). lib 48 (+2 chat non-stream/aggregate),
      mock 8, llama-cpp 29.
- [x] Conformance re-run 2026-10-04 (local admin + mock provider):
      `basic-response`, `streaming-response`, `system-prompt`,
      `multi-turn` — **4/4 pass**; Phase 6 not regressed by the
      `mode: "responses"` → `mode: "chat"` registration change.

### Deviations
- Chat turns share the `responses` table (`ResponseRecord`) rather than a
  new table — the column set (input/output items, usage, status,
  parameters) fits; `api_format` distinguishes rows.
- The Phase 7 checklist item "generate-client.sh after routes land" is
  **deferred to Phase 10** (hook disabled per plan); no frontend touched.
- `ensure_registered` keeps a single process-wide cache; switching an
  already-registered alias's mode only happens on process restart — fine
  because ALL aliases use the same union entry.

---

## Phase 8 — Provider Type Ports ✅

Port remaining engines from `legacy/agent/services/*` to
`provider/<type>/`, overriding only what differs from the lib. **All
three packages are fake-driver-tested only (no real NPU/ROCm/gufo
hardware in this environment); real-hardware validation is deferred.**

- [x] **gufo** (`provider/gufo`, `matrix-provider-gufo`) — `gufo serve
      llm` argv map (`VALUE_FLAGS`/`BOOL_FLAGS`/`cache_disk` per-instance
      dir), per-request `model` forwarded as-is (multi-model), native
      OpenResponses passthrough, aux spec files (mmproj/dflash/dspark/
      mtp) resolved via `ensure_artifact`, rate-gauge parsing
      (`llamacpp:*_tokens_seconds` gauges scraped from `/metrics` and
      merged into terminal `usage.completion_tokens_details` before the
      event is yielded), effective capacity = `options.sessions`.
- [x] **halogen** (`provider/halogen`, `matrix-provider-halogen`) —
      two ports (api = PROVIDER_PORT+1, engine = +2), env-configured:
      12-option `HALOGEN_*` map + fixed wiring (CHECKPOINT/TOKENIZER/
      BIND/ENGINE/ports), capacity from `kv_slots` (reported as
      `effective_capacity` in the `backend.start` ack), `stdbuf`
      entrypoint spawn with merged stdout, SSE proxy from the API port.
- [x] **halogen-flash** (`provider/halogen-flash`,
      `matrix-provider-halogen-flash`) — native `/v1/responses`
      passthrough, 43-key `HALOGEN_*` env map, **static ports for the
      disk-cache fingerprint** (explicit in backend_config, else
      SHA-256(MACHINE_UID) into 8200–8289 adjacent pairs), **NPU
      small-model pinning** (ported `npu_models` suffix/pins logic +
      `npu_probe` device/XRT/engine checks; no NPU → env omitted
      gracefully, never an error; probe result reported in registration
      hardware under `npu`), and the **canonical `calculate_usage`
      override** (`usage.py`: normalizes chat-style counts / native
      timing chunks / partial details / missing usage into the spec
      usage dict before the terminal event is yielded; reported zeros
      trusted, char//4 estimate only when no count keys at all; rates
      land in `completion_tokens_details`).
- [x] Each: Dockerfile in `provider/<type>/` (repo-root build context,
      `uv sync --frozen --package matrix-provider-<type>`; engine binary
      layered by the real-engine image per deployment.md), uv workspace
      members registered in root pyproject, commented-out compose
      examples, log ring + `backend.logs` forwarding, version gate
      (`VERSION = "dev"`), exact `PROVIDER_TYPE` strings.
- [x] Tests (fake-backend subprocess pattern from llama-cpp): gufo 34,
      halogen 38, halogen-flash 97. Existing suites unchanged:
      admin 109, lib 48, mock 8, llama-cpp 29.

### Deviations
- The legacy `GUFO_MAX_INSTANCES` / `HALOGEN_MAX_INSTANCES` multi-
  server-per-agent limits are **dropped**: the new model is exactly one
  backend per provider instance (ARCHITECTURE.md invariant 1).
- Legacy allocate-and-persist Flash ports in the admin DB are replaced
  by deterministic `SHA-256(MACHINE_UID)` derivation into the static
  8200–8289 range (provider has no DB); explicit `api_port`/
  `engine_port` in `backend_config` still win, which is where the
  Phase 9 fingerprint flow can persist an allocation.
- `calculate_usage` is wired into `BackendOverrides.calculate_usage`
  for visibility, but the authoritative call site is the driver's
  stream generator (the generic `/v1` layer never recomputes usage).
- NPU pins-file downloads (`_prepare_npu_models`) are surfaced as
  `resolve_npu_download_set()` + suffix maps in `npu.py`; the actual
  per-file download wiring **landed with Phase 9** (`driver.
  _prepare_npu_models()` is called from `start()` when the probe
  passes, `ensure_artifact`s each pinned file into
  `MODELS_DIR/npu/<id>/` with the record's revision before spawn, and
  failure degrades to omitting `HALOGEN_NPU_MODELS` — never a start
  failure). [CLOSED]
- `metrics.inference` (ARCHITECTURE.md §8, reserved frame kind) did
  **not** land in Phase 8 — the gufo rate gauges feed per-request
  `TokenUsageSample` persistence instead of the reserved always-on
  telemetry frame. It remains reserved for a future phase.

---

## Phase 9 — Config Update / Fingerprint / Cache ✅

**Admin:** `app/api/admin/machines.py`, `app/api/admin/definitions.py`,
`app/api/admin/instances.py`, `app/services/config_update.py`,
`app/api/ws.py` (connect-path heal), `app/services/presence_sweep.py`
(sweep heal). **Provider lib:** `provider_lib/config_update.py`
(shared `provider.config.update` / `cache.clear` /
`storage.prune_unused` handlers + the `CACHE_DIR/prompt_cache/<fp>/`
convention), `BackendDriver.apply_config` ABC hook +
`resolved_artifacts` contract, `downloader.ensure_artifact` gained
`revision` + `local_subdir`. **Docs:** `docs/ws-protocol.md` §4,
`provider/README.md` ("Phase 9" section).

- [x] `provider.config.update` command → provider enters `initializing`:
      capacity adopt (no restart, before the gates), fingerprint compare
      (noop ack on match — safely ackable under load), atomic drain
      (`stop_if_idle()` NAKs `backend_in_use` + `retry_after` while in
      flight), stop backend, **auto-clear
      prompt cache only** (old fp dir + engine cache dirs; never model
      files), `driver.apply_config`, start backend (artifact
      downloads/`download.progress` flow), scrape `list_models()`, ack
      `{"config_fingerprint", "capacity", "model_metadata",
      "prompt_cache_deleted", "prompt_cache_bytes_freed"}`.
      Per-step failure → NAK `{"step": "...", "error": "..."}` +
      `provider.status error`.
- [x] Admin push + persistence: PATCH `backend_config` or `capacity`
      (provider-visible fields) pushes concurrently to all connected
      instances
      (300s timeout), updates `ProviderInstance.config_fingerprint` on
      ok ack, persists discovered `model_metadata`
      (`{"models": [...]}`) onto the definition; failed provider update
      is reported in the PATCH body (`config_update_results`), not fatal
      to the admin row. **Retry policy:** drain-refused only, 3 attempts,
      10s apart (settings `CONFIG_UPDATE_*`); other failures reported
      immediately.
- [x] Stale-fingerprint **self-heal**: WS connect background task +
      presence sweep compare the instance fingerprint to the definition's
      and auto-push on mismatch **to the stale instance only**, under a
      per-instance in-flight push guard (cheap no-op on match).
- [x] `cache.clear` — prompt-cache dirs only; ack `{deleted,
      bytes_freed}`, `dry_run` supported. Admin endpoint
      `POST /admin/api/instances/{id}/cache/clear`.
- [x] `storage.prune_unused` — deletes `MODELS_DIR` files not in
      `driver.resolved_artifacts` (recorded during start; referenced
      dirs protect subtrees); **refuses** when the reference set is empty
      (never deletes blind); `dry_run` supported. Admin endpoint
      `POST /admin/api/instances/{id}/storage/prune`.
- [x] Admin CRUD: `POST/GET/PATCH/DELETE /admin/api/machines` (uid
      immutable; DELETE refused 409 while instances attached) and
      `/admin/api/definitions` (light admin-side validation:
      `provider_type` ∈ PROVIDER_TYPES, capacity ≥ 1, vram/idle ≥ 0,
      JSON-serializable `backend_config`; deep validation stays
      provider-side; DELETE allowed only with **no** websocket-connected
      instance, else 409 suggesting disable; offline instance rows
      cascade-delete with the definition).
- [x] `ensure_registered(alias)` called on definition create **and**
      update so litellm aliases are warm before first use.
- [x] All five providers wire `install_config_handlers` (mock/llama-cpp/
      gufo/halogen/halogen-flash); gufo adds its `--cache-disk` dir and
      halogen-flash adds `HALOGEN_CACHE_DIR` as `extra_cache_dirs`.
- [x] NPU pre-download wiring for halogen-flash (closes Phase 8
      deviation #4): pins planning → `ensure_artifact` per file into
      `MODELS_DIR/npu/<id>/` before spawn; graceful omission on any
      planning/download failure.
- [x] Tests: admin 140 (31 new: `test_admin_machines.py`,
      `test_admin_definitions.py`, `test_config_update_flow.py` —
      includes a real-WS PATCH→frame→ack round-trip, drain-retry,
      reconnect self-heal, capacity-only push, heal-stale-instance-only +
      in-flight guard, explicit-null 422, provider_type-change refusal),
      lib 71 (+23: `test_config_update.py`, `test_cache_clear.py`,
      `test_prune.py`, `stop_if_idle` lifecycle tests), mock 11 (+3),
      llama-cpp 31 (+2), gufo 36 (+2), halogen 40 (+2),
      halogen-flash 101 (+4, NPU prep). All fake/local-tested; no real
      hardware involved.
- [x] Live validation 2026-10-04 (local uvicorn admin :8000 + mock
      provider :8081, dev DB): register+connect → PATCH
      `backend_config` (`delta_count` 3→7→5) observed over the real WS
      (provider went initializing→running, acked new fingerprint, DB
      instance fingerprint + definition `model_metadata` updated) →
      `POST /v1/responses` still works and reflects the new config
      (`output_tokens` tracks `delta_count`) → `cache.clear` dry-run +
      real via the admin endpoint deleted only the prompt-cache dir
      (`provider_config.json` untouched) → kill provider, PATCH while
      down (no push), force the DB row stale, presence sweep healed it
      back to the current fingerprint.

### Phase 9 review fixes (2026-10-04)
- **SF-1 (racy drain)**: `BackendLifecycle.stop_if_idle()` performs the
  busy check + STOPPING transition atomically under the lifecycle lock
  (stop body extracted to `_stop_locked()`; `BackendBusy` now carries
  `in_flight`). `config_update` uses it — a `/v1` acquire can no longer
  slip between the check and a SIGTERM under a live stream.
- **SF-2 (capacity-only PATCH)**: the provider adopts a differing
  `capacity` **before** the noop/drain gates (no restart; ack
  `capacity_adopted`); the admin PATCH triggers a push on fingerprint
  **or** capacity change. `idle_timeout_seconds` is NOT a push trigger —
  the provider consumes no idle setting (admin-side reaper
  `InferenceScheduler._idle_reaper`); documented.
- **SF-3 (heal storm / fan-out)**: provider order swapped (noop check
  before drain refusal — same-fp is always safely ackable); admin heal
  pushes to the **stale instance only** (not the definition fan-out;
  the PATCH flow keeps the fan-out); per-instance in-flight guard
  (module-level `set` in `app/services/config_update.py` — single
  uvicorn worker makes in-process authoritative; always cleared in a
  `finally`).
- **SF-4 (explicit null → 409)**: PATCH with explicit `null` on any
  non-nullable column (`alias`, `provider_type`, `registration_token`,
  etc.) is rejected with a clear 422 "field cannot be null"; the setattr
  loop only applies validated non-null fields.
- **N-1**: unused `_STEP_PRE_START` / `_STEP_STOP` constants removed.
- **N-2**: `backend.stop` docs softened — the command has **no** forced
  drain (in-flight streams release as the upstream closes); real drain
  semantics are `provider.config.update` only.
- **N-3**: PATCH changing `provider_type` while instances are attached →
  409 (old containers would keep a silently broken binding).
- **N-4**: `cache.clear` refuses while `backend_status == IN_USE`
  (NAK `backend_in_use` / step `drain`) unless `force: true`; `dry_run`
  always allowed. Admin endpoint accepts `force`.
- **N-5**: prune reference matching realpath-normalizes both the
  referenced set and walked paths (`..`/`//`/symlinked mounts can no
  longer cause a live file to be deleted).
- **N-8**: prune admin timeout raised to 300s (`PRUNE_TIMEOUT_SECONDS`).
- **N-6 / N-7**: comments/doc notes only (transient double-PATCH
  self-heals; fingerprint means "adopted", not "running").

### Deviations
- The Phase 9 checklist's "stop backend → `running`" final transition is
  implemented as start→`running` (the stop happens before the apply);
  the provider ends in `running` as intended.
- `backend.metadata` (reserved event) was **not** used: discovered
  metadata rides in the `provider.config.update` ack detail instead,
  which keeps the flow request/response-ordered (no cross-frame race).
- Config-update results are surfaced synchronously in the PATCH body;
  there is no async job table (the provider has no DB and the admin stays
  stateless-per-request; the self-heal covers missed pushes).

---

## Phase 10 — Admin UI Rework ✅

**Admin reads (new):** `app/api/admin/responses.py` (response log, usage
stats, dashboard overview) + `GET /admin/api/instances[/{id}]` added to
`app/api/admin/instances.py`. **Frontend:** full rebuild of
`admin/frontend` around the new data model on the regenerated SDK
(`AdminService`, `ResponsesService`, `ChatService`, `ModelsService`).

- [x] **Machines** page (`/machines`): table (uid, name, address,
    total VRAM, instance count) with expandable merged-hardware view
    (gpus/cpu/ram). Create + Edit dialogs (uid disabled/immutable on
    edit), Delete with inline 409 "instances attached" surfacing.
- [x] **Provider Definitions** page (`/definitions`): table (alias,
    provider_type, enabled, capacity, vram req, idle, connected/total
    instances) + expandable detail (copyable masked registration token,
    config fingerprint, discovered `model_metadata`, full `backend_config`,
    instance list). Create/Edit form: provider-type Select, numeric
    validation (capacity ≥ 1, vram/idle ≥ 0), **backend_config JSON
    editor** validated on submit with per-type schema hints + "Load
    example" (mock/llama-cpp/halogen/halogen-flash/gufo, from
    `provider/README.md`). PATCH that changes `backend_config`/`capacity`
    surfaces `config_update_results` (ok/noop/error per instance) in the
    toast. Delete surfaces the 409 (connected) → "disable instead".
- [x] **Provider Instances** page (`/instances`): table (machine uid,
    alias, dual status, ws badge, version, port, epoch, last_seen,
    last_request_at, config fingerprint). Row actions: **Clear cache**
    (dry-run toggle + force) and **Prune storage** (dry-run preview of
    files + bytes) with result rendering from the ack detail.
    **Live logs skipped** — `backend.logs`/`provider.logs` are WS-only;
    no REST read endpoint exists (noted in the page).
- [x] **Responses / Usage** page (`/responses`): paginated recent
    `ResponseRecord`s (response_id, model, chat/responses format,
    status + error, in/out/total tokens, previous_response_id chain,
    created) + a Token-usage tab (all-time totals, avg rates, CSS-bar
    chart of prompt/cached/completion per recent request — no charting
    dep added).
- [x] **Dashboard** (`/`): overview cards (machines, definitions,
    connected instances, live active/queued), instance status strip,
    token-usage summary, recent responses, link tiles.
- [x] **Playground** (`/playground`): minimal `/v1/responses` +
    `/v1/chat/completions` streaming panel (SSE deltas + event chips)
    against any enabled alias.
- [x] **Settings** (`/settings`): admin VERSION (from `/admin/api/health`),
    connected-provider + enabled-alias counts, provider-env deploy
    pointer to `deployment.md`.
- [x] Regenerated the API client (`scripts/generate-client.sh`) and
    **re-enabled the `generate-frontend-sdk` pre-commit hook** in
    `.pre-commit-config.yaml` (was deferred from Phase 7).
- [x] Removed the old UI: routes `agents`, `server-instances`,
    `benchmarks`, `models`, `files`, `audio`, `chat`, `completions`,
    `embeddings`, old `responses`; components `Agents/`, `Benchmarks/`,
    `Files/`, `Models/`, `Pending/`, `Queue/`, `ServerInstances/`; dead
    hooks (`useQueueStatus`, `useLogFeed`, `useAgentEvents`,
    `useTokenStats`, `logMerge`) and libs (`benchmarkApi`, `gpuMetrics`).
    Kept `ui/`, `Common/`, `Sidebar/`, `theme-provider`. Sidebar nav
    rewritten to the new page set; no dangling imports (build clean).
- [x] TanStack Query polling (3–5s) on dashboard + instances +
    definitions; mutations invalidate the right keys; loading/empty/error
    states throughout; dark-mode-aware status badges via the existing
    theme-provider.

### Deviations
- **New read endpoints added** (Phase 9 shipped CRUD + actions only, no
  list reads for the UI): `GET /admin/api/instances[/{id}]`,
  `GET /admin/api/responses`, `GET /admin/api/stats/usage`,
  `GET /admin/api/stats/overview`. Read-only; the overview reads the
  scheduler's Redis mirror for live queue/active counts (best-effort,
  observability only).
- **Live logs not shown**: `backend.logs`/`provider.logs` flow over the
  WS only; there is no REST read endpoint, so the Instances page omits a
  log stream rather than inventing one (per task scope).
- **Usage chart is CSS bars**, not a charting library — the repo had no
  charting pattern; kept the bundle lean per the task's guidance.
- **backend_config is a validated JSON editor with per-type examples**,
  not a per-field form — provider-specific deep validation stays
  provider-side (ARCHITECTURE.md §4 / provider/README.md).

---

## Phase 11 — Docs Consolidation + E2E ✅

**Completed 2026-10-05.** Dead-code removal, self-seeding dev bootstrap,
doc consolidation, and the final full test + conformance gate.

### Dead code removed
- `legacy/` (whole tree — pre-overhaul backend/agent/recipes; lives in
  git history only). References purged from `.pre-commit-config.yaml`,
  root `pyproject.toml` (typos exclude), `.dockerignore`, and all docs.
- Old-architecture scripts: `scripts/test.sh`, `scripts/test-local.sh`,
  `scripts/bombard_chat_completions.py`,
  `scripts/probe_lease_release_disconnect.py`,
  `scripts/proxy_connection_leak_probe.py`,
  `scripts/test_responses_keepalive.py`, `debug-container.sh`.
- FastAPI-template leftovers: `hooks/post_gen_project.py`,
  `.fastapicloudignore`, `packages/react-email/` (+ the `packages/*`
  workspaces entry in root `package.json` — no email feature exists in
  the trusted-LAN model).
- **Kept (live tooling):** `scripts/prepare_release.py` (referenced by
  the release workflows), `scripts/add_latest_release_date.py`
  (pre-commit hook), `scripts/generate-client.sh`,
  `.claude/skills/` (library-skills agent-integration symlinks into
  `.venv`, not a duplicate of `.agents/skills`), `img/` (README
  screenshots).
  - `scripts/local/run-deployment.sh` is a **gitignored** personal
    SSH/deploy helper (under `scripts/.gitignore`), not tracked repo
    tooling — it was never part of the committed tree.

### `scripts/dev.sh` — one-command local dev
- Auto-detects **docker mode** (compose: postgres + redis + admin +
  provider-mock) vs **local-processes mode** (migrations via
  `prestart.sh`, `uvicorn` admin :8000, mock provider :8081 against a
  local Postgres + Redis).
- Seeds via the **admin API** (idempotent, 409-tolerant): Machine
  `mock-machine-1` + ProviderDefinition `mock-model`
  (`mock-registration-token`, capacity 4).
- Smoke test: polls instance `websocket_connected`, then asserts
  `response.completed` in a streamed `/v1/responses`.
- `up` / `status` / `down` subcommands; logs + PIDs in `.dev-run/`
  (gitignored); re-run while up detects and reports (no double-start).
- Verified in this environment (local-processes mode): up PASS → status
  shows both RUNNING + `ws=True` → re-run reports already-up → down stops
  both cleanly.

### Doc updates
- `development.md`: dev.sh quick start replaces the "until Phase 10"
  manual seeding section (manual curl/psql kept as reference); provider
  test env-var requirement documented.
- `AGENTS.md`: dev.sh added to Commands; test.sh/test-local.sh warning
  removed; legacy/ sentences rewritten to "removed; lives in git
  history"; litellm registration note corrected to `mode: "chat"`.
- `ARCHITECTURE.md`: §2 tree drops `legacy/`; header reworded.
- `README.md`: Quick start now `./scripts/dev.sh`; endpoint table
  updated (chat/completions + models shipped); layout section refreshed.
- `CONTRIBUTING.md`: rewritten for this repo (template text + stale
  architecture references removed).

### Final validation gate (2026-10-05)
- Admin **156 passed** · lib **71** · mock **11** · llama-cpp **31** ·
  gufo **36** · halogen **40** · halogen-flash **101** (all baselines).
- Frontend `bun run build` + `bun run lint` clean.
- `ruff check` + `ruff format --check` clean over provider,
  admin/backend/app, admin/backend/tests, scripts/.
- OpenResponses conformance vs `./scripts/dev.sh` stack: **8 passed /
  7 WS N/A / 2 compaction failed** — all KEY tests green
  (basic/streaming/system/assistant-phase/output-phase/multi-turn +
  tool-calling + image-input); the 2 failures are the accepted
  `/responses/compact` scope decision. Run log:
  `docs/integration-testing.md`.

---

---

## Phase 12 — Schema-driven backend config ✅

**Completed 2026-10-06** (commits: docs `e4a56a2`, B `94565e5`, C `4f1894d` +
`078ff10`, D1 `391a6bf`, D2 `347ca13`, D3 `7858e70`, D4 `e6c1332`, E `2c5fa9a`).
Moves `backend_config` from a free-form JSON blob (with
a hand-maintained doc) to a **JSON Schema (2020-12)** owned by each
provider package and registered in the admin under a new `ProviderType`
table. The schema is the single source of truth for admin validation,
the UI form render, and the HuggingFace file picker. Docs updated first
(this section, `ARCHITECTURE.md` §4/§5, `docs/ws-protocol.md` §2/§4,
`provider/README.md`, `admin/backend/docs/redis-keys.md`, `AGENTS.md`).

**Decisions locked:**
- Format: standard JSON Schema 2020-12.
- Bootstrap: first registration of an unknown type creates the
  `ProviderType` row (committed immediately).
- Consensus: a new schema must be presented by **all known
  `ProviderInstance` rows of the type** before it commits.
- While pending: registration **refused 409 `schema_pending`**; agent
  enters `waiting_schema` and retries via normal backoff.
- Force-commit (operator override) affects **future registrations +
  new/edited definitions only** — no forced fleet convergence.
- Existing `backend_config` rows that don't validate: **report-only**
  prestart migration check (UI banner), never auto-mutated.
- Raw-JSON editor toggle **kept** alongside the schema-driven form.
- UI renderer: `@rjsf/core` + `@rjsf/validator-ajv8`, custom theme.

### Sub-tasks

- [x] **A. Docs** — this section + ARCHITECTURE/WS-protocol/provider README/redis-keys/AGENTS.
- [x] **B. Admin registry** — `ProviderType` table + Alembic migration
      (`provider_types`, plus `ProviderInstance.reported_schema_fingerprint`);
      `providers.py` accepts `schema` + consensus gate (409
      `schema_pending`/`schema_conflict`, structured detail);
      `definitions.py` validates `backend_config` with
      `jsonschema.Draft202012Validator` on create/PATCH (422 per-field);
      `/admin/api/provider-types*` endpoints (list/get/commit/dismiss);
      prestart report-only migration check. `PROVIDER_TYPES`
      constant removed. Tests: consensus matrix, bootstrap, validation, overrides.
- [x] **C. HF proxy** — `admin/backend/app/api/admin/huggingface.py`
      ported from legacy `huggingface.py` (search + files with sizes +
      `cardData.gguf` quantization; generalized beyond GGUF-only to
      `.hgn`/tokenizer dirs). Tests with mocked HF API.
- [x] **D. Provider schemas** — `provider_lib/schema.py` (load bundled
      `schema.json` via importlib.resources, fingerprint, send in
      register); `admin_client.py` surfaces `schema_pending`/`conflict`
      as `SchemaPendingError` (readable vote message); shipped
      `schema.json` for all five providers — llama-cpp (from the
      upstream server README flag list), halogen-flash (from HALOGEN_*
      FLAGS.md — full list, unwired flags `x-supported: false`),
      halogen, gufo, mock. Sectioned `backend_config` shape with
      `command.py`/`env.py` flatten adapters.
- [x] **E. Frontend** — rjsf + custom Tailwind/Radix theme;
      collapseable-section `ObjectFieldTemplate`; `HfFileWidget`
      (picker dialog over C's endpoints, `x-widget: hf-file`);
      `definitions.tsx` form replaces the textarea (raw-JSON toggle kept);
      Provider Types page (schema viewer + consensus banner +
      commit/dismiss); `waiting_schema` badge on Instances.
      `scripts/generate-client.sh` regenerated.

### Implementation deviations (deliberate, reviewed)
- **Full config restructure** (operator decision): `backend_config` is
  now NAMED COLLAPSEABLE SECTIONS per provider, not the old flat
  `args{}`/`options{}` blob. Drivers read the sectioned shape via a
  `flatten_args`/`flatten_options` adapter (legacy flat input still
  honored as a fallback). This is a **breaking migration**: old flat
  configs fail schema validation and are flagged by the report-only
  prestart check; fix them in the UI.
- **TRANSITION path (schema-omitted)**: `RegistrationRequest.schema` is
  optional until every provider ships a real schema. Omitted + unknown
  type → permissive `{"type":"object"}` bootstrap; omitted + known type
  → gate skipped, `reported_schema_fingerprint=None`. Remove once all
  providers ship `schema.json` (documented in ws-protocol §2).
- **llama-cpp default changes** (upstream parity): `gpu_layers` 35→
  `auto`; `flash_attn` now tri-state `on|off|auto` (AGENTS.md
  corrected); `batch_size` default reconciled to 2048 (schema == driver).
  **Operator action needed:** audit existing llama-cpp definitions'
  `vram_required_bytes` — `auto` offloads more layers than the old 35.
- **halogen/halogen-flash ATOMIC port resolution**: a port pair is
  honored only when BOTH come from the same source (the `networking`
  section or the legacy top-level keys); a lone half-pair derives the
  full default pair. Prevents cross-source mixes that would shift the
  on-disk prompt-cache fingerprint.
- **gufo `cache_disk` directory** keyed by the real registered
  `instance_id` (MACHINE_UID only as a pre-registration fallback), so
  two gufo definitions on one machine no longer share a cache dir.
- **Frontend stack correction**: the plan said "Chakra theme" — the app is
  actually Tailwind 4 + Radix (shadcn-style), so a custom Tailwind/Radix
  rjsf theme was built (no Chakra/MUI package installed).
- **Secret handling**: `x-secret` fields are write-only — stripped from
  the detail view and the raw-JSON seed (never rendered in DOM),
  `restoreSecrets` on both form and raw submit (empty = keep current,
  type a value = rotate). No "remove secret" affordance by design.
- **No-fingerprint-churn saves**: `pruneUntouchedDefaults` ensures saving
  an untouched definition emits byte-identical `backend_config` (rjsf's
  default materialization is pruned against the stored config).
- **Fingerprints pinned** in each provider's `test_schema_sections.py`
  (the fleet consensus contract): llama-cpp `0af969e6…`,
  halogen-flash `7d2a8f49…`, halogen `07e4a3b4…`, gufo `3e5882d7…`,
  mock `19144f76…`. Any deliberate schema edit must bump the pin and
  coordinate fleet re-registration.
- The `x-flag`/`x-widget`/`x-numeric-effect`/`x-secret`/`x-supported`
  keywords are additive UI/validation hints; the driver still maps real
  flags in `command.py`/`env.py`.
- **Not done**: the llama-cpp enum-drift guard that parses `llama-server
  --help` at runtime (the schema enums are hand-verified against the
  upstream README; a CI smoke test against the built binary is a
  follow-up nicety, not blocking).

**Test totals after Phase 12:** admin **245** · provider/lib **87** ·
mock **23** · llama-cpp **60** · halogen-flash **139** · halogen **66** ·
gufo **66** (all with required env; a few `build_app()`/`ProviderSettings`
tests need `MACHINE_UID`/`PROVIDER_REGISTRATION_TOKEN`/`ADMIN_BASE_URL`
exported — pre-existing pattern). Frontend `bun run build` + biome lint
clean. Live dev-stack (`./scripts/dev.sh`) round-trip verified the mock
registers through the real gate with the committed fingerprint.

**Gotchas:** `flash_attn` is now `on|off|auto` (upstream default `auto`)
— AGENTS.md corrected. Do NOT import provider packages from the admin or
vice versa (the schema files are data, not code). The `x-flag` /
`x-widget` / `x-numeric-effect` / `x-secret` / `x-supported` keywords
are additive UI/validation hints; the driver still maps real flags in
`command.py`/`env.py`.

---

## Phase 13 — Server + backend log views ✅

**Completed 2026-10-06** (commits: F1+F2 `aac73e5`, F3 `536025c`).
Fills in the `backend.logs` / `provider.logs` frame kinds
(previously reserved) with a Redis-backed tail and UI log views.

**Decisions locked:**
- Storage: **Redis-only** (`im:logs:backend:{id}`, `im:logs:provider:{id}`),
  ~2000-line cap, 1h TTL. Never Postgres.
- Provider ring buffer; flush throttle ~1s / 100 lines per
  frame; `dropped` cumulative counter.
- `backend.logs.get` command for catch-up after admin restart /
  long disconnect (Redis tail may lag the provider ring).

### Sub-tasks

- [x] **F1. Provider capture** — consolidated the four duplicated
      per-provider `log_ring.py` into one canonical
      `provider_lib.log_ring.CursorLogRing` (seq-addressable, `ts`,
      `dropped` on eviction, `threading.Lock`); added
      `provider_lib.log_stream.LogStreamer` (batched ~1s / ≤100 lines
      per `backend.logs` frame, cursor resume across stop/start) +
      `ProviderLogHandler` (captures the provider's own logger →
      `provider.logs`, reentrancy-guarded, no recursion) +
      `install_log_streaming()` wired into every provider `main.py`
      (start on connect, stop on disconnect). `backend.logs.get` command
      handler replies with ring + seq + dropped.
- [x] **F2. Admin storage + read** — `connection_manager` consumes
      `backend.logs`/`provider.logs` (best-effort, tolerates the legacy
      single-line payload); `app/services/log_store.py` LPUSH+LTRIM(2000)
      +EXPIRE(1h) with a per-entry monotonic **ingest seq** (Redis
      `INCRBY im:logs:seq:{id}`, shared across both lists so `kind=all`
      merges on one cursor); `GET /admin/api/instances/{id}/logs?kind=
      backend|provider|all&since=&limit=` returns
      `{entries (newest-first), cursor, dropped, gap, oldest_seq,
      unseen_total}`.
- [x] **F3. UI logs view** — `components/Common/LogsSheet.tsx` (opened
      from a "Logs" button per instance row): Backend/Provider/All tabs,
      live tail (poll `since` ~2s), newest-first API rendered
      chronologically (Map<seq> merge, ascending, newest at bottom),
      auto-scroll + pause-on-scroll-up with "Jump to latest (N new)",
      stdout/stderr filter, client-side search with `<mark>` (React
      children — no XSS), level color hints, gap + dropped + skipped
      banners, download-visible-lines `.log`, MAX_LINES=2000 cap.
- [x] **Tests** — ring bounds/drop/atomic-after, streamer batching +
      cursor resume + shared-counter `kind=all` no-gap/no-dup, provider
      handler recursion guard, Redis cap/cursor/gap/unseen_total,
      endpoint + legacy-payload tolerance. Both `wire.py` FrameKind
      mirrors carry `BACKEND_LOGS_GET` (drift guard passes).

### Implementation notes / deviations
- **Consolidation**: the four near-duplicate provider `log_ring.py`
  files were deleted in favor of the shared `provider_lib.log_ring`
  (single source). Drivers keep their "recent stderr on startup failure"
  behavior (tail of the shared ring).
- **Shared seq space (provider side)**: `install_log_streaming` gives the
  backend ring and provider ring a SHARED `SeqCounter`, so
  `backend.logs.get kind=all` resumes correctly with a single cursor.
- **Two unrelated seq spaces** (IMPORTANT for any client): the provider
  `backend.logs.get` ack `seq` is the **provider-side ring sequence**
  (starts at 0 per provider process/connect lifetime); the admin REST
  `GET /logs?since=`/`cursor` is the **admin ingest sequence**
  (`im:logs:seq:{id}`, Redis INCRBY, 1-based). They MUST NOT be passed
  interchangeably. (Orchestrator: add this note to ws-protocol §4.)
- **F2.3 connect-catchup skipped**: the provider streamer resumes from
  `last_flushed_cursor` on WS reconnect, so the disconnect-window lines
  flush automatically; the admin does NOT fire `backend.logs.get` at
  connect (avoids timing races alongside metrics-assign + config-heal).
  `backend.logs.get` remains available for future use.
- **Post-provider-process-restart gap**: the ring is in-memory, so a
  provider *process* restart loses the un-flushed tail; Redis holds the
  historical tail (live gap ≤ ring lifetime). Accepted — logs are
  ephemeral ops telemetry.
- **`unseen_total`** added to `read_logs` so the UI can warn when >limit
  lines arrive between polls (realistic on verbose backends) rather than
  silently skipping them.
- **UI is a Sheet, not a route**: matches the app's existing
  row-triggered Dialog/Sheet pattern (Cache/Prune) and avoids
  `routeTree.gen.ts` churn.

**Test totals after Phase 13:** admin **257** · provider/lib **105** ·
mock **23** · llama-cpp **61** · halogen-flash **140** · halogen **67** ·
gufo **67**. Frontend `bun run build` + biome lint clean. Logs are
best-effort end-to-end: capture, transport, storage, and read never block
or crash the request/WS path.

---

## Phase 14 — Shell definitions: deferred typing + `awaiting_config` ✅

**Goal:** remove the fresh-install bootstrap ceremony (definitions require a
registered `ProviderType` whose registry only materializes at first provider
registration — the chicken-and-egg solved today by `seed_provider_type` in
`scripts/dev.sh` and the manual SQL in `development.md`). New operator flow:

1. Create **machine** in the UI.
2. Create a **shell definition**: alias + optional registration_token +
   scheduler hints — `provider_type: null`, `backend_config: null`.
3. Start the provider container. Registration binds the instance and
   **adopts** the container's `provider_type` onto the definition; the type's
   `ProviderType` row bootstraps from the container's shipped `schema.json`
   (existing `_bootstrap_type` path, unchanged).
4. Instance sits in a new **`awaiting_config`** pre-state (admin-owned, sits
   before `running` — registered, connected, metrics flowing, but explicitly
   not schedulable).
5. Operator configures `backend_config` in the UI — now rendered from the
   *actual committed schema* instead of a free JSON editor. The standard
   Phase 9 PATCH push (`provider.config.update`) delivers the config; ack
   clears `awaiting_config` → `initializing/running`. Everything downstream
   (`/v1/models`, scheduler, alias registry) activates only here.

**Non-goals:** no change to the registration-token model (§12 stays: token
still binds instance→definition, still secrets-checked 401/403/409, still
minted per definition), no wire-protocol envelope changes (`v: 1` untouched),
no worker-pool reassignment semantics. One definition = one type, still
immutable while instances are attached.

### Locked decisions

- **Type is adopted at registration, not deferred past it.** The container's
  reported `provider_type` is authoritative for an untyped definition
  (registration is the only place a container and definition meet).
  Downstream joins (`ProviderInstance`→definition→type, voter universe,
  `owned_by`) require the definition typed before an instance row exists —
  which registration guarantees.
- **`backend_config: null` ≠ `{}`.** Null = "no config authored yet"
  (definition not runnable, never pushed). `{}` = a real (possibly empty)
  config, valid under permissive schemas. Fingerprint helpers treat only
  the NULL-config case as unconfigured; `_instance_fingerprint`/pushes fire
  only when config is non-null.
- **`awaiting_config` is admin-owned** (like `disconnected`): the provider
  never emits it. Added to `InstanceStatusValue` in BOTH wire mirrors.
- **Unconfigured definitions never appear client-facing**: excluded from
  `/v1/models`, litellm alias registration, and all scheduler candidate
  queries. A request for such an alias is a clean `NoProviderAvailable` →
  503 (or 404 at route resolution — same as today's unknown alias).
- **First PATCH-with-config validates, then pushes** through the existing
  `config_update.push_config_update` — no new push mechanism. The heal
  sweep covers a config authored while the instance was disconnected.
- **Old behavior fully preserved** when a definition is created typed with
  a config (existing tests keep passing unchanged).

### Sub-tasks

- [x] **A. Model + migration** — `NullableJSON` TypeDecorator
      (`JSON(none_as_null=True)` wrapper — REQUIRED: the plain postgres
      `JSON` type serializes a top-level Python `None` to the JSON
      null *VALUE* (text `null`), silently defeating `backend_config IS
      NULL` shell detection); `provider_definitions.provider_type` +
      `backend_config` nullable; migration `b8e4f7d2a9c1`
      (downgrade refuses while shells exist);
      `models.backend_config_is_authored()` is the single truth source
      for the null≠{} distinction.
- [x] **B. Definitions CRUD** — shell create (config refused on a
      typeless definition, 422); typed create validates as before;
      serializers emit null type/config/fingerprint for shells; PATCH:
      unset-to-null refused 422 (never returns to shell state),
      config-set requires a type (same-PATCH type+config allowed),
      retype validates the stored config, alias warms exactly when
      config is authored.
- [x] **C. Registration adoption** — shell definition adopts the
      container's `provider_type` (same atomic commit; log +
      `type_adopted: true` in the response; `RegistrationResult.type_adopted`
      provider-side); registry bootstrap reuses `_bootstrap_type`
      untouched; instance fingerprint stays null for shells.
- [x] **D. `awaiting_config` state** — added before `registering` in
      BOTH `InstanceStatusValue` mirrors (admin + provider lib);
      drift guard unchanged/passing.
- [x] **E. Connect path** — `_mark_connected` sets `awaiting_config`
      when the definition is unconfigured (re-asserted on reconnect);
      `_persist_status` coerces provider-reported
      running/initializing/registering → `awaiting_config` while
      unconfigured (error/unhealthy stay honest);
      `heal_stale_fingerprint` + `push_config_update` early-return for
      shells (never push a fabricated `{}` config).
- [x] **F. Scheduler + /v1 gates** — `_candidates` +
      `_idle_stop_candidates` filter `backend_config IS NOT NULL`;
      `_try_admit` raises `NoProviderAvailable` for an unconfigured
      alias; `/v1/models` hides shells; `/v1/responses` +
      `/v1/chat/completions` return 404 (`not configured`) mirroring
      the disabled-alias path.
- [x] **G. Provider lib + packages** — `RegistrationResult` persists
      null fingerprint honestly (`_persist_config`);
      `install_config_handlers` exposes `client.no_config_nak` (NAK
      `no_config` @ validate when `applied_fingerprint is None`);
      wired into `on_backend_start` of all five provider packages
      (defense-in-depth behind the admin gates); `apply_registration`
      keeps None fingerprints as None in all packages (pre-existing
      behavior verified).
- [x] **H. Admin UI** — `StatusBadge` `awaiting_config` (violet);
      definition row shows the state badge for null types; create form
      gains the "(shell — type adopted at registration)" option
      (omits `backend_config`; PATCH never sends explicit nulls);
      detail view explains the shell + null fingerprint state;
      rjsf form path only activates for typed definitions;
      `scripts/generate-client.sh` run (client types now nullable).
- [x] **I. Bootstrap cleanup** — `seed_provider_type` removed from
      `scripts/dev.sh`; seeding creates a shell, waits for mock
      registration (type adoption), then PATCHes the canonical mock
      `backend_config` through the real Phase 9 push flow
      (`configure_mock_definition`); the smoke test is unchanged and
      end-to-end validates the whole shell lifecycle.
- [x] **J. Docs** — ARCHITECTURE.md §3 (definitions row), §4
      (ProviderDefinition fields + instance statuses), §5 (sequence +
      shell-adoption + awaiting_config notes); docs/ws-protocol.md §2
      (validation rule 3 rewritten, response example + type_adopted),
      §3 (hello/status coercion note), §4 (provider.status enum +
      config.update shell note + backend.start no_config NAK);
      provider/README.md (step 4 note + "Shell definitions" section).
- [x] **K. Tests** — admin: shell create/config-refusal/unknown-type,
      shell PATCH paths (config-needs-type, validate+push, no-unset),
      capacity-only-on-shell never pushes, registration adoption
      matrix (unknown-type bootstrap + known-type + typed flag +
      adopt-before-gate), scheduler (never admitted, excluded from
      candidates, reaper skip), /v1/models hidden, ws `awaiting_config`
      persist + non-clobber + honest errors + configured-path
      unchanged. Provider/lib: null-fingerprint persist+round-trip,
      `type_adopted` flag, `no_config` fence NAK+pass-through.

### Implementation notes / deviations

- **`none_as_null` discovery (important for any future nullable JSON
  column):** SQLAlchemy's postgres `JSON` type serializes a top-level
  Python `None` to the **JSON null value** (text `null`), NOT SQL NULL.
  The `NullableJSON` TypeDecorator (models.py) must wrap every nullable
  JSON column; the initial migration's `postgresql.JSON` stays (the
  type is Python-side only).
- **PATCH null-handling:** `.get()` vs `in changes` — explicit null
  checks on `provider_type`/`backend_config` use `key in changes`
  (absent ≠ null) after an early version broke every normal config
  PATCH; SF-4's loop for the remaining non-nullable fields keeps
  `changes[key] is None` iteration.
- **Consensus interplay:** adoption happens BEFORE `_schema_gate`
  consumes the type, so a first-ever registration via a shell adopts
  AND participates in consensus identically to a typed registration
  (solo voter → presented schema commits immediately).
- **Mock e2e tests need env locally** (`PROVIDER_REGISTRATION_TOKEN` /
  `MACHINE_UID` / `ADMIN_BASE_URL` — CI provides them; bare `pytest`
  runs of `test_mock_phase4.py`/`test_main_wiring.py` fail in
  pydantic-settings at `ProviderSettings()` — pre-existing, not a
  Phase 14 regression).
- **N-8 post-deploy fix (prod incident 2026-10-07):** the UI PATCH on
  `rocinante-testing` returned **500 idle-in-transaction** — the route
  session sat inside an open write transaction across the awaited
  `provider.config.update` (llama.cpp apply ≫ 15s) and Postgres's
  `idle_in_transaction_session_timeout` (15s, `app/core/db.py`) killed
  the connection before the final `instances` re-read. The admin row and
  the push themselves succeeded (config was live; the 500 was cosmetic
  on that path — but any future PATCH during a long apply would fail the
  same way, and local tests (≈1s boots) can never reproduce it). Fix:
  the PATCH route commits + `expire_all()` + `commit()` (ends the
  transaction) BEFORE awaiting, pushes via the new
  `config_update.push_config_update_by_id` (resolves the definition in
  its own short session), and re-binds the row after the await
  (`get` may return None if the definition was deleted mid-push —
  handled with a `deleted_during_push` acknowledgment). Regression test:
  `test_patch_push_slower_than_idle_tx_timeout_still_200` (route-side
  connections disposed mid-push; asserts 200 + ack-echoed fingerprint
  persisted).

### Accepted risks (per spec)

1. Typo'd container `PROVIDER_TYPE` poisons a shell on first
   registration — visible (`type_adopted: true` + log); repair = PATCH
   the type while no instances attached (or delete + recreate the
   shell).
2. Never-configured shell definitions linger with an
   `awaiting_config` badge — intentional (no reaper touches them,
   nothing is loaded).
3. `null` vs `{}` remains the cardinal footgun —
   `backend_config_is_authored()` + `NullableJSON` centralize it;
   both are doc-commented in models.py.

**Test totals after Phase 14:** admin **272** · provider/lib **108** ·
mock **23** (env-provided) · llama-cpp **61** · halogen-flash **140** ·
halogen **67** · gufo **67**. `bun run build` + lint clean.
Client regenerated from the updated OpenAPI (nullable
provider_type/backend_config on DefinitionCreate/DefinitionPatch).

### Implementation notes / ordering

- Deploy order (§10 lockstep): admin image first, then recreate every
  provider container — version gate forces this anyway; old providers
  registering against a shell definition are the only new path and they
  speak the same registration body.
- DB migration is additive (nullable columns); prestart applies it.
  Existing rows all have type+config → zero behavior change until a
  shell is created.
- The two wire mirrors MUST be edited in the same commit
  (`test_wire_drift_guard.py`).
- Client generation (`scripts/generate-client.sh`) after A/B/C/E/F route
  changes.

### Accepted risks / regressions

1. **Typo'd container `PROVIDER_TYPE` poisons a shell** (adopted on first
   registration). Mitigation: `type_adopted: true` in the response +
   registration log line; operator repair = DELETE shell definition and
   recreate (no instances of consequence attached pre-config) or PATCH
   type while instance detached. Cheaper than the old system's total
   block (unknown type → 422 with "(none registered yet)").
2. **Shell definitions linger if never configured.** Visible
   (`awaiting_config` badge); no reaper touches them; acceptable.
3. **Null vs `{}` distinction is a footgun for future contributors.**
   Locked decision above; `backend_config_is_authored()` helper +
   `NullableJSON` column type centralize it; doc notes in models.py.

---

## Accepted Regressions (do NOT restore from `legacy/`)

These existed in the pre-overhaul system and are intentionally removed or
stubbed. Re-implement against the new model only when a feature needs it.

| Feature | Status | Why |
| --- | --- | --- |
| `/v1/embeddings` | 501 stub | Out of scope for the litellm Responses/Chat core |
| `/v1/completions` (legacy) | 501 stub | Legacy text completions |
| `/v1/rerank`, `/v1/moderations`, `/v1/decisions` | 501 stub | Not in core path |
| `/v1/audio/*`, `/v1/files`, `/v1/batches` | 501 stub | File/audio/batch subsystem dropped with old `files`/`batch_jobs`/`audio_jobs` tables |
| Responses-over-WebSocket transport | Removed | Not part of the OpenResponses spec |
| Benchmarks (`llama-bench`) | Removed | All tables/services/UI/events dropped |
| `PromptCache` table / hybrid cache tracking | Removed | Cache is provider-local via fingerprint (Phase 9) |
| `Model` registry table | Removed | Artifacts folded into `backend_config` JSON |
| `InferenceLease` / reservation / slot_generation | Removed | Replaced by scheduler + provider connection-lifecycle admission |
| Users / API keys | Removed | Trusted-LAN model |
| Prometheus `/metrics` | Never existed in new code | Old `monitoring-guide.md` deleted |

## Known Limitations

1. Trusted-LAN only — `/admin/api` and `/v1` unauthenticated.
2. Version hard-fail requires coordinated admin+provider deploys.
3. `MACHINE_UID` must not be reused across physical hosts (no host
   fingerprint).
4. VRAM eviction reclaims only idle different-alias backends on the
   request's machine — there is no cross-machine rebalancing and no
   workload priority scheme (see ARCHITECTURE.md §6).
5. Single uvicorn worker (in-process scheduler authority); multi-worker
   needs the Redis-queue swap behind the `acquire`/`release` interface.
