# Inference Matrix — Implementation Status

**Overhaul branch:** `litellm-architecture-overhaul`
**Last updated:** 2026-10-04 (Phase 8 complete: gufo / halogen /
halogen-flash provider ports — fake-driver-tested; hardware validation
deferred)

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
| 6 | Scheduler + `/v1/responses` via litellm + SSE emitter + persistence | 🟡 In progress |
| 7 | `/v1/chat/completions` + `/v1/models` + 501 stubs | ✅ Complete |
| 8 | gufo / halogen / halogen-flash provider ports | ✅ Complete |
| 9 | `provider.config.update` / fingerprint / cache-clear flow | ⬜ Pending |
| 10 | Admin UI rework | ⬜ Pending |
| 11 | Docs consolidation + full E2E validation | ⬜ Pending |

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
  the admin.
- Old code parked in `legacy/` (reference only; never import).

---

## Phase 2 — Schema + Redis Keys ✅

**Models:** `admin/backend/app/models.py`
**Redis doc:** `admin/backend/docs/redis-keys.md`
**Migration:** `admin/backend/alembic/versions/0140d6ad9f48_initial_squashed_schema.py`

Five tables: `Machine`, `ProviderDefinition`, `ProviderInstance`,
`ResponseRecord`, `TokenUsageSample` (fields in ARCHITECTURE.md §4).
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

## Phase 6 — Scheduler + `/v1/responses` 🟡 IN PROGRESS

**Admin:** `app/services/scheduler.py`, `app/services/sse.py`,
`app/services/alias_registry.py`, `app/api/v1/responses.py`.
**Design:** ARCHITECTURE.md §6, §7; FINDINGS.md.

### Done (tests green)
- `InferenceScheduler` — in-process per-alias FIFO + Redis mirror.
  `acquire`/`release`; `NoProviderAvailable`→503, `QueueTimeout`→504.
  Slot = capacity + machine free VRAM. Prefers already-running instances.
  Boots via `backend.start` under `im:sched:lock`. Remembers `_booted` to
  avoid redundant boots. `release` cancellation-safe (shielded).
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
  (72 admin tests pass).

### Remaining for Phase 6
- [ ] **VRAM eviction** — `TODO(phase6-eviction)`: stop idle
      different-alias instances on a machine to free VRAM instead of
      waiting. Priority: LRU-idle first.
- [ ] **Idle-timeout reaper** — `TODO(phase6-idle-reaper)`: stop
      instances past `idle_timeout_seconds` with no active requests
      (background task stub exists; `start_background`/`stop` wired).
- [x] **Tool forwarding** — client `tools`/`tool_choice` are forwarded to
      litellm (`PASSTHROUGH_FIELDS`); tool *execution* is provider-side
      (the backend emits `function_call` items; the mock emits a canned
      one). An admin-side multi-turn tool-execution loop is a Phase 7
      concern, not Phase 6.
- [ ] **Keepalive during cold boot** — emit SSE `: keep-alive` comments
      while `scheduler.acquire` blocks on a boot, so Traefik doesn't kill
      long cold starts.
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

Phase 6 code items are complete except eviction, the idle reaper, and
cold-boot SSE keepalive (all explicitly-TODO, non-blocking for the
conformance contract).

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
  per-file download wiring lands with Phase 9's config/update flow
  (the driver only gates `HALOGEN_NPU_MODELS` on the probe today).
- `metrics.inference` (ARCHITECTURE.md §8, reserved frame kind) did
  **not** land in Phase 8 — the gufo rate gauges feed per-request
  `TokenUsageSample` persistence instead of the reserved always-on
  telemetry frame. It remains reserved for a future phase.

---

## Phase 9 — Config Update / Fingerprint / Cache ⬜

- [ ] `provider.config.update` command → provider enters `initializing`:
      drain+stop backend, recompute `config_fingerprint`, **auto-clear
      prompt cache if changed** (`cache.clear` = prompt cache only, never
      model files), download/update model artifacts, start backend,
      scrape metadata, stop backend → `running`.
- [ ] `storage.prune_unused` — delete orphaned files not referenced by
      the active fingerprint.
- [ ] Per-step failure states + progress reporting.
- [ ] Admin calls `ensure_registered` on definition create/update so
      aliases are warm before first use.

---

## Phase 10 — Admin UI Rework ⬜

- [ ] **Machines** page: create (uid, name, host/dns/ip, total_vram),
    view merged hardware.
- [ ] **Provider Definitions** page: alias, provider_type, `backend_config`
    editor (JSON schema-validated), vram/idle/capacity, **copyable
    registration token**, discovered metadata.
- [ ] **Provider Instances** page: dual status (instance + backend),
    version, port, epoch, last_seen, assigned metrics, live logs.
- [ ] Wire to regenerated client; TanStack routes under `/admin` base.
- [ ] Remove old Agents / Server Instances / Benchmarks pages.

---

## Phase 11 — Docs Consolidation + E2E ⬜

- [ ] Final pass over all docs; fix drift from Phases 6–10.
- [ ] Full local run with mock provider end-to-end.
- [ ] Integration + conformance suite green against a deployed admin.
- [ ] Provider authoring guide complete (`provider/README.md`).

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
4. Phase 6 eviction + idle reaper are TODO (see Phase 6 remaining).
5. Single uvicorn worker (in-process scheduler authority); multi-worker
   needs the Redis-queue swap behind the `acquire`/`release` interface.
