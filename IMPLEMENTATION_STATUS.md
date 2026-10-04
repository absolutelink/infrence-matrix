# Inference Matrix — Implementation Status

**Overhaul branch:** `litellm-architecture-overhaul`
**Last updated:** 2026-10-04 (Phase 6 in progress)

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
| 7 | `/v1/chat/completions` + `/v1/models` + 501 stubs | ⬜ Pending |
| 8 | gufo / halogen / halogen-flash provider ports | ⬜ Pending |
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

### Done (uncommitted, tests green)
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
- [ ] **Tool / agent loop** — forward client `tools` to litellm; execute
      platform local tools; feed results back (multi-turn within one
      request). Decide explicit-loop vs litellm agentic hooks.
- [ ] **Keepalive during cold boot** — emit SSE `: keep-alive` comments
      while `scheduler.acquire` blocks on a boot, so Traefik doesn't kill
      long cold starts.
- [ ] **Conformance gate** — run openresponses.org compliance suite
      against `/v1/responses`; must pass before Phase 6 is ✅.
- [ ] **Non-stream path** — verify JSON (non-`stream`) response shape +
      persistence parity.

Feature spec for each remaining item: see ARCHITECTURE.md §6–§7 and
`docs/ws-protocol.md`; ask which to take next.

---

## Phase 7 — Chat Completions + Models + Stubs ⬜

- [ ] `POST /v1/chat/completions` via `litellm.acompletion` against the
      same scheduler admission + provider port. Persist usage.
- [ ] `GET /v1/models` — derive from enabled `ProviderDefinition`s
      (alias + `model_metadata`).
- [ ] 501 stubs with OpenAI error envelope for: `/v1/embeddings`,
      `/v1/completions` (legacy), `/v1/rerank`, `/v1/moderations`,
      `/v1/decisions`, `/v1/audio/*`, `/v1/files`, `/v1/batches`.
- [ ] `generate-client.sh` after routes land.
- [ ] Conformance/integration suite green.

---

## Phase 8 — Provider Type Ports ⬜

Port remaining engines from `legacy/agent/services/*` to
`provider/<type>/`, overriding only what differs from the lib.

- [ ] **gufo** — `gufo serve llm` argv maps, per-request `model`
      injection, aux spec files (mmproj/dflash/dspark/mtp) by path,
      native responses passthrough, rate-gauge parsing.
- [ ] **halogen** — `HALOGEN_*` env map, two ports (api/engine),
      `kv_slots` capacity.
- [ ] **halogen-flash** — native `/v1/responses`, ~45 `HALOGEN_*` env
      vars, static port range for disk-cache fingerprint, NPU small-model
      pinning, **`calculate_usage` override** (backend not spec-compliant
      on usage — the canonical override example).
- [ ] Each: Dockerfile in `provider/<type>/`, tests, mock-equivalent.

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
