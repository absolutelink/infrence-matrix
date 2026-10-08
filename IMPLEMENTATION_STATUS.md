# Inference Matrix — Implementation Status

**Overhaul branch:** `litellm-architecture-overhaul`
**Last updated:** 2026-10-08 (**Port model overhaul — agent-owned ports +
model-routed `/v1`: ✅ SHIPPED (all 5 slices done).**
Fixes the production `port_conflict` cross-agent collision by moving to one
published `PROVIDER_PORT` per agent that routes `/v1` by model; the admin stops
allocating/policing ports and `ProviderInstance.port` is dropped (migration
`b7d3f0a1c9e2`).
`ARCHITECTURE.md` rewritten to the new model; see the "Port model overhaul"
section below for the locked decisions + slice plan. Prior: **Phase 18 ✅
embeddings + modality-scoped
endpoints — SHIPPED (all 7 slices green + verified end-to-end against a local
admin + mock). Adds `ProviderDefinition.modality` (`llm`|`embedding`, `audio`
reserved) as the admin routing key, `ProviderType.serves_modalities` (declared
via top-level `x-serves-modalities` in `schema.json`, mirroring
`x-max-running-backends`), a spec `POST /v1/embeddings` route driven by
`litellm.aembedding`, `modality` pushed to the provider on the wire, and
llama-cpp `--embedding`/`--pooling` engine support + mock fake embeddings. See
the Phase 18 section below for the shipped details + verification.** Prior: **Phase 16 ✅
machine-scoped provider agents —
slices 7 + 8 landed + agent-delete follow-up**: slice 7 = React Agents page +
definition placement controls; **slice 8 = the remaining law docs rewritten to
the agent model** (`docs/ws-protocol.md` full rewrite, `provider/README.md`
agent-authoring guide, `admin/backend/docs/redis-keys.md` agent-keyed layout,
`AGENTS.md` Architecture/Gotchas bullets) — documentation-only; **follow-up:
`DELETE /admin/api/agents/{agent_id}`** removes decommissioned/renamed agents
(409 while connected; cascades backends + placement links, clears Redis WS/metrics
keys, releases scheduler holds) + Agents-UI Delete button + `AdminService.deleteAgent`;
admin suite now 334 green. Prior: **slice 6**: real-engine
**multi-backend-per-process** hosting +
per-port serving — each hardware provider (llama-cpp/gufo/halogen/halogen-flash)
now builds a `BackendRegistry` of drivable lifecycles (one per placed backend),
threads the assignment `port` into its engine driver so subprocess ports stay
distinct, and serves every hosted backend's `/v1` on its own port through the new
`provider_lib.serve.MultiPortServer` (dynamic add/remove sync on registry change);
halogen-flash declares `x-max-running-backends: 1` so the admin places at most one
backend per its agent; single-backend behavior is byte-identical to before.
Prior: **slice 5**: `agent.assignments.update` push + event-driven placement
reconciliation — a placement change now propagates to a live agent over its
socket without a re-registration via the single shared `reconcile_agent_placement`
diff (busy-safe on both sides) + the slice-4 warm-up re-trigger; provider_lib
reconciles its `BackendRegistry` and every package acks. Prior: slice 3 atomic
cutover — a provider container is now an *agent*
bound to a machine + one provider type; auth moves to a shared
`Machine.registration_secret` + `AGENT_ID`; one agent-level WS multiplexes
per-backend frames addressed by `ProviderInstance` id; definitions get
placement (`any_of_type` / cherry-picked agents); the per-definition
`registration_token` and Phase 14 shells/`awaiting_config` are removed;
`ProviderInstance` is re-keyed onto `ProviderAgent` (migration `f1b6c2d84a97`).
Admin + provider lib + all five provider packages migrated and green
**verified in a clean shell (no exported env)**: admin 306, lib 128, mock 25,
llama-cpp 63, gufo 69, halogen 72, halogen-flash 161; ruff check + format
clean across admin + all provider packages; alembic
upgrade/downgrade/upgrade/check clean; frontend `generate-client.sh` + `tsc`
clean. Provider tests now set the Phase 16 settings contract
(`MACHINE_SECRET`/`AGENT_ID`, no `PROVIDER_REGISTRATION_TOKEN`) via a shared
`provider/lib/tests/conftest.py` + per-package conftests; `build_app()` takes
explicit settings instead of re-parsing env.
**Code-review follow-up (slice 3 hardening):** fixed the `uvicorn.Config`
`base_port`→`port` startup crash in all four hardware packages (+ per-package
`test_serve_config.py` guard); corrected `_placed_definitions` to the
ARCHITECTURE §5 union (any_of_type ∪ specific-linked) with a mixed-placement
test; `provider.config.update`/`backend.*` now dispatch per-backend by
`instance_id` via a new `provider_lib.registry.BackendRegistry` (single-backend
agents pass through; multi-backend agents NAK `unknown_instance`) with a
2-backend routing test; `backend.metadata` + `backend.logs` frames stamp
`instance_id`; placement agent type-mismatch → 422 (M1); un-placed backends are
pruned at registration and on placement PATCH (M2); migration copies
`error_message` to agents + documents downgrade safety (M3); frontend types +
reads moved to `agent_status` and the `registration_token` UI removed (H4);
`scripts/dev.sh` + `compose.yml` migrated to the agent contract (H5); dead
Phase-14 no-config machinery removed (L3); doc staleness banners added (L4);
`assigned_gpus` stored structured + hardware UNION (L5).
**Round-2 review follow-up:** the single-handle dispatcher no longer
pass-throughs a foreign `instance_id` — a per-backend command naming an id the
agent does not host is NAK'd `unknown_instance` (matched against the handle's
live `lifecycle.instance_id`, so single-backend agents installed before
registration still serve their own id) with a test; prune (registration +
placement PATCH) now SKIPS `running`/`in_use` backends so a live engine's VRAM
hold is never orphaned, pruned on a later pass once stopped, with a test;
`connection_manager` ignores `backend.status`/`backend.metadata` frames whose
`instance_id` belongs to a different agent (cross-agent guard) with a test;
`settings.tsx` help text + CI `build-and-push.yml` env migrated to
`MACHINE_SECRET`/`AGENT_ID`.
Deferred: React Agents/placement UI + removing the
inert `awaiting_config`/shell-create UI remnants (slice 7), `provider/README.md`
`no_config_nak` references + full `docs/ws-protocol.md` rewrite (slice 8). See
the Phase 16 section below for the full plan.). Prior:
Phase 15 manual backend control, Phase 14 shell definitions, Phase 13 log
views, Phase 12 schema-driven backend config.

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
| 14 | Shell definitions: deferred typing + `awaiting_config` pre-state | ✅ Complete (superseded by 16) |
| 15 | Manual backend control + `provider.initialize` + download-bound boot budget | ✅ Complete |
| 16 | Machine-scoped provider **agents**: one container → many same-type backends, placement, `max_running_backends` | 🟡 slices 1–6 landed (agents, placement, `max_running` hot-swap + proactive warm-up, `agent.assignments.update` push, real-engine multi-backend-per-process + per-port serving); 7–8 pending |
| 17 | Per-GPU machine metrics + hardware union: device-isolated agents (one GPU each) merge into a full machine inventory and live snapshot | ✅ Complete |
| 18 | Embeddings + modality-scoped endpoints: `ProviderDefinition.modality` (`llm`/`embedding`, `audio` reserved), `ProviderType.serves_modalities`, spec `POST /v1/embeddings` via litellm, llama-cpp `--embedding`/`--pooling` + mock fake embeddings | ✅ shipped (all 7 slices green + e2e verified vs local admin + mock) |
| 19 | Live stats bar (tokens/sec, queue/active, VRAM/GPU + popovers incl. queue clear) + UI ergonomics: create/edit forms → right-side drawers, logs → bottom-docked tabbed panel | 🟡 S0–S1 landed |

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
  `full-vulkan` base) and CUDA 12 (`full-cuda12` base) — via
  `provider/llama-cpp/{Dockerfile,Dockerfile.cuda12}`; `provider/llama-cpp`
  tests run in the `test-provider` job. No `:latest` tag is produced.
  - **CUDA 12 driver-baked (2026-10-07):** `Dockerfile.cuda12` now copies the
    NVIDIA userspace driver (`libcuda`, NVML `libnvidia-ml`,
    `libnvidia-ptxjitcompiler`) and `nvidia-smi` from RPM Fusion (Fedora 44
    builder stage) into `/usr/local/nvidia` and prepends it to
    `LD_LIBRARY_PATH`, so the image runs **without the NVIDIA Container
    Toolkit** — pass the `/dev/nvidia*` device nodes instead of `--gpus all`.
    The baked driver version is pinned by build-args `NVIDIA_DRIVER_BRANCH`
    (default `580`) / `NVIDIA_VERSION` (default `580.178.04`) and **must match
    the host kernel driver**. The previously disabled
    `build-provider-llama-cpp-cuda12` CI job is re-enabled with those
    build-args. `nvidia-smi` in-image feeds the `vram`/`gpu_usage` metrics.
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
  idle clock — `max(last_request_at, backend_loaded_at)`, fallback
  `created_at` (see the hotfix section; the boot re-arms the window) —
  is older than the definition's `idle_timeout_seconds`;
  `idle_timeout_seconds == 0` means never idle-stop. Clears ledger +
  mirror + DB status on success; a NAK/exception is retried next tick;
  each tick is try/except-guarded. Each tick also refreshes the
  `im:vram:used` TTL for this process's
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
- **halogen-flash image was unbootable (found while re-deploying the
  Flash instance on `lfh-ai-node-01`, 2026-10-07):** two stacked defects
  in `provider/halogen-flash/Dockerfile`, neither visible in CI (which
  builds the image but never runs it) and both fatal on the host — the
  quadlet restart loop hit `start request repeated too quickly` in ~2s.
  (1) `CMD` only: `peonist-ai/halogen-flash-server` ships
  `ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]`, so the provider command
  was passed to the **engine** as its mode word
  (`halogen: halogen-flash-server 0.16.2, mode sh` → usage → exit 2).
  llama-cpp's Dockerfile already overrides this; halogen-flash must too.
  (2) The venv was built in a `python:3.14-slim` stage and copied to
  `/agent/.venv` — but uv's venv is **not relocatable**
  (`bin/python -> /usr/local/bin/python3`, and the editable `.pth` files
  name build-stage paths). In the final image that symlink resolves to
  the base's **3.12**, which cannot see `lib/python3.14/site-packages`
  (`ModuleNotFoundError: provider_halogen_flash`), so even a corrected
  entrypoint would have died on import. Fix: copy the standalone 3.14
  interpreter + stdlib only, then `uv sync` **in the final image** at
  `/app/.venv` (the legacy `recipes/halogen-flash` build did exactly
  this), with `python3.14 -m uv` (uv installed into the copied 3.14
  tree, never into the engine's interpreter). The base's global
  python3.12 + ROCm site-packages are left untouched and PATH is not
  prepended — `entrypoint.sh` resolves `python3` from PATH, so venv-first
  PATH would break the engine. `HF_HUB_OFFLINE` is left exactly as the engine
  image bakes it (`1`, which is what stops the front-end reaching for the Hub
  at request time): provider-side fetches clear huggingface_hub's *in-process*
  offline switch in `provider_lib.downloader` instead of mutating the
  environment the backend child inherits. Verified locally by reproducing the
  restricted `COPY` set +
  `uv sync --frozen --no-dev --package matrix-provider-halogen-flash`
  (imports resolve, `.pth` paths correct).
- **HF artifact picker always failed validation (Phase 12 E regression,
  found 2026-10-07 on both `llama-cpp` and `halogen-flash`):** the admin
  rejected every picker selection with
  `{'repo': ..., 'file': ...} is not valid under any of the given schemas`.
  Cause was client-side, not the schema: `$defs/hfFile`'s HF branch marks
  `source` with `const: "hf"`, rjsf materializes that const into formData,
  and `pruneUntouchedDefaults` (the H2 "ship only operator-changed leaves"
  rule) then compared the leaf, saw `current.source === populated.source`,
  and dropped it — leaving a descriptor matching neither `oneOf` branch.
  Fix: `pruneUntouchedDefaults` now takes the schema and treats a
  descriptor node (`isHfFileSchema`, including a union branch that offers
  `hfFile`) as the picker's **atomic value** — a complete descriptor ships
  verbatim, an incomplete one ships nothing (so a half-materialized
  `{"source":"hf"}` can never be saved), and scalar branches
  (`vision_tower: true`) keep the old leaf rules. Probe-checked against the
  shipped llama-cpp + halogen-flash schemas: fresh HF selection keeps
  `source`, an untouched save of a stored descriptor stays byte-identical
  (no fingerprint churn), local-path and clear cases behave, and the
  pre-fix call still reproduces `{"repo","file"}`.
- **halogen-flash artifacts are now optional (blank = image default):**
  `artifacts.model` / `artifacts.tokenizer` no longer carry a `required`
  entry; when blank the driver resolves nothing and `build_env` OMITS
  `HALOGEN_CHECKPOINT` / `HALOGEN_TOKENIZER` entirely (never an empty
  string — the image's `resolve_checkpoint` picks `qwen38-flash-next-v2.hgn`
  under MODELS_DIR, falling back to `...-w4b.hgn`, and `need_tokenizer`
  uses the checkpoint's sidecar `tokenizer/`). `resolved_artifacts` then
  holds only what was resolved, so `storage.prune_unused` keeps refusing to
  run with an empty reference set (the guard is load-bearing, not new).
   Shipped schema fingerprint: `79584684e7f4…a03a98b8` (pin in
   `test_schema_sections.py`; the `download` wiring below re-pinned it from
   the optional-artifacts value). Safe to change now because the fleet has no
   committed `halogen-flash` ProviderType row yet (only `llama-cpp` is
   registered); once Flash instances exist this is a consensus change.
   `llama-cpp` and `gufo` still require `artifacts.model`. The plain `halogen`
   provider keeps its required checkpoint too — its entrypoint discovers a
   default under `/models` but has no `maybe_download` pass.
- **Downloads belong to the engine, not the provider (2026-10-07, operator
  rule).** `HALOGEN_DOWNLOAD` is now wired (`ENV_MAP["download"]`, value
  driver-resolved by `HalogenFlashBackend._download_repo`: explicit
  `troubleshooting.download` → the `artifacts.model` HF descriptor's repo →
  `env.DEFAULT_DOWNLOAD_REPO` = `peonist-ai/halogen-qwen3.8-flash-next`;
  `""` opts out and emits nothing). The entrypoint's `maybe_download` pass
  then pulls what is missing — the default checkpoint for a blank selection,
  plus the overlay sidecar / ngram table / vision tower / tokenizer — into
  the checkpoint's own directory under `MODELS_DIR`. The provider only ever
  calls `ensure_artifact` for an operator-picked HF descriptor, so a blank
  selection touches no network from Python. `networking.bind` stays unwired;
  `troubleshooting` is no longer "the whole section unwired". Because the
  driver cannot know the engine's fetched paths, a resolved checkpoint
  contributes its **parent directory** to `resolved_artifacts` (skipped when
  the checkpoint sits directly in the models root, which would make
  `storage.prune_unused` a no-op); with a fully blank selection
  `resolved_artifacts` stays empty and pruning refuses to run by design.
  Provider-side downloads also no longer depend on the ambient
  `HF_HUB_OFFLINE`: `provider_lib.downloader._download_hf` clears
  huggingface_hub's in-process offline switch (read per request in
  `utils/_http`) without touching `os.environ`, so the engine child still
  inherits the image's `HF_HUB_OFFLINE=1`.

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

## Phase 15 — Manual backend control + reinitialize ✅

Operators can now drive a backend without sending a request, and a boot
that spends an hour downloading weights no longer breaks the protocol.

**Why the shape.** `backend.start` acked only after the lifecycle reached
running — correct for the scheduler ("acked == /v1 is live"), impossible
for a halogen-flash cold boot: with `HALOGEN_DOWNLOAD` the engine's own
entrypoint pulls the checkpoint plus its overlay sidecar, ngram table,
vision tower and tokenizer (tens of GB) before `/health` ever answers, far
beyond the 30 s command-ack window, the 120 s llama.cpp-era health budget,
and any HTTP request's patience.

- **`provider_lib/ops.py` (new)** — one installer for every provider:
  `install_backend_ops(client, lifecycle, backend_name=..., re_register=...)`
  registers `backend.start` / `backend.stop` / `backend.restart` /
  `provider.initialize` (both previously "Reserved" in the wire doc) and
  deletes the per-package start/stop handlers. Ack detail is built
  generically from `capacity` + whichever of `effective_capacity`,
  `api_port`, `engine_port`, `backend_port` the driver exposes; the Phase 14
  `client.no_config_nak` fence is read per command (install order-free).
- **`wait_for_running`** on start/restart/initialize. Default `true` keeps
  the scheduler's contract unchanged. `false` (what the UI sends) acks
  `{"accepted": true, "backend_status": ...}` and boots in a background
  task; terminal state arrives via `provider.status`, and a successful boot
  also emits `backend.metadata` (`{"models": [...]}` from `list_models()`;
  a scrape failure never fails the boot, and the admin ignores an empty
  list). A waiting caller arriving during an in-flight boot **joins** it
  (`asyncio.shield`, never a second spawn); a second accept-style caller
  NAKs `boot_in_progress`; stop-triggering ops NAK `backend_in_use` +
  `retry_after` while slots are held.
- **`provider.initialize`** = re-register (POST `/register`: re-checks the
  version + schema gates, re-adopts capacity/`backend_config`/fingerprint,
  mints a fresh instance secret, rewrites `CACHE_DIR/provider_config.json`)
  → drain check → background stop → start → metadata. Only
  `re_register` is provider-supplied (it knows type/version/hardware/schema).
  The live socket is deliberately NOT recycled: it is already authenticated
  at the current epoch and carries the events the operator is watching.
  A shell definition refreshes but never boots (ack carries
  `no_config: true`) — this is the escape hatch for an instance stuck in
  `awaiting_config` after a config edit that never pushed.
- **Long-init heartbeats.** `BackendLifecycle.report(status, reason)` emits
  `backend.status` without touching the state machine (so a heartbeat can
  never make a non-serving backend look servable — slots keep refusing
  until the real `RUNNING`), and `BackendLifecycle.__init__` binds it to any
  driver declaring `attach_status_callback`. halogen + halogen-flash
  heartbeat every 15 s while waiting for `/health`, with the engine's newest
  log line as the reason (first beat after the grace period, so a warm boot
  stays as quiet as before; a broken callback can never break a boot).
- **Boot budgets** (`provider_lib/config.py`): new `ENGINE_BOOT_TIMEOUT`
  (env-tunable, default **3600 s**) for the halogen family's health wait;
  `SERVER_START_HEALTH_TIMEOUT` (120 s) stays for llama.cpp/gufo local
  loads. Admin mirror: `BACKEND_BOOT_TIMEOUT_SECONDS` (3600 s) for a waiting
  `backend.start` (the scheduler's `_boot` now sends it explicitly) and
  `BACKEND_STOP_TIMEOUT_SECONDS` (120 s).
- **Admin routes** (`app/api/admin/instances.py`, client regenerated):
  `POST /admin/api/instances/{id}/backend/start|stop|backend/restart` and
  `.../initialize`. Start/restart refuse 409 on a shell (mirroring the
  provider fence locally so the operator gets a readable message);
  `initialize` and `stop` do not. 202 by default; `{"wait_for_running":
  true}` blocks instead. Provider NAKs surface as 502 with `error`/`step`/
  `retry_after` (existing `_send_action` mapping).
- **`backend.metadata` ingest** (`connection_manager`): stores
  `{"models": [...]}` on the definition — the same column and shape as the
  `provider.config.update` ack echo — so manual boots keep the roster fresh.
- **Scheduler interaction**: the idle reaper only targets
  `running`/`in_use`, so it never kills a downloading boot; a client request
  still gives up at its own `queue_timeout` (504) while the boot continues,
  and the next request adopts the now-running instance through the existing
  "running per the DB but this process never booted it" ledger path.
- **UI** (`routes/_layout/instances.tsx`): a per-row **Backend** menu
  (Start / Stop / Restart / Reinitialize; Start+Restart disabled with an
  explanation while the definition is `awaiting_config`, the whole menu
  while the WS is down) and a confirm dialog carrying the
  `wait_for_running` switch + an ack result (accepted vs final status,
  capacity, ports, `no_config` hint). The existing 4 s instance poll shows
  the `initializing → running` progression; the log tail shows the engine's
  own download output.
- **Registration port-clash guard** (`app/api/admin/providers.py`):
  409 `port_conflict` when a *different* **connected** instance on the same
  machine already claims the reported port. The admin addresses instances as
  `machine address:instance port`, so a clash silently serves one alias from
  another provider's engine — found live while debugging this phase
  (halogen-flash registered 8081 on a box whose llama.cpp container owns
  host 8081, so its traffic reached the wrong backend). Re-registering the
  same definition (the initialize path) is never a clash, and disconnected
  rows do not hold a port (the presence sweep clears them), so a replacement
  instance can always come online.

Tests: `provider/lib/tests/test_backend_ops.py` (16: ack shapes, join vs
duplicate spawn, fences, drain, initialize order + failures, metadata),
`provider/lib/tests/test_lifecycle.py` (report/attach), halogen +
halogen-flash driver heartbeat/budget tests, `tests/test_main_wiring.py`
(accept-style boot + initialize over a **real** socket, counting
registration POSTs), per-provider `test_handlers_installed` (all four
commands), `admin/backend/tests/test_instance_actions.py` (14: routes,
payloads, timeouts, fences, NAK mapping, metadata ingest, port clash).
Admin 290 · lib 128 · halogen-flash 153 · halogen 70 · gufo 67 ·
llama-cpp 61 · mock 23 pass; ruff + biome + tsc + vite build clean.

---

## Hotfix — halogen-flash chat surface + reaper load clock (2026-10-07)

Production report: playground chat completions against the halogen-flash
alias died with a bare "network error"; `/v1/responses` worked but the
engine only saw the message a few seconds after submit. Three defects,
one per layer:

- **halogen-flash had no chat surface (501 → the whole failure chain).**
  The API port speaks OpenAI `/v1/chat/completions` natively, but
  `HalogenFlashBackend` only implemented `stream_responses`, so the
  provider app returned 501 and litellm burned 3 retries before giving
  up. Fix: `stream_chat_completions` proxies the API port's chat SSE
  (`stream` forced, `[DONE]` skipped, chunks relayed verbatim; slot
  release stays driven by upstream close). Tests: fake engine now serves
  chat SSE; `provider/halogen-flash/tests/test_chat_stream.py` (5:
  relay + force-stream + upstream-501 raise + cancel-on-aclose +
  lifecycle slot release).
- **Admin streaming chat aborted the SSE response on pre-first-chunk
  errors.** `_stream_chat` awaited `litellm.acompletion` inside the
  generator, **after** the route had committed 200 +
  `text/event-stream` headers: a raising call (any upstream 4xx/5xx
  after litellm's retries) escaped the generator, uvicorn cut the
  chunked body mid-flight, and clients surfaced a raw network error.
  Fix: the route awaits the call *before* returning the
  StreamingResponse — a call-time failure maps to a real HTTP 502 JSON
  (like the non-stream path), persists a failed `ResponseRecord`, and
  releases the slot (client disconnect during the await releases under
  shield). Only failures raised while iterating the committed stream
  keep the in-stream framing (`data: {"error": ...}` + `[DONE]`).
  Test: `test_prefirstchunk_failure_is_http_502_and_failed_record`.
- **The idle reaper killed freshly booted backends.** Its clock was
  `last_request_at` alone (fallback `created_at`), so a backend loaded
  via the UI or a scheduler boot that had never served a request was
  reaped on the next tick ("stopped ... after 5877s idle" while the
  boot was 15s old) — and every following request paid a full engine
  re-boot before the engine saw traffic (the responses delay). Fix: new
  `ProviderInstance.backend_loaded_at` (nullable additive migration
  `d3a9c6e1f842`), stamped by `connection_manager._persist_status` on
  transitions into running/in_use (also when a loaded report arrives
  with a missing clock, e.g. after an admin restart), cleared when the
  backend leaves the loaded set, when the socket dies (`ws.py`
  `_mark_disconnected` + the presence sweep — a dead socket makes the
  backend state unknown, so the reconnect's `running` re-arms at the
  real boot time), and by the scheduler's `_persist_stopped` mirror.
  Never re-armed by running→in_use→running idle heartbeats. The reaper
  baseline is `max(last_request_at, backend_loaded_at)` with the old
  `created_at` fallback (tz-normalized before comparing). Exposed in
  the instance API payload for debugging. Tests: reaper (fresh load not
  reaped; load-clock reap), WS status-ingest clock assertions, and
  disconnect→reconnect clock re-arm.

- **halogen-flash disk cache never got its directory (found on
  `rocinante` @ matrix.thelink.family, 2026-10-07):** `build_env` exported
  `HALOGEN_CACHE_DIR` only when `cache_dir_enabled` was **explicitly**
  `True`, but the UI prunes untouched defaults and the schema default is
  `true` — a config of `{"disk_cache": {"cache_disk_gib": 512}}` started
  the engine with `HALOGEN_CACHE_DISK_GIB=512` (and the image's
  `HALOGEN_PROMPT_CACHE=2`) but **no cache dir**, so the disk cache had
  nowhere to persist. Fix: absent/null `cache_dir_enabled` now follow the
  schema default (export + create); only an explicit `false` disables.
  `schema.json` untouched (description wording aside, the gate change is
  provider-side; the committed fingerprint stays stable). Test:
  `test_cache_dir_emitted_when_key_pruned_away`. Apply by recreating the
  provider container after deploying the image.

Suites after the fix: admin **294** · lib **128** · mock **23** ·
llama-cpp **61** · halogen **70** · halogen-flash **158** · gufo **67**.
Client regenerated (no surface change: instance payloads are free-form
dicts).

---

## Phase 16 — Machine-scoped provider agents ✅

**Status: COMPLETE (slices 1–8 landed).** Slice 7 = React Agents page +
definition placement controls. **Slice 8 = the remaining law docs rewritten to
the agent model** (documentation-only): `docs/ws-protocol.md` (full rewrite —
agent socket, `agent.assignments.update`, per-backend `instance_id`, agent-level
`provider.status` vs `backend.status`, agent-keyed Redis), `provider/README.md`
(agent authoring guide — env vars, registration flow + `backends` response,
`BackendRegistry` + `make_handle` + `MultiPortServer`, `x-max-running-backends`;
Phase-14 shell/`registration_token`/`awaiting_config`/`no_config_nak` removed),
`admin/backend/docs/redis-keys.md` (every key re-addressed to the
`ProviderAgent` PK uuid; owner no-TTL, presence 60s; phantom
`im:vram:total`/`im:metrics:instance`/`im:metrics:alias` removed;
`im:logs:provider`/`im:logs:seq` re-keyed to agent), and `AGENTS.md`
(Architecture/Gotchas bullets). No code changed; admin suite still 330 green.

**Agent-delete path — landed (follow-up).** `DELETE /admin/api/agents/{agent_id}`
removes a decommissioned/renamed `ProviderAgent` (previously operators had to
clear the ghost row via raw SQL + Redis). Safety gate: **409 while
`websocket_connected`** (never delete a live agent); a disconnected agent's
backends are unschedulable ghosts regardless of `backend_status`, so they go
with it. The endpoint deletes the agent's `ProviderInstance` + `DefinitionAgent`
children **explicitly** (the ORM relationships carry no delete-orphan cascade /
`passive_deletes`, so `session.delete` would try to NULL the non-nullable child
FKs and fail before the DB `ON DELETE CASCADE` fires), clears the agent's Redis
WS/metrics keys (`im:ws:secret|epoch|owner|presence`, `im:metrics:cats` — keyed
by the PK uuid), and releases any in-process scheduler VRAM/`_booted` hold for
the removed backends via the existing `note_backend_stopped` hook. Response:
`{ok, agent_id, deleted_instances}`. The Agents UI gained a Delete danger button
(disabled + tooltip while connected) with a confirm dialog. Client regenerated
(`AdminService.deleteAgent`). Admin suite now **334 green** (4 new tests).

Prior — **slice 5 (`agent.assignments.update` push + event-driven placement
reconciliation) implemented.** The agent data model, migration, registration/
auth, agent-level WS, scheduler joins, placement CRUD, config-push addressing,
and the provider lib + all five provider packages speak the agent protocol; the
scheduler enforces the per-agent `max_running_backends` cap (hot-swap eviction)
and proactively warms an agent's assigned backends on connect; and a placement
change now propagates to a LIVE agent over its socket without a re-registration
(the slice-3 "prune only on next registration" interim is replaced by the shared
`reconcile_agent_placement` diff). **All suites green** (admin 328, lib 137,
mock 29, llama-cpp 63, gufo 69, halogen 72, halogen-flash 161; ruff + format
clean; no model/route-schema change this slice so alembic and the generated
client are unchanged — the added `request: Request` params do not alter
OpenAPI).

**Slice 6 (real-engine multi-backend-per-process) — landed.** Each hardware
provider now hosts a `BackendRegistry` of drivable `BackendLifecycle`s (one per
placed backend) and serves every backend's `/v1` on its own assignment port via
the new `provider_lib.serve.MultiPortServer` (dynamic listener sync on registry
change; a single-backend agent yields exactly one listener on `PROVIDER_PORT`,
byte-identical to pre-slice-6). The assignment `port` is threaded into each
engine driver so subprocess ports stay distinct (llama-cpp/gufo `serve_port+1`;
halogen `serve_port+1/+2`; halogen-flash uses static engine ports). halogen-flash
declares `x-max-running-backends: 1` in its `schema.json`, read into
`ProviderType.max_running_backends` so the admin places at most one backend per
its agent. Per-backend `config.update`/`cache.clear` resolve to the correct
lifecycle (gufo's per-instance cache dir now resolves from the target handle).
**Slice 6 review follow-up:** `MultiPortServer.sync` now log-and-continues per
backend so one failing app-build can't strand the others (M1, + test); gufo's
`extra_cache_dirs` skips the per-instance dir when `driver.instance_id` is unset
instead of falling back to the shared `MACHINE_UID` (M2, + test — two pre-registration
backends no longer clobber one dir); stale "single-backend-per-process" docstrings in
`assignments.py` updated and the no-factory refuse branch reworded as purely defensive
(N1). The bare `except A, B:` form is **correct** under the project's `>=3.14` floor
(PEP 758) and is what `ruff format` (target py314) canonicalizes to — parenthesizing it
is stripped by the formatter, so clarity is added via inline comments instead; the five
`run_async` clauses were simplified to `except asyncio.CancelledError:` (L1: with uvicorn
signal capture disabled, SIGINT exits via `main()`, never reaching `await serve_task`).
**Landed since:** the React Agents/placement UI (slice 7) and the
`provider/README.md` + `docs/ws-protocol.md` + `redis-keys.md` + `AGENTS.md`
rewrite (slice 8).


**Goal.** Stop binding a provider container to a single `ProviderDefinition`.
A container becomes a **provider agent** bound to a **machine + one provider
type**, running **1..N backends** (one per assigned definition) of that same
type. Add a per-type **`max_running_backends`** cap so engines that can only
host one model at a time (halogen-flash / single NPU) are limited to one
running backend per agent, hot-swapped by the scheduler.

**Why now.** Today one container = one alias = one backend = one port = one
WS. Scaling a machine to several models means several containers, several
registration tokens, and (for single-NPU engines) no clean way to say "this
engine can only hold one model at a time." The agent model collapses that
into one container per (machine, type) that owns its backends and one socket.

### Locked decisions (from operator Q&A)

1. **Agent identity** — new `ProviderAgent` row keyed
   `(machine_id, provider_type, agent_id)`. **Multiple agents may share a
   `(machine, type)`**; the operator-supplied `AGENT_ID` env discriminates
   them and keeps cherry-picked placements stable across restarts.
2. **Auth** — container authenticates with `MACHINE_UID` + a shared
   **`Machine.registration_secret`** + `provider_type` + `AGENT_ID`. The
   per-definition `registration_token` is **retired**. One secret per machine
   (all agents/types on it share it). Per-connection secret (`agent_secret`)
   is minted at registration into Redis, as before but agent-scoped.
3. **Placement** — `ProviderDefinition.agent_placement` ∈
   `any_of_type` | `specific`; `specific` uses a `definition_agents` link
   table (cherry-picked agent ids). UI dropdown drives it.
4. **Backend rows** — `ProviderInstance` = one backend on one agent for one
   definition, re-keyed to `(agent_id, provider_definition_id)`. Fan-out: a
   definition placed on several eligible agents gets one backend per agent;
   the scheduler balances across them.
5. **WebSocket** — **one agent-level socket**. Secret/epoch/presence and
   machine-metrics ownership move to the agent. Commands/events address a
   specific backend by `instance_id` in the payload. New
   `agent.assignments.update` command pushes add/remove of backends.
6. **Boot** — the **admin scheduler stays the boot + VRAM authority**. On
   agent (re)connect it **proactively warms the agent's assigned backends one
   at a time** (serialized per agent, gated by VRAM + `max_running`) until
   they no longer fit; the rest boot on demand. Idle reaper unchanged.
7. **`max_running_backends`** — declared in each type's shipped `schema.json`
   (`x-max-running-backends`, default unlimited), stored on `ProviderType`,
   enforced **per agent**. halogen-flash = `1`. To boot past the cap the
   scheduler **hot-swaps** (evicts the LRU running backend of that type on
   that agent).
8. **Ports** — per-container base port via `PROVIDER_PORT`; backends take
   `base_port + offset`. Admin keeps the per-machine `port_conflict` guard
   (now across all backends of all agents on the machine).
9. **Shells removed** — Phase 14 shell definitions + type adoption +
   `awaiting_config` are **dropped**; a definition is always typed and
   configured at create.
10. **Schema consensus** — voter universe is now **`ProviderAgent` rows of
    the type** (agents ship the schema). Decommissioned agents: keep rows,
    rely on **force-commit** (same posture as today's instance rows).
11. **Migration** — full replacement (no dual path). Version hard-fail forces
    admin-first then redeploy every agent.

### Sub-tasks (implementation order)

- [x] **A. Remaining law docs** — `docs/ws-protocol.md` (§2 registration body
      + validation + response = agent/machine-secret/assignments; §3 connect
      = agent secret/epoch/presence; §4 add `agent.assignments.update`,
      per-backend `instance_id` addressing, `provider.status` agent-level vs
      `backend.status` per-backend), `provider/README.md` (agent lifecycle,
      multi-backend, base+offset ports, `max_running`), `admin/backend/docs/redis-keys.md`
      (`im:ws:*`/`im:metrics:cats` → `{agent_id}`, `im:logs:provider` →
      `{agent_id}`), and **`AGENTS.md`** (its Architecture bullets still say
      "each own one inference backend" / "instance_secret" — must be
      corrected to the agent model). *(landed in slice 8 — all four docs
      rewritten to the agent model; documentation-only, admin suite still 330)*
- [x] **B. Models + migration** — add `ProviderAgent`, `definition_agents`;
      `Machine.registration_secret`; `ProviderType.max_running_backends`;
      `ProviderDefinition.agent_placement` (+ drop `registration_token`,
      make `provider_type`/`backend_config` non-null again); re-key
      `ProviderInstance` to `(agent_id, provider_definition_id)` and move
      `instance_status`/`version`/`websocket_connected`/`epoch`/
      `reported_schema_fingerprint` onto `ProviderAgent`. Data migration:
      for each existing instance, synthesize an agent
      `(machine, definition.provider_type, agent_id=<new>)` and repoint.
      Alembic revision + downgrade. *(rev `f1b6c2d84a97`; upgrade/downgrade/
      upgrade/check verified on a throwaway UTF8 DB)*
- [x] **C. Registration + auth** — `app/api/admin/providers.py`: accept
      machine-secret auth (401/404), upsert `ProviderAgent`, resolve
      placement → upsert `ProviderInstance` backends (stopped), mint
      `im:ws:secret:{agent_id}`, return the assignment set. Retire the
      definition-token path.
- [x] **D. Agent WS + wire** — `app/api/ws.py`, `app/services/connection_manager.py`,
      both `wire.py` mirrors: socket keyed by `agent_id`; epoch/presence/
      owner at agent level; `send_command(agent_id, type, payload)` with
      `instance_id` in payload for per-backend commands; add
      `AGENT_ASSIGNMENTS_UPDATE` FrameKind; `provider.status` → agent_status,
      `backend.status` → per-backend. Presence sweep marks the **agent**
       disconnected (and its backends unschedulable). *(`AGENT_ASSIGNMENTS_UPDATE`
       FrameKind landed in slice 5 (both mirrors + drift guard green);
       `AWAITING_CONFIG` removed from both mirrors)*
- [x] **E. Scheduler** — `app/services/scheduler.py`: candidates join through
      agent connectivity + placement; `max_running_backends` gate + hot-swap
      eviction (per agent, per type); serialized **proactive init warm-up** on
      agent connect (per-agent `_warming` guard + per-alias `state.lock` under
      `im:sched:lock`); VRAM ledger stays per booted backend.
      *(join-through-agent + eviction/idle keyed by agent done in slice 3;
      slice 4 landed: `_max_running_for_type`/`_running_on_agent`/
      `_ensure_agent_capacity` gate the boot path and hot-swap an idle
      same-agent backend LRU-first (never busy, never the target, never
      cross-agent) reusing `_eviction_candidates` (agent-scoped, loaded-not-hold
      filter) + `_stop_instance` + the `_evicting` guard; `warm_up_agent` boots
      an agent's assigned backends one at a time MRU-first, bounded by VRAM +
      `max_running`, never evicting (stops at the first that doesn't fit),
      triggered from the WS accept path behind
      `settings.SCHEDULER_WARMUP_ON_CONNECT` (default on; off in the test app so
      raw-socket tests keep their frame contract). `x-max-running-backends` is
       now read from the committed schema into `ProviderType.max_running_backends`
       at every commit point (bootstrap / sole-voter / consensus / force-commit).
       Review hardening: warm-up re-checks `_is_loaded` inside the per-alias
       `state.lock` (no redundant `backend.start` for a backend a concurrent
       request booted after the target list was built), breaks the pass when the
       agent's mirrored `websocket_connected` drops mid-pass, and the WS accept
       path keeps strong references to its detached background tasks
       (`_spawn_background` + a module-level set) so a long boot is never GC'd
       mid-flight; the single-type-per-agent invariant the cap scoping relies on
        is documented on `_running_on_agent`/`_ensure_agent_capacity`.
        Slice 5 wired the residual seam: `push_agent_assignments` calls
        `warm_up_agent(agent_id)` (as a tracked background task) after a
        placement push that added backends.)*
- [x] **F. Placement CRUD + push** — `app/api/admin/definitions.py`:
      `agent_placement` + `agents` on create/PATCH; on placement change push
      `agent.assignments.update` to affected agents; config PATCH still
      pushes `provider.config.update` per backend. New
      `app/api/admin/agents.py` reads. Machines route exposes/rotates
      `registration_secret`. *(placement persistence + agent reads + machine
      secret done in slice 3; **slice 5 landed the live push**: new
      `app/services/assignments.py` is the ONE placement-diff path
      (`reconcile_agent_placement` — create stopped rows at `base_port + next
      free offset`, retire de-placed rows busy-safe, build the full assignment
      set), shared by registration (`prune_unplaced_backends` is now a thin
      wrapper) and `push_agent_assignments`. Create/PATCH (placement/enabled/
      alias) fan out via `push_agent_assignments_for_type`; a connected agent
      gets `agent.assignments.update` (epoch-fenced) then the slice-4
      `warm_up_agent` re-warm seam fires as a background task; a disconnected
      agent still has ghost rows pruned but receives no frame. Busy-safe
      removal mirrors on both sides (admin leaves a `running`/`in_use` row +
      reports `refused`; provider refuses to stop a busy handle). DELETE needs
      no live push (the connected-host guard already forbids it). provider_lib
      `assignments.install_assignment_handler` reconciles the `BackendRegistry`
      (add via a package `make_handle` factory, busy-safe remove, ack
       `{ok, added, removed, refused}`) and is auto-installed by
       `install_backend_ops` so every package acks (no 30s timeout); the mock
       supplies a factory and demonstrates N add/remove.
       *Slice 5 review hardening:* **B1** the provider remove-loop now resolves
       each handle's identity via `lifecycle.instance_id or` registry key (like
       `resolve_target`) so a single-backend agent whose handle is still keyed
       `""` (pre-`apply_registration`) is NOT torn down on a same-type push;
       **H1** `push_agent_assignments` runs the per-machine connected-peer port
       clash check on newly-created rows and rolls back + refuses (`port_conflict`)
       any collision (registration already had `_reject_port_clash`); **M1**
       registration renumber reserves retained busy-ghost ports so two rows on
       one agent never share a port; **M2** warm-up is gated on the provider's
       `ack_added ∩ diff.added` (not admin `diff.added` alone) so a refused add
       on a single-backend agent triggers no futile `backend.start`; **L1** a
       `provider_type` change is a placement trigger (pushes the new type);
       **L3** the push also catches `TimeoutError` (30s stall) for a clean skip;
       **L2** documented that an already-hosted handle's alias rides the next
       `provider.config.update`/re-registration (not refreshed by
       `assignments.update`). New discriminating tests: admin
       `test_push_no_warm_when_provider_refuses_add`,
       `test_push_refuses_port_collision_with_connected_peer`,
       `test_renumber_avoids_retained_busy_ghost`; provider-lib
       `test_single_backend_placeholder_key_is_kept`.)*
- [~] **G. Provider lib** — `provider_lib/admin_client.py`: register as agent
      (machine secret + agent_id + type), one socket, dispatch per-backend
      commands by `instance_id`, handle `agent.assignments.update`
      (spawn/retire `BackendLifecycle`s). `config.py`: `MACHINE_SECRET`,
      `AGENT_ID`, `PROVIDER_TYPE`, base `PROVIDER_PORT`. `app_factory.py`:
      serve N backend apps (one per backend port). `schema.py`: read
      `x-max-running-backends`. *(agent registration + WS + `MACHINE_SECRET`/
      `AGENT_ID` + per-backend `instance_id` status/commands done; slice 5 added
      `assignments.install_assignment_handler` (auto-installed by
      `install_backend_ops`, busy-safe registry reconcile + ack) and the mock's
      `make_handle` factory; **slice 6 landed** N-app serving via
      `provider_lib.serve.MultiPortServer` (one uvicorn listener per hosted
      backend on its assignment port, dynamic sync on registry change) and
      `x-max-running-backends` is read from the committed schema into
      `ProviderType.max_running_backends`)*
- [x] **H. Provider packages** — mock/llama-cpp/gufo/halogen/halogen-flash:
       multi-backend driver instances keyed by definition; halogen-flash sets
       `x-max-running-backends: 1` in its `schema.json`; each keeps its
       `install_config_handlers`/`install_backend_ops` per backend.
       *(all five migrated to the agent registration/WS/status contract;
       **slice 6 landed**: each real package builds a `BackendRegistry` of
       drivable lifecycles (one per placed backend) via `build_registry`/
       `make_handle`, threads the assignment `port` into its driver so engine
       ports stay distinct (llama-cpp/gufo `serve_port+1`; halogen
       `serve_port+1/+2`; halogen-flash uses static engine ports), serves each
       backend's `/v1` on its own port through `MultiPortServer`, and routes
       per-backend `config.update`/`cache.clear` to the correct lifecycle;
       halogen-flash declares `x-max-running-backends: 1` so the admin places
       at most one backend per its agent. Single-backend behavior is unchanged.)*
- [x] **I. Admin UI** — Machines page: show/rotate secret. New **Agents**
      page (or expand Machines): list agents per machine/type, hosted
      backends, `waiting_schema`. Definitions page: placement dropdown
      (any-of-type / pick agents). Instances page: group backends under their
      agent; drop the `awaiting_config` badge. *(deferred to slice 7; the
      existing UI was only patched to keep `tsc` green after the client
      regen — shell/token fields removed)*
- [x] **J. Client** — `bash scripts/generate-client.sh` after B/C/D/F routes.
- [~] **K. Tests** — admin: agent registration + placement resolution +
      machine-secret auth + `max_running` hot-swap + proactive warm-up +
      per-backend frame routing + presence-sweep-disconnects-agent; provider
      lib: multi-backend dispatch + `agent.assignments.update`; each
      provider package: N backends + halogen-flash singleton. Wire drift
      guard for the new FrameKind. *(admin + provider suites rewritten for the
      agent contract and green; slice 4 added `test_scheduler.py` coverage for
      `max_running` cap=1 hot-swap / busy-never-evicted / cap=0 unlimited /
      same-agent-only / LRU order and for serialized VRAM+cap-bounded proactive
      warm-up (MRU order, skip-already-running, per-agent guard, boot-failure
       resilience) plus a WS connect-trigger test; **slice 5 added**
       `test_assignments_push.py` (admin: create-row + warm-up, busy-safe
       refusal keeps the row, idle-drop removes it, disconnected prune-only,
       CREATE fan-out to connected agents, shared-diff equivalence),
       provider-lib `test_assignments.py` (add/remove/idempotent/busy-refuse/
       single-backend-refuses-unknown/ack-shape) and mock
       `test_assignments_wiring.py` (N add+remove reconcile + per-backend
        `provider.config.update` reaches the correct lifecycle on a 2-backend
        agent); **slice 5 review added** discriminating tests — admin
        `test_push_no_warm_when_provider_refuses_add` (M2),
        `test_push_refuses_port_collision_with_connected_peer` (H1),
        `test_renumber_avoids_retained_busy_ghost` (M1) and provider-lib
        `test_single_backend_placeholder_key_is_kept` (B1); **slice 6 added**
        provider-lib `test_serve.py` (`MultiPortServer` per-port listener
        bookkeeping: start/add/remove reconcile, base-port fallback, dynamic
        sync via the registry change listener) and per-package
        `test_multi_backend.py` (llama-cpp/gufo/halogen: `build_registry`/
        `make_handle` thread the assignment port into the driver so engine
        ports stay distinct, `agent.assignments.update` spawns a fresh
        drivable lifecycle, and per-backend `config.update` reaches the correct
        driver only) plus `test_serve_config.py` guards that `run_async` serves
        via `MultiPortServer` (never `base_port=`) and halogen-flash's
        `x-max-running-backends: 1` schema pin)*
- [ ] **L. Conformance** — re-run openresponses suite on the mock agent
      (now hosting the mock definition) to confirm the request path is
      unchanged by the agent indirection. *(requires a deployed admin +
      mock agent; run after deploy)*

### Accepted risks / notes

- **Breaking change, no dual path.** Every provider container must be
  redeployed with the new env (`MACHINE_SECRET`, `AGENT_ID`, `PROVIDER_TYPE`);
  the version hard-fail enforces admin-first deploys anyway.
- **`AGENT_ID` discipline.** Two containers on one machine+type that forget
  to set distinct `AGENT_ID`s collide onto one agent row. Mitigation: the
  base-port `port_conflict` guard catches the common case; document loudly.
- **Proactive warm-up vs VRAM thrash.** Warming many assigned backends on a
  small machine can evict as fast as it boots; the one-at-a-time + fit-stop
  bound keeps it from looping (stop at first non-fit).
- **Phase 14 reversal.** Removing shells/`awaiting_config` reverts part of
  Phase 14; its tests are deleted/rewritten, not kept.

---

## Phase 17 — Per-GPU machine metrics + hardware union ✅

**Phase 17 landed (slices 1–6).** Device-isolated agents (one GPU each) now
merge into a full machine inventory and live snapshot:

- **Admin hardware union (slice 1):** per-GPU-uuid union at registration
  (`merge_hardware_union`, `normalize_gpu_uuids`) with survivor-aware stale-drop
  and `machine.total_vram_bytes` as the auto-sum of the union; `assigned_gpus`
  normalized to uuid strings; agent DELETE recomputes the union + re-sums VRAM.
- **Provider GPU-scoped split emitter (slice 2):** `ASSIGNED_GPU_UUIDS` env
  (implicit visible==owned + explicit), `GPU_CATEGORIES`/`MACHINE_WIDE_CATEGORIES`
  split, `filter_gpus`/`parse_gpu_assignment`/`gpu_uuid`, AMD stable PCI-slot
  uuid; the emitter loop runs from connect and `set_owned` toggles only the
  machine-wide categories.
- **Admin per-GPU metrics merge (slice 3):** GPU categories accepted from every
  agent into per-agent partials `im:metrics:machine:{uid}:agent:{id}`,
  machine-wide owner-gated, merged on read (`read_machine_metrics`) + new
  `GET /admin/api/machines/{id}/metrics`.
- **UI (slice 4):** live per-GPU metrics panel + read-only auto-sum VRAM field.
- **Mock + compose (slice 5):** synthetic two-GPU emitter + `two-gpu` compose
  profile for end-to-end demo with no GPU present.
- **Law docs (slice 6):** this section + `AGENTS.md`, `ARCHITECTURE.md`
  §4/§8/§9/§13, `docs/ws-protocol.md`, `admin/backend/docs/redis-keys.md`,
  `provider/README.md` rewritten to the shipped split model.

**Final test counts:** admin 364 · provider lib 159 · mock 42 · llama-cpp 72 ·
gufo 78 · halogen 80 · halogen-flash 167.

**Goal.** On a machine whose provider agents each see only a **subset** of the
GPUs (device-isolated containers — one dedicated GPU per agent, the production
layout on the provider host), the machine inventory and live metrics show
**all** GPUs, and `machine.total_vram_bytes` (the scheduler's admission
budget) reflects the full union instead of a single agent's view.

**Why now.** The deployment runs two agents on one machine, each pinned to its
own GPU. Each container's `nvidia-smi`/sysfs only sees its own device, and two
coupled bugs collapse the machine to 1 GPU:

1. **Registration hardware merge is last-writer-wins per top-level key**
   (`app/api/admin/providers.py` step 5): `merged.update(body.hardware)`
   replaces the whole `gpus` list and overwrites `total_vram_bytes` with the
   last registrant's single-GPU report. The inline comment claims UNION, but
   only top-level keys (`cpu`, `ram`) survive from earlier agents. Side
   effect: the scheduler's VRAM budget
   (`machine.total_vram_bytes - held_on(machine)`, `scheduler.py`) is
   **halved** on a 2-GPU box — admission is wrong, not just display.
2. **Machine-metrics ownership is all-or-nothing** (`services/metrics_service.py`):
   exactly one owner agent per machine emits the full snapshot and
   `handle_machine_metrics` **drops** frames from every other agent. The
   owner's `vram`/`gpu_usage` sections only ever contain its own GPU.
   `ProviderAgent.assigned_gpus` is persisted at registration and documented
   as "VRAM accounting + metrics dedup" but is used by neither.

Related gaps found while diagnosing: the live snapshot
`im:metrics:machine:{machine_uid}` has **no reader** (no admin endpoint or UI
consumes it — the machines page renders only the Postgres `hardware`
inventory); `assigned_gpus` is typed `list[str]` (UUIDs) in `models.py` but
registration stores full GPU dicts; `agents.py` DELETE leaves the dead
agent's GPU entries in the machine union.

### Locked decisions (from operator Q&A)

1. **GPU ownership = implicit + explicit override.** By default an agent
   reports exactly the GPUs it can see inside its container (device
   isolation makes visible == owned — zero new config for this deployment).
   Optional `ASSIGNED_GPU_UUIDS` env (space-delimited UUIDs or indices,
   `METRICS_CATEGORIES` style) filters the report for containers that see all
   GPUs but should own a subset.
2. **`machine.total_vram_bytes` = auto-sum** of the unioned per-GPU
   `total_vram_bytes`, recomputed on every hardware merge at registration.
   The UI field becomes read-only display (it already advertises "refreshed
   from provider hardware").
3. **Split category ownership.** `cpu` / `os_ram` / `storage` are machine-wide
   and visible from any container → keep the existing single-owner lease
   (`im:metrics:owner:{machine_uid}`). `vram` / `gpu_usage` are per-GPU →
   emitted by **every** agent that declares them (filtered to its assigned
   GPUs) and merged per-GPU-UUID by the admin.

### Sub-tasks (implementation order)

- [x] **A. Provider lib — GPU scoping + split emitter** ✅ (Slice 2 landed)
  - `ProviderSettings`: new optional `ASSIGNED_GPU_UUIDS` env (space-delimited
    GPU UUIDs or decimal indices).
  - `metrics.py`: `filter_gpus(sample, assignment)` — empty assignment =
    pass-through (implicit); otherwise keep GPUs whose `uuid` or `id` matches.
    Category split constants: `GPU_CATEGORIES = {vram, gpu_usage}`,
    `MACHINE_WIDE_CATEGORIES = {os_ram, cpu, storage}`.
  - `MachineMetricsEmitter`: GPU categories emit from **every** agent that
    declares them (filtered), independent of ownership; machine-wide
    categories only between `metrics.assign`/`unassign` (unchanged lease
    semantics). Payload gains `assigned_gpus: [uuid, ...]` for attribution.
  - `build_hardware_report` in every hardware provider (llama-cpp, gufo,
    halogen, halogen-flash) + mock: filter `gpus` through the assignment;
    per-agent `total_vram_bytes` = sum of the filtered list.
  - **Slice 2 note (provider-side only):** `ASSIGNED_GPU_UUIDS` +
    `assigned_gpu_tokens` land in `provider_lib/config.py`; `GPU_CATEGORIES` /
    `MACHINE_WIDE_CATEGORIES`, `parse_gpu_assignment`, `filter_gpus`, and the
    `gpu_assignment` threading through `collect_vram`/`collect_gpu_usage`/
    `collect_machine_snapshot` land in `provider_lib/metrics.py`. The emitter
    now runs whenever `start()`ed (started on connect in all four hardware
    providers' `run_async.on_connected` + `register_and_connect`), with
    `set_owned(bool)` toggling only the machine-wide categories; GPU categories
    emit regardless of ownership and an empty composed snapshot is skipped
    (no frame sent). `on_metrics_assign` → `set_owned(True)`,
    `on_metrics_unassign` → `set_owned(False)` (never stops the loop);
    disconnect resets `_owned` and stops the loop. `build_hardware_report`
    filters the GPU sample + sums `total_vram_bytes` from the filtered set in
    llama-cpp/gufo/halogen/halogen-flash; the mock report is assignment-aware
    over its single fake GPU (default output unchanged). Admin merge is Slice 3.
- [x] **B. Admin — hardware union at registration** ✅ (Slice 1 landed)
  - `providers.py` step 5: merge `machine.hardware["gpus"]` **by UUID**
    (latest report wins per UUID; never erases other agents' entries);
    recompute `machine.total_vram_bytes = sum(g["total_vram_bytes"])`;
    `cpu`/`ram` top-level keys keep last-writer-wins.
  - Normalize `ProviderAgent.assigned_gpus` to UUID **strings** (matches the
    `list[str]` model + ARCHITECTURE §4; today dicts are stored).
  - `agents.py` DELETE: recompute the machine union (and VRAM sum) from the
    remaining agents' `assigned_gpus`; delete the agent's metrics partial.
  - **Slice 1 note:** the registration union (`merge_hardware_union`),
    `assigned_gpus` UUID normalization (`normalize_gpu_uuids`), and the
    `agents.py` DELETE union recompute are landed and covered by new admin
    tests. The pure helpers live in `app/services/hardware.py` (shared by the
    registration and delete paths). Two review fixes landed: the registration
    stale-drop is **survivor-aware** (a GPU another agent still claims is never
    dropped by one agent's re-registration, mirroring the DELETE path), and the
    auto-sum only overwrites `total_vram_bytes` when a GPU list is actually in
    play — a report with no `gpus` key against a machine with a manually-set
    budget and no existing union **preserves** that budget (consistent with the
    DELETE no-op). The "delete the agent's metrics partial" sub-step is deferred
    to Slice 3 (Redis metrics keys) — the DELETE endpoint does not touch Redis
    metrics keys yet.
- [x] **C. Admin — per-GPU metrics merge** ✅ (Slice 3 landed)
  - `metrics_service.handle_machine_metrics`: accept GPU-category sections
    from any connected agent (no owner gate); store per-source partial
    `im:metrics:machine:{machine_uid}:agent:{agent_id}` (TTL 30s, refreshed
    per frame). Machine-wide sections stay owner-gated and refresh the lease
    as today.
  - Merge-on-read helper: union per-GPU entries across live partials;
    recompute aggregates (`vram.total/used/free/gpu_count`, `gpu_usage`
    average + per-GPU list).
  - New read path: `GET /admin/api/machines/{id}/metrics` returning the
    merged snapshot with per-GPU `agent_id` attribution (the snapshot
    finally gets a consumer).
  - **Slice 3 note (admin-side only):** `redis_keys` gains
    `metrics_agent_partial_key`/`metrics_agent_partial_prefix`
    (`im:metrics:machine:{uid}:agent:{id}`, nested under the machine
    snapshot namespace). `handle_machine_metrics` now splits the frame:
    GPU categories (`GPU_CATEGORIES = {vram, gpu_usage}`) are written to a
    minimal per-agent partial regardless of ownership (`assigned_gpus` is
    NOT stored — attribution comes from the unioned `vram.gpus` uuids);
    machine-wide categories (`MACHINE_WIDE_CATEGORIES = {os_ram, cpu,
    storage}`) stay owner-gated and refresh the lease + write
    `im:metrics:machine:{uid}` (with `owner_agent_id`). A non-owner's
    GPU-only frame is now stored (previously the whole frame was dropped);
    a non-owner's machine-only frame is still dropped. `read_machine_metrics`
    SCANs the partials, unions GPUs by their real `uuid` (entries without a
    uuid are skipped, matching `hardware.merge_hardware_union`; latest-writer-
    wins per uuid, each tagged with its `agent_id`), recomputes
    `vram`/`gpu_usage` from the union with defensive numeric coercion,
    overlays the owner machine-wide snapshot, and lists only agents that
    actually contributed a GPU in `reporting_agents`. New endpoint
    `GET /admin/api/machines/{machine_id}/metrics` (in `machines.py`) exposes
    it; the client was regenerated. `assign_ownership` now claims the
    machine-wide lease ONLY when the agent declares a machine-wide category
    (a GPU-only agent no longer starves the lease); `release_ownership` is
    unchanged. No Alembic migration. Law-doc updates (redis-keys.md §,
    ws-protocol.md, ARCHITECTURE §8/§9) are deferred to Slice 6.
- [x] **D. Admin UI + client regen** ✅ (Slice 4 landed)
  - `bash scripts/generate-client.sh` after the new route.
  - `machines.tsx`: verify the GPU panel shows both GPUs post-merge; surface
    live per-GPU used/total + utilization from the new endpoint.
  - Machine edit dialog: `total_vram_bytes` read-only with an auto-sum hint.
  - **Slice 4 note (frontend-only):** `types/admin.ts` gains a hand-written
    `MachineMetrics` shape (loose `vram`/`gpu_usage`/`os_ram`/`cpu`/`storage`/
    `owner_agent_id`/`reporting_agents` mirroring `read_machine_metrics`).
    `useAdminData.ts` gains `machineKeys.metrics(machineId)` +
    `useMachineMetrics(machineId, { enabled, refetchInterval })` (default 5s,
    gated on `machineId != null`). `machines.tsx` adds a `LiveMetricsPanel` in
    the expanded row (fetches only while expanded) showing per-GPU
    name/vendor + used/total GiB + utilization % (vram `utilization` or
    `gpu_usage` matched by uuid) + contributing `agent_id` (short), the
    `gpu_count`/machine totals, machine-wide `cpu`/`os_ram` via `KvList`, the
    `owner_agent_id`, and a muted "No live metrics yet — agents report every
    ~10s." fallback (never crashes on missing sections). `total_vram_bytes` is
    now disabled/read-only on edit with the auto-sum `FormDescription` and is
    omitted from the PATCH body (still editable on create; zod unchanged).
    Client already had `getMachineMetrics` — no regen needed.
- [x] **E. Mock provider + compose e2e** ✅ (Slice 5 landed)
  - Mock supports a fake two-GPU inventory + `ASSIGNED_GPU_UUIDS` filtering,
    so compose can run two mock agents on one machine (distinct `AGENT_ID`,
    one fake GPU each) and the UI must show 2 GPUs / summed VRAM.
- [x] **F. Law docs** ✅ (Slice 6 landed)
  - `ARCHITECTURE.md` §4 (union semantics, auto-sum, `assigned_gpus` = UUIDs),
    §8 rewrite (per-GPU merge + owner-gated machine-wide), §9 key table,
    §13 known limitations (overlapping-visibility last-writer-wins note).
  - `docs/ws-protocol.md` (`metrics.machine` payload + `assigned_gpus`,
    split assignment semantics), `admin/backend/docs/redis-keys.md`
    (new `im:metrics:machine:{uid}:agent:{id}` partial),
    `provider/README.md` (`ASSIGNED_GPU_UUIDS` env), `AGENTS.md`
    (metrics gotcha: "exactly one agent per machine" → machine-wide only).

### Tests

- Admin: two-agent registration union (2 GPUs, `total_vram_bytes` = sum);
  `assigned_gpus` UUID normalization; agent-delete recompute;
  `handle_machine_metrics` accepts non-owner GPU frames but still drops
  non-owner machine-wide frames; merge-on-read aggregates; new metrics
  endpoint. Update `test_metrics_ownership.py` + `test_registration.py` for
  the split semantics.
- Provider lib: `filter_gpus` implicit/explicit (uuid + index forms); emitter
  gating (GPU categories without ownership, machine-wide only while owned).
- Mock: two-agent split-machine scenario via compose.

### Accepted risks / notes

- **Version hard-fail applies**: deploy the admin first, then recreate both
  agents (standard order).
- **No Alembic migration** — JSON columns unchanged; the `assigned_gpus`
  content fix self-heals at each agent's next registration.
- **Overlapping visibility misconfig** (two agents with `--gpus all` and no
  `ASSIGNED_GPU_UUIDS`): both report the same UUIDs → per-UUID
  last-writer-wins, values still correct (same physical GPU); attribution
  may flap between the two reporters. Documented; the explicit env resolves
  it.
- **Out of scope**: per-GPU VRAM admission (the scheduler still budgets per
   machine — correct once the union is fixed); per-backend
   `ProviderInstance.assigned_gpus` stays unused.

---

## Phase 18 — Embeddings + modality-scoped endpoints ✅

**Status: SHIPPED (2026-10-08; design locked via operator Q&A, implemented
slice-by-slice via the `delegated-slice-delivery` skill, all 7 slices green +
verified end-to-end against a local admin + mock).** Adds the first non-chat
modality: OpenAI-spec `POST /v1/embeddings`, served through litellm exactly
like `/v1/responses`/`/v1/chat/completions`, and the **modality** concept that
scopes which definition serves which endpoint family (so `/v1/audio/*` can land
later without a second redesign).

**Goal.** Let an operator publish an embedding model (e.g. a GGUF `bge-m3`)
as a normal `ProviderDefinition` and serve `POST /v1/embeddings` to spec.
Every definition declares a **modality** (`llm` | `embedding`; `audio`
reserved); every provider **type** declares which modalities it can host. The
admin routes a request to a definition only when the endpoint family matches
the definition's modality, and the scheduler treats an embedding backend as
just another bootable instance.

**Why now.** Embeddings were an accepted 501 regression after the overhaul.
The litellm + scheduler + agent model now makes them a thin addition (a new
route + a new litellm `mode` + a per-type capability flag + one engine flag),
and doing it behind a `modality` enum (rather than a boolean `is_embedding`)
is what keeps the door open for the OpenAI audio endpoints without re-cutting
the data model again.

### Locked decisions (from operator Q&A)

1. **Field name + values** — `ProviderDefinition.modality` ∈ `llm` (default) |
   `embedding`; `audio` reserved (single value, not granular speech/
   transcription — the endpoint family is the unit). This is the admin's
   **routing key**: `/v1/embeddings` accepts only `embedding` aliases;
   `/v1/responses` + `/v1/chat/completions` accept only `llm` aliases (the
   wrong kind is a clean 404, mirroring the disabled-alias path).
2. **Type capability** — `ProviderType.serves_modalities` (JSON list, default
   `["llm"]`), declared via a top-level `x-serves-modalities` in the shipped
   `schema.json` — the exact mechanism already used for `x-max-running-backends`
   (read into the column at every schema-commit point: bootstrap / sole-voter /
   consensus / force-commit). Definition create/PATCH validates
   `modality ∈ serves_modalities` → 422 otherwise. `llama-cpp` declares
   `["llm","embedding"]`; halogen / halogen-flash / gufo stay `["llm"]`.
3. **Provider learns modality via the wire** — `modality` is added to the
   `agent.assignments.update` assignment entry, the `provider.config.update`
   payload, and the registration `backends[].definition` response (single
   source of truth = the definition row; admin routing and engine boot can
   never drift). llama-cpp maps `modality=="embedding"` → boots `llama-server
   --embedding`.
4. **llama-cpp engine flags** — `--embedding` is auto-added from modality;
   `--pooling` (none|last|mean|cls|…) is a new schema option so per-model
   pooling is settable in the same schema change (avoids a second consensus
   bump).
5. **litellm** — embedding aliases registered with `mode: "embedding"`
   (`ensure_registered(alias, mode=...)`; cache keyed by alias since an alias
   is exactly one modality). Verified against litellm 1.103.2:
   `aembedding(custom_llm_provider="openai", api_base=…)` posts to
   `{api_base}/embeddings`. Non-streaming only.
6. **Persistence** — `TokenUsageSample` only (prompt tokens); **no
   `ResponseRecord`** (embeddings are not conversation turns).
7. **`/v1/models`** — lists `llm` **and** `embedding` definitions; each object
   carries a `modality` marker.
8. **Mock serves embeddings** — deterministic fake vectors so the full local
   path (compose `dev.sh` + integration tests) exercises embeddings with no
   GPU.
9. **Immutability** — `modality` is refused (409) on PATCH while backends are
   attached, same gate as `provider_type` (a llama-cpp agent booted
   `--embedding` cannot silently become a chat backend).

### Sub-tasks (implementation order — additive; each slice leaves the tree green)

- [x] **Slice 1 — Admin model + type capability (no behavior change).**
      `ProviderDefinition.modality` (default `llm`, NOT NULL) +
      `ProviderType.serves_modalities` (default `["llm"]`) columns; Alembic
      migration (additive, backfills existing rows to `llm`; downgrade safe).
      Read `x-serves-modalities` into the column at every commit point (extend
      the `x-max-running-backends` sync helper in `providers.py`). Definition
      CRUD: accept `modality` on create/PATCH, validate
      `modality ∈ serves_modalities` (422), add to `DefinitionCreate`/
      `DefinitionPatch`/`definition_dict`/`_NON_NULLABLE_FIELDS`, refuse
      modality change 409 while backends attached. `scripts/generate-client.sh`.
      Tests: migration up/down/check, validation matrix, serves_modalities
      sync, immutability gate.
- [x] **Slice 2 — litellm embedding registration + `/v1/embeddings` route.**
      `alias_registry.ensure_registered(alias, mode="chat")` gains a `mode`
      param (`"embedding"` registers `mode: "embedding"`). New
      `app/api/v1/embeddings.py`: resolve definition, 404 unless
      `modality==embedding`, 400 on missing/empty `input`, `scheduler.acquire`,
      `litellm.aembedding(api_base=f"{base}/v1", custom_llm_provider="openai",
      api_key="unused", input=…, dimensions=…, user=…)`, return spec
      `CreateEmbeddingResponse`, persist `TokenUsageSample`, shielded
      `scheduler.release` in `finally`; error contract = chat non-stream
      (400/404/503/504/502). Remove `("POST","/v1/embeddings")` from
      `stubs.py`. Gate `/v1/responses` + `/v1/chat/completions` to reject an
      `embedding` alias (404). `/v1/models` adds the `modality` marker.
      `scripts/generate-client.sh`. Tests: happy path (fake/mock upstream),
      both-direction type gating, usage persistence, error contract, models
      marker.
- [x] **Slice 3 — Wire: push modality to the provider.** Add `modality` to the
      assignment entry (`app/services/assignments.py`), the
      `provider.config.update` payload (`app/services/config_update.py`), and
      the registration `backends[].definition` (`app/api/admin/providers.py`);
      provider_lib `admin_client.py`/`config.py`/`registry.py`/`ops.py`/
      `config_update.py` read + persist it into `provider_config.json` and the
      per-backend handle. FrameKind unchanged (payload field only) — wire drift
      guard stays green. Tests: payload round-trip carries modality; provider
      persists it; drift guard.
- [x] **Slice 4 — provider_lib embeddings surface.** `BackendDriver.embeddings()`
      ABC default raises `NotImplementedError` (→ 501); `BackendLifecycle.embeddings()`
      slot-admitted wrapper (acquire → await driver → release in `finally`,
      BackendBusy/NotReady → 429/503); `app_factory.py` `POST /v1/embeddings`
      route. Tests: 501 when driver lacks it, slot acquire/release on a fake
      driver, busy/not-ready mapping.
- [x] **Slice 5 — llama-cpp embeddings.** `schema.json`: add top-level
      `x-serves-modalities: ["llm","embedding"]` + a `--pooling` option
      (`x-flag: --pooling`); bump the pinned fingerprint in
      `test_schema_sections.py`. `command.py`: `build_llama_command` adds
      `--embedding` when the handle's modality is `embedding` + `--pooling`
      when set. `driver.py`: read modality, add `embeddings()` proxying
      upstream `/v1/embeddings`. Tests: command emits flags, driver proxy,
      fingerprint pin.
- [x] **Slice 6 — mock embeddings.** Mock driver implements `embeddings()`
      (deterministic fake vectors, spec `CreateEmbeddingResponse` + usage);
      mock `schema.json` declares `x-serves-modalities: ["llm","embedding"]`.
      Enables full local e2e via `./scripts/dev.sh`. Tests: mock embeddings
      shape + determinism.
- [x] **Slice 7 — Frontend + docs-to-shipped + conformance.** Definitions
      create/edit: modality selector gated by the chosen provider type's
      `serves_modalities`; Provider Types page shows `serves_modalities`;
      `/v1/models` display shows modality. `scripts/generate-client.sh`.
      Re-run openresponses conformance (chat/responses must not regress) +
      an embeddings integration check against the deployed admin + mock.
      Rewrite this Phase 18 section + ARCHITECTURE.md to the shipped state.

### Shipped verification (2026-10-08)

- **Clean-env suites (all green together):** admin **399**, provider/lib **169**,
  mock **49**, llama-cpp **82**, gufo **78**, halogen **80**, halogen-flash
  **167**.
- **Schema fingerprints bumped** (fleet re-present required): llama-cpp
  `03373ece…` (`x-serves-modalities` + `server.pooling`, enum includes `null`,
  `--pooling` gated to embedding modality only), mock `017addc9…`
  (`x-serves-modalities`).
- **End-to-end (local `dev.sh` admin + mock):** created a `modality=embedding`
  mock definition → `POST /v1/embeddings` returns a spec `CreateEmbeddingResponse`
  (deterministic 8-dim vectors; `str` and `list` inputs → correct indices);
  `TokenUsageSample` written with `prompt_tokens` and `completion_tokens=0`;
  **no** `responses` row created. Type-gating both directions → 404
  (`/v1/responses` on an embedding alias, `/v1/embeddings` on an llm alias).
  `/v1/models` carries the `modality` marker per object.
- **Frontend:** definitions create/edit modality selector gated by the chosen
  type's `serves_modalities` (locked while backends attached), Modality column +
  detail, provider-types Serves column, playground `/v1/models` marker.
  `bun run build` + biome lint clean.
- **Local dev note:** the schema-consensus gate correctly 409-`schema_pending`
  when ghost agent rows from prior dev sessions pin the old fingerprint; resolve
  with `POST /admin/api/provider-types/{name}/pending/commit` (or a fresh DB).
- **Production follow-up (`--jinja` gating):** a real embedding backend
  (`rocinante-embed`, llama-cpp) booted with `--embedding` **and** `--jinja`.
  `--jinja` applies the model's chat template to inputs, which is wrong for
  embeddings (template tokens pollute the vector). `command.py` now defaults
  `--jinja` **OFF** for `modality=="embedding"` (explicit `jinja: true` still
  wins; llm behavior unchanged). Command-construction only — `schema.json`
  untouched, so **no fingerprint/consensus bump**; the llama-cpp provider image
  just needs a rebuild + recreate.

### Accepted risks / notes

- **Schema-consensus churn**: adding `x-serves-modalities` (+ `--pooling`) to
  `llama-cpp`'s `schema.json` changes its committed fingerprint → 409
  `schema_pending` until **every** llama-cpp agent (vulkan **and** cuda share
  the one `llama-cpp` type) re-presents. This is business-as-usual with the
  version hard-fail (admin-first, then recreate every provider); the operator
  force-commits if a llama-cpp machine is permanently gone.
- **`modality` vs `provider_type` are orthogonal**: a definition is
  `(provider_type=llama-cpp, modality=embedding)`. The type gates *which
  agents* can host it; the modality gates *which endpoint* serves it.
- **No new scheduler machinery**: embedding backends are ordinary booted
  instances (VRAM + `max_running` + idle reaper all apply unchanged).
- **`audio` is reserved, not built**: the enum + `serves_modalities` list make
  a future `/v1/audio/*` phase additive; nothing in Phase 18 emits audio.

---

## Port model overhaul — agent-owned ports + model-routed `/v1` (2026-10-08)

**Status: ✅ SHIPPED — all 5 slices done.** Slice 1 landed the
admin half: the admin no longer allocates or polices per-backend ports, dials
`machine.reachable_address():agent.base_port`, and the `ProviderInstance.port`
column is dropped (Alembic `b7d3f0a1c9e2`). Slice 2 landed the provider_lib
multiplexing core + the mock reference cutover: `BackendRegistry` indexes handles
by alias (`resolve_by_model`), `create_provider_app` routes `/v1` by the request
`model` when a registry is supplied (single-`lifecycle` path kept for
backward-compat), and a new `AgentServer` serves one `/v1` on the env
`PROVIDER_PORT`. Slice 3 migrated **llama-cpp** and Slice 4 migrated **gufo,
halogen, and halogen-flash**: each `run_async` uses `AgentServer`, handles carry
`alias` (no `port`/`serve_port` threading), and each engine binds OS-assigned
free loopback ports (gufo/llama-cpp one `backend_port`; halogen a distinct
`(api, engine)` pair via `env.allocate_ports()`), `None` until start and reset on
stop. halogen-flash KEEPS its env/`MACHINE_UID`-derived static `(api, engine)`
pair (one-per-machine, fingerprint-stable) — only its schema port fields were
removed. The shared Vulkan+CUDA llama-cpp schema and the gufo/halogen/halogen-flash
schemas no longer expose port fields (fingerprints bumped: llama-cpp
`03373ece…`→`a695f7de…`, gufo `3e5882d7…`→`cbe7abf1…`, halogen
`07e4a3b4…`→`ae943493…`, halogen-flash `d3b144d4…`→`3d17412b…`). Slice 5 removed
the now-dead per-backend-port machinery from provider_lib (`MultiPortServer`,
`BackendHandle.port`, the registry change-listener API + the `notify_changed`
call in the assignments reconcile) and rewrote `docs/ws-protocol.md`,
`provider/README.md`, and `deployment.md` to the shipped single-env-port +
model-routing model (also correcting the stale `PROVIDER_REGISTRATION_TOKEN`
provider env examples → `MACHINE_SECRET` + `AGENT_ID`). `ARCHITECTURE.md` is
rewritten to the new model.

**The production bug.** Adding a 2nd `ProviderDefinition` to an agent that shares
a machine with another agent fails with
`assignments push: refused new backend ... — port 8082 already held by a
connected peer on the machine`. Root cause: the admin allocates each backend an
admin-facing port `base_port + offset` and enforces machine-wide uniqueness
(cross-agent clash guards on both registration and the live
`agent.assignments.update` push). Two agents on one machine with overlapping
`base_port` ranges collide, so a legitimate second placement is refused.

**The new canonical model (locked decisions).**

1. **Admin dials the agent's env-driven port.** Each provider agent publishes
   exactly **one** admin-facing HTTP port = its container env `PROVIDER_PORT`
   (recorded on the agent as `base_port`). The admin reaches every backend of
   that agent at `http://{machine.reachable_address()}:{agent.base_port}/v1/...`.
   There is **no** per-backend admin-facing port and **no** `base_port + offset`
   scheme.
2. **The agent routes `/v1` by model.** The agent serves a single
   OpenAI/OpenResponses-compliant `/v1` surface on its env port and dispatches
   each request to the correct backend by the request's `model` field (which
   equals the `ProviderDefinition.alias`). The admin carries **zero**
   provider-specific logic — it just points litellm at the agent's env port with
   the alias as the model.
3. **Engine (backend) ports are the agent's private concern.** Each backend's
   engine process (llama-server, halogen, gufo, …) binds a **local** port inside
   the container. Default allocation = ask the OS for a free port (bind
   `127.0.0.1:0`, read back the assigned port). The admin never learns or dials
   these.
4. **The admin stops policing ports entirely.** No `port_conflict` 409 at
   registration; no cross-agent clash refusal on the live
   `agent.assignments.update` push. The `ProviderInstance.port` column is being
   **dropped** (Alembic migration lands in Slice 1).
5. **Schema port config is removed** from llama-cpp (Vulkan + CUDA share one
   schema) and halogen-flash. halogen-flash keeps its existing env/`MACHINE_UID`-
   derived static `(api_port, engine_port)` pair (one-per-machine, deterministic)
   — that logic is unchanged; only the operator-facing schema fields go away.
6. **Bridge networking with published ports remains the deployment model:** one
   published `PROVIDER_PORT` per agent container.

### Slices

- [x] **Slice 0 — docs / spec (this).** `ARCHITECTURE.md` rewritten to the new
      model (data-plane diagram, components table, key invariant 4, URL-layout
      register row + paragraph, Machine/Agent/ProviderInstance tables — `port`
      row removed, env table `PROVIDER_PORT`, registration sequence diagram,
      `agent.assignments.update` prose + payload, §7 flow steps, §13 known
      limitation). This IMPLEMENTATION_STATUS.md section added. **No code,
      schema, test, or other-doc changes.**
- [x] **Slice 1 — admin stops allocating/policing ports.** DONE: removed the
      `base_port + offset` allocation, the registration `port_conflict` 409
      (`_reject_port_clash`), and the cross-agent clash refusal on the
      `agent.assignments.update` push; dropped the `ProviderInstance.port` column
      + Alembic migration `b7d3f0a1c9e2`; pointed litellm's `api_base` at the
      agent's env port (`machine.reachable_address():agent.base_port`) with the
      alias as the model. Admin tests updated (396 green); client regenerated.
- [x] **Slice 2 — provider_lib single env-port listener + model routing.** DONE:
      `BackendRegistry` gained an alias index + `resolve_by_model`;
      `create_provider_app` routes `/v1` by the request `model` when a registry
      is supplied (404 unknown model, 400 non-object body; single-`lifecycle`
      path preserved for backward-compat); new `AgentServer` serves one `/v1` on
      the env `PROVIDER_PORT` (no per-backend listener churn). Mock cut over to
      `AgentServer` + alias-carrying handles. provider_lib 184 + mock 51 green;
      the four un-migrated providers stay green (additive).
- [x] **Slice 3 — llama-cpp cutover + random engine port + schema field removal.**
      DONE: `run_async` uses `AgentServer`; handles carry `alias` and no longer
      thread `port`/`serve_port`; each llama-server engine binds an OS-assigned
      free loopback port (`_pick_free_port`; `backend_port` is `None` until start
      and reset on `stop`; no in-process retry — the scheduler re-drives a rare
      TOCTOU); `server.backend_port` removed from the shared Vulkan+CUDA
      `schema.json` (shipped fingerprint `03373ece…` → `a695f7de…`). llama-cpp 87
      green; provider_lib 184 + mock 51 stay green.
- [x] **Slice 4 — halogen + halogen-flash + gufo cutover.** DONE: all three
      `run_async` use `AgentServer` with alias-carrying handles (no `port`/
      `serve_port`). gufo mirrors llama-cpp (OS-assigned free `backend_port`,
      `server.backend_port` removed from schema). halogen replaces
      `env.resolve_ports` with `env.allocate_ports()` (two distinct OS-assigned
      free loopback ports per backend; `networking.{api_port,engine_port}`
      removed from schema). halogen-flash keeps its env/`MACHINE_UID` static
      `(api, engine)` pair unchanged; only the schema port fields were removed.
      Schema fingerprints bumped (gufo `cbe7abf1…`, halogen `ae943493…`,
      halogen-flash `3d17412b…`). gufo 83 + halogen 84 + halogen-flash 170 green;
      provider_lib 184 + mock 51 + llama-cpp 87 stay green.
- [x] **Slice 5 — provider_lib cleanup + docs.** DONE: removed the dead
      per-backend-port machinery from provider_lib (`MultiPortServer`,
      `BackendHandle.port`, the registry change-listener API + the
      `notify_changed` call in the assignments reconcile); rewrote
      `docs/ws-protocol.md`, `provider/README.md`, and `deployment.md` to the
      single-env-port + model-routing model (and corrected the stale
      `PROVIDER_REGISTRATION_TOKEN` provider env examples → `MACHINE_SECRET` +
      `AGENT_ID`). No admin API surface changed since Slice 1, so no client regen
      was needed. All six provider suites green (lib 183, mock 51, llama-cpp 87,
      gufo 83, halogen 84, halogen-flash 170).

**Follow-ups (optional, not part of the overhaul):** the mock schema
(`provider/mock/provider_mock/schema.json`) still declares a vestigial
`server.backend_port` (unused — the mock has no subprocess); `development.md`
still references the retired `PROVIDER_REGISTRATION_TOKEN`; `deployment.md`'s
idle-eviction note predates Phase 6. None affect the shipped port model.

---

## Phase 19 — Live stats bar + UI ergonomics 🟡

**Goal:** a global statistics bar in the admin UI header, create/edit forms
as right-side drawers, and a bottom-docked tabbed logs panel.

**Features:**

1. **Stats bar** (`_layout.tsx` header, new `StatsBar` component):
   - current `tok/s` + `prompt tok/s` (rolling avg over the last ~50
     `TokenUsageSample`s via `stats/usage?limit=50`); click → popover
     with per-model rates/token rollups.
   - queued / processing counts; click → popover with per-definition
     queue + active rows, a **Clear** button per definition, and a
     **Clear all** button at the bottom.
   - fleet VRAM `used/total` + GPU utilization %.
2. **Forms → drawers:** every create/edit form (`DefinitionFormDialog`,
   `MachineFormDialog`, `StorageActionDialog`, `BackendActionDialog`,
   provider-type schema dialog) moves from centered `Dialog` to a
   scrollable right-side `Sheet`. Delete confirmations stay dialogs.
3. **Logs → bottom-docked tabbed panel:** `LogsSheet` refactored into a
   reusable `LogsPanel` mounted in a **non-modal** dock in `_layout.tsx`
   (flex sibling above `<main>` — page content is pushed up, not
   overlaid). One tab per open instance, add/close tabs, fixed default
   height + drag resize, tabs persist for the session (in-memory).

**Locked decisions (operator Q&A):**

- Queue clear cancels **waiters only** — admitted/active requests, boots,
  and VRAM holds are untouched. Cleared clients get **503 `queue
  cleared`** (pre-stream JSON; in-stream `response.failed` with code
  `queue_cleared` on `/v1/responses`). New `QueueCleared(SchedulerError)`
  + `InferenceScheduler.queue_snapshot()` / `clear_queue(alias|None)` —
  authoritative in-process state, never the Redis mirror.
- Rates window = last ~50 completed requests (no time-filter backend
  change).
- Logs dock: fixed height + drag resize; session-persistent tabs (not
  localStorage).
- Orchestrator auto-commits each slice at green+CLEAN and continues;
  law docs are updated by the orchestrator only (S0 first).

**Slices (delegated-slice-delivery loop per slice):**

- [x] **S0 — law docs (this).** Phase 19 planned in
      `IMPLEMENTATION_STATUS.md`; `ARCHITECTURE.md` updated: §3 URL rows
      for `stats/scheduler` (+queue clear) and `stats/metrics`, §6
      "Queue observability & operator clear", §7 non-stream error
      mapping gains `QueueCleared` → 503. No code.
- [x] **S1 — scheduler:** DONE: `QueueCleared(SchedulerError)` +
      `_AliasState.cleared` set (waiter re-check at top of the acquire loop;
      `_drop_waiter` is the single cleanup point); `queue_snapshot()` →
      per-alias `{alias, queued, active}` from in-process state;
      `clear_queue(alias|None)` pops waiters under `state.lock` (never
      races a mid-admission head), bumps `changed` + `notify_all`, removes
      mirror keys best-effort; admitted slots/`_booted`/VRAM untouched.
      Routes: 503 in responses (pre-stream) / chat / embeddings; in-stream
      `response.failed` code `queue_cleared`. Admin suite 406 green
      (+10 tests: scheduler +6, responses +2, chat +1, embeddings +1);
      all provider suites green; ruff clean. Reviewer: CLEAN (2 nits,
      pre-existing lock-site pattern only).
- [ ] **S2 — admin stats endpoints:** `GET /admin/api/stats/scheduler`,
      `DELETE /admin/api/stats/scheduler/queue/{alias}`,
      `DELETE /admin/api/stats/scheduler/queue`,
      `GET /admin/api/stats/metrics` (fleet VRAM/GPU rollup), `alias` on
      `stats/usage` samples; tests + client regen.
- [ ] **S3 — StatsBar UI** + `popover.tsx` primitive
      (`@radix-ui/react-popover`) + per-model & per-definition popovers.
- [ ] **S4 — forms → right-side `Sheet` drawers.**
- [ ] **S5 — bottom-docked tabbed logs panel.**

---

## Accepted Regressions (do NOT restore from `legacy/`)

These existed in the pre-overhaul system and are intentionally removed or
stubbed. Re-implement against the new model only when a feature needs it.

| Feature | Status | Why |
| --- | --- | --- |
| `/v1/embeddings` | ~~501 stub~~ → **implemented Phase 18** | Was out of scope for the Responses/Chat core; now served via `litellm.aembedding` for `modality=embedding` definitions |
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
