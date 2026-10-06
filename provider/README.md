# Provider Authoring Guide

This is the guide to the **provider** side of Inference Matrix: the
hardware-local containers that each own one inference backend. Architecture
context: [../ARCHITECTURE.md](../ARCHITECTURE.md); the admin ⇄ provider
wire protocol: [../docs/ws-protocol.md](../docs/ws-protocol.md).

A **provider instance** runs on a hardware machine, owns a real inference
backend as a subprocess (or fake, for the mock), and exposes:

- An **OpenAI-compatible HTTP API on `PROVIDER_PORT`** (default 8081).
  The admin's litellm client targets
  `http://<machine.reachable_address()>:<PROVIDER_PORT>/v1/...`.
- A **dial-out WebSocket** to the admin (`/provider/ws`) for lifecycle
  commands/events.

`provider/lib/provider_lib` holds the generic machinery; each provider
package (`provider/mock`, `provider/llama-cpp`, `provider/halogen`,
`provider/halogen-flash`, `provider/gufo`) implements the per-type
driver and overrides only what its backend does differently.

## What you get from the library vs. what you implement

| Provided by `provider_lib` | You implement per provider type |
| --- | --- |
| Registration + WS client (`admin_client.py`) | `BackendDriver` (start/stop/health/list_models/stream_*) |
| Backend state machine + slot admission (`backend.py`) | Command/env construction for the real backend |
| Provider-port FastAPI app + `/v1` surface (`app_factory.py`) | Translation of the backend's native output → spec |
| Model downloader + progress events (`downloader.py`) | Artifact layout for your model files |
| Metrics collectors (`metrics.py`) | Any provider-specific metric quirks |
| Config fingerprint (`config.py`) | **Overrides** — e.g. halogen-flash `calculate_usage` |
| Phase 9 shared command handlers (`config_update.py`) | Wiring only — `install_config_handlers(client, lifecycle, state, settings, extra_cache_dirs=...)` |

The goal: a new provider type is usually just a `BackendDriver` subclass
plus a `main.py` that wires it into `BackendLifecycle` and
`create_provider_app`. If only one behavior differs (e.g. usage
normalization), override just that.

## Writing a new provider type

1. Create `provider/<type>/` as a uv workspace member (`pyproject.toml`
   depending on `provider_lib`), add it to the root
   `[tool.uv.workspace].members`.
2. Implement a `BackendDriver` subclass (see next section).
3. Add `main.py` mirroring `provider/mock/provider_mock/main.py`: build
   the driver, wrap in `BackendLifecycle`, install WS command handlers,
   register + connect, serve `create_provider_app(...)`.
4. Set `PROVIDER_TYPE` for the package; the admin cross-checks it against
    the registration token's definition.
5. **Ship `provider_<type>/schema.json`** (Phase 12) — a JSON Schema
    (2020-12) describing this type's `backend_config`. See "Authoring
    schema.json" below. `provider_lib` loads it and sends it in the
    registration body; the first registration of a type creates the
    admin's `ProviderType` registry entry.
6. Add a `Dockerfile` and tests.
7. Do **not** re-implement lifecycle/slot/WS semantics — they come from
    the lib.

## Authoring schema.json (Phase 12)

Each provider package ships `provider_<type>/schema.json`: a standard
**JSON Schema 2020-12** describing the shape of that type's
`backend_config`. It has three consumers, all read-only except the
driver:

- **Admin validation** — `ProviderDefinition.backend_config` is
  validated against the **committed** schema on create/PATCH (422 with
  per-field errors).
- **Admin UI** — the definition form is rendered from the schema:
  collapseable sections + custom widgets (below).
- **The driver** — still receives the plain dict in `apply_config()` /
  `start()`. The schema never generates code; `command.py` / `env.py`
  remain the mapping.

The schema is a **consensus contract**: the fingerprint of the committed
schema gates registrations (see `docs/ws-protocol.md` §2). Changing it
requires every known instance of the type to re-register with the new
file, or an operator force-commit.

### Sections (collapseable groups)

Group fields into top-level objects; the UI renders each as a
collapseable section. Use `title` for the label and `x-order` for
ordering. Recommended canonical sections: `artifacts`, `context` /
`rope`, `kv_cache` / `disk_cache`, `gpu`, `speculative` (MTP),
`sampling`, `reasoning`, `vision`, `server`, `security`, `logging` —
adapt to the backend. Example:

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "type": "object",
  "properties": {
    "artifacts": {
      "type": "object", "title": "Model files", "x-order": 1,
      "properties": {
        "model":  {"$ref": "#/$defs/hfFile", "title": "Main model", "x-order": 1},
        "mmproj": {"$ref": "#/$defs/hfFile", "title": "Vision projector", "x-order": 2},
        "draft":  {"$ref": "#/$defs/hfFile", "title": "Draft model", "x-order": 3}
      },
      "required": ["model"]
    },
    "speculative": {
      "type": "object", "title": "MTP / Speculative decoding", "x-order": 5,
      "properties": {
        "spec_draft_n_max": {"type": "integer", "default": 3, "minimum": 1, "maximum": 8,
                             "title": "Draft tokens", "x-flag": "--spec-draft-n-max"}
      }
    }
  },
  "$defs": {
    "hfFile": {
      "title": "HuggingFace file",
      "oneOf": [
        {"type": "object", "x-widget": "hf-file",
         "properties": {
           "source": {"const": "hf"},
           "repo":   {"type": "string"},
           "file":   {"type": "string"},
           "revision": {"type": "string"}
         },
         "required": ["source", "repo", "file"],
         "additionalProperties": false},
        {"type": "object",
         "properties": {"path": {"type": "string"}},
         "required": ["path"], "additionalProperties": false}
      ]
    }
  }
}
```

### Custom keywords (all optional, `x-` prefixed)

| Keyword | Meaning |
| --- | --- |
| `x-flag` | The exact upstream CLI flag / env var this maps to (e.g. `"--ngl"`, `"HALOGEN_KV_SLOTS"`). Rendered as a UI tooltip; keeps `command.py`/`env.py` reviewable against the schema. |
| `x-widget` | UI widget override. **`"hf-file"`** = HuggingFace file picker (search → repo → file list with sizes, or a local path) producing the `{"source":"hf","repo","file"}` / `{"path":...}` descriptor that `provider_lib.downloader.ensure_artifact` resolves. |
| `x-numeric-effect` | `"bitwise"` (output byte-identical; speed/policy only) or `"numeric"` (output can change). From the halogen FLAGS.md convention; the UI warns on numeric fields. |
| `x-secret` | `true` = write-only field (e.g. `api_key`): masked in UI reads, never logged. |
| `x-supported` | `false` = documented but not wired by this driver version; shown disabled in the UI. Used for the full upstream flag list where `command.py`/`env.py` doesn't map everything yet. |

### Rules

- Give **every** field a `default`, `description`, and range/enum where
  applicable — the form is only as good as its metadata.
- Put only fields the driver actually consumes (or `x-supported: false`
  documents) in the schema; don't invent ad-hoc keys beyond it.
- Obsolete upstream flags (`--prompt-cache`, removed `--draft*` aliases,
  deprecated `--defrag-thold`) must **not** appear in the schema.
- Backend binary paths (`LLAMA_SERVER_PATH`, etc.) stay in the container
  env — never schema fields.
- Test that the schema parses, validates the example configs in this
  README, and has a stable fingerprint.

## BackendDriver (provider_lib/backend.py)

A provider package implements this ABC around its real backend:

```python
class BackendDriver(ABC):
    async def start(self) -> None: ...  # bring the backend up; raise on failure
    async def stop(self) -> None: ...  # tear it down
    async def health(self) -> bool: ...  # ready to serve?
    async def list_models(self) -> list[dict]: ...  # OpenAI model objects
    def stream_responses(self, request: dict) -> AsyncIterator[dict]:
        ...
        # OpenResponses SSE event dicts: response.created, response.in_progress,
        # response.output_item.added, response.output_text.delta*, ...,
        # response.output_item.done, response.completed (with usage).
        # Must be a closeable async generator.

    def stream_chat_completions(self, request: dict) -> AsyncIterator[dict]:
        ...
        # Optional; default raises NotImplementedError -> HTTP 501.

    def apply_config(self, backend_config: dict) -> None:
        ...
        # Phase 9: adopt an updated backend_config before the next start
        # (default no-op; all five providers implement it). Must also
        # reset `resolved_artifacts` (see storage.prune_unused).

    resolved_artifacts: list[str]
    # Phase 9: local artifact paths recorded during the last start
    # (main/mmproj/draft/tokenizer/NPU pins). This is the reference
    # set storage.prune_unused keeps.

    async def aclose(self) -> None: ...  # optional cleanup (default no-op)
```

## BackendLifecycle

Wraps a driver with the admin-visible state machine and slot accounting:

```python
BackendLifecycle(driver, *, capacity=1, status_callback=None)
# status_callback: async (new_status: str, reason: str | None) -> None
#   wired to AdminClient.send_event("backend.status", {...}).

await lifecycle.start()            # STOPPED→STARTING→(health)→RUNNING | ERROR
await lifecycle.ensure_started()   # idempotent
await lifecycle.stop()             # *→STOPPING→STOPPED | ERROR
await lifecycle.acquire_slot()     # BackendNotReady | BackendBusy; RUNNING→IN_USE
await lifecycle.release_slot()     # last release: IN_USE→RUNNING
async with lifecycle.slot(): ...   # context-manager form
stream = await lifecycle.stream_responses(request)    # slot-owned async iterator
stream = await lifecycle.stream_chat_completions(request)
lifecycle.backend_status / .in_flight / .capacity / .last_request_at
```

States follow `BackendStatusValue` (`stopped|initializing|starting|running|
in_use|stopping|error`); every transition is emitted as a `backend.status`
WS event.

### Slot admission rule

- Backend must be `RUNNING` or `IN_USE`; otherwise `BackendNotReady` → **503**.
- `in_flight < capacity`; otherwise `BackendBusy` → **429**. (The admin
  scheduler queues globally in Phase 6; provider-side admission is
  defense-in-depth.)
- First acquire: `RUNNING → IN_USE` (emitted). Last release:
  `IN_USE → RUNNING` (emitted). `last_request_at` recorded per acquire
  (admin idle-timeout consumes it later).
- `capacity` comes from the registration response
  `provider_definition.capacity` (default 1).

### Release-on-upstream-close invariant (CRITICAL)

**A slot is released when the UPSTREAM driver stream closes — never when
the downstream HTTP client finishes consuming.** A completed llama
request must not hold a slot because a client connection lingers (the
legacy agent's documented failure).

Implementation: `BackendLifecycle.stream_responses()` acquires the slot
eagerly, then starts a background **pump task** that drains the driver
generator into an unbounded queue. The pump owns the release in its
`finally`:

```python
async def _pump(self, upstream, queue):
    try:
        async for event in upstream:
            await queue.put(event)
    except Exception as exc:
        await queue.put(_StreamFailure(exc))
    finally:
        await upstream.aclose()  # close the driver generator...
        await queue.put(_STREAM_END)
        await self.release_slot()  # ...and free the slot, always
```

The consumer generator returned to the HTTP layer just reads the queue.
If the consumer abandons the stream (client disconnect), closing the
consumer cancels the pump, which closes the upstream and releases. If
the consumer never iterates, the eager pump still drains and releases.
Upstream exceptions surface to the consumer as an SSE `error` event.

Covered by `provider/lib/tests/test_slot_release_on_stream_close.py`
(exhaustion, early `aclose()`, never-iterated, raw-ASGI
`http.disconnect`, two independent streams) and a real-socket uvicorn
disconnect test in `provider/mock/tests/test_mock_phase4.py`.

## /v1 surface (provider_lib/app_factory.py)

Mounted when `BackendOverrides.lifecycle` is set; otherwise 503.

| Route | Behavior |
| --- | --- |
| `GET /v1/models` | `{"object":"list","data":driver.list_models()}`. **503** unless backend RUNNING/IN_USE (a stopped backend has no cheap source of truth; the admin starts it on demand). |
| `POST /v1/responses` | **`stream: true`**: slot-admitted SSE (`text/event-stream`). Each event: `event: <type>` + `data: <json>` lines (OpenResponses style, no `[DONE]`). **Default/false**: the event stream is drained and the terminal `response.*` object returned as JSON (spec default). **429** busy, **503** not ready. |
| `POST /v1/chat/completions` | Same discipline; chunks are `chat.completion.chunk` events, terminated with `data: [DONE]`. **501** if the driver has no chat surface. |
| `GET /health` | Provider identity + `backend_status`, `in_flight`, `capacity`. |

Slots are acquired **before** the `StreamingResponse` starts so
busy/not-ready are real HTTP statuses; no slot leaks on early errors.

## Overrides (BackendOverrides, app_factory.py)

`create_provider_app(settings, overrides)` takes a `BackendOverrides`
object. Unset hooks fall back to library defaults; set only what your
provider type needs to change.

| Override | Purpose |
| --- | --- |
| `provider_type` / `version` | Identity reported to the admin. |
| `lifecycle` | The `BackendLifecycle` to mount the `/v1` surface on. Omit → 503. |
| `calculate_usage` | `Callable[[Any], dict[str, int]]`. Compute spec usage when the backend does not report it spec-compliantly. **Canonical example: halogen-flash overrides this** because its backend's usage shape isn't OpenResponses-clean. |

Keep overrides minimal — if `calculate_usage` is the only thing that
differs, that's the only code your package adds beyond the driver.

## Mock provider (provider/mock)

`MockBackend` implements the driver: instant start, canned
`output_text.delta` streams (`delta_count`/`delta_delay`/`hold`
configurable), model alias adopted from the registration response. When
the request carries a non-empty `tools` list it emits a canned
`function_call` turn instead, so the tool-forwarding path is exercisable
without a real model. Terminal response objects are spec-complete
(`created_at`, `completed_at`, `usage` details, parameter echoes) so the
conformance suite validates against them. `provider_mock.main` builds
**one** `BackendLifecycle` shared between the admin WS command handlers
(`backend.start`/`backend.stop` drive it) and the FastAPI `/v1` app. Boot
is admin-driven: after connect the backend stays STOPPED.

The entrypoint registers over HTTP once (`register_provider`) and then
keeps the admin WS alive with `AdminClient.run_forever()`, re-emitting
`provider.status` on every (re)connect, so the mock survives admin
restarts like a real hardware provider (`register_and_connect` remains as
the one-shot connect used by tests).

## Worked example: llama-cpp provider (provider/llama-cpp)

`LlamaCppBackend` (provider_llama_cpp/driver.py) manages one
`llama-server` subprocess. Entry point `provider_llama_cpp.main` follows
the mock's admin-driven boot pattern.

### backend_config schema

> **Canonical (Phase 12):** the authoritative schema is the shipped
> `provider_llama_cpp/schema.json` (loaded by `provider_lib.schema` and
> sent at registration). The example below is the starting point for it —
> keep both in sync. See "Authoring schema.json" above for section /
> `x-flag` / `x-widget` conventions.

```json
{
  "model":  {"source": "hf", "repo": "ggml-org/models", "file": "gemma/ggml-model.gguf"},
  "mmproj": {"source": "hf", "repo": "...", "file": "..."},
  "draft":  {"source": "hf", "repo": "...", "file": "..."},
  "args":   {"ctx": 8192, "gpu_layers": 35, "flash_attn": "on", "parallel": 1,
             "threads": 8, "batch_size": 512, "ubatch_size": 128,
             "reasoning": true, "jinja": true, "mtp_draft_max": 4},
  "backend_port": 9999
}
```

- Artifacts also accept `{"path": "/local/file.gguf"}` (no download; the
  file must exist).
- `model`/`mmproj`/`draft` are separate `ensure_artifact` calls
  (provider_lib.downloader). Candidate resolution: `MODELS_DIR/<repo>/<file>`,
  then FLAT `MODELS_DIR/<file>`, then HF download into
  `MODELS_DIR/<repo>`.
- The `llama-server` binary path comes from env `LLAMA_SERVER_PATH`
  (default `llama-server`), never from backend_config.
- `backend_port` optional; default `PROVIDER_PORT + 1`.
- Health wait: env `SERVER_START_HEALTH_TIMEOUT` (default 120s).

### Command mapping (command.py)

| Config (args.*) | CLI |
| --- | --- |
| `model` (resolved) | `--model <path>` |
| (backend_port) | `--port <p>` |
| `gpu_layers` (default 35) | `--n-gpu-layers` |
| `ctx` / `context_size` (default 4096) | `--ctx-size` |
| `batch_size` (default 512) | `--batch-size` |
| (always) | `--metrics` |
| `threads`, `threads_batch`, `ubatch_size`, `keep`, `predict`, `cache_type_k/v`, `cache_reuse`, `ctx_checkpoints`, `checkpoint_every`→`--checkpoint-min-step`, `cache_ram`, `slot_save_path`, `device`, `split_mode`, `tensor_split`, `main_gpu`, `fit`, `fit_target`, `fit_ctx`, `temperature`, `top_k`, `top_p`, `min_p`, `repeat_penalty`, `presence_penalty`, `frequency_penalty`, `seed`, `parallel`, `reasoning`, `reasoning_budget`, `spec_draft_p_min` | `--<flag> <value>` when present and not None |
| `swa_full`/`kv_unified` | emitted when truthy |
| `kv_offload`, `cache_prompt`, `cont_batching`, `warmup`, `context_shift`, `no_cache_idle_slots` | `--x` / `--no-x` by boolean. NOTE: `--cache-prompt` is a valid modern flag; the OBSOLETE `--prompt-cache` is NEVER emitted. |
| `load_mode` (auto|mmap|mlock|none) | `--load-mode <m>` when present and not `auto`. `no_mmap` and `strict_mtp_qwen` are REMOVED in modern llama.cpp (`--mmap`/`--no-mmap`/`--spec-mtp-strict-qwen` no longer exist) and are silently ignored |
| `flash_attn` (bool or "on"/"off") | `--flash-attn on|off`; skipped when None |
| `draft` artifact present | `--spec-type draft-dflash` + `-md <path>` |
| else `mtp_draft_max > 0` | `--spec-type draft-mtp --spec-draft-n-max N` |
| `jinja` (default true) | `--jinja` |
| `mmproj` artifact | `--mmproj <path>` |

Spawn env: `LD_LIBRARY_PATH` defaults to the binary's directory
(`env.setdefault`) — Vulkan/containers may not provide it.

### Downloads + machine metrics

- `provider_lib.downloader.ensure_artifact()` — HF downloads with a
  throttled tqdm→`download.progress` publisher, in-flight dedup by
  `repo/file`, path-traversal guards.
- `provider_lib.metrics.collect_machine_snapshot(categories, ...)` —
  `vram`/`gpu_usage` (shared nvidia-smi/AMD-sysfs sampler), `os_ram`
  (/proc/meminfo), `cpu` (loadavg), `storage` (disk_usage). Never
  raises; failed collectors omit their key.
- `provider_lib.metrics.MachineMetricsEmitter` — sends
  `metrics.machine` every `MACHINE_METRICS_INTERVAL` (default 10s)
  **only while the admin has assigned ownership** via the
  `metrics.assign` WS command (revoked by `metrics.unassign`).
  Admin-side ownership lease: see `admin/backend/docs/redis-keys.md`.

## Overriding usage normalization (the `calculate_usage` pattern)

Some backends are **not spec-compliant on usage**: they report chat-style
counts (`prompt_tokens`/`completion_tokens`), native timing keys
(`tokens_evaluated`, `timings.cache_n`), partial detail objects, or
nothing in-stream. The admin's `persist_turn` reads exactly one shape —
the OpenResponses spec usage dict on the terminal
`response.completed`/`response.incomplete` event:

```json
{
  "input_tokens": 100, "output_tokens": 8, "total_tokens": 108,
  "input_tokens_details":  {"cached_tokens": 80},
  "output_tokens_details": {"reasoning_tokens": 0},
  "completion_tokens_details": {"prompt_per_second": 1412.21,
                                "predicted_per_second": 36.25}
}
```

The override lives in the **driver's stream generator**, not in the lib:
before yielding the terminal event, the driver normalizes whatever the
backend emitted into the shape above. Rules (canonical implementation:
`provider_halogen_flash.usage.calculate_usage`):

1. **Trust reported values verbatim — including reported zeros.** A
   backend that reports `cached_tokens: 0` must not have that replaced
   by a timing fallback.
2. **Estimate only when nothing was reported.** If the raw blob contains
   no count keys at all, output tokens are estimated from accumulated
   streamed characters (`chars // 4`); input is left 0.
3. **Fallback ladder for cached tokens:** `prompt_tokens_details` /
   `input_tokens_details` → `timings.cache_n` → 0.
4. **Reasoning tokens:** the backend's reported detail wins over the
   driver's own count of reasoning deltas.
5. **Rates land in `completion_tokens_details`**
   (`prompt_per_second` / `predicted_per_second`) — that's where the
   admin's `TokenUsageSample` reads them.

The function is *also* exposed on `BackendOverrides.calculate_usage` so
the hook is visible at the app-factory level, but the authoritative call
site is the driver (the generic `/v1` layer never recomputes usage — it
passes the driver's normalized events through). New providers with a
non-compliant backend should copy this: a pure
`calculate_usage(raw, fallback_chars=…, reasoning_tokens=…) -> dict` in
`usage.py`, unit-tested with a (raw shape → spec shape) table, called
from the driver right before yielding the terminal event.

## gufo provider (provider/gufo)

`GufoBackend` (provider_gufo/driver.py) manages one
`gufo serve llm` subprocess. gufo is **multi-model** (one instance can
serve several model names) and speaks **native OpenResponses** — the
driver passes events through unmodified except for rate enrichment.

### backend_config schema
> **Canonical (Phase 12):** the authoritative schema is the shipped
> `schema.json` for this provider package (loaded by `provider_lib.schema`,
> sent at registration). The example below is its starting point — keep both
> in sync. See "Authoring schema.json" above for conventions.


```json
{
  "model":  {"source": "hf", "repo": "...", "file": "model.gguf"},
  "backend_port": 8082,
  "options": {
    "context": 131072,
    "served_model_name": "my-alias",
    "mmproj":       {"source": "hf", "repo": "...", "file": "mmproj.gguf"},
    "dflash_model": {"source": "hf", "repo": "...", "file": "draft.gguf"},
    "dspark_model": "...", "mtp_model": "...",
    "cache_disk": true, "cache_disk_bytes": 1073741824,
    "sessions": 4,
    "temperature": 0.7, "top_k": 40, "top_p": 0.9,
    "think": "on", "reasoning_effort": "high",
    "speculative": "dflash2", "draft_policy": "adaptive",
    "api_key": "...", "verbose": true, "log_progress": false
  }
}
```

- `model` accepts the same descriptor shapes as llama-cpp
  (`{"path": ...}` or HF `{"source","repo","file"}`), resolved via
  `provider_lib.downloader.ensure_artifact` (repo layout + FLAT
  `MODELS_DIR/<file>` candidates).
- Aux spec files (**mmproj / dflash_model / dspark_model / mtp_model**)
  may be *artifact descriptors* in `options`; the driver resolves each
  through the downloader and rewrites the option to the concrete local
  path before spawn (legacy "aux by path" semantics). Plain strings pass
  through untouched.
- `backend_port` optional; default `PROVIDER_PORT + 1`. Health:
  `GET /health` on that port. Binary from env `GUFO_SERVER_PATH`
  (default `gufo`).
- The subprocess runs in its own session (`start_new_session=True`);
  `stop()` kills the whole process group.

### Command mapping (command.py)

Base argv: `gufo serve llm --host 127.0.0.1 --port <backend_port>
--model <resolved model path>`, then per-key from `options`:

| Option | CLI | Notes |
| --- | --- | --- |
| `context` | `--context <v>` | |
| `served_model_name` | `--served-model-name <v>` | gufo otherwise advertises the GGUF filename |
| `mmproj` | `--mmproj <path>` | descriptor resolved first |
| sampling (`max_tokens temperature top_k top_p min_p min_keep seed repeat_penalty repeat_last_n frequency_penalty presence_penalty`) | `--<flag> <v>` | each only when present and not None |
| reasoning (`think reasoning_effort preserve_thinking`) | `--<flag> <v>` | tri-state/enum values passed as strings |
| speculative (`speculative dflash_model dspark_model mtp_model draft_policy draft_tokens min_draft_tokens`) | `--<flag> <v>` | aux descriptors resolved to paths |
| limits (`prefill_chunk max_pending max_pending_per_client request_timeout_ms max_output_bytes max_buffered_output_bytes max_buffered_output_total`) | `--<flag> <v>` | |
| `cache_disk_bytes` / `cache_disk_staging_bytes` | `--<flag> <v>` | |
| `cache_disk: true` | `--cache-disk <CACHE_DIR>/<instance_id>` | directory created on demand; anything else omits the flag |
| server (`sessions max_connections max_request_bytes api_key`) | `--<flag> <v>` | `sessions` = GPU concurrency |
| `verbose` / `log_progress` | `--verbose` / `--log-progress` | **bool-only**: emitted only when `true`; false/None omitted so gufo keeps its default |

**Per-request `model` forwarding:** unlike llama-cpp (single-model
server), the driver forwards the request body's `model` field to the
upstream `/v1/responses` / `/v1/chat/completions` **as-is** — gufo
routes to the requested served model itself.

**Rate gauges (rates.py):** gufo exposes instantaneous throughput on
`GET /metrics` as Prometheus *gauges* instead of `*_seconds_total`
counters. Before yielding a terminal `response.completed`/`incomplete`,
the driver scrapes `/metrics` and merges
`llamacpp:prompt_tokens_seconds` / `llamacpp:predicted_tokens_seconds`
into the usage's `completion_tokens_details` as `prompt_per_second` /
`predicted_per_second` (gauges take precedence; counter-derived
lifetime averages are the fallback; scrape failures are silent — rates
are telemetry, never a request failure).

**Effective capacity:** `options.sessions` (default 1) is the engine's
GPU-session concurrency, reported in the `backend.start` ack's
`detail.effective_capacity`. The lifecycle capacity itself comes from
`provider_definition.capacity` (admin-side scheduling contract).

## halogen provider (provider/halogen)

`HalogenBackend` (provider_halogen/driver.py) manages one halogen
process. Halogen is **env-configured** (not argv): `backend_config`
holds the semantic config and `env.py` maps it to `HALOGEN_*`
environment variables. The process exposes **two ports**: an
OpenAI-compatible **API port** (all HTTP traffic) and a private
**engine port** (referenced only via `HALOGEN_ENGINE`).

### backend_config schema
> **Canonical (Phase 12):** the authoritative schema is the shipped
> `schema.json` for this provider package (loaded by `provider_lib.schema`,
> sent at registration). The example below is its starting point — keep both
> in sync. See "Authoring schema.json" above for conventions.


```json
{
  "model":     {"source": "hf", "repo": "peonist-ai/halogen-qwen3.8-27b", "file": "qwen3.8-27b-p1w4d-d2.hgn"},
  "tokenizer": {"source": "hf", "repo": "peonist-ai/halogen-qwen3.8-27b", "file": "tokenizer"},
  "api_port": 8082,
  "engine_port": 8083,
  "options": {
    "kv_slots": 4,
    "drafter": "mtp",
    "cache_mb": 2048,
    "cache_reserve_mb": 256,
    "slot_ctx": 4096,
    "cache_align": 16,
    "max_tokens_cap": 8192,
    "queue_timeout": 300,
    "w4a4": 1, "w4a4_excl": "attn",
    "keepalive_timeout": 60, "sse_keepalive_s": 15
  }
}
```

- `model` = the `.hgn` **checkpoint**, `tokenizer` = the tokenizer
  **directory**. Both accept `{"path": ...}` (file *or* directory, no
  download) or HF descriptors resolved via the lib downloader.
- Ports: `api_port` defaults to `PROVIDER_PORT + 1`, `engine_port` to
  `PROVIDER_PORT + 2`. Health = `GET /health` on the **API port**.
- Binary/entrypoint from env `HALOGEN_SERVER_PATH` (default
  `halogen-server`); spawned as
  `stdbuf -oL -eL <entrypoint> all` with stderr merged into stdout
  (single log pipe, legacy anti-deadlock), in its own process group.

### Env mapping (env.py)

Fixed (always set):

| Env | Value |
| --- | --- |
| `HALOGEN_API_PORT` | `api_port` |
| `HALOGEN_PORT` | `engine_port` |
| `HALOGEN_BIND` | `127.0.0.1` |
| `HALOGEN_ENGINE` | `127.0.0.1:<engine_port>` |
| `HALOGEN_CHECKPOINT` | resolved checkpoint path |
| `HALOGEN_TOKENIZER` | resolved tokenizer path |

From `options` (each emitted **only when present and not None**,
stringified):

| Option | Env |
| --- | --- |
| `drafter` | `HALOGEN_DRAFTER` |
| `cache_align` | `HALOGEN_CACHE_ALIGN` |
| `kv_slots` | `HALOGEN_KV_SLOTS` |
| `slot_ctx` | `HALOGEN_SLOT_CTX` |
| `cache_mb` | `HALOGEN_CACHE_MB` |
| `cache_reserve_mb` | `HALOGEN_CACHE_RESERVE_MB` |
| `max_tokens_cap` | `HALOGEN_MAX_TOKENS_CAP` |
| `queue_timeout` | `HALOGEN_QUEUE_TIMEOUT` |
| `w4a4` | `HALOGEN_W4A4` |
| `w4a4_excl` | `HALOGEN_W4A4_EXCL` |
| `keepalive_timeout` | `HALOGEN_KEEPALIVE_TIMEOUT` |
| `sse_keepalive_s` | `HALOGEN_SSE_KEEPALIVE_S` |

**Capacity:** the engine's request slots = `options.kv_slots` (default
1), reported as `detail.effective_capacity` in the `backend.start` ack
(plus `api_port`/`engine_port`). The lifecycle capacity comes from
`provider_definition.capacity`.

**Streaming:** the driver proxies SSE from the API port's
`/v1/responses` / `/v1/chat/completions` with the standard
close-on-`GeneratorExit` semantics (early consumer close cancels the
upstream request and frees the slot).

## halogen-flash provider (provider/halogen-flash)

`HalogenFlashBackend` (provider_halogen_flash/driver.py) manages one
halogen-flash process. Key differences from plain halogen:

- The backend speaks **native `/v1/responses`** (OpenResponses SSE); the
  driver mostly passes events through.
- The backend is **NOT spec-compliant on usage** — this package carries
  the canonical `calculate_usage` override (see "Overriding usage
  normalization" above).
- **Static ports** keep the engine's disk-cache fingerprint stable.
- **NPU small-model pinning** (Ryzen AI), gated by a host probe.

### backend_config schema
> **Canonical (Phase 12):** the authoritative schema is the shipped
> `schema.json` for this provider package (loaded by `provider_lib.schema`,
> sent at registration). The example below is its starting point — keep both
> in sync. See "Authoring schema.json" above for conventions.


```json
{
  "model":     {"source": "hf", "repo": "peonist-ai/halogen-qwen3.8-flash-next", "file": "qwen38-flash-next-w4b.hgn"},
  "tokenizer": {"source": "hf", "repo": "peonist-ai/halogen-qwen3.8-flash-next", "file": "tokenizer"},
  "api_port": 8200,
  "engine_port": 8201,
  "options": {
    "kv_slots": 8, "kv_pool_positions": 524288, "kv_pool_fit": 1,
    "host_reserve_gib": 20, "ctx": 131072, "max_tok": 16384,
    "rope_yarn": 4, "admit_chunk": 4096, "indexer_budget": 64,
    "cache_dir_enabled": true, "cache_disk_gib": 64, "cache_prune_old": 1,
    "prompt_cache": 2, "cache_inplace": 1, "prefill_chunk": 8192,
    "npu_models": ["qwen3-embedding-0.6b", "qwen3.5-2b"],
    "vision_tower": 1, "vision_max_pixels": 1048576
  }
}
```

### Env mapping (env.py)

Fixed wiring is identical to halogen (`HALOGEN_API_PORT`,
`HALOGEN_PORT`, `HALOGEN_BIND`, `HALOGEN_ENGINE`, `HALOGEN_CHECKPOINT`,
`HALOGEN_TOKENIZER`). The 43 semantic `options` → env mappings (each
emitted only when present and not None):

| Option | Env | Group |
| --- | --- | --- |
| `kv_slots` | `HALOGEN_KV_SLOTS` | KV cache & admission |
| `kv_pool_positions` | `HALOGEN_KV_POOL_POSITIONS` | |
| `kv_pool_fit` | `HALOGEN_KV_POOL_FIT` | |
| `host_reserve_gib` | `HALOGEN_HOST_RESERVE_GIB` | |
| `ctx` | `HALOGEN_CTX` | context / length |
| `max_tok` | `HALOGEN_MAX_TOK` | |
| `rope_yarn` | `HALOGEN_ROPE_YARN` | |
| `admit_chunk` | `HALOGEN_ADMIT_CHUNK` | |
| `indexer_budget` | `HALOGEN_INDEXER_BUDGET` | |
| `max_tokens_cap` | `HALOGEN_MAX_TOKENS_CAP` | |
| `max_tokens_default` | `HALOGEN_MAX_TOKENS_DEFAULT` | |
| `queue_timeout` | `HALOGEN_QUEUE_TIMEOUT` | timeouts / keepalive |
| `keepalive_timeout` | `HALOGEN_KEEPALIVE_TIMEOUT` | |
| `sse_keepalive_s` | `HALOGEN_SSE_KEEPALIVE_S` | |
| `temperature` | `HALOGEN_TEMPERATURE` | sampling defaults |
| `top_p` | `HALOGEN_TOP_P` | |
| `top_k` | `HALOGEN_TOP_K` | |
| `min_p` | `HALOGEN_MIN_P` | |
| `presence_penalty` | `HALOGEN_PRESENCE_PENALTY` | |
| `frequency_penalty` | `HALOGEN_FREQUENCY_PENALTY` | |
| `reasoning_effort` | `HALOGEN_REASONING_EFFORT` | reasoning |
| `enable_thinking` | `HALOGEN_ENABLE_THINKING` | |
| `max_thinking_tokens` | `HALOGEN_MAX_THINKING_TOKENS` | |
| `thinking_answer_room` | `HALOGEN_THINKING_ANSWER_ROOM` | |
| `drafter_default` | `HALOGEN_DRAFTER_DEFAULT` | speculative decoding |
| `mtp_depth` | `HALOGEN_MTP_DEPTH` | |
| `pld` | `HALOGEN_PLD` | |
| `spec_adapt` | `HALOGEN_SPEC_ADAPT` | |
| `prompt_cache` | `HALOGEN_PROMPT_CACHE` | prompt cache |
| `cache_inplace` | `HALOGEN_CACHE_INPLACE` | |
| `prefill_chunk` | `HALOGEN_PREFILL_CHUNK` | |
| `cache_entries` | `HALOGEN_CACHE_ENTRIES` | |
| `cache_branches` | `HALOGEN_CACHE_BRANCHES` | |
| `cache_snap3` | `HALOGEN_CACHE_SNAP3` | |
| `cache_full` | `HALOGEN_CACHE_FULL` | |
| `cache_disk_gib` | `HALOGEN_CACHE_DISK_GIB` | |
| `cache_prune_old` | `HALOGEN_CACHE_PRUNE_OLD` | |
| `composable_context` | `HALOGEN_COMPOSABLE_CONTEXT` | composable context |
| `composable_context_floor` | `HALOGEN_COMPOSABLE_CONTEXT_FLOOR` | |
| `composable_context_bytes` | `HALOGEN_COMPOSABLE_CONTEXT_BYTES` | |
| `grammar` | `HALOGEN_GRAMMAR` | structured output |
| `vision_tower` | `HALOGEN_VISION_TOWER` | vision |
| `vision_max_pixels` | `HALOGEN_VISION_MAX_PIXELS` | |

Plus two conditional vars:

| Condition | Env |
| --- | --- |
| `options.cache_dir_enabled` is `true` | `HALOGEN_CACHE_DIR = <CACHE_DIR>/halogen-flash` (created on start) |
| `options.npu_models` non-empty **and** the NPU probe passes | `HALOGEN_NPU_MODELS = <id>,<id>` (bare comma list, never a Python repr) |

Binary from env `HALOGEN_FLASH_SERVER_PATH` (default
`halogen-flash-server`); same `stdbuf -oL -eL <entrypoint> all` spawn
as halogen. Health = `GET /health` on the API port.

### Static ports for the disk-cache fingerprint

The Flash engine hashes its server variables — **ports included** —
into its on-disk prompt-cache fingerprint. Random per-start ports would
invalidate the disk cache on every boot, so the `(api, engine)` pair is
**pinned**:

1. Explicit `api_port` + `engine_port` in `backend_config` win (the
   admin persists the pair in the definition — Phase 9's fingerprint
   flow may re-derive it).
2. Otherwise `derive_static_ports(MACHINE_UID)` hashes the machine uid
   (SHA-256) into the dedicated static range **8200–8289**
   (`FLASH_PORT_START`/`FLASH_PORT_END`, 45 adjacent pairs), giving the
   same stability property the legacy backend got from allocate-once-
   and-persist: the same machine always boots Flash on the same pair.
3. `PROVIDER_PORT` never participates in the pair (it's the instance's
   own admin-facing surface).

### NPU small-model pinning

`npu.py` ports the legacy `npu_models` + `npu_probe` knowledge:

- `NPU_SUFFIXES` maps upstream ids (`qwen3-embedding-0.6b`,
  `qwen3-reranker-0.6b`, `qwen3.5-2b`, `decider-0.8b`,
  `qwen3guard-gen-0.6b`) to client-facing suffixes
  (`<alias>-embed`, …); `client_name()`/`split_client_name()` convert
  both ways.
- `parse_npu_pins()` / `resolve_npu_download_set()` read the image's
  pins file (`NPU_PINS_FILE`, default `/opt/halogen/npu/models.txt`)
  and plan downloads into `MODELS_DIR/npu/<id>/`, honoring shared
  `devices/` programs (a model running on another's device program
  pulls the owner's `devices/*` files into the owner's dir).
- `probe_npu(device_path, xrt_lib_dir, binary_path)` gates pinning:
  device node (`NPU_DEVICE_PATH`) writable, XRT libs with the
  `xdna` plugin present (`NPU_XRT_LIB_DIR`), and the engine binary
  (`NPU_BINARY_PATH`) answering its usage contract (exit 1 +
  `usage: halogen-npu`). The GPU fabric-clock state is *reported*
  (`fabric_clock_held`) but never gates (start-time requirement, not a
  capability).
- **No NPU present → graceful degradation:** the probe returns
  `available: false`, the driver logs the reasons and **omits
  `HALOGEN_NPU_MODELS`** — start proceeds without NPU models, never
  an error. A crashing probe degrades the same way. The probe result is
  also included in the registration hardware report under `npu` so the
  admin UI can gate NPU options per machine.

### Usage override (the canonical example)

See "Overriding usage normalization" above; the implementation is
`provider_halogen_flash/usage.py` and the driver applies it in
`stream_responses` right before yielding `response.completed`/
`response.incomplete`, accumulating `output_text.delta` characters and
reasoning-delta tokens for the estimate inputs. Table-tested with every
observed raw shape in `tests/test_calculate_usage.py`.

## Phase 9: config update / cache clear / storage prune

The admin owns the `backend_config`; a definition change must reach the
running provider without operator intervention. The shared machinery
lives in `provider_lib/config_update.py` and every provider wires it
with one call in `install_command_handlers`:

```python
from provider_lib.config_update import ConfigState, install_config_handlers

state = ConfigState()  # applied-fingerprint tracker
install_command_handlers(client, lifecycle, emitter, state)  # per-package
# inside that:
install_config_handlers(
    client,
    lifecycle,
    state,
    settings,
    extra_cache_dirs=[...],  # optional: engine cache dirs this provider owns
)
```

`apply_registration(...)` seeds `state.applied_fingerprint` from the
registration response's `provider_definition.config_fingerprint`, so
the provider and the admin start with the same view.

### `provider.config.update` flow

Payload: `{"backend_config": {...}, "config_fingerprint": "<sha256>",
"idle_timeout_seconds": int, "capacity": int}`.

Order matters: capacity adoption and the noop check run **before** the
drain gate, so a same-fingerprint (or capacity-only) update is always
safely ackable regardless of load and never restarts the backend.

1. **Validate** — fingerprint must be a non-empty string and
   `backend_config` an object; otherwise NAK `{"ok": false, "step":
   "validate"}`.
2. **Capacity adopt** — if the pushed `capacity` differs from
   `lifecycle.capacity`, adopt it immediately. Capacity is enforced at
   the provider (slot admission) and never requires a restart.
   `idle_timeout_seconds` is **not** consumed by the provider (the idle
   reaper is admin-side, `InferenceScheduler._idle_reaper`); it rides in
   the payload for observability only.
3. **Fingerprint compare** — received == applied → ack
   `{"ok": true, "detail": {"noop": true, "capacity_adopted": bool,
   ...}}`; nothing else touched (a capacity-only change lands here with
   `capacity_adopted: true` and the backend still running).
4. **Drain (atomic)** — `lifecycle.stop_if_idle()` checks
   `in_flight > 0 or status == IN_USE` and performs the STOPPING
   transition under the **same lifecycle lock** with no intervening
   await, so a concurrent `/v1` `acquire_slot()` can never slip in
   between check and stop and get SIGTERMed mid-stream. Busy → NAK
   `{"ok": false, "error": "backend_in_use", "detail": {"step":
   "drain", "retry_after": 10, "in_flight": N}}` (no error status
   emitted — the provider is simply refusing, not broken).
5. **Announce** — emit `provider.status initializing` (backend already
   stopped by step 4; skipped work when it was already stopped).
6. **Clear the prompt cache** — delete the OLD fingerprint's
   prompt-cache dir (see the convention below) plus any
   `extra_cache_dirs`. Model files are **never** touched.
7. **Apply config** — `driver.apply_config(backend_config)` (now a
   `BackendDriver` ABC hook with a default no-op; all five providers
   implement it).
8. **Start** — `lifecycle.start()`; artifact resolution/downloads
   happen inside the driver and stream `download.progress` events.
9. **Scrape metadata** — `driver.list_models()` (best-effort: a scrape
   failure logs and yields `[]`, never fails the update).
10. **Ack** — `{"ok": true, "detail": {"config_fingerprint": fp,
   "capacity": c, "model_metadata": [...],
   "prompt_cache_deleted": [...], "prompt_cache_bytes_freed": N}}`.
   The admin persists the echoed fingerprint onto
   `ProviderInstance.config_fingerprint` and stores
   `model_metadata` (as `{"models": [...]}`) on the definition.

On failure at any step (after the drain refusal): emit `provider.status
error` and NAK `{"ok": false, "error": "<message>", "detail": {"step":
"<step>"}}` where step ∈ `validate|drain|cache_clear|apply_config|
start`. The provider stays in whatever state the lifecycle reached
(usually `error`); the admin records the per-instance failure — it is
visible in the PATCH response, not fatal to the admin row.

Note: the fingerprint means the config is **adopted**, not that a
backend is *running* under it — registration seeds
`state.applied_fingerprint` before the first start, and a later stop
leaves it in place.

**Retry policy (admin side, documented choice):** the push uses a
generous per-instance timeout (`CONFIG_UPDATE_TIMEOUT_SECONDS`, default
300s — drain + download + boot can take minutes) and awaits instances
concurrently. A `backend_in_use` NAK is retried up to
`CONFIG_UPDATE_RETRIES` total attempts (default 3) spaced
`CONFIG_UPDATE_RETRY_DELAY_SECONDS` (default 10s); any other failure is
reported immediately. Re-sync is idempotent: re-PATCHing, or simply a
provider reconnect, re-triggers the same flow (see self-heal below).

### Prompt-cache directory convention

**`CACHE_DIR/prompt_cache/<config_fingerprint>/`** is the canonical
per-config prompt-cache location. Providers that persist engine prompt
caches write them under the directory keyed by the fingerprint they
were configured with, so:

- A fingerprint change naturally orphans the old dir; clearing on
  update = deleting the old fingerprint's dir (and the shared root).
- `cache.clear` deletes **only** the prompt-cache root
  (`CACHE_DIR/prompt_cache`) plus the provider's `extra_cache_dirs`
  (gufo: `CACHE_DIR/<MACHINE_UID>` per-instance `--cache-disk` dir;
  halogen-flash: `CACHE_DIR/halogen-flash` engine disk cache).
  `MODELS_DIR` and `CACHE_DIR/provider_config.json` are never touched.
  The ack carries `{"deleted": [...], "bytes_freed": N}` and supports
  `{"dry_run": true}` (plan only). **Refused while the backend is
  `in_use`** (`{"ok": false, "error": "backend_in_use", "detail":
  {"step": "drain", ...}}`) unless `{"force": true}` — clearing engine
  caches under live streams can cause I/O errors. `dry_run` never
  touches files and is always allowed.
- llama-cpp has no on-disk prompt cache in this design (its cache is
  process-RAM via `--cache-ram`/slot save); the convention applies to
  disk-caching engines. The handler is still installed and clears the
  shared root harmlessly.

### `storage.prune_unused`

Payload: `{"dry_run": bool = false}`. Deletes files under `MODELS_DIR`
**not referenced by the current backend_config's resolved artifact
set**. The reference set is `driver.resolved_artifacts` — the local
paths each driver recorded during the last successful artifact
resolution (main GGUF, mmproj, draft, tokenizer dir, prepared NPU
pin files). Safety rules:

- If the driver has no resolved artifact set yet (backend never
  started under this config), the provider **refuses** (`{"ok": false,
  "error": "no_resolved_artifacts..."}`) rather than deleting every
  model on the box.
- A referenced **directory** (tokenizer) protects its whole subtree.
- Reference paths are `os.path.realpath`-normalized on both sides of the
  comparison, so a recorded path with `..` / double slashes / symlinked
  mounts still protects the live file.
- `dry_run: true` returns the plan (`deleted`, `bytes_freed`, `kept`)
  without deleting.
- Ack: `{"ok": true, "deleted": [...], "bytes_freed": N, "kept": [...]}`.

Drivers must reset `resolved_artifacts` in `apply_config` and
repopulate it during `start()` (all five packages do).

### NPU pre-download wiring (halogen-flash)

Closes the Phase 8 deferral: in `HalogenFlashBackend.start()`, when
`options.npu_models` is requested **and** the host NPU probe passes,
the driver reads the image's pins file (`NPU_PINS_FILE`), plans the
download set with `resolve_npu_download_set()` (shared `devices/`
programs pulled from the owner's record), and `ensure_artifact`s each
file into `MODELS_DIR/npu/<id>/` **before spawn** (revision pinned;
`download.progress` events flow through the driver's progress
callback). Prepared files join `resolved_artifacts` so prune keeps
them. Graceful by design: a missing/unparsable pins file or any
download failure logs a warning and **omits the pins**
(`HALOGEN_NPU_MODELS` unset) — the start never fails on NPU prep.

### Self-heal (stale fingerprints)

If the admin PATCHed a definition while a provider was disconnected,
the instance's stored fingerprint lags. On every WS accept the admin
schedules a cheap background check (and the presence sweep repeats it):
`instance.config_fingerprint != sha256(canonical(definition.backend_config))`
→ push `provider.config.update` for that instance **only** (never the
whole-definition fan-out — a current sibling must not be drained or
churned by a heal). A per-instance in-flight guard in
`app/services/config_update.py` ensures a slow (300s) apply is never
re-pushed by the next 30s sweep into the same provider's command queue;
the skipped heal simply retries on a later sweep. Matching fingerprints
cost one DB read and no traffic.

