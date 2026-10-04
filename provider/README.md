# Provider packages — Phase 4: backend lifecycle + /v1 surface

A **provider instance** runs on a hardware machine, owns a real inference
backend as a subprocess (or fake, for the mock), and exposes:

- An **OpenAI-compatible HTTP API on `PROVIDER_PORT`** (default 8081).
  The admin's litellm client targets `http://<machine>:<PROVIDER_PORT>/v1/...`.
- A **dial-out WebSocket** to the admin (`/provider/ws`) for lifecycle
  commands/events (see `docs/ws-protocol.md`).

`provider/lib/provider_lib` holds the generic machinery; each provider
package (`provider/mock`, later `provider/llama-cpp`) implements the
per-type driver.

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
| `POST /v1/responses` | Slot-admitted SSE (`text/event-stream`). Each event: `event: <type>` + `data: <json>` lines (OpenResponses style, no `[DONE]`). **429** busy, **503** not ready. |
| `POST /v1/chat/completions` | Same discipline; chunks are `chat.completion.chunk` events, terminated with `data: [DONE]`. **501** if the driver has no chat surface. |
| `GET /health` | Provider identity + `backend_status`, `in_flight`, `capacity`. |

Slots are acquired **before** the `StreamingResponse` starts so
busy/not-ready are real HTTP statuses; no slot leaks on early errors.

## Mock provider (provider/mock)

`MockBackend` implements the driver: instant start, canned
`output_text.delta` streams (`delta_count`/`delta_delay`/`hold`
configurable), model alias adopted from the registration response.
`provider_mock.main` builds **one** `BackendLifecycle` shared between
the admin WS command handlers (`backend.start`/`backend.stop` drive it)
and the FastAPI `/v1` app. Boot is admin-driven: after connect the
backend stays STOPPED.

## Phase 5 (llama-cpp) checklist

Implement `BackendDriver` over a managed `llama-server` subprocess:
`start` spawns + waits for health, `stream_responses` proxies the
upstream SSE and normalizes to spec events (usage in
`response.completed`), `stop` terminates the process. Pass your driver
into `BackendLifecycle` and mount via `BackendOverrides(lifecycle=...)`.
All slot/state-machine semantics come for free — do not re-implement
them in the package.

## llama-cpp provider (provider/llama-cpp) — Phase 5

`LlamaCppBackend` (provider_llama_cpp/driver.py) manages one
`llama-server` subprocess. Entry point `provider_llama_cpp.main` follows
the mock's admin-driven boot pattern.

### backend_config schema

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
| `swa_full`/`kv_unified`/`strict_mtp_qwen`→`--spec-mtp-strict-qwen` | emitted when truthy |
| `kv_offload`, `cache_prompt`, `cont_batching`, `warmup`, `context_shift`, `no_mmap`, `no_cache_idle_slots` | `--x` / `--no-x` by boolean. NOTE: `--cache-prompt` is a valid modern flag; the OBSOLETE `--prompt-cache` is NEVER emitted. |
| `flash_attn` (bool or "on"/"off") | `--flash-attn on|off`; skipped when None |
| `draft` artifact present | `--spec-type draft-dflash` + `-md <path>` |
| else `mtp_draft_max > 0` | `--spec-type draft-mtp --spec-draft-n-max N` |
| `jinja` (default true) | `--jinja` |
| `mmproj` artifact | `--mmproj <path>` |
| `strict_mtp_qwen: true` | forces `--parallel 1` |

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
