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
