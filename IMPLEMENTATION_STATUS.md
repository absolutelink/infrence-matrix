# Inference Matrix - Implementation Status

Last Updated: September 24, 2026

## ✅ Completed Features

### 🏗️ Infrastructure & DevOps

- [x] **Multi-stage Docker build**
  - Frontend build with Bun/Vite
  - Backend with Python/uv
  - Single image serving both on port 8000
  - Image name: `ghcr.io/absolutelink/matrix-app:main`

- [x] **Modular Entrypoint System**
  - `/etc/entrypoint.d/` with numbered scripts (010, 020...)
  - 010-generate-config.sh: Runtime API URL configuration
  - 020-run-migrations.sh: Database migrations
  - Support for `/opt/rootfs` overlay for custom files
  - CMD override via `/tmp/docker_cmd_override`

- [x] **Database Migrations**
  - Alembic configured with environment variable support
  - BigInteger columns for `size_bytes` and `parameter_count`
  - Automatic migration on container startup

- [x] **Production Deployment**
  - Systemd service on 10.100.2.100
  - Traefik reverse proxy with Let's Encrypt
  - Domain: `matrix.thelink.family`
  - PostgreSQL database connection

### 📊 Models Management

- [x] **Models CRUD**
  - List all models with DataTable
  - Add model manually
  - Edit model details
  - Delete model with confirmation
  - Model actions menu (⋮)

- [x] **HuggingFace Integration**
  - Search HuggingFace for GGUF models
  - Auto-detect split model files (e.g., `model-00001-of-00003.gguf`)
  - Group split files as single download option
  - Fetch actual parameter count from `config.json`
  - Display file sizes and counts

- [x] **Model Information Display**
  - Name, architecture, quantization
  - Size (formatted: GB, MB)
  - Parameter count (formatted: B, M)
  - Context length
  - Source information
  - Tags

### 🎨 Frontend UI Components

- [x] Dashboard layout with sidebar navigation
- [x] DataTable component with sorting, pagination
- [x] Dialog components (Add, Edit, Delete, Search)
- [x] Form components with validation
- [x] Toast notifications (success, error)
- [x] Loading states and skeletons
- [x] Dark/Light theme support
- [x] Responsive design

### 🔌 API Endpoints (Backend)

- [x] `/api/v1/models/` - CRUD operations
- [x] `/api/v1/huggingface/search` - Search HF models
- [x] `/api/v1/huggingface/models/files` - List GGUF files
- [x] `/api/v1/huggingface/models/params` - Get parameter count
- [x] `/api/health` - Health check

### 🌐 Frontend Configuration

- [x] Runtime API URL configuration via `window.APP_CONFIG`
- [x] Fallback chain: runtime → build env → window.location.origin
- [x] Vite dev server proxy for local development
- [x] No hardcoded localhost URLs

---

## 🚧 In Progress

- [x] **Chat Page** - UI complete with backend streaming integration
- [x] **Server instance lifecycle** - start/stop via UI, events, live logs (see below)

---

## 📋 Planned Features

### 🎯 High Priority (MVP)

#### Agents Management
- [x] Agents list page
- [x] Add agent dialog
- [x] Edit agent configuration
- [x] Delete agent
- [x] Agent status monitoring
- [x] Agent logs viewer (llama-server stdout/stderr, 5s auto-refresh)
- [x] Agent metrics viewer (GPU, VRAM, running servers)
- [x] Agent auto-registration from agent service (retry until success)
- [x] Periodic re-registration (heals backend->agent event WS after backend restarts)
- [x] Configurable advertised address (AGENT_HOST / AGENT_PORT)

#### Recipes (Agent Backend Images)
- [x] Vulkan recipe: agent app layered on `ghcr.io/ggml-org/llama.cpp:full-vulkan`
- [x] CI `build-recipes` job chained after `build-agent` (AGENT_IMAGE build-arg)
- [x] Quadlet `.container` deployment file for systemd/podman

#### Server Instances
- [x] Server instances list (live, 5s auto-refresh)
- [x] Start server dialog (model + agent picker)
- [x] Start/stop existing instances from row menu
- [x] Automatic model download on server start (agent fetches from HuggingFace if missing)
- [x] Server lifecycle events (server.started/stopped/error) with DB state fallback (dispatch ack)
- [x] Live llama-server log streaming (log.lines events) with 100-line history seeding + polling fallback
- [x] Per-server llama.cpp logs sheet in UI (live tail, auto-scroll, stderr highlight)
- [x] Stale instance cleanup: agents report running_server_ids on registration; backend marks unreported instances stopped
- [x] Server health monitoring (periodic health checks, transition events, crash detection)
- [ ] Server configuration editing (in-place)
- [ ] Connection testing

#### Real-time Event Streaming
- [x] Agent event bus with buffered pub/sub (`server.*`, `download.*`, `gpu.usage`, `log.lines`)
- [x] Agent `/ws/status` streams real events (was heartbeat-only)
- [x] Backend fans agent events out to UI clients via `/api/ws/events/{agent_id}`
- [x] Backend `server.error` handling, `started_at` on `server.started`
- [x] UI live event feed sheet per agent
- [x] Download progress events with speed_mbps (tqdm shim for hf_hub)
- [x] Real GPU metrics via nvidia-smi sampling + periodic gpu.usage emission

#### Inference Features
- [x] Chat completion UI (OpenAI-compatible `/v1/chat/completions` via agent proxy)
- [ ] Text completion UI
- [ ] Embeddings generator
- [ ] File upload for processing
- [ ] Batch job management
- [ ] Audio transcription UI

#### OpenResponses API (`/v1/responses`) — ✅ implemented (core scope + conformance)
Target: [Open Responses spec v2026-04-24](https://www.openresponses.org/specification) (full core scope incl. WebSocket transport + `/responses/compact`; `service_tier` accepted, maps to default).

- [x] **Schemas module** (`backend/app/api/routes/v1/responses/schemas.py`): content parts (input_text/image/file/video, output_text, refusal, reasoning_text, summary_text), item params (user/system/developer/assistant messages, function_call, function_call_output, reasoning, item_reference, compaction), FunctionToolParam, tool_choice union incl. `allowed_tools`, text.format (text/json_object/json_schema), ReasoningParam, TextParam, StreamOptionsParam, Usage (+details), CreateResponseBody, ResponseResource, CompactRequestBody/CompactResource
- [x] **Spec serialization** (`serialize()` helper): `exclude_none=True, by_alias=True` at every emission point — conformance harness's Zod schemas mark temperature/top_p/penalties/top_logprobs/parallel_tool_calls strictly non-nullable and use optional() (key-absent, not null) for phase/logprobs/encrypted_content; text.format echo matches the TextFormat union (`type:"text"` → `{type}` only, `json_schema` → real `schema` key via alias + `strict` defaulting `true`, `verbosity` omitted when unset); tools echo carries `strict:true` default
- [x] **Streaming events** (`events.py`): all ~24 spec events (response.created/queued/in_progress/completed/failed/incomplete, output_item.added/done, content_part.added/done, output_text.delta/done, refusal.*, reasoning.* + reasoning_text.* (dual-name: spec conformance schemas + OpenWebUI), reasoning_summary_*, function_call_arguments.delta/done, error) with monotonic `sequence_number`, `event:`/`data:` SSE framing, terminal `[DONE]`
- [x] **Persistence**: new `responses` table mirroring ResponseResource 1:1 (replaces `conversations` table, migration `b1f8c2a47d90`); `previous_response_id` chaining = previous.input + previous.output + new input; `store=false` → no persistence, `previous_response_not_found` error on missing chain
- [x] **Translator** (`translator.py`, Responses items ↔ llama.cpp chat messages/chunks): input items → messages (instructions → system, function_call/output replay as text, reasoning skipped); llama.cpp tool_calls + reasoning_content → function_call/reasoning items; `allowed_tools` → backend enforcement (violating calls suppressed into a note message)
- [x] **POST /v1/responses**: alias resolution (incl. stopped/errored auto-start) + name/id fallback, non-streaming + SSE, spec error envelopes (previous_response_not_found, model_not_found); `background=true` → 400; no GET retrieval endpoint (spec doesn't document one)
- [x] **Agent `--jinja` always-on**: llama.cpp tool calling requires `--jinja`; agent starts every llama-server with it (behavior change: templates drive formatting/parsing for chat/completions too)
- [x] **Reasoning**: map llama.cpp `reasoning_content` deltas → reasoning items + dual-name events (`response.reasoning.delta/done` for spec conformance + `response.reasoning_text.delta/done` for OpenWebUI, identical payloads); reasoning item closed (`output_item.done`, completed) as soon as content/tool-call deltas start — llama.cpp sends no explicit end-of-reasoning marker mid-stream, and OpenWebUI only closes the thinking block on `output_item.done`
- [x] **Assistant phase passthrough** (spec 2026-04-24): `commentary`/`final_answer` accepted on assistant input items (replayed as bracketed cue), last assistant phase echoed on output message items
- [x] **Multimodal (real)**: `input_image` parts → llama.cpp multimodal `image_url` message parts (base64 data URLs/remote URLs); mmproj plumbing end-to-end — projector is selected per server instance (start/edit dialogs; default None = no `--mmproj` flag), agent downloads the projector on demand and spawns llama-server with `--mmproj` (servers without a projector get llama.cpp's clean error, surfaced as model_error)
- [x] **POST /v1/responses/compact** (spec 2026-04-24): real compaction — model-summarization sampling pass through the same llama.cpp proxy, returns `response.compaction` (compaction item `encrypted_content` = summary, `created_at`, `usage`), stateless (no DB write); spec error envelopes for missing model / not-found chain
- [x] **WebSocket transport** (`ws.py`, spec 2026-04-24): `POST /v1/responses` also served over WS — `{"type":"response.create"}` turns, same streaming events as SSE, spec error envelope, sequential processing (one in-flight response), transport-specific fields (`stream`/`stream_options`/`background`) rejected, 60-min connection limit (`websocket_connection_limit_reached`); connection-local cache enables `store=false` chaining via `previous_response_id` with `previous_response_not_found` on miss and cache eviction on failed turns; mounted directly in main.py (FastAPI 1.3 `_IncludedRouter` breaks WS handshakes through the v1 include)
- [x] **Tests**: `tests/api/routes/test_responses_unit.py` (schemas/translator/StreamState/SSE framing + `TestConformanceSchemaRules`/`TestConformanceStreamingEvents` mirroring the harness's Zod rules), `tests/api/routes/test_v1_responses.py` (routes), regenerated frontend client
- [x] **UI test page** (`/responses`): dedicated playground — server selector, spec SSE parsing (`event:`+`data:` frames), rendered items (text, Thinking, tool-call chips), **raw streaming-event inspector** (sequence_number + collapsible JSON payloads), `store` toggle with `previous_response_id` chaining ("New conversation" resets), temperature/max_tokens sliders, non-streaming mode rendering the full ResponseResource; added Switch component (`radix-ui`); sidebar entry "Responses API" after Chat
- [ ] Deferred: WebSocket compaction tests (CLI-only), service_tier behavior (accepted, maps to default)

Key llama.cpp facts (researched):
- Tool calling on `/v1/chat/completions` only works with `--jinja` (template autoparser builds PEG grammar; parses model output into structured tool_calls; grammar-constrains arguments from tool JSON schemas)
- Wire shapes: request `tools:[{type:"function",function:{name,description,parameters}}]`, `tool_choice:"auto"|"none"|"required"|{type:"function",function:{name}}`; non-streaming result `message.tool_calls[]` + `finish_reason:"tool_calls"`; streaming via `delta.tool_calls[]` fragments
- We translate flat Responses `FunctionToolParam` ↔ llama's nested `function:{...}` shape; `allowed_tools` is ours to enforce (llama.cpp unaware)
- Images: llama.cpp accepts `{"type":"image_url","image_url":{"url":"data:image/png;base64,..."}}` content parts; without an mmproj projector the server 500s ("image input is not supported") — we surface it as a model_error
- Conformance harness quirks (openresponses repo, `src/lib/sse-parser.ts`): its generated schemas validate reasoning deltas as `response.reasoning.delta/done` while its own reference/spec text and OpenWebUI use `reasoning_text.*` — hence dual-name emission

#### Dashboard Improvements
- [ ] System metrics dashboard
- [ ] Model usage statistics
- [ ] Recent activity feed
- [ ] Quick actions panel
- [ ] Resource usage charts

### 🔧 Medium Priority

#### Benchmarking
- [ ] **Benchmark definitions**
  - Persist benchmark definitions with CRUD support
  - Select an existing server definition and retain a configuration snapshot for reproducibility
  - Reuse server definition add/edit/delete UI patterns
  - Configure core `llama-bench` options, including prompt/context sizes, batch sizes, repetitions, GPU layers, and flash attention
  - Soft-copy source settings while retaining the source definition reference
- [ ] **Benchmark execution**
  - Run `llama-bench` directly on the agent instead of starting a `llama-server`
  - Configure the `llama-bench` executable path per agent, with a PATH-based default
  - Queue runs until the selected agent and model are available
  - Execute all selected prompt/context combinations in a run
  - Parse and persist per-case results and aggregate summary metrics
  - Persist run history, raw output, errors, timestamps, and status transitions until explicitly deleted
- [ ] **Benchmark server isolation**
  - Allow only one active benchmark globally and process queued runs FIFO
  - Wait for all running servers to become idle before stopping them
  - Prompt on idle timeout to abort or force-stop
  - Force-stop terminates active requests before proceeding
  - Keep all servers stopped while a benchmark is running
  - Block server starts/restarts during benchmark execution
  - Leave automatically stopped servers stopped after the run
- [ ] **Benchmark UI and history**
  - Add a dedicated Benchmarks page with definitions, queue, active runs, history, and result details
  - Add benchmark actions and latest result summaries to server definitions
  - Open a dedicated log tab automatically when a run starts
  - Identify log tabs by benchmark run
- [ ] **Benchmark tests**
  - Definition CRUD, snapshots, deletion, and validation
  - Queueing, idle detection, timeout prompts, force-stop, and cleanup
  - Agent subprocess execution, command construction, output parsing, and failures
  - Result persistence and frontend benchmark workflows

#### Server and Log Panel Improvements
- [ ] **Reliable log streaming**
  - Preserve an explicit connected/disconnected/reconnecting state for every log stream
  - Automatically reconnect after WebSocket failures without losing the visible tail
  - Poll log history while disconnected and merge history/live output without duplicates
  - Show a green connected or red disconnected indicator in every log tab title, including inactive tabs
- [ ] **Log panel layout**
  - Reserve main-content space for the open bottom log panel instead of overlaying page inputs
  - Update reserved space while the panel is resized or collapsed
  - Verify Responses API, Chat, and Completions forms remain fully accessible with logs open

#### Prompts Library
- [ ] Saved prompts
- [ ] Prompt templates
- [ ] Prompt categories
- [ ] Share prompts

#### Model Enhancements
- [x] Model download progress (download.progress events with speed_mbps)
- [ ] Model validation checker
- [ ] Model compatibility test
- [ ] Model benchmarking (see the Benchmarking implementation plan above)

### 🌟 Low Priority (Nice to Have)

#### Advanced Features
- [ ] Conversation history
- [ ] Chat export/import
- [ ] Custom themes
- [ ] Keyboard shortcuts
- [ ] Mobile responsive improvements
- [ ] PWA support

#### Monitoring & Analytics
- [ ] Request logging
- [ ] Performance metrics
- [ ] Error tracking
- [ ] Usage analytics dashboard
- [ ] Cost tracking

#### Integrations
- [ ] Webhook configurations
- [ ] API webhooks
- [ ] Third-party integrations

---

## 📁 File Structure

```
frontend/src/
├── components/
│   ├── Common/
│   │   ├── DataTable.tsx ✅
│   │   └── Pending*.tsx ✅
│   ├── Models/
│   │   ├── AddModel.tsx ✅
│   │   ├── AddModelFromHF.tsx ✅
│   │   ├── columns.tsx ✅
│   │   ├── DeleteModel.tsx ✅
│   │   ├── EditModel.tsx ✅
│   │   ├── ModelActionsMenu.tsx ✅
│   │   └── SearchHuggingFace.tsx ✅
│   └── ui/ (Shadcn components) ✅
├── routes/
│   ├── _layout/
│   │   ├── index.tsx ✅ (Dashboard)
│   │   ├── models.tsx ✅
│   │   ├── agents.tsx ✅
│   │   ├── server-instances.tsx ✅
│   │   └── chat/ ✅
│   └── client/ (Auto-generated API client) ✅

backend/app/
├── api/
│   ├── routes/
│   │   ├── models.py ✅
│   │   ├── huggingface.py ✅
│   │   ├── server_instances.py ✅
│   │   ├── agents.py ✅
│   │   └── v1/ (OpenAI-compatible endpoints) ✅
│   └── deps.py ✅ (get_db/SessionDep)
├── models.py ✅
└── core/ (Config, DB) ✅
```

---

## 🛠️ Development Setup

### Local Development
```bash
# Backend
cd backend
uv sync
uv run python -m app.main

# Frontend
cd frontend
bun install
bun run dev
# Access at http://localhost:5173
# API proxied to http://localhost:8000
```

### Docker Build
```bash
docker build -t ghcr.io/absolutelink/matrix-app:main .
```

### Environment Variables
```bash
# Database
POSTGRES_USER=inference-matrix
POSTGRES_PASSWORD=<secret>
POSTGRES_DB=inference-matrix
POSTGRES_HOST=postgres
POSTGRES_PORT=5432

# Application
AGENT_DISCOVERY_ENABLED=true
FRONTEND_HOST=https://matrix.thelink.family

# Runtime (optional - uses current origin if not set)
API_URL=https://matrix.thelink.family
```

---

## 🗄️ Change Archive

### September 22, 2026 (afternoon - server lifecycle & streaming)
- ✅ Real agent event bus: `/ws/status` now streams server.started/stopped/error, download.progress, gpu.usage, log.lines (was heartbeat-only)
- ✅ Implemented `/server-instances/start` (port allocation 8090-8190, DB row up front, background dispatch) and `/{id}/stop` (agent command proxy)
- ✅ Added `/server-instances/{id}/start` to restart stopped/errored instances reusing stored port/config
- ✅ Automatic model download on server start: agent fetches missing GGUF from source_repo_id before launching llama-server
- ✅ Live log streaming: per-server ring buffer in agent, log.lines events every 1s, UI live tail with history seeding (last 100 lines) and polling fallback
- ✅ UI events WS: new `/api/ws/events/{agent_id}` forwarding agent events to UI clients
- ✅ Agents report running_server_ids on registration; backend only resets instances not actually running (fixes stopped/error loop from periodic re-register)
- ✅ Periodic agent re-registration every 60s (heals event channel after backend restarts)
- ✅ Start Server dialog in UI (model + agent picker), Start/Stop actions per instance row
- ✅ Fixed llama-server flags for new llama.cpp: `--flash-attn on|off` (removed `--prompt-cache`)
- ✅ Set LD_LIBRARY_PATH when spawning llama-server (vulkan image env not guaranteed)
- ✅ Fixed 'str' object has no attribute 'get': backend now parses agent WS messages before handling; consolidated duplicate connection loop
- ✅ Fixed log drain KeyError (silent buffer discard); download.progress via tqdm shim (added refresh/set_postfix_str for xet)
- ✅ Removed tracked __pycache__ files; repo-wide gitignore rules
- ✅ Server Instances page wired to real API (was stub)

### September 22, 2026 (earlier)
- ✅ Vulkan recipe rebuilt: layers agent app on official `ghcr.io/ggml-org/llama.cpp:full-vulkan` (no llama.cpp compilation), CI chained after agent build
- ✅ Fixed agent registration 500 (UUID PK upsert on name), wired registration loop on agent startup
- ✅ Fixed WebSocket URLs: agent→frontend scheme mapping (https→wss), backend→agent `/ws/status` (no /api prefix)
- ✅ Mounted agent WS endpoint at `/api/ws/agents/{agent_id}` on backend
- ✅ Implemented `send_command` HTTP proxy to agent (was TODO stub); agent routes at root (no `/api` prefix)
- ✅ Agent list/get now read from database (survives backend restarts)
- ✅ Added View Logs (llama-server stdout/stderr) and View Metrics (GPU/servers) sheets in UI
- ✅ Added `AGENT_HOST`/`AGENT_PORT` settings for advertised agent address
- ✅ Fixed agent startup import errors (model_manager singleton, server_manager alias, response_model)
- ✅ Added missing backend settings (WS_RECONNECT_INTERVAL, AGENT_MAX_RECONNECT_ATTEMPTS)

### September 21, 2026
- ✅ Removed users, authentication, and API keys (not needed for this project)
- ✅ Dropped `user`, `item`, and `api_keys` database tables via migration
- ✅ Regenerated API client without auth endpoints
- ✅ Simplified vite proxy to `/api/v1` and `/v1` (fixed 404s on `/api-keys` page path)
- ✅ Forced `client_encoding=utf8` for SQL_ASCII postgres compatibility

### September 19, 2026
- ✅ Fixed frontend API calls to use runtime config instead of localhost
- ✅ Implemented auto-grouping of split GGUF files from HuggingFace
- ✅ Added endpoint to fetch actual parameter count from HF config.json
- ✅ Fixed BigInteger migration for large model sizes (>2GB)
- ✅ Implemented modular entrypoint system with numbered scripts
- ✅ Fixed database migrations running on container startup
- ✅ Added Vite proxy for local development
- ✅ Updated GitHub Actions to build from root Dockerfile
- ✅ Changed image name to `matrix-app` (was `frontend`)

---

## 🎯 Next Steps

1. ~~OpenResponses API~~ ✅ implemented + conformance pass (see Inference Features section)
2. **Server health monitoring** - periodic health checks; auto-mark instances unhealthy/stopped
3. **Dashboard Metrics** - Add charts and statistics (gpu.usage events already streaming)
4. ~~Schedule cleanup loop~~ - run cleanup_offline_agents periodically on startup ✅
5. **Inference UI** - embeddings, audio transcription pages (text completion done)

---

## 📝 Recent Changes

### September 25, 2026 (server health monitoring)
- ✅ Agent-side health monitor checks active llama-server processes every 10s with failure/recovery thresholds
- ✅ Unexpected llama-server exits emit `server.error` and are removed from the agent's active registry
- ✅ `server.health` transition events persist health state and last-check timestamps in the backend
- ✅ Unhealthy inference instances get a short recovery window instead of the full cold-start timeout
- ✅ Server instance UI shows the last health-check time and agent event feed includes health transitions
- ✅ Added agent and backend regression tests for health thresholds, recovery, crashes, and persistence

### September 23, 2026 (Open Responses conformance pass)
- ✅ Ran the openresponses.org conformance suite (cloned spec repo, validated against its generated Zod schemas); fixed all reported schema failures
- ✅ Spec serialization: `serialize()` helper (`exclude_none`, `by_alias`) — unset temperature/top_p/penalties/top_logprobs/parallel_tool_calls omitted (were `null`, which fails the harness's non-nullable fields); optional() keys (phase/logprobs/encrypted_content) absent, not null; text.format echo matches the TextFormat union (`schema` alias + `strict:true` default); tools echo `strict:true`
- ✅ Reasoning dual-name emission: `response.reasoning.delta/done` (spec conformance schemas) + `response.reasoning_text.delta/done` (OpenWebUI) with identical payloads
- ✅ Assistant phase (`commentary`/`final_answer`) full passthrough; output message echo
- ✅ Real multimodal: `input_image` → llama.cpp `image_url` parts; mmproj end-to-end (projector selected per server instance, agent downloads + `--mmproj` flag; None = flag omitted)
- ✅ `POST /v1/responses/compact`: real model-summarization pass, spec `response.compaction` shape, stateless
- ✅ WebSocket transport at `/v1/responses`: `response.create` turns, sequential, spec error envelope, 60-min limit, connection-local `store:false` cache with eviction-on-failure (spec reconnect-recovery semantics); mounted directly — FastAPI 1.3 `_IncludedRouter` breaks WS handshakes
- ✅ Conformance tests added mirroring harness Zod rules (65 backend tests green); client regenerated, frontend build green

### September 23, 2026 (reasoning passthrough fixes)
- ✅ `/v1/chat/completions` dropped llama.cpp's `delta.reasoning_content` before re-emitting chunks — thinking tokens never reached clients (Matrix Chat page or OpenWebUI); now forwarded in stream deltas and non-streaming `message.reasoning_content`
- ✅ Responses API emitted reasoning as custom `response.reasoning.*` events — OpenWebUI ignores those (expects spec `response.reasoning_text.delta/done`); renamed, UI listener + generated client updated
- ✅ Translator never closed the reasoning item mid-stream (llama.cpp has no end-of-reasoning marker; the `finish_reason == "reasoning_content"` legacy signal never fires) → OpenWebUI kept thinking blocks open until `response.completed`; now closed on first content/tool-call delta

### September 23, 2026 (Responses API test page)
- ✅ New UI page `/responses` ("Responses API" in sidebar after Chat): playground for the OpenResponses endpoint
- ✅ Spec-framed SSE parsing (`event:` + `data:` blocks, `[DONE]` terminator — richer than the chat page's data-line-only parser)
- ✅ Rendered output items: text bubbles, Thinking disclosures (response.reasoning_text.delta), tool-call chips (function_call items)
- ✅ Raw streaming-event inspector panel: every event with sequence_number, type color coding, collapsible JSON payload
- ✅ Test affordances: `store` toggle with `previous_response_id` chaining (New-conversation button resets), temperature / max_output_tokens sliders, non-streaming mode via generated client
- ✅ Added missing `switch.tsx` Shadcn component (unified `radix-ui` package); empty `Sparkles` cleanup
- ✅ tsc clean, biome clean, production build green, routeTree regenerated

### September 23, 2026 (invisible cold starts)
- ✅ New shared service `backend/app/services/server_startup.py`: `ensure_server_ready` / `ensure_server_ready_by_id` / `find_alias_instance` — running servers pass through, `starting` instances are waited on (dedup: parallel requests share one startup), `stopped`/`error` instances are auto-started via agent dispatch and the request waits for health
- ✅ Client connection is held during the entire cold start (including model download, 900s budget) so the first token arrives as soon as the server is healthy — start is invisible to users
- ✅ `/v1/chat/completions`: alias resolution now includes stopped/errored instances (auto-restarts them); `_get_or_create_server` dedups in-flight startups and passes `jinja` + 900s timeout; streaming generator awaits readiness itself
- ✅ `/v1/responses`: same alias auto-start + ensure-ready for both SSE and JSON paths
- ✅ Chat + completions UI selectors now list **all** server instances (running first), stopped ones annotated "(will start on first message)" — no more dead selections
- ✅ Tests: `tests/services/test_server_startup.py` (7 cases: running/starting/error/stopped/missing paths)

### September 23, 2026 (OpenResponses API)
- ✅ Planned + implemented OpenResponses (`/v1/responses`) targeting spec v2026-04-24 — schemas, streaming events, `responses` table (drops `conversations`), translator, agent `--jinja` always-on, unit + route tests, regenerated client (see Inference Features section for the full checklist)

---

## 📊 Statistics

- **Frontend Routes**: 9 (dashboard, models, agents, server-instances, chat, responses, completions, embeddings, audio)
- **API Endpoints**: 30+ implemented (incl. server-instances start/stop/restart, `/v1/responses` + `/responses/compact` + WS transport, models status)
- **UI Components**: 50+ Shadcn components (incl. new Switch)
- **Database Models**: 8 SQLModel classes (ResponseRecord replaced Conversation)
- **Docker Layers**: 2-stage build
- **Entrypoint Scripts**: 3 modular scripts
- **Event Types**: 8 agent events + 26 OpenResponses streaming events (incl. dual-name reasoning)
- **Agent Recipes**: 1 (vulkan on AMD Strix Halo)

---

## 🔗 Links

- **Production**: https://matrix.thelink.family
- **GitHub**: https://github.com/absolutelink/infrence-matrix
- **Container Registry**: ghcr.io/absolutelink/matrix-app
- **Documentation**: See `/docs` folder
