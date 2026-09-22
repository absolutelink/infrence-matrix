# Inference Matrix - Implementation Status

Last Updated: September 22, 2026 (evening)

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
- [ ] Server health monitoring (periodic health checks)
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

#### Dashboard Improvements
- [ ] System metrics dashboard
- [ ] Model usage statistics
- [ ] Recent activity feed
- [ ] Quick actions panel
- [ ] Resource usage charts

### 🔧 Medium Priority

#### Prompts Library
- [ ] Saved prompts
- [ ] Prompt templates
- [ ] Prompt categories
- [ ] Share prompts

#### Model Enhancements
- [x] Model download progress (download.progress events with speed_mbps)
- [ ] Model validation checker
- [ ] Model compatibility test
- [ ] Model benchmarking

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

## 📝 Recent Changes

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

1. **Server health monitoring** - periodic health checks; auto-mark instances unhealthy/stopped
2. **Dashboard Metrics** - Add charts and statistics (gpu.usage events already streaming)
3. ~~Schedule cleanup loop~~ - run cleanup_offline_agents periodically on startup ✅
4. **Inference UI** - text completion, embeddings, audio transcription pages

---

## 📊 Statistics

- **Frontend Routes**: 6 (dashboard, models, agents, server-instances, chat, 404)
- **API Endpoints**: 25+ implemented (incl. server-instances start/stop/restart, models status)
- **UI Components**: 50+ Shadcn components
- **Database Models**: 8 SQLModel classes
- **Docker Layers**: 2-stage build
- **Entrypoint Scripts**: 3 modular scripts
- **Event Types**: 8 (server.started/stopped/error, download.started/progress/completed/failed, gpu.usage, log.lines)
- **Agent Recipes**: 1 (vulkan on AMD Strix Halo)

---

## 🔗 Links

- **Production**: https://matrix.thelink.family
- **GitHub**: https://github.com/absolutelink/infrence-matrix
- **Container Registry**: ghcr.io/absolutelink/matrix-app
- **Documentation**: See `/docs` folder
