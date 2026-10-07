# Inference Matrix — Development

Local development of the overhauled stack (`admin/` + `provider/`).
Architecture: [ARCHITECTURE.md](ARCHITECTURE.md). Agent conventions:
[AGENTS.md](AGENTS.md).

Everything you need — **PostgreSQL, Redis, the admin, and the mock
provider — runs locally via Docker Compose with no hardware.** For the
fastest inner loop, run the admin (and/or a provider) as a local dev
server against the Compose postgres + redis.

## Prerequisites

- Python 3.14 + [uv](https://docs.astral.sh/uv/)
- [Bun](https://bun.sh/)
- Docker + Docker Compose v2

## Full stack via Compose

```bash
docker compose up -d --build
```

| URL | What |
| --- | --- |
| http://localhost:8000/ | Swagger UI |
| http://localhost:8000/openapi.json | OpenAPI schema |
| http://localhost:8000/admin | React admin UI |
| http://localhost:8000/admin/api/health | Health check |
| http://localhost:8000/v1/... | Public inference API |

Services: `postgres` (5432), `redis` (6379), `admin` (8000),
`provider-mock` (8081). `compose.override.yml` exposes postgres/redis
ports to the host and adds hot-reload watch for the admin;
`compose.deploy.yml` adds Traefik TLS (see deployment.md).

## Quick start: `./scripts/dev.sh`

One command for a fully working local dev instance:

```bash
./scripts/dev.sh          # start + seed + smoke test
./scripts/dev.sh status   # what's running (PIDs + instance WS state)
./scripts/dev.sh down     # stop everything (data kept; reset hints printed)
```

What it does:

- **Docker mode** (when `docker compose` is available): brings up
  postgres + redis + admin + the mock provider container.
- **Local-processes mode** (no docker — uses a local Postgres + Redis per
  `.env`): runs migrations (`admin/backend/scripts/prestart.sh`),
  starts the admin (`uvicorn app.main:app :8000`) and the mock provider
  (`python -m provider_mock.main :8081`) as background processes.
- **Seeds via the admin API** (idempotent — 409 is fine): Machine
  `mock-machine-1` + a **shell** ProviderDefinition `mock-model` (no
  type/config — Phase 14: the type is adopted at registration token
  `mock-registration-token`, capacity 4). After the provider registers
  and connects, the canonical mock `backend_config` is pushed via the
  standard definitions PATCH (`configure_mock_definition`).
- **Smoke tests**: waits for the mock instance's websocket to connect,
  then streams `POST /v1/responses` and asserts `response.completed`.

Logs and PIDs live in `.dev-run/` (gitignored). Re-running while up
detects the running stack and reports instead of double-starting.

### Manual seeding (reference)

The dev script seeds through the API; since Phase 14 definitions may be
created as **shells** (no `provider_type`/`backend_config` — the type is
adopted at first registration and the config is authored via the UI),
so no `provider_types` pre-seed is needed. If you prefer raw SQL
(compose stack), the equivalent is:

```bash
docker compose exec -T postgres psql -U inference -d inference_matrix <<'SQL'
INSERT INTO machines (id, uid, name, host, total_vram_bytes, hardware, created_at)
VALUES (gen_random_uuid(), 'mock-machine-1', 'Mock Machine', 'provider-mock', 32000000000, '{}', now());

-- Phase 14: a shell definition — no provider_type/backend_config. The
-- mock container adopts the type at registration; then configure via
-- the definitions UI (or a PATCH; config push flows over the WS).
INSERT INTO provider_definitions (id, alias, provider_type, backend_config,
    vram_required_bytes, idle_timeout_seconds, capacity, registration_token,
    model_metadata, enabled, status, created_at)
VALUES (gen_random_uuid(), 'mock-model', NULL, NULL,
    8000000000, 300, 4, 'mock-registration-token',
    '{}', true, 'stopped', now());
SQL

docker compose restart provider-mock   # provider registers on startup only
```

Prefer the API path (`./scripts/dev.sh`) — it seeds the shell, waits for
the provider registration (type adopts), pushes the canonical config
and smoke-tests consistently.

The admin reaches the mock provider via `machine.host` — use the Compose
service name `provider-mock` (as in the snippet above). If you run the
provider outside Compose (the `uv run` loop below), set `host` to
`host.docker.internal` or `127.0.0.1`.

Then:

```bash
curl -N http://localhost:8000/v1/responses \
  -H 'Content-Type: application/json' \
  -d '{"model": "mock-model", "input": "Hello!", "stream": true}'
```

## Admin backend dev loop

`./scripts/dev.sh` covers the common case (admin + seeded mock provider +
smoke test). For a hot-reload loop instead, run the admin yourself with
postgres + redis up (`docker compose up -d postgres redis` or local):

```bash
cd admin/backend
uv sync
uv run bash scripts/prestart.sh   # alembic upgrade head
uv run fastapi dev                # http://localhost:8000
```

`fastapi dev` hot-reloads on changes in `admin/backend/app`.

### Admin checks & tests

From `admin/backend`:

```bash
uv run ruff check app
uv run ruff format --check app
uv run pytest tests/ -q
```

Tests require:
- PostgreSQL **UTF8** database, `TEST_DATABASE_URL` (default
  `postgresql+psycopg://postgres@localhost:5432/inference_matrix_test_utf8`).
  The dev cluster's default DBs are SQL_ASCII — create the test DB once:
  `docker compose exec postgres createdb -U inference inference_matrix_test_utf8`
  (and point `TEST_DATABASE_URL` at `postgres` user or grant rights).
- Redis DB 15: `TEST_REDIS_URL` (default `redis://localhost:6379/15`).
- The background presence sweep is disabled in tests
  (`INSTANCE_SWEEP_ENABLED=false`); `sweep_once` is exercised directly.

## Provider dev loop

Provider packages are uv workspace members. Run the mock provider locally
against a local admin:

```bash
cd provider/mock
MACHINE_UID=mock-machine-1 \
PROVIDER_REGISTRATION_TOKEN=mock-registration-token \
ADMIN_BASE_URL=http://localhost:8000 \
CACHE_DIR=/tmp/im-cache MODELS_DIR=/tmp/im-models \
uv run python -m provider_mock.main
```

Provider tests (from the repo root). Provider-package tests expect the
provider env vars to exist (CI sets them); export a dummy set locally:

```bash
export MACHINE_UID=test-machine PROVIDER_REGISTRATION_TOKEN=test-token \
       ADMIN_BASE_URL=http://localhost:8000 \
       CACHE_DIR=/tmp/cache MODELS_DIR=/tmp/models
uv run --project provider/lib pytest provider/lib/tests -q
uv run --project provider/mock pytest provider/mock/tests -q
uv run --project provider/llama-cpp pytest provider/llama-cpp/tests -q
```

Real-hardware providers (llama-cpp etc.) need `LLAMA_SERVER_PATH` and GPU
access — use the mock provider for hardware-free development.

## Frontend dev loop

From the repo root:

```bash
bun install
bun run dev        # Vite at http://localhost:5173/admin/
```

Vite proxies `/admin/api` and `/v1` to `http://localhost:8000` (see
`admin/frontend/vite.config.ts`; `base: "/admin/"`). Run the admin
backend alongside (fastapi dev) while working on the UI.

`admin/frontend/src/routeTree.gen.ts` is auto-generated by the TanStack
Router plugin — never edit by hand. If routes 404 unexpectedly, delete it
and restart Vite.

## API client generation

After any admin route/schema change:

```bash
bash scripts/generate-client.sh
```

Exports `app.openapi()` from `admin/backend` → `admin/frontend/openapi.json`
→ regenerates `admin/frontend/src/client/` → runs frontend lint. Never
hand-edit generated client files.

## Environment variables

- `.env` (root, tracked): local defaults; compose interpolates from it.
- Admin settings: `admin/backend/app/core/config.py` (Pydantic BaseSettings;
  `POSTGRES_*`, `REDIS_URL`, `VERSION`, `INSTANCE_SWEEP_*`, CORS).
- Provider settings: `provider/lib/provider_lib/config.py` (`MACHINE_UID`,
  `PROVIDER_REGISTRATION_TOKEN`, `ADMIN_BASE_URL`, `PROVIDER_PORT`,
  `CACHE_DIR`, `MODELS_DIR`, `METRICS_CATEGORIES`, `LLAMA_SERVER_PATH`).
- Provider containers read env **at startup only** — recreate
  (`docker compose up -d --force-recreate provider-mock`) to apply
  changes.

## Pre-commit hooks

The project uses [prek](https://prek.j178.dev/) (pre-commit compatible).

```bash
uv run prek install -f     # install git hook
uv run prek run --all-files  # manual full run
```

## Testing the public API

OpenResponses compliance suite: see
[docs/integration-testing.md](docs/integration-testing.md). Run it
against a local admin (`http://localhost:8000/v1`) with the mock
provider for fast feedback, or against the deployment.
