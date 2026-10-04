# Inference Matrix — Admin Backend

FastAPI application: the broker, scheduler, provider registry, public
`/v1` inference API, admin API, and the served React SPA. Python 3.14 +
uv; package name `matrix-admin`.

Full design: [../../ARCHITECTURE.md](../../ARCHITECTURE.md). Agent
conventions: [../../AGENTS.md](../../AGENTS.md). Wire protocol:
[../../docs/ws-protocol.md](../../docs/ws-protocol.md). Redis keys:
[docs/redis-keys.md](docs/redis-keys.md).

## URL layout

| Path | Purpose |
| --- | --- |
| `/` | Swagger UI |
| `/openapi.json` | OpenAPI schema |
| `/admin/*` | React SPA (built into `app/frontend`) |
| `/admin/api/*` | Admin management API (health, provider registration) |
| `/v1/*` | Public OpenAI-compatible inference |
| `/provider/ws` | Provider instance WebSocket dial-in |

## Local development

Postgres + Redis must be running (`docker compose up -d postgres redis`
from the repo root). Then:

```bash
uv sync
uv run bash scripts/prestart.sh   # alembic upgrade head
uv run fastapi dev                # http://localhost:8000
```

## Lint & tests

```bash
uv run ruff check app
uv run ruff format --check app
uv run pytest tests/ -q
```

Tests need a **UTF8** PostgreSQL database (`TEST_DATABASE_URL`, default
`postgresql+psycopg://postgres@localhost:5432/inference_matrix_test_utf8`)
and Redis DB 15 (`TEST_REDIS_URL`). The background presence sweep is
disabled in tests (`INSTANCE_SWEEP_ENABLED=false`).

## Layout

```
app/
  main.py                 App factory + lifespan (scheduler, sweep)
  models.py               SQLModel tables (Machine, ProviderDefinition,
                          ProviderInstance, ResponseRecord, TokenUsageSample)
  core/                   config.py, db.py, redis.py
  api/
    admin/                /admin/api/* (health, providers register)
    v1/                   /v1/* (public inference)
    ws.py                 /provider/ws
    main.py               Router assembly
  services/
    scheduler.py          InferenceScheduler (FIFO + VRAM admission)
    connection_manager.py WS registry, auth, epoch, command/ack
    presence_sweep.py     Stale-connection sweeper
    sse.py                SSEEmitter (client-facing stream framing)
    alias_registry.py     litellm alias registration
    metrics_service.py    Machine metrics ownership
    wire.py               Admin mirror of the provider wire envelope
    redis_keys.py         Redis key helpers
alembic/                  ONE squashed initial migration
scripts/prestart.sh       Applies migrations
tests/                    pytest suite
```

## Client generation

After route/schema changes, run `bash ../../scripts/generate-client.sh`
from the repo root to regenerate the frontend API client.

## Note on provider packages

This app must **not** import from `provider/*`. It builds with
`uv sync --no-install-workspace --package matrix-admin`. The `wire.py`
duplication with `provider_lib.wire` is deliberate — keep both in sync.
