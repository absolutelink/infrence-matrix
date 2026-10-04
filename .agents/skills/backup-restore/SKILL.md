---
name: backup-restore
description: Backup and restore the Inference Matrix admin (Postgres config + responses) and understand Redis rebuildability
---

# Backup & Restore — Inference Matrix

Use this skill when backing up or restoring an Inference Matrix deployment.

**There is no backup service/CLI in the codebase.** Backups are done with
standard PostgreSQL tools against the admin database. Redis is **not**
backed up — all Redis state is rebuildable (see §Redis).

## What is worth backing up

| Store | Contents | Backup? |
| --- | --- | --- |
| **PostgreSQL** (`inference_matrix`) | Machines, ProviderDefinitions (incl. `backend_config` + registration tokens), ProviderInstances, ResponseRecords (conversation history), TokenUsageSamples | **Yes — this is the source of truth** |
| **Redis** | WS secrets/epochs/presence, scheduler mirrors, VRAM ledger, metrics-ownership leases | **No** — rebuildable from Postgres + fresh provider registration |
| Provider `MODELS_DIR` | GGUF model artifacts | Optional (re-downloadable; large) |
| Provider `CACHE_DIR` | Prompt caches, `provider_config.json` | No (regenerated) |
| `.env` / compose files | Deployment config, DB password | Yes (out-of-band; contains secrets) |

## Database backup

The admin DB runs in the `postgres` Compose service (user `inference`,
db `inference_matrix`).

```bash
# Full dump (custom format, compressed, restorable with pg_restore)
docker compose exec -T postgres \
  pg_dump -U inference -d inference_matrix -Fc -f /tmp/matrix.dump
docker compose cp postgres:/tmp/matrix.dump ./backups/matrix-$(date +%Y%m%d).dump

# Plain SQL dump (human-readable, restore with psql)
docker compose exec -T postgres \
  pg_dump -U inference -d inference_matrix > matrix-$(date +%Y%m%d).sql
```

On the production host (`core@10.100.2.100`), use `docker exec` against
the running postgres container (adjust container name to the deployment):

```bash
ssh core@10.100.2.100 \
  'docker exec -i <postgres-container> pg_dump -U inference -d inference_matrix -Fc' \
  > matrix-$(date +%Y%m%d).dump
```

### What the dump contains (overhaul schema)

Five tables (see `ARCHITECTURE.md` §4):

- `machines` — uid, host/dns/ip, total_vram_bytes, hardware JSON.
- `provider_definitions` — alias, provider_type, **backend_config JSON**,
  vram_required_bytes, idle_timeout_seconds, capacity,
  **registration_token**, model_metadata, enabled, status.
- `provider_instances` — machine/definition FKs, port, version,
  instance_status, backend_status, epoch, last_seen/last_request_at,
  config_fingerprint, assigned_gpus. (Live WS state — the row itself is
  fine to restore; connections re-establish.)
- `responses` — ResponseRecord conversation chain (input_items,
  output_items, previous_response_id, tokens, store, status).
- `token_usage_samples` — per-request telemetry.

> **Secrets note:** `registration_token` values are included in the dump.
> Treat backup files as sensitive (trusted-LAN model still relies on them
> being unguessable). Store encrypted / access-controlled. The per-instance
> WS secret lives **only in Redis** and is not in the dump — it is
> re-issued on next provider registration.

## Database restore

```bash
# Custom format
docker compose cp ./backups/matrix-20261004.dump postgres:/tmp/matrix.dump
docker compose exec -T postgres \
  pg_restore -U inference -d inference_matrix --clean --if-exists /tmp/matrix.dump

# Plain SQL
docker compose exec -T postgres \
  psql -U inference -d inference_matrix < matrix-20261004.sql
```

Restore into a fresh database:

```bash
docker compose exec -T postgres createdb -U inference inference_matrix
docker compose exec -T postgres pg_restore -U inference -d inference_matrix < backup.dump
# then run migrations to ensure schema matches the deployed code
docker compose exec -T admin bash scripts/prestart.sh   # alembic upgrade head
```

After restore:
1. Recreate provider instances (they re-register and pick up their
   definitions + get a fresh WS secret).
2. Restart the admin so it reloads config and rebuilds Redis runtime state.

## Redis (no backup needed)

All Redis keys are TTL-bounded operational state (see
`admin/backend/docs/redis-keys.md`): `im:ws:*`, `im:metrics:*`,
`im:sched:*`, `im:vram:*`. On a Redis flush/loss:

- **Config is safe** — it lives in Postgres.
- Provider instances re-register (or already-connected ones keep going)
  and the admin rebuilds secrets/epochs/presence.
- Scheduler queues are in-process (single worker) with Redis as a mirror;
  a flush does not lose the authoritative queue.
- VRAM ledger self-heals via TTLs + re-admission.

To flush Redis deliberately (e.g. clear stale leases after a weird state):

```bash
docker compose exec redis redis-cli FLUSHDB
```

Prefer recreating the admin + providers over restoring Redis.

## Model artifacts

`MODELS_DIR` holds GGUF files (re-downloadable via the provider
downloader from the `backend_config` HF descriptors). Back up only if
bandwidth/availability matters:

```bash
# On the provider host (podman volume)
sudo podman volume export <provider_models_volume> | gzip > models-backup.tgz
```

`CACHE_DIR` (prompt caches + `provider_config.json`) never needs backup —
`provider_config.json` is rewritten at registration and caches regenerate.

## Backup strategy (solo operator)

- **Daily**: `pg_dump -Fc` of `inference_matrix`, keep last 7.
- **Before any schema migration / admin upgrade**: manual dump first.
- **Off-box copy**: since dumps contain registration tokens, store on
  encrypted/ACL'd storage, not world-readable.
- **Test restore quarterly** into a scratch DB.

## Verify a backup

```bash
# Custom format: list contents without restoring
docker compose exec -T postgres \
  pg_restore --list /tmp/matrix.dump | head
# Confirm the five tables appear and row counts look sane
docker compose exec -T postgres \
  psql -U inference -d inference_matrix -c \
  "SELECT 'machines',count(*) FROM machines UNION ALL
   SELECT 'provider_definitions',count(*) FROM provider_definitions UNION ALL
   SELECT 'provider_instances',count(*) FROM provider_instances UNION ALL
   SELECT 'responses',count(*) FROM responses UNION ALL
   SELECT 'token_usage_samples',count(*) FROM token_usage_samples;"
```

## Restore scenarios

| Scenario | Steps |
| --- | --- |
| Accidental definition/machine delete | Restore dump into scratch DB, re-insert the rows, or full restore + recreate providers. |
| Full host rebuild | Bring up compose (postgres+redis+admin), restore dump, `prestart.sh`, redeploy matching-version providers, restart providers to re-register. |
| Corrupt Redis | `FLUSHDB` (or recreate the redis container); no data loss — restart admin + providers. |
| Version mismatch after partial deploy | Every mismatched provider 409s. Deploy the admin, then recreate all providers to the matching version (see `deployment.md`). |

## Anti-patterns

- ❌ Backing up Redis as if it held source of truth — it doesn't.
- ❌ Restoring an old dump onto a newer schema without `prestart.sh`.
- ❌ Leaving dumps (with registration tokens) in world-readable locations.
- ❌ Trying to restore `legacy/`-era tables (`agents`, `server_instances`,
  `models`, `inference_leases`, `prompt_cache`, `users`, `api_keys`) —
  they do not exist in the overhaul schema.
