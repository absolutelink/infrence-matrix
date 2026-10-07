#!/usr/bin/env bash
# Local development bootstrap for Inference Matrix.
#
#   ./scripts/dev.sh          start the stack (auto-detects docker vs local procs)
#   ./scripts/dev.sh status    show what's running
#   ./scripts/dev.sh down      stop background processes / compose stack
#
# Starts the admin + a mock provider, seeds a Machine + ProviderDefinition
# via the admin API (idempotent), and smoke-tests a streamed /v1/responses
# call. Docker mode uses compose; without Docker it runs local processes
# against a local PostgreSQL + Redis.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_DIR="$ROOT/.dev-run"
ADMIN_PID_FILE="$RUN_DIR/admin.pid"
PROVIDER_PID_FILE="$RUN_DIR/provider.pid"

MACHINE_UID="${MOCK_MACHINE_UID:-mock-machine-1}"
AGENT_ID="${MOCK_AGENT_ID:-mock-agent-1}"
DEF_ALIAS="${MOCK_MODEL_ALIAS:-mock-model}"
ADMIN_PORT="${DEV_ADMIN_PORT:-8000}"
ADMIN_URL="http://127.0.0.1:$ADMIN_PORT"
PROVIDER_PORT="${DEV_PROVIDER_PORT:-8081}"
# Populated by seed() from the machine's registration_secret.
MACHINE_SECRET=""

log() { printf '\033[1;36m[dev]\033[0m %s\n' "$*"; }
err() { printf '\033[1;31m[dev]\033[0m %s\n' "$*" >&2; }

# --- helpers ---------------------------------------------------------------

pid_alive() {
  local f="$1"
  [[ -f "$f" ]] && kill -0 "$(cat "$f")" 2>/dev/null
}

load_root_env() {
  # POSTGRES_* / REDIS_URL defaults for prestart.sh + admin settings.
  if [[ -f "$ROOT/.env" ]]; then
    set -a
    # shellcheck disable=SC1091
    source "$ROOT/.env"
    set +a
  fi
  export FASTAPI_ENV=development
}

docker_mode() {
  command -v docker >/dev/null 2>&1 && docker compose version >/dev/null 2>&1 \
    && docker info >/dev/null 2>&1
}

# Phase 16: no provider_types pre-seed and no shell definitions. A typed
# definition (provider_type + backend_config) is created directly; the mock
# agent authenticates with the machine's shared registration_secret.

# Seed machine + a TYPED definition through the admin API. Captures the
# machine's registration_secret into $MACHINE_SECRET for the provider env.
# $1 = host the admin should use to reach the mock provider.
seed() {
  local provider_host="$1"
  log "seeding machine '$MACHINE_UID' + typed definition '$DEF_ALIAS' via admin API..."
  local body code
  body=$(curl -s -w $'\n%{http_code}' -X POST "$ADMIN_URL/admin/api/machines" \
    -H 'Content-Type: application/json' \
    -d "{\"uid\":\"$MACHINE_UID\",\"name\":\"Mock Machine\",\"host\":\"$provider_host\",\"total_vram_bytes\":32000000000}") || true
  code="${body##*$'\n'}"
  body="${body%$'\n'*}"
  case "$code" in
    2*) log "  machine created" ;;
    409) log "  machine already exists" ;;
    *) err "  machine seed failed (HTTP $code)"; return 1 ;;
  esac
  # Read the machine's registration_secret (from the create body if present,
  # else by looking it up in the list).
  MACHINE_SECRET=$(python3 -c 'import json,sys
try:
    d = json.loads(sys.argv[1])
except Exception:
    d = {}
print(d.get("registration_secret") or "")' "$body")
  if [[ -z "$MACHINE_SECRET" ]]; then
    MACHINE_SECRET=$(curl -s "$ADMIN_URL/admin/api/machines" 2>/dev/null \
      | python3 -c 'import json,sys
data = json.load(sys.stdin)
m = next((x for x in data if x.get("uid") == "'"$MACHINE_UID"'"), None)
print((m or {}).get("registration_secret") or "")' 2>/dev/null || echo "")
  fi
  [[ -n "$MACHINE_SECRET" ]] || { err "  could not read machine registration_secret"; return 1; }
  # Phase 16: a TYPED definition — provider_type + backend_config, no token.
  code=$(curl -s -o /dev/null -w '%{http_code}' -X POST "$ADMIN_URL/admin/api/definitions" \
    -H 'Content-Type: application/json' \
    -d "{\"alias\":\"$DEF_ALIAS\",\"provider_type\":\"mock\",\"backend_config\":{\"artifacts\":{\"model\":{\"path\":\"/models/mock.gguf\"}},\"context\":{\"ctx\":4096,\"predict\":-1},\"sampling\":{\"temperature\":0.8,\"top_k\":40,\"top_p\":0.95,\"seed\":-1},\"stream\":{\"delta_count\":3,\"delta_delay\":0.0},\"server\":{\"backend_port\":8082}},\"capacity\":4,\"vram_required_bytes\":0,\"idle_timeout_seconds\":300}") || true
  case "$code" in
    2*) log "  typed definition created" ;;
    409) log "  definition already exists" ;;
    *) err "  definition seed failed (HTTP $code)"; return 1 ;;
  esac
}

# Wait for the mock agent's WS to connect, then stream one response.
smoke_test() {
  log "waiting for mock provider websocket..."
  local i connected
  for i in $(seq 1 60); do
    connected=$(curl -s "$ADMIN_URL/admin/api/instances" 2>/dev/null \
      | python3 -c 'import json,sys
data = json.load(sys.stdin)
print(any(d.get("alias") == "'"$DEF_ALIAS"'" and d.get("websocket_connected") for d in data))' 2>/dev/null || echo False)
    [[ "$connected" == "True" ]] && break
    sleep 1
  done
  if [[ "$connected" != "True" ]]; then
    err "mock provider did not connect within 60s (see $RUN_DIR/*.log)"
    return 1
  fi
  log "  provider connected — smoke-testing streamed /v1/responses..."
  local out
  out=$(curl -sN --max-time 30 -X POST "$ADMIN_URL/v1/responses" \
    -H 'Content-Type: application/json' \
    -d "{\"model\":\"$DEF_ALIAS\",\"input\":\"ping\",\"stream\":true}" || true)
  if grep -q 'response.completed' <<<"$out"; then
    log "  PASS: stream contained response.completed"
  else
    err "  FAIL: no response.completed in stream:"
    head -c 800 <<<"$out" >&2; echo >&2
    return 1
  fi
}

print_summary() {
  cat <<EOF

  ┌────────────────────────── Inference Matrix (dev) ──────────────────────────┐
  Admin UI   $ADMIN_URL/admin
  Swagger    $ADMIN_URL/
  API base   $ADMIN_URL/v1
  Mock alias $DEF_ALIAS   agent $AGENT_ID on machine $MACHINE_UID
  Logs       $RUN_DIR/*.log          Stop:  ./scripts/dev.sh down
  └─────────────────────────────────────────────────────────────────────────────┘
EOF
}

start_bg() {
  # start_bg <pidfile> <workdir> <cmd...>
  # Fully detaches: new session (setsid), own pid recorded = session leader,
  # fds redirected so the caller's terminal/pipe is never held open.
  local pidfile="$1" workdir="$2"
  shift 2
  setsid bash -c 'pf=$1; dir=$2; shift 2; echo $$ > "$pf"; cd "$dir" && exec "$@"' _ "$pidfile" "$workdir" "$@" \
    >>"$RUN_DIR/$(basename "$pidfile" .pid).log" 2>&1 </dev/null &
  disown
}

# --- docker mode -----------------------------------------------------------

up_docker() {
  load_root_env
  # compose runs the admin with FASTAPI_ENV=production, which hard-fails on
  # the default "changethis"/empty password. A dev bootstrap needs a real
  # value; generate one if the operator hasn't set POSTGRES_PASSWORD.
  if [[ -z "${POSTGRES_PASSWORD:-}" || "${POSTGRES_PASSWORD:-}" == "changethis" ]]; then
    export POSTGRES_PASSWORD="$(head -c 24 /dev/urandom | base64 | tr -d '/+=' | head -c 20)"
    log "no POSTGRES_PASSWORD set — generated an ephemeral dev password"
  fi
  log "docker mode: building + starting postgres/redis/admin..."
  docker compose -f "$ROOT/compose.yml" up -d --build postgres redis admin
  log "waiting for admin health..."
  local i
  for i in $(seq 1 60); do
    curl -sf "$ADMIN_URL/admin/api/health" >/dev/null 2>&1 && break
    sleep 1
  done
  curl -sf "$ADMIN_URL/admin/api/health" >/dev/null 2>&1 || { err "admin not healthy"; return 1; }
  seed "provider-mock"
  log "starting mock provider..."
  MOCK_MACHINE_SECRET="$MACHINE_SECRET" MOCK_AGENT_ID="$AGENT_ID" \
    docker compose -f "$ROOT/compose.yml" up -d --build provider-mock
  smoke_test
}

# --- local-processes mode ----------------------------------------------------

up_local() {
  load_root_env
  # Local-processes mode targets a host-local Postgres/Redis. Export the
  # localhost defaults so the admin subprocess (pydantic_settings reads
  # env with highest precedence) doesn't fall back to the compose service
  # names "postgres"/"redis" when .env is absent.
  export POSTGRES_HOST="${POSTGRES_HOST:-localhost}"
  export POSTGRES_PORT="${POSTGRES_PORT:-5432}"
  export REDIS_URL="${REDIS_URL:-redis://localhost:6379/0}"
  log "local mode: checking PostgreSQL + Redis..."
  pg_isready -h "${POSTGRES_HOST:-localhost}" -p "${POSTGRES_PORT:-5432}" >/dev/null 2>&1 \
    || { err "Postgres not reachable at ${POSTGRES_HOST:-localhost}:${POSTGRES_PORT:-5432} — start it first"; return 1; }
  redis-cli -u "${REDIS_URL:-redis://localhost:6379/0}" ping >/dev/null 2>&1 \
    || { err "Redis not reachable at ${REDIS_URL:-redis://localhost:6379/0} — start it first"; return 1; }

  log "running migrations..."
  (cd "$ROOT/admin/backend" && uv run bash scripts/prestart.sh) >"$RUN_DIR/prestart.log" 2>&1 \
    || { err "migrations failed — see $RUN_DIR/prestart.log"; return 1; }

  log "starting admin (uvicorn :$ADMIN_PORT)..."
  start_bg "$ADMIN_PID_FILE" "$ROOT/admin/backend" \
    uv run uvicorn app.main:app --host 127.0.0.1 --port "$ADMIN_PORT"
  local i
  for i in $(seq 1 60); do
    curl -sf "$ADMIN_URL/admin/api/health" >/dev/null 2>&1 && break
    sleep 1
  done
  curl -sf "$ADMIN_URL/admin/api/health" >/dev/null 2>&1 \
    || { err "admin failed to start — see $RUN_DIR/admin.log"; return 1; }

  seed "127.0.0.1"

  log "starting mock provider (:$PROVIDER_PORT)..."
  mkdir -p "$RUN_DIR/provider-cache" "$RUN_DIR/provider-models"
  start_bg "$PROVIDER_PID_FILE" "$ROOT/provider/mock" \
    env MACHINE_UID="$MACHINE_UID" \
      MACHINE_SECRET="$MACHINE_SECRET" \
      AGENT_ID="$AGENT_ID" \
      ADMIN_BASE_URL="$ADMIN_URL" \
      PROVIDER_PORT="$PROVIDER_PORT" \
      CACHE_DIR="$RUN_DIR/provider-cache" \
      MODELS_DIR="$RUN_DIR/provider-models" \
      uv run python -m provider_mock.main

  smoke_test
}

# --- down / status -----------------------------------------------------------

cmd_down() {
  if docker_mode; then
    log "docker mode: compose down..."
    docker compose -f "$ROOT/compose.yml" down || true
  fi
  local stopped=0
  for f in "$PROVIDER_PID_FILE" "$ADMIN_PID_FILE"; do
    if pid_alive "$f"; then
      local pid; pid=$(cat "$f")
      # Processes start via setsid (own process group) — signal the group
      # so uv/uvicorn children die too.
      kill -TERM -"$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true
      log "stopped $(basename "$f" .pid) (pid $pid)"
      stopped=1
    fi
    rm -f "$f"
  done
  [[ $stopped -eq 0 ]] && log "no local dev processes running"
  log "to reset data: psql -U postgres -c 'DROP DATABASE IF EXISTS inference_matrix' (then re-run) "
  log "              redis-cli -n 0 FLUSHDB"
}

cmd_status() {
  if pid_alive "$ADMIN_PID_FILE"; then
    echo "admin:    RUNNING (pid $(cat "$ADMIN_PID_FILE"))  $ADMIN_URL"
  else
    echo "admin:    stopped"
  fi
  if pid_alive "$PROVIDER_PID_FILE"; then
    echo "provider: RUNNING (pid $(cat "$PROVIDER_PID_FILE"))  :$PROVIDER_PORT"
  else
    echo "provider: stopped"
  fi
  if curl -sf "$ADMIN_URL/admin/api/health" >/dev/null 2>&1; then
    curl -s "$ADMIN_URL/admin/api/instances" 2>/dev/null | python3 -c '
import json, sys
for d in json.load(sys.stdin):
    print("instance: {} ws={} status={}/{}".format(
        d.get("alias"), d.get("websocket_connected"),
        d.get("agent_status"), d.get("backend_status")))
' || true
  else
    echo "instance: (admin not reachable)"
  fi
}

# --- main --------------------------------------------------------------------

cmd="${1:-up}"
case "$cmd" in
  up)
    mkdir -p "$RUN_DIR"
    if pid_alive "$ADMIN_PID_FILE" || { docker_mode && docker compose -f "$ROOT/compose.yml" ps --status running admin 2>/dev/null | grep -q admin; }; then
      log "already running — use './scripts/dev.sh status' or './scripts/dev.sh down' first"
      exit 0
    fi
    if docker_mode; then up_docker; else up_local; fi
    print_summary
    ;;
  down)
    cmd_down
    ;;
  status)
    cmd_status
    ;;
  *)
    err "usage: $0 [up|down|status]"
    exit 2
    ;;
esac
