---
name: integration-testing
description: Run and iterate on OpenResponses compliance tests against the deployed inference matrix; fix failing tests one at a time with commit/deploy cycles
---

# Integration Testing Skill

Use this skill when testing the deployment against the OpenResponses
compliance suite, tracking results, or fixing failing tests via
commit → CI → deploy cycles.

**Overhaul note:** the Responses API is now litellm-driven in
`admin/backend/app/api/v1/responses.py` + `admin/backend/app/services/`
(scheduler, `sse.py` emitter, `alias_registry.py`). The
**Responses-over-WebSocket transport was dropped** (not part of the
OpenResponses spec), so `websocket-*` compliance tests are **not
applicable** — record them as N/A, don't chase them.

## Workflow

1. **Run the suite** (full run first, no filter):

   ```bash
   cd /workspaces/openresponses
   bun run test:compliance --base-url https://matrix.thelink.family/v1 \
     --api-key none --model <alias>
   ```

   - Add `--filter <test-id>` to run a single test (repeatable/comma-separated).
   - Add `--verbose` for request/response dumps on failure.
   - Add `--json` for machine-readable output (pipe to a file for diffing).
   - For fast local iteration, point `--base-url` at a local stack
     (`http://localhost:8000/v1`) running the **mock provider** (see
     `development.md` § Seeding) — same contract, no hardware.

2. **Update the tracker**: record results in
   `docs/integration-testing.md` (status table + history row at the
   bottom). Keep failure clusters current — fixes usually unblock whole
   clusters, not single tests.

3. **Fix one test at a time**:
   - Investigate in `admin/backend/app/api/v1/responses.py` (route +
     persistence), `app/services/sse.py` (client framing: id replacement,
     `sequence_number`, terminal `response.failed` synthesis),
     `app/services/alias_registry.py` (litellm native-streaming
     registration), and `app/services/scheduler.py` (admission).
   - Provider-side translation lives in `provider/lib` + `provider/<type>`
     — a spec-shape bug may originate there, not in the admin.
   - Correlate with deployment logs: `tail -n 200 /tmp/lfai-matrix.log`
     (admin) and provider container logs.
   - Run the single test with `--filter` to verify before moving on.

4. **Commit cycle** (user performs commit/push/deploy):
   - Provide a concise commit message for the fix.
   - User pushes → GitHub Actions builds images → user deploys (admin
     first, then providers — version hard-fail) → log wiped and streaming
     resumes → re-run tests (full or filtered).
   - After deploy, verify the provider version matches the admin (see
     `deployment.md`).

## Failure clusters (overhaul-aware)

1. **Null-valued response fields** — serialize concrete values where the
   schema requires them. Blocks: basic-response, system-prompt,
   image-input, tool-calling.
2. **`output.0: Invalid input`** — output item shape mismatches schema.
   Check the emitter's pass-through of litellm `output[]` and the
   provider's normalization. Blocks: basic-response, system-prompt,
   tool-calling, compact-response.
3. **Streaming terminal frame missing required fields** — ensure
   `response.completed` carries `completed_at`, echoed params, `usage`,
   etc. The `SSEEmitter` reassigns `sequence_number` and swaps the
   client id; verify nothing in that path drops required fields. Blocks:
   streaming-response.
4. **Alias not registered with litellm** — if native streaming isn't
   selected you get fake-streaming / APIError. Confirm
   `ensure_registered(alias)` runs before the litellm call.
5. **HTTP 500 on assistant `phase` labels / multi-turn** — blocks
   assistant-phase, multi-turn. Check `build_litellm_input` chain
   reconstruction (each record stores the FULL input up to that turn).
6. **`previous_response_id` continuation** — the admin owns the chain and
   never passes `previous_response_id` to litellm; a missing prior record
   → 404. Blocks: previous-response-not-found tests (HTTP path only).

**Not applicable (do not fix):**
- `websocket-*` clusters — the WS Responses transport was removed.

## Useful pointers

- Test definitions: `/workspaces/openresponses/src/lib/compliance-tests.ts`
  (test IDs, request shapes, validators).
- Fidelity ground truth: `spike/litellm-fidelity/FINDINGS.md` — which
  events/fields survive litellm native streaming.
- Deployment log: `/tmp/lfai-matrix.log`.
- Tracker: `docs/integration-testing.md`.
- Architecture: `ARCHITECTURE.md` §7 (Inference Flow).

## Conventions

- Never run the full suite more than needed — it takes minutes and some
  tests are slow.
- `--api-key none` is correct: the deployment is trusted-LAN and does not
  enforce auth on `/v1`.
- Use the mock provider for fast local compliance runs before a deploy.
- mypy/ty have pre-existing errors; only keep touched files at/below their
  count.
