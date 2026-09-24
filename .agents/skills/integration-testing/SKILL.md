---
name: integration-testing
description: Run and iterate on OpenResponses compliance tests against the deployed inference matrix; fix failing tests one at a time with commit/deploy cycles
---

# Integration Testing Skill

Use this skill when testing the deployment against the OpenResponses compliance
suite, tracking results, or fixing failing tests via commit → CI → deploy cycles.

## Workflow

1. **Run the suite** (full run first, no filter):

   ```bash
   cd /workspaces/openresponses
   bun run test:compliance --base-url https://matrix.thelink.family/v1 \
     --api-key none --model voyager
   ```

   - Add `--filter <test-id>` to run a single test (repeatable/comma-separated).
   - Add `--verbose` for request/response dumps on failure.
   - Add `--json` for machine-readable output (pipe to a file for diffing).

2. **Update the tracker**: record results in `docs/integration-testing.md`
   (status table + history row at the bottom). Keep the failure-cluster section
   current — fixes usually unblock whole clusters, not single tests.

3. **Fix one test at a time**:
   - Investigate in `backend/app` (Responses API implementation lives there).
   - Correlate failures with deployment logs: `tail -n 200 /tmp/lfai-matrix.log`.
   - The log is wiped before each deploy; deployment cycles stream to it again.
   - Run the single test with `--filter` to verify before moving on.

4. **Commit cycle** (user performs commit/push/deploy):
   - Provide a concise commit message for the fix.
   - User pushes → GitHub Actions builds images → user deploys → log wiped and
     streaming resumes → re-run tests (full or filtered).

## Failure clusters (fix order matters)

Fixing clusters unblocks multiple tests at once:

1. **Null-valued response fields** — backend serializes `null` for
   `parallel_tool_calls`, `top_p`, `temperature`, `presence_penalty`,
   `frequency_penalty`, `top_logprobs`; schema requires concrete values.
   Blocks: basic-response, system-prompt, image-input, tool-calling.
2. **`output.0: Invalid input`** — output item shape mismatches schema
   (missing/incorrectly typed item fields). Blocks: basic-response, system-prompt,
   tool-calling, compact-response.
3. **Streaming final response missing required fields** — `response.completed`
   payload lacks `completed_at`, `instructions`, `previous_response_id`, `error`,
   `reasoning`, `max_output_tokens`, `max_tool_calls`. Blocks: streaming-response.
4. **WebSocket framing** — server sends SSE-style `event:`/`data:` frames over
   WS; clients expect one raw JSON object per WS message. Blocks: all websocket-*
   tests (websocket-previous-response-not-found passes because it tolerates it).
5. **HTTP 500 on assistant `phase` labels / multi-turn** — blocks assistant-phase,
   multi-turn.
6. **Compaction items rejected as input** — blocks websocket-compact-new-chain.

## Useful pointers

- Test definitions: `/workspaces/openresponses/src/lib/compliance-tests.ts`
  (test IDs, request shapes, validators).
- WS client behavior: `/workspaces/openresponses/src/lib/compliance-tests.ts`
  (`runWebSocket*` helpers) — see what frame format it expects.
- Deployment log: `/tmp/lfai-matrix.log`.
- Tracker: `docs/integration-testing.md`.

## Conventions

- Never run the full suite more than needed — it takes ~5 min and several tests
  are slow (websocket tests time out at 30s each).
- A test timing out "waiting for terminal WebSocket response event" is almost
  always the framing bug, not a real timeout.
- `--api-key none` is correct: the deployment currently does not enforce auth.
- mypy/ty have pre-existing errors; only keep touched files at/below their count.