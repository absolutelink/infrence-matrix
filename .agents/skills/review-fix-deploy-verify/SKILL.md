---
name: review-fix-deploy-verify
description: Use when working through an Inference Matrix code-review finding one at a time via delegated implementation, independent review, local verification, user-led commit/deploy, and post-deploy regression and integration tests.
---

# Review → fix → deploy → verify

Use this skill when the user wants to take a finding from a repository review to a
verified deployment, then move to the next finding. Keep each cycle focused on
**one issue**. For an integration-suite-only request, use `integration-testing`
instead; for exploration-only requests, return the diagnosis without editing.

## 1. Scope and reproduce

- Check `git status`, the relevant diff, and the current implementation first.
  Treat changes made during the session as user work. Check again before handoff:
  the user may have committed or deployed while you were reviewing.
- Translate the finding into an observable failure, expected behavior, affected
  transport(s), and a bounded change. Verify claims against source and, where
  practical, reproduce them. Do not assume a passing test proves the deployed
  version has the fix.
- Ask only necessary clarifications using the **questions tool**. For example:
  whether to include adjacent routes, whether to preserve backpressure, or how
  to exercise an idle gap remotely. Recommend a conservative option first.
- If the user requests research with `@explore`, delegate research, distinguish
  evidence from hypotheses, and do not implement until requested.

## 2. Delegate the implementation

For this workflow, hand the scoped fix to a **`general` subagent** using the
task tool. Give it a self-contained prompt with:

1. Exact paths, call chain, observed failure, expected result, and explicit
   scope boundaries. Include relevant `AGENTS.md` invariants (especially
   slot release on upstream connection close, admin scheduler release in a
   cancellation-safe `finally`, provider instance process identity, the
   admin-owned `resp_<uuid>`, and generated files).
2. Required cancellation/error/resource-cleanup behavior and how to preserve
   existing semantics. State which behavior is *not yet proven* if diagnosis is
   uncertain.
3. A regression test that fails before the fix and passes after, plus the
   smallest appropriate checks. Ask it to report changed files, test results,
   and residual risks. Do not ask for tests that merely mirror implementation.
4. **No commit, push, deploy, production SSH, or remote test**. Preserve
   pre-existing work; do not hand-edit generated clients.

If the user explicitly requests a different agent or no delegation, follow
that instruction. Do not launch parallel agents just to accelerate the cycle.

## 3. Independently review and verify

- Read the *actual diff*, not only the agent summary. Trace the production
  caller as well as the helper under test; look for early break, disconnect,
  concurrent cancellation, timeouts, resource/slot release, and response
  framing. Check that failures keep their original type/status and that
  bounded queues retain backpressure.
- Challenge assumptions with evidence. In the C3 cycle, local tests proved
  `wait_for(anext())` truncated an async generator, but a 48-second
  **client-visible** gap did not prove an idle upstream body read: it could
   precede provider response headers or consist of discarded upstream comments.
  Correct the diagnosis when observations disagree.
- Strengthen missing *meaningful* tests and fix review findings, or send the
  agent a focused follow-up. Where safe, demonstrate that the regression test
  fails on the prior implementation. Confirm prompt/JSON shapes used by tests
  and probes against the real API schema.
- Run the relevant checks once after the final changes. Examples:
  - Admin (workdir `admin/backend/`): `uv run ruff check app`,
    `uv run ruff format --check app`, `uv run pytest tests/ -q --tb=short`.
    Check touched test files with Ruff too; `ruff ... app` does not include them.
  - Provider packages (repo root): `uv run --project provider/lib pytest
    provider/lib/tests -q` (same for `provider/mock`, `provider/llama-cpp`),
    and Ruff on touched provider files.
  - Frontend: run relevant build/lint checks. Admin **route/schema** changes
    require `bash scripts/generate-client.sh`; never edit generated files.
  - Database changes: apply migrations before DB-backed tests as documented in
    `AGENTS.md` (`admin/backend/scripts/prestart.sh`). Never run destructive
    `scripts/test.sh` on a shared database.
- Inspect `git diff --check` and `git status` at handoff. Do not commit unless
  the user explicitly asks. Provide a **concise commit message in a fenced
  code block** when the change is ready for the user to commit/deploy.

## 4. Write a targeted remote probe when it adds evidence

Create a small script in `scripts/` only if the deployed behavior is remotely
observable and the test can distinguish a regression from a normal outcome.
Make endpoint/model/base URL configurable and use a valid request shape. For
cache-sensitive tests, use novel input so warm prompt caches cannot turn the
intended stall into a fast request. Include diagnostics and exit codes:

- `0`: observed the trigger **and** the expected recovery/completion.
- `1`: observed the trigger and the regression, or a meaningful client-visible
  violation.
- `2`: trigger not observed; **inconclusive**, not a false pass.
- `3`: HTTP/transport/setup error; not proof of the product bug.

Measure at the boundary relevant to the finding. For SSE, a client-visible
gap is not proof of an upstream-body idle; header acquisition and intermediary
buffering can produce the same symptom. Allow a small timing tolerance.
Validate probe parsing and exit statuses locally with controlled responses.
If a reliable remote trigger is unavailable, prefer a deterministic local
test and explain why the remote probe would be inconclusive.

## 5. Wait for deployment, then verify once

- The **user** commits and deploys unless they explicitly request those actions
  from you. Never assume a new commit is live; wait for deployment confirmation.
  Do not SSH into production or alter production data without explicit request.
- Run the targeted probe against the deployed endpoint first; report whether
  it actually exercised the trigger, and retain the raw result/exit status.
  If inconclusive, investigate stage boundaries, caching, and buffering before
  claiming success or failure. Do not repeatedly load a model without reason.
- For OpenResponses/Responses changes, also load `integration-testing` and
  run the **full** suite once (workdir `/workspaces/openresponses`):

  ```bash
  bun run test:compliance --base-url https://matrix.thelink.family/v1 \
    --api-key none --model rocinante
  ```

  Run it when this cycle includes integration verification or the user asks.
  Respect a user-aborted run: do not restart it without a new request. Record
  results in `docs/integration-testing.md` (status table, history, clusters).
  Preserve historical failures rather than claiming a separate problem was
  fixed merely because one clean run did not reproduce it.
- Report pass/fail/inconclusive plainly. If remote verification fails, trace
  the **single** failing path, revise the fix, and repeat the user-led
  commit/deploy cycle. Move to the next review item only after reporting this
  cycle's outcome and receiving the user's direction.
