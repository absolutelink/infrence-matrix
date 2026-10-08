---
name: delegated-slice-delivery
description: Use when delivering a large multi-slice feature or refactor (e.g. an ARCHITECTURE phase) by delegating each slice to a general subagent, having the code-reviewer review it, sending findings back to the subagent to fix, looping until clean, then committing — keeping the todo list and the law docs current throughout.
---

# Delegated slice delivery (`@general → @code-reviewer → fix → repeat → commit`)

Use this skill to drive a big, multi-slice change as an **orchestrator**: you do not
write the feature code yourself. For each slice you delegate implementation to a
**`general`** subagent, get an **independent** green check, run a **`code-reviewer`**
pass, send findings back to `general` to fix, repeat until the reviewer says
**CLEAN**, then commit and move to the next slice.

This is the loop used for Phase 16 (machine-scoped provider agents). It assumes the
repo's `AGENTS.md` workflow rules (law docs first, generated clients, no unrequested
commits/deploys/production writes).

## 0. Before the first slice

- Read `ARCHITECTURE.md` and `IMPLEMENTATION_STATUS.md`, plus the phase's linked docs
  (`docs/ws-protocol.md`, `provider/README.md`, `admin/backend/docs/redis-keys.md`,
  etc.). The architecture is canonical; if code and docs disagree, decide which is
  wrong and record the change in `IMPLEMENTATION_STATUS.md`.
- Break the work into **ordered slices** that each leave the tree green. Prefer
  additive slices first, then the atomic breaking cutover, then polish, then docs.
- Create a **todo list** with one item per slice (status `pending`), plus the commit
  hash once each is committed. Keep exactly one slice `in_progress`. Update status in
  real time; **clear the list when all slices are done.**
- Confirm with the user (questions tool) whether to commit automatically at each
  green+CLEAN boundary or wait for go-ahead. Default per `AGENTS.md`: do not commit
  unless the user has asked for it.

## 1. Delegate the implementation to `@general`

Launch a `general` subagent (task tool) with a **self-contained, context-rich** prompt.
Always instruct it to **read the law docs first** (`ARCHITECTURE.md`,
`IMPLEMENTATION_STATUS.md`, the phase doc). The prompt must contain:

- The exact slice scope: what to change, file-by-file, and what is explicitly **out of
  scope** (deferred to later slices) so it does not over-reach.
- The locked design decisions for this slice (restate them; do not assume memory).
- The **verification commands** and the **definition of done** (all suites green, ruff
  check+format, alembic upgrade→downgrade→upgrade→`check` if models change,
  `bash scripts/generate-client.sh` + frontend typecheck if routes change).
- A hard instruction: **"Do NOT claim green unless you paste the actual command
  output."** Subagents have previously reported success while suites were red.
- A request to report every file changed, deviations from `ARCHITECTURE.md`, and
  anything unfinished.

Reuse the same subagent session (`task_id`) for the fix rounds so it keeps context.

## 2. Independently verify — never trust the agent's green claim

Re-run everything yourself in a **clean env** (strip provider env so tests that rely on
exported vars fail loudly if the code requires them):

```bash
env -u MACHINE_UID -u ADMIN_BASE_URL -u MACHINE_SECRET -u AGENT_ID -u PROVIDER_REGISTRATION_TOKEN bash -c '
  (cd admin/backend && uv run pytest tests/ -q)
  uv run --project provider/lib        pytest provider/lib/tests        -q
  uv run --project provider/mock       pytest provider/mock/tests       -q
  uv run --project provider/llama-cpp  pytest provider/llama-cpp/tests  -q
  uv run --project provider/gufo       pytest provider/gufo/tests       -q
  uv run --project provider/halogen    pytest provider/halogen/tests    -q
  uv run --project provider/halogen-flash pytest provider/halogen-flash/tests -q'
```

Also check `git status`/`git diff --stat` (did it touch what it claimed? did it delete
tests to go green?), ruff, and the alembic chain on a throwaway UTF8 DB when models
changed. Recreate the admin test DB if the schema changed. If the agent's counts do
not reproduce, send it back before reviewing.

## 3. Review with `@code-reviewer`

Send the reviewer a prompt with: the slice scope, what to read first, the diff/new
files, your independent verification results, and **focus areas** (correctness,
concurrency, security, migrations, test quality — "were tests weakened?"). Tell it not
to modify files and to return findings by **Blocker/High/Medium/Low/Nit** with
`file:line` + a concrete fix, and a final **CLEAN** verdict or the list.

**Challenge the reviewer too.** Reviewers have raised false "Blockers" — e.g. flagging
bare `except A, B:` as a Python 2 syntax error when the project targets Python 3.14
(PEP 758) and `ruff format` enforces the unparenthesized form. Verify a claimed
blocker against the real interpreter/config (`py_compile`, `import`, the CI
`python-version`, `requires-python`) before sending it downstream.

## 4. Send findings back to `@general` to fix

Resume the same `general` session with the findings. Require it to fix all
Blockers/Highs (and the cheap Mediums/Lows), add a **discriminating** test for each
real fix (revert-the-fix → test fails), keep every suite green, and update
`IMPLEMENTATION_STATUS.md`. Explicitly list what is **legitimately deferred** to later
slices so it does not gold-plate. Re-run step 2 yourself.

## 5. Loop until CLEAN

Repeat steps 3–4 (reviewer → general fixes → independent verify) until the reviewer
returns **CLEAN** with only items that are genuinely deferred. Track each round; do not
stop on the first "looks good."

## 6. Commit the slice, then the next

On CLEAN + green, stage exactly the intended files (verify no secrets, no generated-file
hand-edits) and commit with a concise message that matches repo style. Record the short
hash in the todo item. Mark the slice `completed`, set the next slice `in_progress`, and
go to step 1.

## 7. Keep the law docs current

After each slice, update `IMPLEMENTATION_STATUS.md` (what landed, residual seams, test
counts). Update `ARCHITECTURE.md` only for a factual mismatch with what was built (it is
canonical). Reserve a final docs slice to rewrite the remaining law docs
(`docs/ws-protocol.md`, `provider/README.md`, `redis-keys.md`, `AGENTS.md`) to match the
shipped code.

## Guardrails

- One slice at a time; one `in_progress` todo; clear the list at the end.
- Never commit/push/deploy/SSH-to-prod unless the user asked. Provide the commit message
  in a fenced block when the user commits themselves.
- Never hand-edit `admin/frontend/src/client/` or `routeTree.gen.ts`; regenerate.
- Keep both `wire.py` mirrors in sync (admin + provider_lib); the drift-guard test is
  the check.
- Prefer resuming the same subagent session for fix rounds over fresh ones.
