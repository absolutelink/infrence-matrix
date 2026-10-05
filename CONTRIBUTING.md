# Contributing

Contributions to Inference Matrix are welcome. Please read
[ARCHITECTURE.md](ARCHITECTURE.md) and [development.md](development.md)
before starting work — the architecture doc is canonical, and
[IMPLEMENTATION_STATUS.md](IMPLEMENTATION_STATUS.md) tracks what each
phase of the litellm overhaul covers.

## Discussions First

For **big changes** (new features, architectural changes, significant
refactoring), open a GitHub Discussion first so the approach can be
reviewed before implementation time is invested. Small fixes (typos,
reproducible bugs, lint issues) can go straight to a PR.

Note that PRs from non-team members are not allowed to modify
`pyproject.toml` or `uv.lock`, to prevent supply chain risk. Propose new
dependencies via a Discussion.

## Developing

```bash
./scripts/dev.sh        # full local stack (docker) or local procs + seeded mock
```

See [development.md](development.md) for the backend/frontend/provider
dev loops, test prerequisites (UTF8 Postgres test DB, Redis DB 15), and
client generation.

Before submitting:

- Admin: `uv run ruff check app && uv run ruff format --check app && uv run pytest tests/ -q` (from `admin/backend`)
- Providers: `uv run --project provider/<pkg> pytest provider/<pkg>/tests -q` (from the repo root, with provider env vars set — see development.md)
- Frontend: `bun run build && bun run lint` (from the repo root)
- Any admin route/schema change: `bash scripts/generate-client.sh` (never hand-edit `admin/frontend/src/client/`)

## Architecture rules that matter for PRs

- The admin knows nothing provider-specific; provider quirks live in
  `provider/<type>/` behind the `provider_lib` contracts.
- Do not import provider packages from the admin or vice versa (the
  admin image builds with `--no-install-workspace --package matrix-admin`).
- `admin/backend/app/services/wire.py` deliberately mirrors
  `provider/lib/provider_lib/wire.py` — keep both in sync with
  `docs/ws-protocol.md` (canonical).
- The pre-overhaul code was removed in the final cleanup and lives only
  in git history. Accepted regressions (embeddings, files, batches,
  benchmarks, users/API keys, etc.) must be re-implemented against the
  new provider/scheduler model, never restored from history. See
  IMPLEMENTATION_STATUS.md for the full list.

## Pull Requests

1. Make sure all tests pass before submitting.
2. Keep PRs focused on a single change.
3. Update tests when changing functionality.
4. Reference related issues in the PR description.
5. Update `IMPLEMENTATION_STATUS.md` / `ARCHITECTURE.md` when behavior
   or boundaries change.

## Automated Code and AI

You are encouraged to use tools — including AI — to work efficiently,
but contributions need meaningful human judgement. If the human effort
to produce a PR is less than the effort to review it, don't submit it.
Spammy automated PRs get accounts blocked.

## Questions?

Open a GitHub Discussion.
