# GitHub Workflows

GitHub Actions for the Inference Matrix monorepo (litellm-based
architecture).

## Active

- **`build-and-push.yml`** — the main pipeline:
  1. `test-admin` — starts **postgres 16 + redis 7** service containers,
     `uv sync`s `admin/backend`, runs Ruff (`ruff check` +
     `format --check`), applies migrations (`scripts/prestart.sh`), then
     `pytest`.
  2. `test-provider` — `uv sync`s the provider packages and runs Ruff +
     tests for `provider/lib` and `provider/mock`.
  3. `build-matrix-app` — needs both test jobs; builds and pushes the
     admin image (`ghcr.io/<owner>/matrix-app`) from the root `Dockerfile`
     (multi-stage: bun builds the frontend → python 3.14 + uv installs
     `matrix-admin` only).

  Runs on push to `main`/`develop` and `v*` tags, and on pull requests to
  `main`.

## Per-provider images

Provider-type images (`llama-cpp`, `gufo`, `halogen`, `halogen-flash`)
build from `provider/<type>/Dockerfile`. The mock provider builds from
`provider/mock/Dockerfile` for local/compose use. Wiring per-type image
builds into this workflow is a **Phase 8** task (see the NOTE in
`build-and-push.yml` and `IMPLEMENTATION_STATUS.md`).

> The old "base agent image + recipes" build model (Vulkan/ROCm recipes
> layering the agent) is **gone** — replaced by one Dockerfile per provider
> type under `provider/`.

## Disabled workflows

The `*.disabled` files (release notes, issue manager, pre-commit, zizmor,
etc.) are inherited template automation, currently switched off. Re-enable
individually if needed.

## Deploy note

Because provider registration **hard-fails on version mismatch** (409),
the admin image must be deployed before provider images — see
`deployment.md`.
