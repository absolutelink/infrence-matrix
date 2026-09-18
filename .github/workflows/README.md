# GitHub Actions Workflows

## Workflows

### 1. Build and Push (`build-and-push.yml`)

**Triggers:**
- Push to `main` or `develop` branches
- Tag pushes (semver)
- Pull requests (build only, no push)

**Jobs:**
1. **build-frontend** - Builds and pushes Frontend Service Docker image
2. **build-agent** - Builds and pushes Agent Service Docker image  
3. **test-deployment** - Deploys both services and verifies health

**Output:**
- Images pushed to `ghcr.io/inference-matrix/frontend`
- Images pushed to `ghcr.io/inference-matrix/agent`
- Tags: branch name, PR number, semver (for tags), SHA

**Example tags:**
- `main` - Latest from main branch
- `develop` - Latest from develop branch
- `v1.2.3` - Release version
- `v1.2` - Major.minor release
- `abc123def` - Commit SHA

### 2. Test and Lint (`test.yml`)

**Triggers:**
- Push to `main` or `develop`
- All pull requests

**Jobs:**
1. **test-backend** - Backend tests with PostgreSQL
   - Runs linter (ruff)
   - Runs type checker (mypy)
   - Runs pytest tests
   
2. **test-agent** - Agent service tests
   - Runs linter (ruff)
   - Runs type checker (mypy)
   - Runs pytest tests
   
3. **check-docker** - Docker build validation
   - Validates Frontend Dockerfile builds
   - Validates Agent Dockerfile builds

## Required Secrets

No secrets required for PR builds.

For production builds (main/develop/tags):
- `GITHUB_TOKEN` - Automatically provided by GitHub Actions

## Container Registry

Images are published to GitHub Container Registry:
- Frontend: `ghcr.io/inference-matrix/frontend`
- Agent: `ghcr.io/inference-matrix/agent`

## Usage in compose.yml

```yaml
services:
  frontend:
    image: ghcr.io/inference-matrix/frontend:main
    # ... rest of config
  
  agent:
    image: ghcr.io/inference-matrix/agent:main
    # ... rest of config
```

## Manual Trigger

To manually trigger a build:

```bash
# Push to main
git push origin main

# Create a tag
git tag v1.0.0
git push origin v1.0.0
```

## Debugging Failed Builds

1. Check GitHub Actions logs
2. Download failed build artifacts
3. Reproduce locally:
   ```bash
   docker build -t test-image ./backend
   docker run --rm test-image
   ```

## Cache Strategy

- Uses GitHub Actions cache for faster builds
- Docker layer caching via BuildKit
- Python package caching via uv
