# Agent Recipes

This folder contains Docker recipes for building Inference Matrix Agent images with different llama.cpp backends.

## Available Recipes

### 1. Base Agent (`agent/Dockerfile`)

Minimal agent image without llama.cpp. Use this as a base for custom builds.

```bash
docker build -t inference-matrix-agent:base -f agent/Dockerfile .
```

**Use case:** When you want to provide your own llama.cpp binary or use system-installed version.

### 2. llama.cpp Vulkan (`recipes/llama-cpp-vulkan/`)

Agent layered on the official llama.cpp Vulkan image (`ghcr.io/ggml-org/llama.cpp:full-vulkan`), with the agent application copied from the built agent image.

```bash
docker build \
  --build-arg AGENT_IMAGE=ghcr.io/absolutelink/agent:main \
  -t inference-matrix-agent:vulkan \
  -f recipes/llama-cpp-vulkan/Dockerfile .
```

**GPU Support:**
- AMD RDNA2+ (RX 6000 series+)
- Intel Arc+
- NVIDIA (via Vulkan)

**Use case:** Cross-platform GPU acceleration, especially for AMD GPUs.

### 3. llama.cpp q38rocm (`recipes/llama-cpp-q38rocm/`)

Agent layered on `ghcr.io/julianmb/q38rocm:latest`, using the ROCmFP4 engine
for AMD Strix Halo systems.

```bash
docker build \
  --build-arg AGENT_IMAGE=ghcr.io/<owner>/agent:main \
  -t inference-matrix-agent:q38rocm \
  -f recipes/llama-cpp-q38rocm/Dockerfile .
```

**Use case:** ROCmFP4 inference on AMD Strix Halo hardware. See the recipe
README for required device mappings and permissions.

## Future Recipes (Planned)

### llama.cpp CUDA (`recipes/llama-cpp-cuda/`)
- NVIDIA GPU support via CUDA
- Best performance for NVIDIA GPUs
- Requires CUDA toolkit

### llama.cpp Metal (`recipes/llama-cpp-metal/`)
- Apple Silicon support
- macOS only
- Best performance for M1/M2/M3 chips

### llama.cpp ROCm (`recipes/llama-cpp-rocm/`)
- AMD GPU support via ROCm
- Linux only
- Alternative to Vulkan for AMD

## Building Recipes

### Build All Images

```bash
# Build base
docker build -t inference-matrix-agent:base -f agent/Dockerfile .

# Build Vulkan (requires the agent image as base)
docker build \
  --build-arg AGENT_IMAGE=inference-matrix-agent:base \
  -t inference-matrix-agent:vulkan \
  -f recipes/llama-cpp-vulkan/Dockerfile .

# Build q38rocm (requires the agent image as base)
docker build \
  --build-arg AGENT_IMAGE=inference-matrix-agent:base \
  -t inference-matrix-agent:q38rocm \
  -f recipes/llama-cpp-q38rocm/Dockerfile .
```

### Using Pre-built Images

Images are automatically built and pushed to GitHub Container Registry:

```bash
# Pull base image
docker pull ghcr.io/your-org/inference-matrix-agent:base

# Pull Vulkan image
docker pull ghcr.io/your-org/inference-matrix-agent:vulkan
```

## Creating Custom Recipes

To create a new recipe:

1. Create folder: `recipes/your-backend/`
2. Add `Dockerfile` starting from an official llama.cpp image (e.g. `ghcr.io/ggml-org/llama.cpp:full-cuda`) and copy the agent app from the `AGENT_IMAGE` build-arg:
   ```dockerfile
   ARG AGENT_IMAGE
   FROM ${AGENT_IMAGE} AS agent
   FROM ghcr.io/ggml-org/llama.cpp:full-cuda
   # Copy agent app and install deps
   ```
3. Add `README.md` with usage instructions
4. Update `.github/workflows/build-and-push.yml` to build your recipe

## Recipe Structure

```
recipes/
├── llama-cpp-vulkan/
│   ├── Dockerfile          # Build instructions
│   └── README.md           # Usage documentation
├── llama-cpp-q38rocm/
│   ├── Dockerfile          # Build instructions
│   └── README.md           # Usage documentation
├── llama-cpp-cuda/         # (future)
│   ├── Dockerfile
│   └── README.md
└── llama-cpp-metal/        # (future)
    ├── Dockerfile
    └── README.md
```

## CI/CD

GitHub Actions automatically builds all recipes on:
- Push to main branch
- New version tags
- Pull requests (no push)

See `.github/workflows/build-and-push.yml` for configuration. The
`build-vulkan-recipe` and `build-q38rocm-recipe` jobs run in parallel after
`build-agent` and pass the freshly built agent image via `AGENT_IMAGE`.

## Choosing a Backend

| Backend | Best For | Platform | Performance |
|---------|----------|----------|-------------|
| Vulkan | AMD/Intel GPUs | Cross-platform | Good |
| CUDA | NVIDIA GPUs | Linux/Windows | Excellent |
| Metal | Apple Silicon | macOS | Excellent |
| ROCm | AMD GPUs | Linux | Very Good |

## Troubleshooting

### Build Fails

- Check Docker is running: `docker info`
- Verify base image exists before building recipes
- Check build logs for missing dependencies

### Runtime Issues

- Ensure GPU drivers are installed on host
- Pass `--gpus all` to docker run
- Check Vulkan/CUDA/Metal installation: `vulkaninfo` / `nvidia-smi`

## Contributing

When adding a new recipe:

1. Test locally first
2. Add comprehensive README
3. Update this index file
4. Ensure CI builds successfully
5. Document GPU requirements
