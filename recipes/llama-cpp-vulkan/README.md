# Inference Matrix Agent with llama.cpp Vulkan Support

This recipe builds an Agent image with llama.cpp compiled with Vulkan GPU acceleration.

## Features

- Vulkan GPU support for AMD and Intel GPUs
- Also works with NVIDIA GPUs via Vulkan
- Cross-platform compatibility (Linux, Windows, macOS)

## Building

```bash
docker build -t inference-matrix-agent:llama-cpp-vulkan -f recipes/llama-cpp-vulkan/Dockerfile .
```

## Usage

```bash
docker run --gpus all \
  -e AGENT_ID=agent-1 \
  -e FRONTEND_URL=http://host.docker.internal:8000 \
  -v ./models:/models \
  -v ./cache:/cache \
  -p 8080:8080 \
  inference-matrix-agent:llama-cpp-vulkan
```

## GPU Requirements

- **AMD**: RDNA2 or newer (RX 6000 series+)
- **Intel**: Arc or newer
- **NVIDIA**: GTX 1000 series or newer (Vulkan 1.2+)

## Vulkan Installation

### Ubuntu/Debian

```bash
sudo apt-get update
sudo apt-get install -y \
  vulkan-tools \
  libvulkan1 \
  libvulkan-dev \
  mesa-vulkan-drivers
```

### AMD GPU

```bash
# Install AMDGPU-PRO drivers
sudo apt-get install -y amdgpu-dkms rocm-opencl-runtime
```

### Intel GPU

```bash
# Intel Vulkan drivers are typically included
sudo apt-get install -y intel-media-va-driver
```

## Verification

Check Vulkan installation:

```bash
vulkaninfo | head -20
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `AGENT_ID` | Unique agent identifier | *(required)* |
| `AGENT_NAME` | Human-readable name | `inference-agent` |
| `FRONTEND_URL` | Frontend Service URL | *(required)* |
| `DEFAULT_GPU_LAYERS` | Default GPU layers | `35` |
| `GPU_BACKEND` | Set to `vulkan` | `vulkan` |

## Performance Tips

1. **GPU Layers**: Start with 35, adjust based on VRAM
2. **Context Size**: Reduce if out of memory
3. **Batch Size**: 512 is good default

## Troubleshooting

**Vulkan not detected:**
```bash
# Check Vulkan installation
vulkaninfo

# Check GPU visibility
lspci | grep -i vga
```

**Out of memory:**
- Reduce `DEFAULT_GPU_LAYERS`
- Reduce `DEFAULT_CONTEXT_SIZE`
- Use smaller quantization (Q4_K_M instead of Q8_0)
