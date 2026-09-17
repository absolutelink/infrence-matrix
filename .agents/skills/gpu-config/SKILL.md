---
name: gpu-config
description: Configure and monitor GPU acceleration for llama.cpp inference
---

# GPU Configuration Skill

Use this skill when configuring, monitoring, or troubleshooting GPU acceleration for inference.

## Commands

### Check GPU Availability

```bash
# Detect available GPUs
uv run python -m app.services.gpu detect

# Show GPU info
uv run python -m app.services.gpu info

# Test GPU backend
uv run python -m app.services.gpu test-backend --backend cuda
uv run python -m app.services.gpu test-backend --backend metal
uv run python -m app.services.gpu test-backend --backend vulkan
```

### Configure GPU Layers

```bash
# Set global default
uv run python -m app.services.gpu config --gpu-layers 35

# Set per-model override
uv run python -m app.services.gpu config \
  --model-id <model-uuid> \
  --gpu-layers 50

# Reset to default
uv run python -m app.services.gpu config --model-id <model-uuid> --reset
```

### Monitor GPU Usage

```bash
# Real-time VRAM usage
uv run python -m app.services.gpu monitor --interval 2

# VRAM by model
uv run python -m app.services.gpu usage --by-model

# Historical usage
uv run python -m app.services.gpu history --timeframe 1h
```

### Optimize GPU Settings

```bash
# Auto-optimize for current GPU
uv run python -m app.services.gpu optimize --model-id <model-uuid>

# Test different layer counts
uv run python -m app.services.gpu benchmark \
  --model-id <model-uuid> \
  --layers 20,30,40,50
```

## Python API

```python
from app.services.gpu import GPUManager

gpu_manager = GPUManager()

# Detect GPUs
gpus = await gpu_manager.detect_gpus()
# Returns: [{id, name, vram_total, vram_free, backend}, ...]

# Get GPU info
info = await gpu_manager.get_gpu_info(gpu_id=0)
# Returns: {name, vram, backend, compute_capability, ...}

# Configure layers
await gpu_manager.set_gpu_layers(
    model_id=model_id,
    layers=35
)

# Get recommended layers
recommended = await gpu_manager.recommend_layers(
    model_id=model_id,
    target_vram_usage_percent=80
)

# Monitor VRAM
usage = await gpu_manager.get_vram_usage()
# Returns: {total, used, free, percent}
```

## GPU Backends

### CUDA (NVIDIA)

**Requirements:**
- NVIDIA GPU
- CUDA drivers installed
- CUDA toolkit 11.0+

**Detection:**
```bash
nvidia-smi
```

**Configuration:**
```yaml
gpu:
  backend: cuda
  main_gpu: 0
  tensor_split: null  # For multi-GPU
```

**Optimal Settings:**
- Small GPU (4GB): 20-30 layers
- Medium GPU (8GB): 35-45 layers
- Large GPU (12GB+): 50+ layers

### Metal (Apple Silicon)

**Requirements:**
- Apple Silicon (M1/M2/M3)
- macOS 12.0+

**Detection:**
```bash
system_profiler SPDisplaysDataType
```

**Configuration:**
```yaml
gpu:
  backend: metal
  main_gpu: 0
```

**Optimal Settings:**
- M1/M2 (8GB): 25-35 layers
- M1/M2 (16GB+): 40-50 layers
- M2/M3 Max: 50+ layers

### Vulkan (AMD/Intel)

**Requirements:**
- AMD or Intel GPU
- Vulkan drivers installed

**Detection:**
```bash
vulkaninfo
```

**Configuration:**
```yaml
gpu:
  backend: vulkan
  main_gpu: 0
```

**Optimal Settings:**
- Varies by GPU, test with benchmark command

### CPU-Only

**When to Use:**
- No GPU available
- GPU debugging
- Testing

**Configuration:**
```yaml
gpu:
  backend: cpu
  n_threads: 8  # Number of CPU threads
```

## Multi-GPU Configuration

### Tensor Parallelism

Split model across multiple GPUs:

```yaml
gpu:
  backend: cuda
  main_gpu: 0
  tensor_split: [0.5, 0.5]  # Split evenly across 2 GPUs
```

**Example for 3 GPUs:**
```yaml
tensor_split: [0.33, 0.33, 0.34]
```

### Layer Splitting

Different GPUs handle different layers:

```python
# Manual layer assignment
await gpu_manager.set_tensor_split(
    model_id=model_id,
    split=[0.6, 0.4]  # 60% on GPU 0, 40% on GPU 1
)
```

## VRAM Management

### Calculate VRAM Usage

```python
from app.services.gpu import calculate_vram_usage

vram_needed = calculate_vram_usage(
    model_params=7_000_000_000,  # 7B model
    quantization="Q4_K_M",
    gpu_layers=35,
    context_size=4096,
    batch_size=512
)
# Returns: vram_bytes (e.g., 6_442_450_944 for ~6GB)
```

**Rule of Thumb:**
- Q4_K_M: ~0.7 GB per billion params (full GPU offload)
- Each GPU layer: ~0.1-0.2 GB
- Context: ~1GB per 4K tokens

### VRAM Monitoring

```python
# Real-time monitoring
async for usage in gpu_manager.monitor_vram(interval_seconds=1):
    print(f"VRAM: {usage.percent}% used")
    if usage.percent > 90:
        print("Warning: Low VRAM!")
```

### Auto-Adjust Layers

```python
# Automatically set layers based on available VRAM
layers = await gpu_manager.auto_configure_layers(
    model_id=model_id,
    max_vram_percent=85,
    min_layers=10
)
```

## Benchmarking

### Test Performance

```python
from app.services.gpu import benchmark_model

results = await benchmark_model(
    model_id=model_id,
    gpu_layers_list=[20, 30, 40, 50],
    context_size=4096,
    prompt="Hello, how are you?",
    max_tokens=100
)

# Results include:
# - tokens/second for each layer count
# - VRAM usage for each
# - Optimal recommendation
```

### Compare Backends

```python
# Compare CUDA vs CPU
cuda_result = await benchmark_model(model_id, gpu_layers=35, backend="cuda")
cpu_result = await benchmark_model(model_id, gpu_layers=0, backend="cpu")

speedup = cuda_result.tokens_per_second / cpu_result.tokens_per_second
print(f"CUDA is {speedup:.1f}x faster")
```

## Troubleshooting

### GPU Not Detected

**CUDA:**
```bash
# Check drivers
nvidia-smi

# Check CUDA version
nvcc --version

# Restart Docker with GPU
docker compose restart backend
```

**Metal:**
```bash
# Check Metal support
system_profiler SPDisplaysDataType | grep Metal
```

**Vulkan:**
```bash
# Check Vulkan support
vulkaninfo | grep GPU
```

### Out of Memory

**Solutions:**
1. Reduce GPU layers
2. Reduce context size
3. Use smaller quantization
4. Close other GPU applications
5. Use tensor splitting across multiple GPUs

```bash
# Reduce layers
uv run python -m app.services.gpu config --gpu-layers 25

# Reduce context
uv run python -m app.services.gpu config --context-size 2048
```

### Slow Performance

**Check:**
1. GPU utilization (should be >80%)
2. VRAM usage (shouldn't be swapping)
3. GPU temperature (thermal throttling?)
4. PCIe bandwidth (for discrete GPUs)

```bash
# Monitor in real-time
watch -n 1 nvidia-smi  # NVIDIA
watch -n 1 powermetrics --samplers gpu  # Apple
```

### CUDA Out of Memory

**Error:** `CUDA out of memory. Tried to allocate...`

**Solutions:**
```bash
# Immediate: Reduce batch size
uv run python -m app.services.gpu config --batch-size 256

# Reduce context
uv run python -m app.services.gpu config --context-size 2048

# Reduce GPU layers
uv run python -m app.services.gpu config --gpu-layers 20
```

## Configuration

GPU config in `/etc/inference-matrix/gpu.yaml`:

```yaml
gpu:
  # Backend selection: auto, cuda, metal, vulkan, cpu
  backend: auto
  
  # Primary GPU (for multi-GPU)
  main_gpu: 0
  
  # Multi-GPU splitting
  tensor_split: null  # [0.5, 0.5] for 2 GPUs
  
  # Defaults
  default_gpu_layers: 35
  max_gpu_layers: 100
  
  # VRAM limits
  max_vram_usage_percent: 90
  reserve_vram_gb: 1
  
  # Monitoring
  monitor_interval_seconds: 5
  log_vram_usage: true
  
  # Auto-optimization
  auto_optimize: true
  benchmark_on_start: false
```

## Best Practices

1. **Start conservative**: Begin with 35 layers, increase gradually
2. **Monitor VRAM**: Keep usage under 90% to avoid OOM
3. **Benchmark**: Test different layer counts for your use case
4. **Use appropriate quantization**: Q4_K_M for balance, Q5+ for quality
5. **Reserve VRAM**: Leave 1GB for display/system (if using same GPU)
6. **Multi-GPU**: Use tensor splitting for large models

## Performance Tips

**Maximize tokens/second:**
- Maximize GPU layers (until VRAM limit)
- Use larger batch sizes (if VRAM allows)
- Keep context size reasonable
- Use Q4_K_M or Q5_K_M quantization

**Minimize latency:**
- Pre-load models (don't wait for first request)
- Use prompt caching
- Keep context warm (don't let server shutdown)
- Use smaller models for simple tasks
