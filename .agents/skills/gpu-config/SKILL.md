---
name: gpu-config
description: Configure and monitor GPU acceleration for Inference Matrix providers (VRAM admission, gpu_layers, backend metrics)
---

# GPU Configuration — Inference Matrix

Use this skill when configuring GPU acceleration, VRAM budgeting, or
monitoring GPU usage in Inference Matrix.

**There is no `app.services.gpu` CLI and no `/etc/inference-matrix/gpu.yaml`.**
GPU configuration is split across three real places:

1. **`ProviderDefinition.backend_config.args`** — per-model llama.cpp GPU
   flags (`gpu_layers`, `flash_attn`, `parallel`, etc.). Set via the admin.
2. **`Machine.total_vram_bytes` + `ProviderDefinition.vram_required_bytes`**
   — the scheduler's VRAM admission budget.
3. **`provider_lib/metrics.py`** — the GPU sampler (nvidia-smi / AMD sysfs)
   that emits `metrics.machine` events.

The backend binary path (`LLAMA_SERVER_PATH`) is a provider-container env
var, **never** in `backend_config`.

## GPU layers (llama-cpp)

Set in `backend_config.args` on the ProviderDefinition. The llama-cpp
driver maps them to CLI flags (see `provider/README.md` command mapping):

```json
{
  "args": {
    "gpu_layers": 35,
    "ctx": 8192,
    "flash_attn": "on",
    "parallel": 1,
    "tensor_split": null,
    "main_gpu": 0,
    "split_mode": "layer"
  }
}
```

| Field | CLI | Notes |
| --- | --- | --- |
| `gpu_layers` | `--n-gpu-layers` | Default 35. Higher = more offload, more VRAM. |
| `ctx` / `context_size` | `--ctx-size` | VRAM grows with context. |
| `flash_attn` | `--flash-attn on\|off` | bool or string. |
| `parallel` | `--parallel` | Concurrency slots. `strict_mtp_qwen:true` forces 1. |
| `tensor_split` | `--tensor-split` | e.g. `[0.5,0.5]` for 2 GPUs. |
| `main_gpu` | `--main-gpu` | Primary GPU index. |
| `split_mode` | `--split-mode` | `layer` / `row` / `none`. |

**Rule of thumb for `gpu_layers`:** 4GB GPU → 20–30; 8GB → 35–45;
12GB+ → 50+ or full offload. Increase gradually and watch VRAM.

> ⚠️ `--prompt-cache` is OBSOLETE — never emit it. `--cache-prompt` is the
> valid modern flag; the driver already handles this distinction.

## VRAM admission (the scheduler)

The scheduler decides whether a backend can boot on a machine based on
VRAM, not layers. See `ARCHITECTURE.md` §6.

- **`Machine.total_vram_bytes`** — the machine's VRAM budget. Set when you
  create the Machine in the admin; refined from provider-reported
  hardware at registration. Size it honestly (leave headroom for OS/display).
- **`ProviderDefinition.vram_required_bytes`** — what this model needs to
  load. Estimate from the GGUF size + context + KV cache.
- On `acquire`, free VRAM = `total_vram_bytes` − VRAM held by other
  instances on the machine. If `vram_required_bytes` > free, the request
  does **not** boot and instead waits (or 504s on timeout).

**Estimating `vram_required_bytes`:**
- Q4_K_M ≈ 0.7 GB per billion params (full offload) + ~1 GB per 4K ctx.
- Add a margin; over-reporting is safer than OOM.

> Idle-backend **eviction** to free VRAM is implemented (Phase 6): when a
> needed boot doesn't fit, the scheduler stops idle **different-alias**
> backends on the same machine LRU-first (`backend.stop`). Requests still
> wait if no evictable victim frees enough. Packing models near 100% of a
> GPU is now possible but risks eviction churn — leave headroom where a
> backend must stay resident.

## Monitoring GPU / VRAM

Machine-level metrics are collected by the provider and emitted over the WS
only when the admin assigns ownership (`im:metrics:owner:{machine_uid}`,
TTL 30s). Categories: `gpu_usage`, `vram`, `os_ram`, `cpu`, `storage`.
Inference metrics (`metrics.inference`) are always emitted per-instance.

**Sampler** (`provider_lib/metrics.py`):
- NVIDIA: `nvidia-smi --query-gpu=...` (CSV).
- AMD: sysfs (`/sys/class/drm/card*/device/mem_info_vram_used`, `busy_percent`).
- Failed collectors omit their key (never raise).

**Ad-hoc host checks:**
```bash
nvidia-smi                          # NVIDIA
watch -n1 nvidia-smi
cat /sys/class/drm/card0/device/mem_info_vram_used   # AMD (bytes)
vulkaninfo | grep GPU               # Vulkan enumeration
```

**Via the platform:** the admin WebUI (Phase 10) shows per-machine VRAM
used/free and per-instance `metrics.inference` (token speed, slots). Until
then, inspect Redis:
```bash
docker compose exec redis redis-cli GET im:metrics:machine:<machine_uid>
docker compose exec redis redis-cli HGETALL im:vram:used:<machine_uid>
```

## Container GPU access

Provider containers need the device nodes exposed:

```bash
# NVIDIA (container toolkit present)
docker run --gpus all ...

# NVIDIA (no container toolkit — the llama-cpp CUDA12 image bakes the
# userspace driver, so just pass the device nodes; the baked driver version
# must match the host kernel driver):
podman run --device /dev/nvidia0 --device /dev/nvidiactl \
  --device /dev/nvidia-uvm --device /dev/nvidia-uvm-tools ...

# AMD (ROCm / halogen)
podman run --device /dev/kfd --device /dev/dri ...

# Vulkan
podman run --device /dev/dri ...
```

`LD_LIBRARY_PATH` is defaulted to the `llama-server` binary's directory by
the driver (`env.setdefault`) — Vulkan/ROCm containers may need it set
explicitly for driver libs. The CUDA12 provider image sets
`LD_LIBRARY_PATH=/usr/local/nvidia/lib64` (baked driver libs) in the image.

## Multi-GPU

- **Layer/tensor split within one backend:** use `tensor_split` +
  `split_mode` in `backend_config.args`.
- **One backend per GPU:** create separate ProviderDefinitions/instances,
  each with `assigned_gpus` bound to a single GPU UUID; the scheduler treats
  each instance's VRAM need against the machine budget.
- A provider instance owns exactly **one** backend, so multi-GPU =
  multiple instances on the same Machine.

## Troubleshooting

### GPU not detected
```bash
nvidia-smi                 # driver present?
ls /dev/kfd /dev/dri       # AMD nodes present + passed to container?
```
Recreate the provider container with the right `--device`/`--gpus` flags.

### Out of memory / boot fails
1. Lower `gpu_layers` in `backend_config.args`.
2. Lower `ctx`.
3. Use a smaller quantization (larger Q# = smaller).
4. Reduce `parallel` (each slot uses KV cache).
5. Increase `Machine.total_vram_bytes` headroom or reduce
   `vram_required_bytes` over-packing.
Check the failed boot's `backend.logs` in the admin / provider container
logs.

### Slow performance
- Confirm `gpu_layers` is high enough that the model actually fits on GPU.
- Ensure `flash_attn: "on"`.
- Check VRAM isn't thrashing (monitor `im:vram:used`).
- Thermal throttling: check temps with `nvidia-smi` / `radeontop`.

### Two backends over-provisioned the same GPU
- Verify `Machine.total_vram_bytes` reflects the real GPU, and each
  `ProviderDefinition.vram_required_bytes` sums correctly.
- Metrics dedup: only one instance per machine emits `metrics.machine` at
  a time (ownership lease); inference metrics are per-instance and never
  deduped.

## Best practices

1. Start `gpu_layers` at 35 and raise until VRAM ~85–90%.
2. Report `vram_required_bytes` honestly; leave the machine under-committed.
3. Don't rely on eviction — it isn't implemented; over-provisioning just
   makes requests wait.
4. Reserve ~1 GB on a GPU shared with display/OS.
5. Use `metrics.machine` VRAM readings to validate your budget assumptions.
