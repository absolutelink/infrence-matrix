---
name: model-management
description: Manage GGUF model artifacts in Inference Matrix (backend_config descriptors, HuggingFace downloads, storage layout)
---

# Model Management — Inference Matrix

Use this skill when managing model files (GGUF artifacts) that provider
instances download and load.

**There is no `app.services.models` CLI and no `/etc/inference-matrix/models.yaml`.**
In the overhauled architecture the admin has **no `Model` table**. Models
are described inside `ProviderDefinition.backend_config` as JSON artifact
descriptors, and the **provider instance** downloads them at init/boot via
`provider_lib.downloader`.

## Where model info lives

- **`ProviderDefinition.backend_config`** (Postgres) — declares the
  artifacts the backend needs: main model, optional mmproj (vision),
  optional draft (speculative). Schema in `provider/README.md`.
- **Provider `MODELS_DIR`** (env) — the actual GGUF files on disk.
- **`ProviderDefinition.model_metadata`** — OpenAI-style metadata scraped
  from the running backend during initialization.

```json
{
  "model":  {"source": "hf", "repo": "TheBloke/Llama-2-7B-Chat-GGUF", "file": "llama-2-7b-chat.Q4_K_M.gguf"},
  "mmproj": {"source": "hf", "repo": "ggml-org/models", "file": "gemma/mmproj-model-f16.gguf"},
  "draft":  {"source": "hf", "repo": "...", "file": "draft.gguf"},
  "args":   {"ctx": 8192, "gpu_layers": 35, "flash_attn": "on", "parallel": 1}
}
```

Artifacts also accept a plain local path (no download):
`{"path": "/models/foo.gguf"}` — the file must already exist.

## Supported source

**HuggingFace only** (`source: "hf"` / `"huggingface"`). The downloader
uses `huggingface_hub.hf_hub_download` (imported lazily). `modelscope` is
**not** supported in the new downloader — if you need it, that's a new
feature to add to `provider_lib/downloader.py`, not a config toggle.

Private HF repos: supply the token via the provider container's
`HF_TOKEN` / `HUGGINGFACE_TOKEN` env (read by `huggingface_hub`), not in
`backend_config`.

## Storage layout + resolution

`MODELS_DIR` mirrors the repo:

```
MODELS_DIR/
  TheBloke/Llama-2-7B-Chat-GGUF/llama-2-7b-chat.Q4_K_M.gguf
  ggml-org/models/gemma/ggml-model.gguf
  mistral-7b.Q5_K_M.gguf          # FLAT: operator-dropped files
```

Resolution order in `ensure_artifact()`:
1. `MODELS_DIR/<repo>/<file>` (repo-mirrored)
2. `MODELS_DIR/<file>` (FLAT — operators drop GGUFs directly here)
3. HuggingFace download into `MODELS_DIR/<repo>`

**The FLAT hit short-circuits the download** — if a file with that name
already exists at the top of `MODELS_DIR`, it's used as-is.

## Downloads + progress events

`ensure_artifact(descriptor, models_dir, progress_cb)` (async):
- Returns the resolved local path.
- Publishes throttled `download.progress` events (default min interval
  1s, `DOWNLOAD_PROGRESS_INTERVAL`) through the provider's event bus →
  admin WS. Payload: `{filename, progress_percent, bytes_downloaded,
  total_bytes, speed_mbps, phase}` (phase: downloading/completed/failed).
- **In-flight dedup** by `repo/file` (module-level registry) — concurrent
  requests for the same artifact share one download.
- **Path-traversal guards** on repo/filename.
- `huggingface_hub` is imported lazily so packages that never download stay
  importable without it.

There is **no pause/resume token** and **no `DownloadJob` table** anymore —
progress is event-only, streamed live to the admin UI. Resume across
process restarts is handled by `hf_hub_download`'s own cache semantics.

## When downloads happen

- On **provider initialization** (Phase 9 `provider.config.update` /
  first boot): the provider downloads/updates the artifacts named in
  `backend_config` before starting the backend.
- On `backend.start`, if an artifact is missing locally, the driver calls
  `ensure_artifact` to fetch it before spawning the process.
- The admin observes progress via `download.progress` WS events (persisted
  to the instance's `download_progress`-style UI state in the admin, not a
  dedicated table).

## Quantization formats (reference)

GGUF quants supported by the backends (this is about the file, not the
platform):
- `Q2_K` … `Q8_0`, `F16`/`BF16`/`F32`.
- **Recommended:** `Q4_K_M` for speed/quality balance.
- Pick the quant that fits your VRAM budget (see `gpu-config` skill);
  the file must match what `Machine.total_vram_bytes` /
  `vram_required_bytes` accounting allows.

## Managing models via the admin

1. Create/edit a **ProviderDefinition** and set `backend_config` with the
   artifact descriptors (validated against the documented JSON schema —
   don't invent ad-hoc keys).
2. The definition's `registration_token` binds it to a provider instance
   of the matching `provider_type`.
3. Trigger init (Phase 9) or a boot; the provider downloads artifacts and
   the admin shows `download.progress`.
4. Removing a model = disabling/deleting the ProviderDefinition. Orphaned
   files are reclaimed by `storage.prune_unused` (Phase 9) — deletes files
   in MODELS_DIR/CACHE_DIR not referenced by any active config fingerprint.

## Validation

- The admin validates `backend_config` shape with Pydantic on write.
- GGUF integrity is the backend's problem at load time; a corrupt file
  surfaces as a boot failure + `backend.logs`. Re-download by removing the
  local file and re-triggering init.

## Troubleshooting

| Symptom | Cause / fix |
| --- | --- |
| "unsupported artifact source" | Only `hf`/`huggingface`/`path` are valid. Fix `backend_config`. |
| Download 401/403 | Private repo — set `HF_TOKEN` in the provider container env. |
| Wrong file used | FLAT `MODELS_DIR/<file>` shadow — remove the stray top-level file. |
| Disk full | Free space or use a smaller quant; downloads go under `MODELS_DIR`. |
| Stale model after config change | Fingerprint should have changed → auto cache-clear + re-download (Phase 9). |

## Anti-patterns

- ❌ Referencing the old `Model`/`DownloadJob` tables — they don't exist.
- ❌ Putting model paths in provider env instead of `backend_config`
  (env is for `MODELS_DIR` root + binary paths, not per-model files).
- ❌ Assuming ModelScope works — it's HuggingFace-only right now.
- ❌ Hand-placing files in `MODELS_DIR` without a matching descriptor and
  expecting the platform to know about them (except the FLAT resolution
  convenience for declared filenames).
