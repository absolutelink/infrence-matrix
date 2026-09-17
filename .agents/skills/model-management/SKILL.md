---
name: model-management
description: Download, manage, and organize GGUF models from HuggingFace and ModelScope
---

# Model Management Skill

Use this skill when managing GGUF models - downloading from sources, organizing files, and maintaining model metadata.

## Commands

### Download a Model

```bash
# From HuggingFace
uv run python -m app.services.models download \
  --source huggingface \
  --repo TheBloke/Llama-2-7B-Chat-GGUF \
  --file llama-2-7b-chat.Q4_K_M.gguf

# From ModelScope
uv run python -m app.services.models download \
  --source modelscope \
  --repo modelscope/Llama-3-8B \
  --file llama-3-8b.Q4_K_M.gguf

# With custom destination
uv run python -m app.services.models download \
  --source huggingface \
  --repo TheBloke/Mistral-7B-v0.1-GGUF \
  --file mistral-7b.Q5_K_M.gguf \
  --destination /custom/path/models
```

**Parameters:**
- `--source`: `huggingface` or `modelscope`
- `--repo`: Repository ID (owner/name)
- `--file`: Specific GGUF file to download
- `--destination`: Custom destination path (optional)
- `--resume`: Resume interrupted download (boolean flag)

### Pause/Resume Download

```bash
# Pause active download
uv run python -m app.services.models pause --job-id <job-uuid>

# Resume paused download
uv run python -m app.services.models resume --job-id <job-uuid>
```

### List Downloads

```bash
# All downloads
uv run python -m app.services.models downloads list

# Active downloads only
uv run python -m app.services.models downloads list --status active

# Failed downloads
uv run python -m app.services.models downloads list --status failed
```

### Delete a Model

```bash
# Delete model file and metadata
uv run python -m app.services.models delete --model-id <model-uuid>

# Delete file only (keep metadata)
uv run python -m app.services.models delete --model-id <model-uuid> --keep-metadata

# Force delete (even if loaded)
uv run python -m app.services.models delete --model-id <model-uuid> --force
```

### Validate Model

```bash
# Check file integrity
uv run python -m app.services.models validate --model-id <model-uuid>

# Full validation with metadata extraction
uv run python -m app.services.models validate --model-id <model-uuid> --extract-metadata
```

### Get Model Info

```bash
# Basic info
uv run python -m app.services.models info --model-id <model-uuid>

# Full metadata
uv run python -m app.services.models info --model-id <model-uuid> --full

# List all models
uv run python -m app.services.models list
```

### Refresh Model Metadata

```bash
# Re-extract metadata from file
uv run python -m app.services.models refresh-metadata --model-id <model-uuid>

# Refresh all models
uv run python -m app.services.models refresh-metadata --all
```

## Python API

Use the `ModelManager` class in Python code:

```python
from app.services.models import ModelManager

manager = ModelManager()

# Download with progress callback
async def on_progress(bytes_downloaded, total_bytes, speed):
    print(f"Progress: {bytes_downloaded/total_bytes*100:.1f}% at {speed/1e6:.1f} MB/s")

job = await manager.download_model(
    source="huggingface",
    repo_id="TheBloke/Llama-2-7B-Chat-GGUF",
    filename="llama-2-7b-chat.Q4_K_M.gguf",
    on_progress=on_progress
)

# Pause/Resume
await manager.pause_download(job.id)
await manager.resume_download(job.id)

# Get model info
model = await manager.get_model(model_id)
models = await manager.list_models()

# Delete model
await manager.delete_model(model_id)

# Validate
is_valid = await manager.validate_model(model_id)
metadata = await manager.extract_metadata(model_id)
```

## Supported Sources

### HuggingFace Hub

**Format:** `owner/repo`

**Examples:**
- `TheBloke/Llama-2-7B-Chat-GGUF`
- `TheBloke/Mistral-7B-Instruct-v0.2-GGUF`
- `bartowski/Llama-3-8B-Instruct-GGUF`

**Features:**
- Public and private repos (with token)
- Progress tracking
- Resume support
- File validation

### ModelScope

**Format:** `owner/repo`

**Examples:**
- `modelscope/Llama-3-8B`
- `qwen/Qwen-7B-Chat-GGUF`

**Features:**
- China-friendly mirror
- Progress tracking
- Resume support

## Model Organization

Models are stored in:
```
/models/
├── llama-2-7b-chat.Q4_K_M.gguf
├── mistral-7b-instruct.Q5_K_M.gguf
└── qwen-7b-chat.Q4_K_M.gguf
```

Metadata stored in PostgreSQL `models` table.

## Quantization Formats

Supported GGUF quantizations:
- `Q2_K` - Smallest, lowest quality
- `Q3_K_S`, `Q3_K_M`, `Q3_K_L` - Small, decent quality
- `Q4_0`, `Q4_1` - Original quants
- `Q4_K_S`, `Q4_K_M` - Recommended balance
- `Q5_0`, `Q5_1`, `Q5_K_S`, `Q5_K_M` - Better quality
- `Q6_K`, `Q8_0` - Near lossless
- `BF16`, `F16`, `F32` - Unquantized

**Recommended:** `Q4_K_M` for best speed/quality balance

## Download Features

### Progress Tracking
- Real-time progress percentage
- Download speed (MB/s)
- ETA calculation
- Bytes downloaded/total

### Pause/Resume
- HTTP range requests for resuming
- Pause token storage
- Automatic retry on network failures

### Error Recovery
- Auto-retry (max 3 attempts)
- Exponential backoff
- Checksum validation after download
- Quarantine failed downloads

### Validation
- GGUF magic number check
- Header parsing
- Metadata extraction
- Optional SHA256 verification

## Metadata Extraction

Automatically extracts:
- Architecture (llama, mistral, qwen, etc.)
- Parameter count
- Context length
- Quantization type
- File size
- Tensor dimensions

## Best Practices

1. **Always validate** after download completes
2. **Use resume** for large models (>5GB)
3. **Check disk space** before downloading
4. **Verify quantization** matches use case
5. **Keep metadata updated** after file operations
6. **Clean up failed downloads** periodically

## Error Handling

Common errors:

**"File not found"**: Check repo ID and filename
**"Out of disk space"**: Free up space or use smaller quantization
**"Invalid GGUF file"**: Redownload, file may be corrupted
**"Repo not found"**: Verify repo is public or provide auth token
**"Download failed"**: Check network, retry with resume

## Configuration

Download config in `/etc/inference-matrix/models.yaml`:

```yaml
downloads:
  max_concurrent: 3
  chunk_size_mb: 64
  max_retries: 3
  timeout_seconds: 3600
  resume_on_failure: true

storage:
  base_path: /models
  organize_by_source: false
  auto_validate: true

sources:
  huggingface:
    use_token: false
    token_env: HUGGINGFACE_TOKEN
  
  modelscope:
    use_mirror: true
```
