# Inference Matrix

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Docker Compose](https://github.com/your-org/inference-matrix/actions/workflows/docker-compose.yml/badge.svg)](https://github.com/your-org/inference-matrix/actions)

**OpenAI API-compatible inference server for local GGUF models with full WebUI management**

Inference Matrix provides a complete OpenAI API-compatible interface that routes inference requests to llama.cpp servers. Manage local models, download from HuggingFace/ModelScope, monitor GPU usage, and maintain conversation history - all through a modern WebUI.

![Dashboard](img/dashboard.png)

## Key Features

### 🚀 OpenAI API Compatible

Drop-in replacement for OpenAI API with full compatibility:
- `/v1/models` - List available models
- `/v1/chat/completions` - Chat completions with SSE streaming
- `/v1/completions` - Legacy completions
- `/v1/embeddings` - Text embeddings
- `/v1/responses` - Advanced Responses API with tree conversations
- `/v1/files` - File management
- `/v1/batches` - Batch processing
- `/v1/audio/*` - Transcription, translation, and speech

### 🤖 Local Model Management

- **Download models** from HuggingFace and ModelScope
- **GGUF format support** (optimized for llama.cpp)
- **Progress tracking** with pause/resume
- **Auto-validation** after download
- **Full metadata** extraction (architecture, capabilities, benchmarks)

### ⚡ High-Performance Inference

- **llama.cpp backend** with llama-server subprocess management
- **GPU acceleration** (CUDA, Metal, Vulkan) with auto-detection
- **Auto start/stop** servers based on demand
- **Prompt caching** with hybrid tracking
- **Multiple models** running simultaneously

### 🎯 WebUI Dashboard

- **Model management** - Download, load, unload, delete
- **Real-time monitoring** - VRAM usage, tokens/sec, queue depth
- **GPU configuration** - Per-model GPU layer settings
- **Conversation history** - Tree-structured for /v1/responses
- **API key management** - Create/revoke keys
- **Backup/restore** - Database and configurations

### 🔒 Optional Authentication

- **API keys** (OpenAI-style) for API endpoints
- **JWT authentication** for admin WebUI
- **Configurable enforcement** per endpoint
- **Rate limiting** support

### 💾 Data Persistence

- **PostgreSQL** for metadata storage
- **Tree-structured conversations** for Responses API
- **Prompt cache tracking** with TTL management
- **Full backup/restore** capabilities

## Quick Start

### 1. Clone and Configure

```bash
git clone https://github.com/your-org/inference-matrix.git
cd inference-matrix
cp .env.example .env
```

Edit `.env` with your settings:
```bash
POSTGRES_PASSWORD=your_secure_password
DEFAULT_GPU_LAYERS=35
MODELS_PATH=/models
```

### 2. Start services

```bash
docker compose up -d
```

### 3. Access the application

- **WebUI**: http://localhost:3000
- **API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs

### 4. Download your first model

1. Open WebUI at http://localhost:3000
2. Navigate to **Models**
3. Click **Download Model**
4. Enter HuggingFace repo: `TheBloke/Llama-2-7B-Chat-GGUF`
5. Select file: `llama-2-7b-chat.Q4_K_M.gguf`
6. Click **Download**

### 5. Make your first API call

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama-2-7b-chat.Q4_K_M.gguf",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

Or use with OpenAI SDK:
```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed"  # Optional if auth disabled
)

response = client.chat.completions.create(
    model="llama-2-7b-chat.Q4_K_M.gguf",
    messages=[{"role": "user", "content": "Hello!"}]
)
print(response.choices[0].message.content)
```

## Documentation

- **[Architecture](docs/architecture.md)** - System design and components
- **[API Reference](docs/api-endpoints.md)** - Complete API specification
- **[Deployment Guide](docs/deployment.md)** - Production deployment with Docker
- **[User Guide](docs/user-guide.md)** - WebUI usage and examples
- **[Models Reference](docs/models.md)** - Database schema and models

## Technology Stack

### Backend
- **FastAPI** - Modern Python web framework
- **SQLModel** - SQLAlchemy + Pydantic ORM
- **PostgreSQL** - Relational database
- **llama.cpp** - Efficient LLM inference

### Frontend
- **React** - UI framework
- **TypeScript** - Type safety
- **shadcn/ui** - Component library
- **Tailwind CSS** - Styling

### Infrastructure
- **Docker Compose** - Container orchestration
- **Traefik** - Reverse proxy (optional)
- **Mailpit** - Email testing (development)

## Model Support

### Supported Formats
- **GGUF only** (llama.cpp native format)

### Supported Quantizations
- Q2_K through Q8_0
- Q4_K_S, Q4_K_M (recommended)
- Q5_K_S, Q5_K_M
- Q6_K, Q8_0
- F16, F32, BF16

### Download Sources
- **HuggingFace Hub** (primary)
- **ModelScope** (secondary)
- Extensible architecture for more sources

### Recommended Models
- **Llama 3.x** - `bartowski/Llama-3-8B-Instruct-GGUF`
- **Mistral** - `TheBloke/Mistral-7B-Instruct-v0.2-GGUF`
- **Qwen** - `Qwen/Qwen-7B-Chat-GGUF`
- **Phi-3** - `TheBloke/Phi-3-mini-4k-instruct-GGUF`

## GPU Support

### NVIDIA (CUDA)
- Requires CUDA drivers and toolkit
- Auto-detected and configured
- Recommended: 35-50 GPU layers

### Apple Silicon (Metal)
- Built-in support on M1/M2/M3
- No additional configuration needed
- Recommended: 25-50 GPU layers

### AMD/Intel (Vulkan)
- Requires Vulkan drivers
- Cross-platform support
- Performance varies by GPU

## Performance Tips

1. **Maximize GPU layers** - Increase until VRAM is 80-90% full
2. **Use Q4_K_M quantization** - Best speed/quality balance
3. **Enable prompt caching** - Reduces latency for repeated patterns
4. **Right-size models** - Use smaller models for simple tasks
5. **Monitor VRAM** - Avoid swapping which kills performance

**Expected Performance** (tokens/second):
- RTX 4090 + Q4_K_M 7B: ~80 tok/s
- M2 Max + Q4_K_M 7B: ~50 tok/s
- CPU only + Q4_K_M 7B: ~5 tok/s

## Configuration

### Environment Variables

Key variables in `.env`:
```bash
# GPU configuration
DEFAULT_GPU_LAYERS=35
DEFAULT_CONTEXT_SIZE=4096
SERVER_INACTIVITY_TIMEOUT=300

# Storage
MODELS_PATH=/models
FILES_PATH=/files
CACHE_PATH=/cache

# Authentication (optional)
API_KEY_AUTH_ENABLED=false
ADMIN_JWT_SECRET=your_secret
```

### llama.cpp Server

Auto-configured with sensible defaults:
- **GPU layers**: 35 (adjust based on VRAM)
- **Context size**: 4096 tokens
- **Batch size**: 512
- **Auto-shutdown**: 5 minutes inactivity

## Backup & Restore

### Create backup
```bash
docker compose exec backend \
  python -m app.services.backup create \
  --output /backups/backup-$(date +%Y%m%d).tar.gz
```

### Restore backup
```bash
docker compose exec backend \
  python -m app.services.backup restore \
  /backups/backup-20240101.tar.gz
```

## Troubleshooting

### Model won't load
- Verify GGUF file exists and is valid
- Check available VRAM: `nvidia-smi` or system monitor
- Review logs: `docker compose logs backend`

### Slow generation
- Increase GPU layers in settings
- Reduce context size
- Use smaller quantization (Q4_K_M)

### Out of memory
- Reduce GPU layers
- Lower context size
- Close other GPU applications

See [docs/deployment.md](docs/deployment.md) for detailed troubleshooting.

## Development

### Backend development
```bash
cd backend
uv sync
uv run python -m app.main
```

### Frontend development
```bash
cd frontend
bun install
bun run dev
```

### Run tests
```bash
cd backend
uv run pytest
```

## Contributing

Contributions welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) first.

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Acknowledgments

- [llama.cpp](https://github.com/ggerganov/llama.cpp) - Efficient LLM inference
- [FastAPI](https://fastapi.tiangolo.com) - Modern web framework
- [HuggingFace](https://huggingface.co) - Model hosting
- This project started from the [Full Stack FastAPI Template](https://github.com/tiangolo/full-stack-fastapi-template)

---

**Built with ❤️ for local AI inference**
