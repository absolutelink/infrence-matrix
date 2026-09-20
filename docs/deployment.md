# Deployment Guide

This guide covers deploying Inference Matrix for home server use with Docker Compose, including the distributed Agent architecture.

## Prerequisites

- Docker and Docker Compose installed
- Python 3.14+ (for local development)
- GPU drivers (if using GPU acceleration)
  - NVIDIA: CUDA drivers and toolkit
  - AMD: ROCm drivers
  - Apple: Metal support (built-in)
- At least 8GB RAM (16GB+ recommended)
- Sufficient disk space for models (varies by model size)

## Architecture Overview

Inference Matrix uses a split-service architecture:

- **Frontend Service** - WebUI, API, database, and orchestration
- **Agent Service** - Runs on GPU-equipped machines, manages llama.cpp servers
- **PostgreSQL** - Central database for state management

Multiple Agents can be deployed across different machines to distribute inference workloads.

## Quick Start

### 1. Clone and configure

```bash
git clone <repository-url> inference-matrix
cd inference-matrix
cp .env.example .env
```

### 2. Edit Environment Variables

Edit `.env` with your configuration:

```bash
# Project settings
PROJECT_NAME=Inference Matrix
BACKEND_HOST=0.0.0.0
BACKEND_PORT=8000
FRONTEND_HOST=http://localhost:5173  # Local development; change for production

# Database
POSTGRES_USER=inference
POSTGRES_PASSWORD=your_secure_password_here
POSTGRES_DB=inference_matrix

# Authentication (optional)
API_KEY_AUTH_ENABLED=false
ADMIN_JWT_SECRET=your_jwt_secret_here

# Storage
MODELS_PATH=/models
FILES_PATH=/files
CACHE_PATH=/cache

# llama.cpp
LLAMA_SERVER_PATH=/usr/local/bin/llama-server
DEFAULT_GPU_LAYERS=35
DEFAULT_CONTEXT_SIZE=4096
SERVER_INACTIVITY_TIMEOUT=300

# Agent configuration (for Agent service)
AGENT_ID=agent-1
AGENT_NAME=GPU-Agent-1
FRONTEND_URL=http://frontend:8000
```

### 3. Start Services

```bash
docker compose up -d
```

This starts the Frontend Service, Agent Service, and PostgreSQL database.

### 4. Access the Application

- **WebUI**: http://localhost:5173 (local development) or your configured FRONTEND_HOST
- **API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs
- **Agent API**: http://localhost:8080

---

## Configuration Options

### Environment Variables

#### Core Settings

| Variable | Description | Default |
|----------|-------------|---------|
| `PROJECT_NAME` | Application name | `Inference Matrix` |
| `BACKEND_HOST` | Backend bind address | `0.0.0.0` |
| `BACKEND_PORT` | Backend port | `8000` |
| `FRONTEND_HOST` | Frontend URL | `http://localhost:5173` (development) |

#### Database

| Variable | Description | Default |
|----------|-------------|---------|
| `POSTGRES_USER` | Database user | `inference` |
| `POSTGRES_PASSWORD` | Database password | *(required)* |
| `POSTGRES_DB` | Database name | `inference_matrix` |
| `POSTGRES_HOST` | Database host | `postgres` |
| `POSTGRES_PORT` | Database port | `5432` |

#### Authentication

| Variable | Description | Default |
|----------|-------------|---------|
| `API_KEY_AUTH_ENABLED` | Enable API key auth | `false` |
| `ADMIN_JWT_SECRET` | JWT secret for admin UI | *(required if auth enabled)* |
| `API_KEY_HEADER` | Custom auth header | `Authorization` |

#### Storage

| Variable | Description | Default |
|----------|-------------|---------|
| `MODELS_PATH` | Model storage path | `/models` |
| `FILES_PATH` | File uploads path | `/files` |
| `CACHE_PATH` | Cache storage path | `/cache` |

#### llama.cpp Configuration

| Variable | Description | Default |
|----------|-------------|---------|
| `LLAMA_SERVER_PATH` | Path to llama-server binary | `/usr/local/bin/llama-server` |
| `DEFAULT_GPU_LAYERS` | Default GPU layers | `35` |
| `DEFAULT_CONTEXT_SIZE` | Default context size | `4096` |
| `DEFAULT_BATCH_SIZE` | Default batch size | `512` |
| `SERVER_INACTIVITY_TIMEOUT` | Auto-shutdown timeout (seconds) | `300` |
| `MAX_SERVER_INSTANCES` | Max concurrent servers | `5` |

#### Agent Configuration

| Variable | Description | Default |
|----------|-------------|---------|
| `AGENT_ID` | Unique agent identifier | *(required for Agent)* |
| `AGENT_NAME` | Human-readable agent name | `inference-agent` |
| `FRONTEND_URL` | Frontend Service URL | *(required for Agent)* |
| `FRONTEND_API_KEY` | Optional API key for auth | `None` |
| `GPU_BACKEND` | GPU backend: auto, cuda, metal, vulkan | `auto` |

#### Audio Processing

| Variable | Description | Default |
|----------|-------------|---------|
| `WHISPER_MODEL_PATH` | Whisper model directory | `/models/whisper` |
| `AUDIO_MAX_FILE_SIZE` | Max audio file size (MB) | `25` |
| `AUDIO_MAX_DURATION` | Max audio duration (minutes) | `60` |

---

## Docker Compose Configuration

### Development Setup

```yaml
# compose.override.yml
services:
  backend:
    build:
      context: ./backend
      target: development
    volumes:
      - ./backend:/app
      - ./models:/models
      - ./files:/files
      - ./cache:/cache
    ports:
      - "8000:8000"
    environment:
      - FASTAPI_ENV=development
    depends_on:
      - postgres
      - mailpit
  
  frontend:
    build:
      context: ./frontend
      target: development
    volumes:
      - ./frontend:/app
      - /app/node_modules
    ports:
      - "3000:3000"
  
  postgres:
    image: postgres:16
    environment:
      - POSTGRES_USER=inference
      - POSTGRES_PASSWORD=inference
      - POSTGRES_DB=inference_matrix
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"
  
  mailpit:
    image: axllent/mailpit
    ports:
      - "8025:8025"
      - "1025:1025"

volumes:
  postgres_data:
```

### Production Setup

```yaml
# compose.deploy.yml
services:
  backend:
    build:
      context: ./backend
      target: production
    restart: unless-stopped
    volumes:
      - models:/models
      - files:/files
      - cache:/cache
    environment:
      - FASTAPI_ENV=production
      - POSTGRES_PASSWORD=${POSTGRES_PASSWORD:?required}
    depends_on:
      postgres:
        condition: service_healthy
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
  
  frontend:
    build:
      context: ./frontend
      target: production
    restart: unless-stopped
  
  postgres:
    image: postgres:16-alpine
    restart: unless-stopped
    environment:
      - POSTGRES_USER=${POSTGRES_USER:?required}
      - POSTGRES_PASSWORD=${POSTGRES_PASSWORD:?required}
      - POSTGRES_DB=${POSTGRES_DB:?required}
    volumes:
      - postgres_data:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U ${POSTGRES_USER}"]
      interval: 10s
      timeout: 5s
      retries: 5
  
  traefik:
    image: traefik:v3.0
    restart: unless-stopped
    command:
      - "--api.insecure=true"
      - "--providers.docker=true"
      - "--entrypoints.web.address=:80"
      - "--entrypoints.websecure.address=:443"
    ports:
      - "80:80"
      - "443:443"
      - "8080:8080"
    volumes:
      - /var/run/docker.sock:/var/run/docker.sock:ro
    labels:
      - "traefik.enable=true"

volumes:
  models:
  files:
  cache:
  postgres_data:
```

### Multi-Agent Setup

Deploy multiple Agents across different GPU-equipped machines:

```yaml
# On GPU Machine 1
services:
  agent:
    build:
      context: ./agent
      dockerfile: Dockerfile
    restart: unless-stopped
    ports:
      - "8080:8080"
    environment:
      - AGENT_ID=agent-1
      - AGENT_NAME=GPU-Machine-1
      - FRONTEND_URL=http://frontend-host:8000
      - DEFAULT_GPU_LAYERS=35
    volumes:
      - ./models:/models
      - ./cache:/cache
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]

# On GPU Machine 2 (different AGENT_ID)
services:
  agent:
    environment:
      - AGENT_ID=agent-2
      - AGENT_NAME=GPU-Machine-2
      - FRONTEND_URL=http://frontend-host:8000
```

**Note:** Each Agent must have a unique `AGENT_ID` and be able to reach the Frontend Service over the network.

---

## GPU Configuration

### NVIDIA GPU (CUDA)

1. Install NVIDIA Container Toolkit:
```bash
# Ubuntu/Debian
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

2. Update `compose.yml`:
```yaml
services:
  agent:
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
```

3. Set GPU layers in `.env`:
```bash
DEFAULT_GPU_LAYERS=35  # Adjust based on VRAM
```

### AMD GPU (ROCm)

1. Install ROCm drivers for your distribution
2. Update `compose.yml`:
```yaml
services:
  agent:
    devices:
      - /dev/kfd:/dev/kfd
      - /dev/dri:/dev/dri
    group_add:
      - video
```

### Apple Silicon (Metal)

Metal is automatically used when running llama.cpp on Apple Silicon. No additional configuration needed.

---

## Model Management

### Downloading Models via WebUI

1. Navigate to **Models** in the WebUI
2. Click **Download Model**
3. Select source (HuggingFace or ModelScope)
4. Enter model repository ID (e.g., `TheBloke/Llama-2-7B-Chat-GGUF`)
5. Select quantization (e.g., `Q4_K_M`)
6. Click **Download**

### Manual Model Installation

1. Download GGUF model file
2. Place in models directory:
```bash
cp llama-3.2-3b-instruct.Q4_K_M.gguf /path/to/models/
```

3. Refresh model list in WebUI or via API:
```bash
curl http://localhost:8000/v1/models
```

### Supported Model Sources

**HuggingFace:**
- Format: `owner/repo`
- Example: `TheBloke/Llama-2-7B-Chat-GGUF`

**ModelScope:**
- Format: `owner/repo`
- Example: `modelscope/Llama-3-8B`

---

## Backup & Restore

### Backup

```bash
# Backup database
docker compose exec postgres pg_dump -U inference inference_matrix > backup.sql

# Backup models list
curl http://localhost:8000/admin/models/export > models.json

# Backup configuration
cp .env backup.env
```

### Restore

```bash
# Restore database
cat backup.sql | docker compose exec -T postgres psql -U inference inference_matrix

# Verify models exist
ls -la /path/to/models/
```

---

## Monitoring

### Health Checks

```bash
# Backend health
curl http://localhost:8000/health

# Database connection
docker compose exec postgres pg_isready -U inference

# llama-server instances
curl http://localhost:8000/admin/servers
```

### Logs

```bash
# All services
docker compose logs -f

# Backend only
docker compose logs -f backend

# Frontend only
docker compose logs -f frontend
```

### Metrics

Access Prometheus metrics at:
```
http://localhost:8000/metrics
```

---

## Troubleshooting

### Common Issues

**Models not loading:**
- Check llama-server binary path
- Verify model file exists and is valid GGUF
- Check GPU memory availability
- Review backend logs for errors

**API returns 503:**
- Model is still loading
- All server instances are busy
- Check `MAX_SERVER_INSTANCES` setting

**Database connection errors:**
- Verify PostgreSQL is running: `docker compose ps`
- Check database credentials in `.env`
- Ensure network connectivity between containers

**GPU not detected:**
- Verify drivers are installed
- Check Docker GPU configuration
- Review llama-server logs for GPU initialization errors

### Performance Tuning

**Increase context size:**
```bash
DEFAULT_CONTEXT_SIZE=8192
```

**Adjust GPU layers:**
- More layers = faster but more VRAM
- Start with 35, adjust based on VRAM usage

**Server concurrency:**
```bash
MAX_SERVER_INSTANCES=10  # Increase for multiple simultaneous models
```

---

## Updating

```bash
# Pull latest changes
git pull

# Rebuild and restart
docker compose down
docker compose build
docker compose up -d
```

---

## Production Hardening

1. **Enable authentication:**
```bash
API_KEY_AUTH_ENABLED=true
ADMIN_JWT_SECRET=$(openssl rand -hex 32)
```

2. **Configure HTTPS with Traefik:**
```yaml
labels:
  - "traefik.http.routers.backend.rule=Host(`inference.example.com`)"
  - "traefik.http.routers.backend.tls=true"
  - "traefik.http.routers.backend.tls.certresolver=letsencrypt"
```

3. **Set up monitoring:**
- Prometheus for metrics
- Grafana for dashboards
- Alertmanager for notifications

4. **Configure backups:**
```bash
# Cron job for daily backups
0 2 * * * /path/to/backup.sh
```

5. **Resource limits:**
```yaml
services:
  backend:
    deploy:
      resources:
        limits:
          memory: 8G
        reservations:
          memory: 4G
```
