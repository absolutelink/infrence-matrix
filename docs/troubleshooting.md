# Troubleshooting Guide

## Common Issues and Solutions

### Agent Won't Register

**Symptoms:**
- Agent shows as offline in WebUI
- `/api/agents` returns empty list
- WebSocket connection fails

**Causes:**
1. Incorrect `FRONTEND_URL` in agent configuration
2. Network connectivity issues between containers
3. Frontend service not running

**Solutions:**

```bash
# 1. Check agent environment
docker compose exec agent env | grep FRONTEND

# Should show: FRONTEND_URL=http://frontend:8000

# 2. Test connectivity from agent
docker compose exec agent curl -v http://frontend:8000/api/health

# 3. Check frontend logs
docker compose logs frontend | grep -i "agent"

# 4. Restart agent
docker compose restart agent
```

**Prevention:**
- Use Docker service names in URLs (not localhost)
- Ensure both services are on same Docker network
- Check firewall rules if running on separate hosts

---

### llama.cpp Server Won't Start

**Symptoms:**
- `/v1/chat/completions` returns 503
- Agent logs show server start failure
- Model file exists but server doesn't start

**Causes:**
1. Model file not found or corrupted
2. GPU drivers not installed
3. Out of VRAM
4. Invalid llama.cpp configuration

**Solutions:**

```bash
# 1. Verify model file exists
docker compose exec agent ls -lh /models/*.gguf

# 2. Check GPU visibility
docker compose exec agent nvidia-smi

# Should show GPU in list

# 3. Check available VRAM
docker compose exec agent nvidia-smi --query-gpu=memory.total,memory.used --format=csv

# 4. Test llama.cpp manually
docker compose exec agent llama-server --model /models/your-model.gguf --port 9999

# Should start without errors

# 5. Check agent logs
docker compose logs agent | grep -i "server"
```

**Prevention:**
- Validate model files after download
- Monitor VRAM usage in WebUI
- Use appropriate quantization (Q4_K_M recommended)
- Set conservative GPU layers initially

---

### WebSocket Disconnects Frequently

**Symptoms:**
- Agent shows online/offline cycling
- Real-time updates stop working
- Logs show frequent reconnections

**Causes:**
1. Network instability
2. Frontend service restarts
3. WebSocket timeout too short
4. Resource exhaustion

**Solutions:**

```bash
# 1. Check WebSocket logs
docker compose logs frontend | grep -i "websocket"
docker compose logs agent | grep -i "websocket"

# 2. Increase timeout (in agent .env)
echo "WS_HEARTBEAT_INTERVAL=60" >> .env
echo "WS_RECONNECT_INTERVAL=10" >> .env

# 3. Check resource usage
docker stats frontend agent

# Look for high CPU/memory usage

# 4. Restart both services
docker compose restart frontend agent
```

**Prevention:**
- Ensure stable network between services
- Monitor system resources
- Use appropriate timeout values
- Implement proper error handling

---

### Inference Very Slow

**Symptoms:**
- Tokens/second < 10
- High latency on chat completions
- GPU utilization low

**Causes:**
1. Running on CPU instead of GPU
2. Too many GPU layers for available VRAM
3. Model too large for hardware
4. Other processes using GPU

**Solutions:**

```bash
# 1. Verify GPU is being used
docker compose exec agent nvidia-smi

# Should show llama-server process with GPU usage

# 2. Check GPU layers configuration
docker compose exec agent env | grep GPU_LAYERS

# 3. Reduce GPU layers if VRAM exhausted
echo "DEFAULT_GPU_LAYERS=20" >> .env
docker compose restart agent

# 4. Monitor token generation
curl http://localhost:8000/metrics | grep tokens

# 5. Check for competing processes
docker compose exec agent nvidia-smi pmon
```

**Prevention:**
- Start with conservative GPU layers (35)
- Monitor VRAM usage regularly
- Use smaller quantization for large models
- Dedicate GPU to inference only

---

### Database Connection Errors

**Symptoms:**
- Frontend won't start
- Logs show "connection refused"
- Agents can't register

**Causes:**
1. PostgreSQL not running
2. Wrong DATABASE_URL
3. Database not initialized
4. Network issues

**Solutions:**

```bash
# 1. Check PostgreSQL status
docker compose ps postgres

# 2. Test database connection
docker compose exec frontend python -c "from app.db.session import Session; Session()"

# 3. Check DATABASE_URL
docker compose exec frontend env | grep DATABASE

# Should be: postgresql+psycopg://inference:PASSWORD@postgres/inference_matrix

# 4. View postgres logs
docker compose logs postgres

# 5. Reset database (WARNING: deletes all data)
docker compose down -v
docker compose up -d postgres
sleep 5
docker compose up -d frontend
```

**Prevention:**
- Use Docker volumes for data persistence
- Set strong POSTGRES_PASSWORD
- Run migrations after schema changes
- Monitor database health

---

### Model Download Fails

**Symptoms:**
- Download stuck at 0%
- Download fails midway
- Invalid model file after download

**Causes:**
1. Network connectivity issues
2. HuggingFace rate limiting
3. Insufficient disk space
4. Repository not found

**Solutions:**

```bash
# 1. Check disk space
docker compose exec agent df -h /models

# Should have > model_size * 2 free

# 2. Test HuggingFace connectivity
docker compose exec agent curl -I https://huggingface.co

# 3. Retry download
curl -X POST http://localhost:8080/api/models/download \
  -H "Content-Type: application/json" \
  -d '{"repo_id": "TheBloke/Llama-2-7B-GGUF", "filename": "llama-2-7b.Q4_K_M.gguf"}'

# 4. Download manually
docker compose exec agent bash
cd /models
huggingface-cli download TheBloke/Llama-2-7B-GGUF llama-2-7b.Q4_K_M.gguf

# 5. Check download logs
docker compose logs agent | grep -i "download"
```

**Prevention:**
- Ensure sufficient disk space (2x model size)
- Use resume-capable downloads
- Implement retry logic
- Consider HuggingFace token for rate limits

---

### Prometheus Metrics Not Working

**Symptoms:**
- `/metrics` endpoint returns 404
- No metrics in Prometheus
- Metrics show stale data

**Solutions:**

```bash
# 1. Verify endpoint exists
curl http://localhost:8000/metrics

# Should return Prometheus format metrics

# 2. Check prometheus-client installed
docker compose exec frontend pip list | grep prometheus

# 3. Restart frontend to load metrics
docker compose restart frontend

# 4. Check Prometheus configuration
# Ensure scrape config includes:
# - job_name: 'inference-matrix'
#   static_configs:
#     - targets: ['frontend:8000']
```

---

## Getting Help

### Collect Debug Information

```bash
# Service status
docker compose ps

# Recent logs
docker compose logs --tail=100

# Specific service logs
docker compose logs -f agent

# Resource usage
docker stats

# Database state
docker compose exec postgres psql -U inference -d inference_matrix -c "\dt"

# Agent registration
curl http://localhost:8000/api/agents | jq

# Model list
curl http://localhost:8000/v1/models | jq
```

### Report Issues

When reporting issues, include:
1. Docker Compose logs (last 100 lines)
2. Service status output
3. Error messages from logs
4. Configuration (.env with secrets redacted)
5. Steps to reproduce

### Additional Resources

- Architecture: `docs/architecture.md`
- API Reference: `docs/api-endpoints.md`
- Deployment Guide: `docs/deployment.md`
- GitHub Issues: Report bugs and feature requests
