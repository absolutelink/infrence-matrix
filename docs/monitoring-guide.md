# Monitoring Guide

## Overview

Inference Matrix provides comprehensive monitoring through:
- **Prometheus metrics** - Time-series data for Grafana
- **WebSocket real-time updates** - Live dashboard updates
- **Health check endpoints** - Service status verification
- **Structured logging** - Debugging and auditing

## Prometheus Metrics

### Endpoint

```
GET /metrics
```

Returns Prometheus-format metrics for scraping.

### Available Metrics

#### Agent Metrics

```prometheus
# Total number of registered agents
inference_matrix_agents_total{status="online|offline|unreachable"}

# Agent uptime
inference_matrix_agent_uptime_seconds{agent_id="uuid"}
```

#### Server Metrics

```prometheus
# Running servers by agent and status
inference_matrix_servers_total{agent_id="uuid",status="running|stopped|starting"}

# Server restart count
inference_matrix_server_restarts_total{agent_id="uuid",server_id="uuid"}
```

#### Inference Metrics

```prometheus
# Total inference requests
inference_matrix_inference_requests_total{model="name",agent_id="uuid",status="success|error"}

# Request latency histogram
inference_matrix_inference_latency_seconds_bucket{model="name",agent_id="uuid",le="0.1|0.5|1.0|..."}
inference_matrix_inference_latency_seconds_sum{model="name",agent_id="uuid"}
inference_matrix_inference_latency_seconds_count{model="name",agent_id="uuid"}

# Tokens generated
inference_matrix_tokens_generated_total{model="name",agent_id="uuid"}
```

#### Resource Metrics

```prometheus
# VRAM usage
inference_matrix_vram_usage_bytes{agent_id="uuid",gpu_id="0"}

# Prompt cache size
inference_matrix_cache_size_bytes{model="name",agent_id="uuid"}
```

### Prometheus Configuration

Add to `prometheus.yml`:

```yaml
scrape_configs:
  - job_name: 'inference-matrix'
    static_configs:
      - targets: ['frontend:8000']
    metrics_path: '/metrics'
    scrape_interval: 15s
```

### Grafana Dashboard

Import this dashboard JSON for visualization:

```json
{
  "dashboard": {
    "title": "Inference Matrix Overview",
    "panels": [
      {
        "title": "Agent Status",
        "targets": [{
          "expr": "inference_matrix_agents_total"
        }]
      },
      {
        "title": "Inference Latency (p95)",
        "targets": [{
          "expr": "histogram_quantile(0.95, rate(inference_matrix_inference_latency_seconds_bucket[5m]))"
        }]
      },
      {
        "title": "Tokens/Second",
        "targets": [{
          "expr": "rate(inference_matrix_tokens_generated_total[1m])"
        }]
      },
      {
        "title": "VRAM Usage",
        "targets": [{
          "expr": "inference_matrix_vram_usage_bytes"
        }]
      }
    ]
  }
}
```

## Real-Time Monitoring

### WebSocket Events

Connect to: `ws://frontend:8000/api/ws/agents/{agent_id}`

**Events:**

```json
// Server started
{
  "event": "server.started",
  "data": {
    "server_id": "uuid",
    "model_id": "uuid",
    "port": 8081
  }
}

// Server stopped
{
  "event": "server.stopped",
  "data": {
    "server_id": "uuid",
    "reason": "graceful|crash|inactivity"
  }
}

// GPU usage update
{
  "event": "gpu.usage",
  "data": {
    "gpu_id": 0,
    "vram_used": 8589934592,
    "vram_free": 15986065408,
    "utilization": 45
  }
}

// Download progress
{
  "event": "download.progress",
  "data": {
    "job_id": "uuid",
    "progress_percent": 45.5,
    "speed_mbps": 12.5
  }
}
```

### WebUI Dashboard

The WebUI provides real-time monitoring:

**Agent Status Page:**
- Online/offline status
- GPU utilization
- VRAM usage
- Running servers
- WebSocket connection status

**Model Management:**
- Download progress
- Model file sizes
- Cache status
- Server assignments

**Inference Monitoring:**
- Active requests
- Token generation rate
- Request latency
- Error rates

## Health Checks

### Service Health

```bash
# Frontend health
curl http://localhost:8000/api/health

# Agent health
curl http://localhost:8080/api/health

# Database health
docker compose exec postgres pg_isready -U inference
```

**Response:**
```json
{
  "status": "healthy",
  "agent_id": "agent-1",
  "uptime_seconds": 3600
}
```

### Docker Health Checks

Configure in `compose.yml`:

```yaml
services:
  frontend:
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/api/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 10s
  
  agent:
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/api/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 30s
```

## Logging

### Log Levels

- **INFO** - Normal operations
- **WARNING** - Potential issues
- **ERROR** - Failures
- **DEBUG** - Detailed debugging (development only)

### Access Logs

```bash
# Frontend logs
docker compose logs frontend

# Agent logs
docker compose logs agent

# Follow logs in real-time
docker compose logs -f agent

# Last 100 lines
docker compose logs --tail=100 agent
```

### Structured Logging

Enable JSON logging for log aggregation:

```python
# In production, use JSON formatter
import json
import logging

class JSONFormatter(logging.Formatter):
    def format(self, record):
        log_entry = {
            "timestamp": self.formatTime(record),
            "level": record.levelname,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
        }
        return json.dumps(log_entry)
```

### Log Aggregation

For production, forward logs to:
- **ELK Stack** (Elasticsearch, Logstash, Kibana)
- **Loki** + Grafana
- **CloudWatch Logs** (AWS)
- **Stackdriver** (GCP)

Example Loki configuration:

```yaml
scrape_configs:
  - job_name: 'inference-matrix'
    static_configs:
      - targets: ['localhost:3100']
    labels:
      job: docker
      __path__: /var/log/containers/*.log
```

## Alerting

### Prometheus Alert Rules

```yaml
groups:
  - name: inference_matrix
    rules:
      - alert: AgentOffline
        expr: inference_matrix_agents_total{status="offline"} > 0
        for: 5m
        annotations:
          summary: "Agent {{ $labels.agent_id }} is offline"
      
      - alert: HighInferenceLatency
        expr: histogram_quantile(0.95, rate(inference_matrix_inference_latency_seconds_bucket[5m])) > 10
        for: 5m
        annotations:
          summary: "P95 latency > 10s"
      
      - alert: LowVRAM
        expr: inference_matrix_vram_usage_bytes / 24576000000 > 0.9
        for: 5m
        annotations:
          summary: "VRAM usage > 90%"
      
      - alert: ServerCrash
        expr: rate(inference_matrix_server_restarts_total[5m]) > 0
        annotations:
          summary: "Server restarted unexpectedly"
```

### Alertmanager Configuration

```yaml
route:
  group_by: ['alertname']
  group_wait: 30s
  group_interval: 5m
  repeat_interval: 4h
  receiver: 'slack'

receivers:
  - name: 'slack'
    slack_configs:
      - api_url: 'https://hooks.slack.com/services/XXX/YYY/ZZZ'
        channel: '#inference-alerts'
        title: 'Inference Matrix Alert'
        text: '{{ range .Alerts }}{{ .Annotations.summary }}{{ end }}'
```

## Performance Tuning

### Monitor These Metrics

1. **Token Generation Rate** - Target: 50+ tok/s for 7B models
2. **P95 Latency** - Target: < 5s for first token
3. **VRAM Usage** - Keep < 90%
4. **Cache Hit Rate** - Target: > 50% for repeated prompts

### Optimization Tips

```bash
# Monitor token generation
watch -n 1 'curl -s http://localhost:8000/metrics | grep tokens_generated'

# Check VRAM trends
curl http://localhost:8000/metrics | grep vram_usage

# Track cache effectiveness
curl http://localhost:8000/metrics | grep cache_size
```

## Troubleshooting

See `troubleshooting.md` for common monitoring issues.

## Next Steps

1. Set up Prometheus scraping
2. Import Grafana dashboard
3. Configure alerting rules
4. Set up log aggregation
5. Create runbooks for common alerts
