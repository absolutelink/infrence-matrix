# User Guide

Welcome to Inference Matrix! This guide covers using the WebUI and API for model management and inference, including multi-agent deployments.

## Table of Contents

1. [Getting Started](#getting-started)
2. [Model Management](#model-management)
3. [Using the API](#using-the-api)
4. [Multi-Agent Workflows](#multi-agent-workflows)
5. [Monitoring & Settings](#monitoring--settings)
6. [Troubleshooting](#troubleshooting)

---

## Getting Started

### First Login

1. Open your browser to http://localhost:5173 (local development) or your configured FRONTEND_HOST
2. If authentication is enabled, log in with your credentials
   - Default admin email: `admin@example.com`
   - Default admin password: `changethis` (change this in production!)
3. You'll see the dashboard with model status and quick actions

### Dashboard Overview

The dashboard shows:
- **Active Models**: Currently loaded models with status
- **Quick Stats**: VRAM usage, tokens/sec, active requests
- **Recent Activity**: Latest downloads, completions, and errors
- **Quick Actions**: Download model, start server, view logs

---

## Model Management

### Downloading Models

#### From HuggingFace

1. Click **Models** in the sidebar
2. Click **Download Model** button
3. Select **HuggingFace** as source
4. Enter repository ID: `TheBloke/Llama-2-7B-Chat-GGUF`
5. Select file: `llama-2-7b-chat.Q4_K_M.gguf`
6. Click **Download**

**Progress Tracking:**
- Download progress bar shows percentage and speed
- Pause/Resume buttons available
- Auto-retry on network failures
- Validation after download completes

#### From ModelScope

1. Follow same steps as HuggingFace
2. Select **ModelScope** as source
3. Enter repository ID: `modelscope/Llama-3-8B`
4. Select desired GGUF file

### Model List

The Models page shows:
- **Name**: Model filename
- **Size**: File size on disk
- **Architecture**: Model type (Llama, Mistral, etc.)
- **Quantization**: Quantization level (Q4_K_M, Q5_K_M, etc.)
- **Status**: Available/Loading/Active
- **Actions**: Load, Unload, Delete, Info

### Model Information

Click on any model to see:
- **Basic Info**: Name, size, architecture
- **Capabilities**: Context length, embeddings support, etc.
- **Recommended Settings**: GPU layers, batch size
- **Source**: Download source and URL
- **License**: Model license information
- **Benchmarks**: Performance metrics (if available)

### Deleting Models

1. Find the model in the list
2. Click the **Delete** button
3. Confirm deletion
4. Model is removed from disk and database

**Note**: Cannot delete models that are currently loaded. Unload first.

---

## Using the API

### Quick Start with curl

#### List Models

```bash
curl http://localhost:8000/v1/models
```

#### Chat Completion

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama-2-7b-chat.Q4_K_M.gguf",
    "messages": [
      {"role": "user", "content": "Hello!"}
    ],
    "temperature": 0.7
  }'
```

#### Streaming Chat

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama-2-7b-chat.Q4_K_M.gguf",
    "messages": [{"role": "user", "content": "Tell me a story"}],
    "stream": true
  }'
```

#### Generate Embeddings

```bash
curl http://localhost:8000/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "model": "nomic-embed-text.Q4_K_M.gguf",
    "input": "Hello world"
  }'
```

### Using with OpenAI SDK

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed"  # Optional if auth disabled
)

# Chat completion
response = client.chat.completions.create(
    model="llama-2-7b-chat.Q4_K_M.gguf",
    messages=[
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Hello!"}
    ]
)
print(response.choices[0].message.content)

# Streaming
stream = client.chat.completions.create(
    model="llama-2-7b-chat.Q4_K_M.gguf",
    messages=[{"role": "user", "content": "Tell a story"}],
    stream=True
)
for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")

# Embeddings
embedding = client.embeddings.create(
    model="nomic-embed-text.Q4_K_M.gguf",
    input="Hello world"
)
print(embedding.data[0].embedding)
```

### Using with LangChain

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed",
    model="llama-2-7b-chat.Q4_K_M.gguf"
)

response = llm.invoke("Hello!")
print(response.content)
```

---

## Monitoring & Settings

### Server Status

The **Servers** page shows:
- **Active Instances**: Running llama-server processes
- **Port**: Server port number
- **Model**: Loaded model name
- **Uptime**: How long server has been running
- **Last Request**: Time since last inference request
- **GPU Usage**: VRAM consumption
- **Actions**: Stop, Restart, View Logs

### GPU Configuration

Navigate to **Settings > GPU**:

**Global Settings:**
- **GPU Layers**: Default layers to offload (0-100)
- **Context Size**: Default context window
- **Batch Size**: Default batch size for processing

**Per-Model Overrides:**
- Click model name to set custom values
- Override global defaults
- Save for automatic application on load

### Prompt Cache

The **Cache** page shows:
- **Active Caches**: Cached prompt prefixes
- **Hit Rate**: Cache effectiveness
- **Size**: Cache memory usage
- **Age**: When cache was created
- **Actions**: Clear cache, Force cache

**Cache Strategies:**
- **Chat Completions**: Automatic conversation caching
- **Responses API**: Hierarchical system prompt caching

### API Keys

If authentication is enabled:

**Create API Key:**
1. Go to **Settings > API Keys**
2. Click **Create Key**
3. Enter name and permissions
4. Copy the key (shown once only)
5. Use in API requests: `Authorization: Bearer sk-...`

**Manage Keys:**
- View key list with last used timestamp
- Revoke keys
- Set expiration dates

### System Monitoring

The **Monitoring** dashboard shows:

**Real-time Metrics:**
- **VRAM Usage**: Per-model GPU memory
- **Tokens/sec**: Generation speed
- **Request Queue**: Pending requests
- **Cache Hit Rate**: Prompt cache effectiveness
- **Active Connections**: Current API clients

**Historical Charts:**
- Token generation over time
- Memory usage trends
- Request volume
- Error rates

### Backup & Restore

**Create Backup:**
1. Go to **Settings > Backup**
2. Click **Create Backup**
3. Select components:
   - Database (conversations, API keys, configs)
   - Model metadata
   - Download job state
4. Download backup file

**Restore Backup:**
1. Go to **Settings > Backup**
2. Click **Restore**
3. Upload backup file
4. Select components to restore
5. Confirm restore

**Note**: Models themselves (GGUF files) are not included in backups. Re-download or restore from separate model backup.

---

## Multi-Agent Workflows

Inference Matrix supports distributed inference across multiple GPU-equipped machines using the Agent architecture.

### Agent Status Dashboard

The **Agents** page shows:

- **Agent List**: All registered agents with status
- **Health Indicators**: Online/offline/unreachable
- **GPU Info**: GPU model, VRAM total/used
- **WebSocket Status**: Real-time connection health
- **Server Count**: Running llama.cpp servers per agent
- **Last Seen**: Most recent heartbeat timestamp

### Starting Servers on Agents

**Automatic Agent Selection:**

1. Go to **Models** page
2. Click **Load** on a model
3. System automatically selects an agent with:
   - The model available
   - Sufficient VRAM
   - Online status
   - Lowest current load

**Manual Agent Selection:**

1. Go to **Models** page
2. Click **Load** dropdown arrow
3. Select specific agent from list
4. Server starts on chosen agent

### Multi-Agent Inference

When making API requests, you can specify which agent to use:

```bash
# Let system choose best agent
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama-3.2-3b-instruct",
    "messages": [{"role": "user", "content": "Hello"}]
  }'

# Specify specific agent
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama-3.2-3b-instruct",
    "agent_id": "agent-2",
    "messages": [{"role": "user", "content": "Hello"}]
  }'
```

### Load Distribution

The Frontend Service distributes requests based on:

1. **Model Availability** - Agent must have the model downloaded
2. **Agent Status** - Only online agents receive requests
3. **Server Availability** - Prefers agents with existing servers
4. **Load Balancing** - Future: distribute based on current load

### Model Synchronization

Models are not automatically synced across agents. To use a model on multiple agents:

**Option 1: Download to each agent separately**
1. Go to **Models** page
2. Click **Download** dropdown
3. Select target agent
4. Repeat for each agent

**Option 2: Manual file copy**
1. Download model to shared storage
2. Mount shared storage on all agents
3. Register model via API on each agent

### Agent Failover

If an agent goes offline:

1. Agent status changes to `offline` or `unreachable`
2. Existing servers on that agent become unavailable
3. New requests automatically route to other agents
4. Manual intervention required to restart servers

**Best Practice:** Keep models downloaded on multiple agents for redundancy.

### Monitoring Agents

**Via WebUI:**

- **Agents Page**: Real-time status overview
- **Dashboard**: Agent count, online/offline breakdown
- **Server List**: Shows which agent hosts each server

**Via API:**

```bash
# List all agents
curl http://localhost:8000/api/agents

# Get specific agent
curl http://localhost:8000/api/agents/agent-1

# Get agent GPU info
curl http://localhost:8000/api/agents/agent-1/gpu
```

### Troubleshooting Agents

**Agent shows offline:**

1. Check agent machine network connectivity
2. Verify agent service is running
3. Check FRONTEND_URL configuration on agent
4. Review agent logs for errors

**Server won't start on agent:**

1. Check agent has sufficient VRAM
2. Verify model file exists on agent machine
3. Check agent logs for llama.cpp errors
4. Try starting server on different agent

**High latency on specific agent:**

1. Check agent GPU utilization
2. Monitor network latency between frontend and agent
3. Consider load balancing to other agents
4. Check agent machine resources (CPU, RAM)

---

## Audio Features

### Transcription

**Via WebUI:**
1. Go to **Audio > Transcription**
2. Upload audio file (mp3, wav, etc.)
3. Select Whisper model
4. Choose language (optional)
5. Click **Transcribe**
6. View/download transcription

**Via API:**
```bash
curl http://localhost:8000/v1/audio/transcriptions \
  -F "file=@audio.mp3" \
  -F "model=whisper-large-v3" \
  -F "language=en"
```

### Translation

**Via WebUI:**
1. Go to **Audio > Translation**
2. Upload audio file (non-English)
3. Select Whisper model
4. Click **Translate**
5. View English translation

**Via API:**
```bash
curl http://localhost:8000/v1/audio/translations \
  -F "file=@spanish_audio.mp3" \
  -F "model=whisper-large-v3"
```

### Text-to-Speech

**Via WebUI:**
1. Go to **Audio > Speech**
2. Enter text
3. Select voice
4. Choose output format (mp3, wav, etc.)
5. Adjust speed (optional)
6. Click **Generate**
7. Download/play audio

**Via API:**
```bash
curl http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "tts-model",
    "input": "Hello world",
    "voice": "alloy",
    "response_format": "mp3"
  }' \
  --output speech.mp3
```

---

## Batch Processing

### Creating Batch Jobs

1. Go to **Batches**
2. Upload input file (JSONL format)
3. Select endpoint (`/v1/chat/completions`)
4. Click **Create Batch**

**Input File Format:**
```jsonl
{"custom_id": "request-1", "method": "POST", "url": "/v1/chat/completions", "body": {"model": "llama-2-7b", "messages": [...]}}
{"custom_id": "request-2", "method": "POST", "url": "/v1/chat/completions", "body": {...}}
```

### Monitoring Batches

Batch status:
- **validating**: Checking input file
- **in_progress**: Processing requests
- **completed**: All requests done
- **failed**: Error occurred
- **cancelled**: User cancelled

### Downloading Results

1. Find completed batch
2. Click **Download Results**
3. Get JSONL with responses

---

## Troubleshooting

### Common Issues

**Model won't load:**
- Check if model file exists in Models page
- Verify it's a valid GGUF format
- Check available VRAM in Monitoring
- Review server logs for errors

**Slow generation:**
- Reduce context size in settings
- Increase GPU layers if VRAM allows
- Check if multiple models are competing for resources
- Monitor tokens/sec in dashboard

**Download fails:**
- Check network connectivity
- Verify HuggingFace/ModelScope access
- Try pausing and resuming download
- Check disk space availability

**API returns errors:**
- Verify model name is correct
- Check if model is loaded (not just downloaded)
- Review error message for specifics
- Check API logs for details

### Getting Help

1. **Check Logs**: Settings > Logs
2. **View Documentation**: docs/ folder in repository
3. **API Reference**: http://localhost:8000/docs
4. **GitHub Issues**: Report bugs or request features

---

## Tips & Best Practices

### Performance

1. **Use appropriate quantization**: Q4_K_M offers good speed/quality balance
2. **Maximize GPU usage**: Increase GPU layers until VRAM is full
3. **Enable prompt caching**: Reduces latency for repeated patterns
4. **Limit concurrent models**: Each model uses VRAM even when idle

### Cost Savings

1. **Auto-shutdown**: Servers automatically stop after inactivity
2. **Right-size models**: Use smaller models when possible
3. **Batch requests**: Use batch endpoint for bulk processing
4. **Monitor usage**: Track tokens generated in Monitoring dashboard

### Security

1. **Enable authentication**: If exposed to network
2. **Use API keys**: Don't share keys publicly
3. **Regular backups**: Protect conversation history
4. **Update regularly**: Keep up with security patches
