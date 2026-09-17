---
name: cache-management
description: Manage prompt caching for chat completions and responses API with hybrid tracking
---

# Cache Management Skill

Use this skill when managing prompt caching for improved inference performance.

## Commands

### View Cache Status

```bash
# Overall cache statistics
uv run python -m app.services.cache status

# Cache by model
uv run python -m app.services.cache status --model-id <model-uuid>

# Cache entries list
uv run python -m app.services.cache list

# Filter by type
uv run python -m app.services.cache list --type conversation
uv run python -m app.services.cache list --type hierarchical
```

### Clear Cache

```bash
# Clear all cache
uv run python -m app.services.cache clear --all

# Clear for specific model
uv run python -m app.services.cache clear --model-id <model-uuid>

# Clear expired entries only
uv run python -m app.services.cache clear --expired

# Clear specific cache entry
uv run python -m app.services.cache clear --cache-id <cache-uuid>
```

### Force Cache

```bash
# Cache a prompt prefix
uv run python -m app.services.cache force-cache \
  --model-id <model-uuid> \
  --content "System prompt here" \
  --ttl 1800

# Cache conversation history
uv run python -m app.services.cache force-cache \
  --model-id <model-uuid> \
  --conversation-id <conv-uuid> \
  --ttl 3600
```

### Cache Analytics

```bash
# Hit/miss statistics
uv run python -m app.services.cache analytics

# Cache efficiency over time
uv run python -m app.services.cache analytics --timeframe 24h

# Top cached prompts
uv run python -m app.services.cache analytics --top 10
```

## Python API

```python
from app.services.cache import PromptCacheManager

cache_manager = PromptCacheManager()

# Check if cache exists
exists = await cache_manager.has_cache(
    model_id=model_id,
    cache_key="conversation:123"
)

# Get cache entry
cache = await cache_manager.get_cache(cache_id)

# Create cache entry
cache = await cache_manager.create_cache(
    model_id=model_id,
    cache_type="conversation",
    content_hash=sha256_hash,
    llama_cache_id=llama_id,
    ttl_seconds=1800,
    conversation_id=conv_id
)

# Update access time
await cache_manager.touch_cache(cache_id)

# Record cache hit
await cache_manager.record_hit(cache_id)

# Delete cache
await cache_manager.delete_cache(cache_id)

# Cleanup expired
await cache_manager.cleanup_expired()
```

## Caching Strategies

### Chat Completions (`/v1/chat/completions`)

**Strategy:** Conversation-based caching

```python
# Cache key format
cache_key = f"chat:{conversation_id}:{message_index}"

# Cache includes
- System message
- Conversation history up to current turn
- Model configuration

# Invalidation
- New message added to conversation
- Model configuration changed
- TTL expired
```

**Example:**
```python
cache = await cache_manager.create_cache(
    model_id=model_id,
    cache_type="conversation",
    cache_key=f"chat:{conv_id}:0",
    content_hash=hash_messages(messages[:3]),
    ttl_seconds=1800  # 30 minutes
)
```

### Responses API (`/v1/responses`)

**Strategy:** Hierarchical caching

```python
# Cache levels
1. System prompt cache (shared across conversations)
2. Conversation prefix cache
3. Turn-level cache

# Cache key formats
- System: f"system:{model_id}:{system_prompt_hash}"
- Conversation: f"response:{response_id}:{turn_index}"
- Hierarchical: f"response:{response_id}:tree:{node_id}"
```

**Example:**
```python
# Cache system prompt separately
system_cache = await cache_manager.create_cache(
    model_id=model_id,
    cache_type="hierarchical",
    cache_key=f"system:{system_hash}",
    content_hash=system_hash,
    ttl_seconds=3600  # 1 hour
)

# Cache conversation with tree awareness
conv_cache = await cache_manager.create_cache(
    model_id=model_id,
    cache_type="hierarchical",
    cache_key=f"response:{response_id}:0",
    content_hash=content_hash,
    ttl_seconds=1800,
    conversation_id=conv_id
)
```

## llama.cpp Integration

### Native Caching

llama.cpp supports prompt caching via:
- **Prompt cache files**: Save/load cache to disk
- **Context reuse**: Keep context alive between requests

**Enable in server config:**
```yaml
llama_server:
  prompt_cache:
    enabled: true
    cache_path: /cache/llama
    max_cache_size_gb: 10
```

### Hybrid Tracking

Track both llama.cpp native cache AND PostgreSQL metadata:

```python
# Start llama.cpp server with cache
server = await start_server(
    model_id=model_id,
    use_prompt_cache=True,
    cache_path="/cache/llama/{model_id}"
)

# When request comes in:
# 1. Check PostgreSQL for cache metadata
cache_meta = await db.get_cache(model_id, cache_key)

# 2. If exists, llama.cpp will use native cache
# 3. Update PostgreSQL metadata
await cache_manager.record_hit(cache_meta.id)
```

## Cache Statistics

### Metrics Tracked

- **Hit Rate**: `hits / (hits + misses)`
- **Size**: Total cache size in bytes
- **Entries**: Number of cache entries
- **Age**: Average/median/max cache age
- **Efficiency**: Tokens saved vs recomputed

### Query Statistics

```sql
-- Overall hit rate
SELECT 
  SUM(hits) as total_hits,
  COUNT(*) as entries,
  AVG(hits) as avg_hits
FROM prompt_cache
WHERE expires_at > NOW();

-- By model
SELECT 
  m.name,
  SUM(pc.hits) as hits,
  COUNT(*) as entries
FROM prompt_cache pc
JOIN models m ON pc.model_id = m.id
GROUP BY m.id, m.name;

-- Expired entries
SELECT COUNT(*) 
FROM prompt_cache 
WHERE expires_at < NOW();
```

## TTL Management

### Default TTLs

- **Conversation cache**: 30 minutes
- **System prompt cache**: 1 hour
- **Hierarchical cache**: 30 minutes

### Custom TTL

```python
# Set per-request
response = await client.chat.completions.create(
    model="llama-2-7b",
    messages=messages,
    prompt_cache_options={
        "ttl": "60m"  # 60 minutes
    }
)

# Set per-model override
await cache_manager.set_model_ttl(
    model_id=model_id,
    default_ttl_seconds=3600
)
```

## Cache Warming

### Pre-cache Common Prompts

```python
# Warm system prompt cache
common_system_prompts = [
    "You are a helpful assistant",
    "You are a coding assistant",
    "You are a translation assistant"
]

for prompt in common_system_prompts:
    await cache_manager.warm_cache(
        model_id=popular_model,
        content=prompt,
        cache_type="hierarchical"
    )
```

### Pre-cache Conversation Starters

```python
# Cache common conversation prefixes
await cache_manager.warm_cache(
    model_id=model_id,
    conversation_id=template_conv,
    messages=template_messages
)
```

## Best Practices

1. **Use appropriate TTLs**: Short enough to save memory, long enough for reuse
2. **Monitor hit rates**: Aim for >50% for frequently used prompts
3. **Clean up expired**: Run cleanup every 5-10 minutes
4. **Cache hierarchically**: System prompts separate from conversations
5. **Track in PostgreSQL**: Even when using llama.cpp native cache
6. **Warm popular caches**: Pre-cache common system prompts

## Troubleshooting

### Low Hit Rate

**Causes:**
- TTL too short
- Prompts not repeated
- Cache invalidation too aggressive

**Solutions:**
- Increase TTL
- Analyze prompt patterns
- Adjust invalidation logic

### High Memory Usage

**Causes:**
- Too many cache entries
- Large context sizes
- Long TTLs

**Solutions:**
- Reduce max cache entries
- Lower context size
- Shorten TTL
- Clear expired more frequently

### Cache Not Working

**Check:**
1. llama.cpp server started with cache enabled
2. Cache path is writable
3. Cache key matches between requests
4. TTL not expired

## Configuration

Cache config in `/etc/inference-matrix/cache.yaml`:

```yaml
cache:
  enabled: true
  
  # Default TTLs
  default_ttl_seconds: 1800
  system_prompt_ttl: 3600
  
  # Limits
  max_entries: 1000
  max_size_gb: 10
  max_age_hours: 24
  
  # Cleanup
  cleanup_interval_seconds: 300
  cleanup_expired: true
  
  # llama.cpp integration
  llama_cache:
    enabled: true
    path: /cache/llama
    max_size_gb: 5
  
  # Analytics
  track_hits: true
  track_misses: true
  log_cache_keys: false
```
