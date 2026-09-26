"""Manages halogen-flash-server subprocesses behind the Matrix agent contract."""

import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.core.config import settings
from app.core.logging import logger
from app.services.event_bus import publish_event
from app.services.halogen_server import HalogenServerManager
from app.services.model_manager import model_manager

HALOGEN_FLASH_REPO_ID = "peonist-ai/halogen-qwen3.8-flash-next"
HALOGEN_FLASH_CHECKPOINT = (
    "/models/peonist-ai/halogen-qwen3.8-flash-next/qwen38-flash-next-w4b.hgn"
)
HALOGEN_FLASH_TOKENIZER = "/models/peonist-ai/halogen-qwen3.8-flash-next/tokenizer"


@dataclass
class HalogenFlashServerConfig:
    """One Flash API/engine pair."""

    model_path: str
    port: int
    api_port: int
    engine_port: int
    options: dict[str, Any]


class HalogenFlashServerManager(HalogenServerManager):
    """Run one independently configured Halogen Flash process per server ID."""

    PROCESS_LABEL = "halogen-flash"

    @property
    def ServerConfig(self) -> type[HalogenFlashServerConfig]:  # noqa: N802
        return HalogenFlashServerConfig

    async def start_server(
        self, server_id: str, config: HalogenFlashServerConfig
    ) -> bool:
        if server_id in self.servers:
            return False
        if len(self.servers) >= settings.HALOGEN_MAX_INSTANCES:
            raise RuntimeError("Halogen Flash instance limit reached")

        await model_manager.download_repository(
            HALOGEN_FLASH_REPO_ID, job_id=f"halogen-flash-{server_id}"
        )

        env = dict(os.environ)
        env.pop("VIRTUAL_ENV", None)
        env["PATH"] = ":".join(
            path
            for path in env.get("PATH", "").split(":")
            if path != "/agent/.venv/bin"
        )
        env.update(
            {
                "HALOGEN_API_PORT": str(config.api_port),
                "HALOGEN_PORT": str(config.engine_port),
                "HALOGEN_BIND": "127.0.0.1",
                "HALOGEN_ENGINE": f"127.0.0.1:{config.engine_port}",
                "HALOGEN_CHECKPOINT": HALOGEN_FLASH_CHECKPOINT,
                "HALOGEN_TOKENIZER": HALOGEN_FLASH_TOKENIZER,
            }
        )
        if config.options.get("cache_dir_enabled", False):
            cache_dir = Path(settings.CACHE_PATH) / server_id
            cache_dir.mkdir(parents=True, exist_ok=True)
            env["HALOGEN_CACHE_DIR"] = str(cache_dir)
        mappings = {
            "kv_slots": "HALOGEN_KV_SLOTS",
            "kv_pool_positions": "HALOGEN_KV_POOL_POSITIONS",
            "kv_pool_fit": "HALOGEN_KV_POOL_FIT",
            "host_reserve_gib": "HALOGEN_HOST_RESERVE_GIB",
            "ctx": "HALOGEN_CTX",
            "max_tok": "HALOGEN_MAX_TOK",
            "max_tokens_cap": "HALOGEN_MAX_TOKENS_CAP",
            "max_tokens_default": "HALOGEN_MAX_TOKENS_DEFAULT",
            "queue_timeout": "HALOGEN_QUEUE_TIMEOUT",
            "keepalive_timeout": "HALOGEN_KEEPALIVE_TIMEOUT",
            "sse_keepalive_s": "HALOGEN_SSE_KEEPALIVE_S",
            "temperature": "HALOGEN_TEMPERATURE",
            "top_p": "HALOGEN_TOP_P",
            "top_k": "HALOGEN_TOP_K",
            "min_p": "HALOGEN_MIN_P",
            "presence_penalty": "HALOGEN_PRESENCE_PENALTY",
            "frequency_penalty": "HALOGEN_FREQUENCY_PENALTY",
            "reasoning_effort": "HALOGEN_REASONING_EFFORT",
            "enable_thinking": "HALOGEN_ENABLE_THINKING",
            "max_thinking_tokens": "HALOGEN_MAX_THINKING_TOKENS",
            "drafter_default": "HALOGEN_DRAFTER_DEFAULT",
            "prompt_cache": "HALOGEN_PROMPT_CACHE",
            "prefill_chunk": "HALOGEN_PREFILL_CHUNK",
            "cache_entries": "HALOGEN_CACHE_ENTRIES",
            "cache_branches": "HALOGEN_CACHE_BRANCHES",
            "cache_snap3": "HALOGEN_CACHE_SNAP3",
            "cache_full": "HALOGEN_CACHE_FULL",
            "cache_disk_gib": "HALOGEN_CACHE_DISK_GIB",
            "cache_prune_old": "HALOGEN_CACHE_PRUNE_OLD",
            "composable_context": "HALOGEN_COMPOSABLE_CONTEXT",
            "composable_context_floor": "HALOGEN_COMPOSABLE_CONTEXT_FLOOR",
            "composable_context_bytes": "HALOGEN_COMPOSABLE_CONTEXT_BYTES",
            "grammar": "HALOGEN_GRAMMAR",
            "vision_tower": "HALOGEN_VISION_TOWER",
        }
        for key, variable in mappings.items():
            if config.options.get(key) is not None:
                env[variable] = str(config.options[key])

        command = ["stdbuf", "-oL", "-eL", settings.HALOGEN_ENTRYPOINT, "all"]
        logger.info(
            "Starting Halogen Flash %s on API port %s (engine port %s)",
            server_id,
            config.api_port,
            config.engine_port,
        )
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
            start_new_session=True,
            bufsize=1,
        )
        self.servers[server_id] = process
        self.configs[server_id] = config
        self.start_times[server_id] = time.time()
        self._health_failures[server_id] = 0
        self.server_health[server_id] = "starting"
        self.healthy_servers.discard(server_id)
        self._start_log_reader(server_id)
        try:
            await self._wait_for_health(server_id, config.api_port)
        except Exception:
            await self.stop_server(server_id, force=True)
            raise
        self.healthy_servers.add(server_id)
        self.server_health[server_id] = "healthy"
        publish_event(
            "server.started",
            {
                "server_id": server_id,
                "status": "running",
                "model_path": config.model_path,
                "port": config.api_port,
            },
        )
        return True


halogen_flash_server_manager = HalogenFlashServerManager()
