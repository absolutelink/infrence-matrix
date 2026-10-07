"""Provider **agent** configuration, derived entirely from environment.

A provider agent has no database of its own; everything comes from these
variables plus the registration response the admin returns (persisted to
CACHE_DIR/provider_config.json).

Phase 16: an agent is a ``(machine, provider_type, agent_id)`` triple that
owns 1..N backends. It authenticates registration with the shared
``MACHINE_SECRET`` (``Machine.registration_secret``) — the old per-definition
``PROVIDER_REGISTRATION_TOKEN`` is gone — and presents its stable ``AGENT_ID``
so several agents can share a machine + type.
"""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class ProviderSettings(BaseSettings):
    model_config = SettingsConfigDict(env_ignore_empty=True, extra="ignore")

    # Mandatory wiring (Phase 16).
    MACHINE_UID: str
    MACHINE_SECRET: str
    ADMIN_BASE_URL: str
    # Stable operator-supplied id discriminating multiple agents that share a
    # (machine, provider_type). Presented on registration and the WS query.
    AGENT_ID: str

    # Base port for this agent's backends. The first backend serves its
    # OpenAI-compatible API here; additional backends increment
    # (``base_port + offset``). Admin points litellm at
    # http://<machine host>:<backend port>/v1.
    PROVIDER_PORT: int = 8081

    # Storage
    CACHE_DIR: Path = Path("/cache")
    MODELS_DIR: Path = Path("/models")

    # Space-delimited metrics categories to emit. Inference metrics are
    # always enabled and must NOT appear in this list.
    # Known categories: gpu_usage, vram, os_ram, cpu, storage
    METRICS_CATEGORIES: str = "gpu_usage vram os_ram cpu storage"

    # llama.cpp backend (provider_llama_cpp). The binary path comes from
    # the environment, never from backend_config.
    LLAMA_SERVER_PATH: str = "llama-server"
    SERVER_START_HEALTH_TIMEOUT: int = 120

    # Boot budget for the halogen family. Their entrypoints may DOWNLOAD
    # before the API answers /health — a cold halogen-flash boot pulls the
    # checkpoint plus its companions (overlay sidecar, ngram table, vision
    # tower, tokenizer) from HuggingFace, which is tens of gigabytes and can
    # legitimately take an hour on a slow link. The health wait therefore
    # gets its own, much larger knob; while it runs the driver heartbeats
    # `backend.status initializing` so the admin shows progress instead of a
    # silent hang. llama.cpp / gufo keep SERVER_START_HEALTH_TIMEOUT (local
    # load only, no download pass).
    ENGINE_BOOT_TIMEOUT: int = 3600

    # gufo backend (provider_gufo). Binary path from env, never backend_config.
    GUFO_SERVER_PATH: str = "gufo"

    # halogen backend (provider_halogen). The engine is launched through
    # this entrypoint script; config flows via HALOGEN_* env vars.
    HALOGEN_SERVER_PATH: str = "halogen-server"

    # halogen-flash backend (provider_halogen_flash).
    HALOGEN_FLASH_SERVER_PATH: str = "halogen-flash-server"

    # halogen-flash NPU (Ryzen AI) host paths, used by the NPU probe to
    # decide whether small models can be pinned to the NPU.
    NPU_DEVICE_PATH: str = "/dev/accel/accel0"
    NPU_XRT_LIB_DIR: str = "/opt/xilinx/xrt/lib"
    NPU_PINS_FILE: str = "/opt/halogen/npu/models.txt"
    NPU_BINARY_PATH: str = "/usr/local/bin/halogen-npu"

    # Machine-level metrics emitter period (seconds), active only while
    # the admin has assigned metrics ownership to this instance.
    MACHINE_METRICS_INTERVAL: float = 10.0

    # Minimum seconds between download.progress events.
    DOWNLOAD_PROGRESS_INTERVAL: float = 1.0

    # Log streaming (Phase 13): backend.logs / provider.logs flush
    # interval in seconds. Ring capacity is owned by each driver
    # (_LOG_BUFFER_LINES) and defaults to LOG_RING_LINES for the
    # provider's own log handler.
    LOG_FLUSH_INTERVAL_SECONDS: float = 1.0
    LOG_RING_LINES: int = 2000

    @property
    def metrics_categories(self) -> set[str]:
        return {c for c in self.METRICS_CATEGORIES.split() if c}

    @property
    def provider_config_path(self) -> Path:
        return self.CACHE_DIR / "provider_config.json"


# Provider type this container implements. Set by each provider package
# (never from the environment) and reported on registration, where the admin
# matches it against the agent's placed definitions.
PROVIDER_TYPE: str = ""
