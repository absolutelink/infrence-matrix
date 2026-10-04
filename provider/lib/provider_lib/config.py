"""Provider instance configuration, derived entirely from environment.

A provider instance has no database of its own; everything comes from these
variables plus the registration response the admin returns (persisted to
CACHE_DIR/provider_config.json).
"""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class ProviderSettings(BaseSettings):
    model_config = SettingsConfigDict(env_ignore_empty=True, extra="ignore")

    # Mandatory wiring
    MACHINE_UID: str
    PROVIDER_REGISTRATION_TOKEN: str
    ADMIN_BASE_URL: str

    # The port this provider instance serves its OpenAI-compatible API on.
    # Admin points litellm at http://<machine host>:<this port>/v1.
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

    @property
    def metrics_categories(self) -> set[str]:
        return {c for c in self.METRICS_CATEGORIES.split() if c}

    @property
    def provider_config_path(self) -> Path:
        return self.CACHE_DIR / "provider_config.json"


# Provider type this container implements. Set by each provider package
# (never from the environment) and validated against the registration
# token's definition on the admin side.
PROVIDER_TYPE: str = ""
