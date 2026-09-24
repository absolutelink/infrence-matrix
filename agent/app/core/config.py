from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_ignore_empty=True,
        extra="ignore",
    )

    # Agent identity
    AGENT_ID: str
    AGENT_NAME: str = "inference-agent"
    AGENT_PLATFORM: str = "llamacpp"
    AGENT_TYPE: str = "generic"

    # Address the main app should use to reach this agent. If unset, the
    # container hostname is advertised (only useful when the main app shares
    # a network with the agent).
    AGENT_HOST: str | None = None
    AGENT_PORT: int = 8080

    # Frontend connection
    FRONTEND_URL: str

    # llama.cpp
    LLAMA_SERVER_PATH: str = "/usr/local/bin/llama-server"
    # ``llama-bench`` is normally installed on PATH in the llama.cpp image.
    LLAMA_BENCH_PATH: str = "llama-bench"
    DEFAULT_GPU_LAYERS: int = 35
    DEFAULT_CONTEXT_SIZE: int = 4096
    DEFAULT_BATCH_SIZE: int = 512
    SERVER_INACTIVITY_TIMEOUT: int = 300

    # GPU
    GPU_BACKEND: str = "auto"

    # Storage
    MODELS_PATH: str = "/models"
    CACHE_PATH: str = "/cache"

    # WebSocket
    WS_HEARTBEAT_INTERVAL: int = 30
    WS_RECONNECT_INTERVAL: int = 5
    WS_MAX_BUFFER_EVENTS: int = 1000

    # Events
    DOWNLOAD_PROGRESS_INTERVAL: float = 1.0
    GPU_USAGE_INTERVAL: int = 10
    LOG_FORWARD_INTERVAL: float = 1.0


settings = Settings()
