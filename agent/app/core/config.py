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

    # Frontend connection
    FRONTEND_URL: str

    # llama.cpp
    LLAMA_SERVER_PATH: str = "/usr/local/bin/llama-server"
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


settings = Settings()
