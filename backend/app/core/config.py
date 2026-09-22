import warnings
from pathlib import Path
from typing import Literal, Self

from pydantic import (
    Field,
    HttpUrl,
    PostgresDsn,
    computed_field,
    field_validator,
    model_validator,
)
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file="../.env",
        env_ignore_empty=True,
        extra="ignore",
    )

    # Core Settings
    PROJECT_NAME: str = "Inference Matrix"
    BACKEND_HOST: str = "0.0.0.0"
    BACKEND_PORT: int = 8000
    FRONTEND_HOST: str = "http://localhost:5174"
    FRONTEND_HOSTS: str = "http://localhost:5174,http://localhost:3000,http://127.0.0.1:5174,http://127.0.0.1:3000"
    CORS_ALLOW_ALL_ORIGINS: bool = True
    API_V1_STR: str = "/api/v1"
    FASTAPI_ENV: Literal["development", "production"] | None = "development"

    # Database
    POSTGRES_USER: str = "inference"
    POSTGRES_PASSWORD: str = Field(default="", min_length=0)
    POSTGRES_DB: str = "inference_matrix"
    POSTGRES_HOST: str = "postgres"
    POSTGRES_PORT: int = 5432

    @computed_field  # type: ignore[prop-decorator]
    @property
    def DATABASE_URL(self) -> PostgresDsn:
        return PostgresDsn.build(
            scheme="postgresql+psycopg",
            username=self.POSTGRES_USER,
            password=self.POSTGRES_PASSWORD,
            host=self.POSTGRES_HOST,
            port=self.POSTGRES_PORT,
            path=self.POSTGRES_DB,
        )

    @computed_field  # type: ignore[prop-decorator]
    @property
    def SYNC_DATABASE_URL(self) -> PostgresDsn:
        return PostgresDsn.build(
            scheme="postgresql+psycopg2",
            username=self.POSTGRES_USER,
            password=self.POSTGRES_PASSWORD,
            host=self.POSTGRES_HOST,
            port=self.POSTGRES_PORT,
            path=self.POSTGRES_DB,
        )

    # Storage Paths
    MODELS_PATH: str = "/models"
    FILES_PATH: str = "/files"
    CACHE_PATH: str = "/cache"

    @field_validator("MODELS_PATH", "FILES_PATH", "CACHE_PATH", mode="before")
    @classmethod
    def _ensure_absolute_path(cls, value: str) -> str:
        return str(Path(value).absolute())

    # llama.cpp Configuration
    LLAMA_SERVER_PATH: str = "/usr/local/bin/llama-server"
    DEFAULT_GPU_LAYERS: int = 35
    DEFAULT_CONTEXT_SIZE: int = 4096
    DEFAULT_BATCH_SIZE: int = 512
    SERVER_INACTIVITY_TIMEOUT: int = 300
    MAX_SERVER_INSTANCES: int = 5

    # Agent WebSocket
    WS_RECONNECT_INTERVAL: int = 5
    AGENT_MAX_RECONNECT_ATTEMPTS: int = 10

    # Audio Processing
    WHISPER_MODEL_PATH: str = "/models/whisper"
    AUDIO_MAX_FILE_SIZE: int = 25
    AUDIO_MAX_DURATION: int = 60

    # Sentry
    SENTRY_DSN: HttpUrl | None = None

    def _check_default_secret(self, var_name: str, value: str | None) -> None:
        if value == "changethis":
            message = (
                f'The value of {var_name} is "changethis", '
                "for security, please change it, at least for deployments."
            )
            if self.FASTAPI_ENV == "development":
                warnings.warn(message, stacklevel=1)
            else:
                raise ValueError(message)

    @model_validator(mode="after")
    def _enforce_non_default_secrets(self) -> Self:
        if self.FASTAPI_ENV == "production":
            self._check_default_secret(
                "POSTGRES_PASSWORD", self.POSTGRES_PASSWORD
            )
        return self


settings = Settings()  # type: ignore[call-arg]
