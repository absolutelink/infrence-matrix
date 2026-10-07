"""Admin settings.

URL layout (see the root ARCHITECTURE.md):
  /                 Swagger UI (built-in)
  /openapi.json     OpenAPI schema
  /admin            React UI (served from app/frontend build)
  /admin/api/*      Admin management API
  /v1/*             Public OpenAI-compatible inference API
  /provider/ws      Provider instances dial in here (bidirectional WS)

Auth model: trusted LAN. /admin/api and /v1 are unauthenticated.
Registration tokens and per-instance secrets protect only the provider WS.
"""

import warnings
from pathlib import Path
from typing import Literal, Self

from pydantic import Field, PostgresDsn, computed_field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=Path(__file__).resolve().parents[4] / ".env",
        env_ignore_empty=True,
        extra="ignore",
    )

    PROJECT_NAME: str = "Inference Matrix"
    FASTAPI_ENV: Literal["development", "production"] = "development"

    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # Version gate: provider instances must match this exactly (hard fail).
    # Overridden at build time from the git commit id.
    VERSION: str = "dev"

    # --- Database -----------------------------------------------------
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

    # --- Redis (scheduler queues, VRAM ledger, metrics ownership) -----
    REDIS_URL: str = "redis://redis:6379/0"

    # --- Provider connection supervision ------------------------------
    # An instance whose WS is dead is marked disconnected immediately; this
    # is only the stale-row sweep interval for bookkeeping.
    INSTANCE_SWEEP_INTERVAL_SECONDS: float = 30.0
    INSTANCE_STALE_AFTER_SECONDS: float = 120.0

    # Set false (e.g. in tests) to skip starting the presence sweep task.
    INSTANCE_SWEEP_ENABLED: bool = True

    # --- Phase 9 config-update pushes ---------------------------------
    # provider.config.update awaits the provider ack; a backend drain +
    # artifact download + boot can take minutes, so the timeout is
    # generous. A drain-refused (backend_in_use) NAK is retried
    # CONFIG_UPDATE_RETRIES total attempts, CONFIG_UPDATE_RETRY_DELAY
    # seconds apart. Any other failure is reported per-instance.
    CONFIG_UPDATE_TIMEOUT_SECONDS: float = 300.0
    CONFIG_UPDATE_RETRIES: int = 3
    CONFIG_UPDATE_RETRY_DELAY_SECONDS: float = 10.0

    # --- Backend boot / manual backend control -------------------------
    # `backend.start` with `wait_for_running: true` (the scheduler's boot
    # path) awaits the provider's whole lifecycle: artifact resolution plus
    # the engine's own start-time download pass — a cold halogen-flash boot
    # pulls the checkpoint, its overlay sidecar / ngram table / vision tower
    # and its tokenizer before the API port answers. The provider heartbeats
    # `backend.status initializing` throughout, so this is a ceiling on
    # waiting for a boot that is visibly progressing, not a probe timeout.
    # A client request is still bounded by its own `queue_timeout`; the boot
    # it triggered continues and a later request adopts the running instance.
    BACKEND_BOOT_TIMEOUT_SECONDS: float = 3600.0
    # `backend.stop` awaits SIGTERM + the engine's exit.
    BACKEND_STOP_TIMEOUT_SECONDS: float = 120.0

    # Phase 16 slice 4: proactively warm an agent's assigned backends (boot
    # them one at a time, bounded by VRAM + max_running) when its WebSocket
    # connects, so the first request is not the one that pays for the boot.
    # Set false for strictly lazy (on-demand-only) booting.
    SCHEDULER_WARMUP_ON_CONNECT: bool = True

    # CORS: the UI is served same-origin from this app; allow Vite dev server.
    CORS_ALLOW_ALL_ORIGINS: bool = True

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
            self._check_default_secret("POSTGRES_PASSWORD", self.POSTGRES_PASSWORD)
        return self


settings = Settings()  # type: ignore[call-arg]
