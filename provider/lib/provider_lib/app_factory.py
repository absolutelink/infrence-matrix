"""FastAPI app factory for a provider instance.

Every provider type gets a base app that serves:
  - GET  /health         liveness + provider type + version
  - /v1/*              OpenAI-compatible surface (translation layer), added in
                       Phase 4; the provider lib launches and fronts the real
                       inference backend here so admin's litellm call targets
                       http://<machine>:<PROVIDER_PORT>/v1.

Per-type overrides (usage calculation, command building, NPU pinning, path
allowlists) are supplied by the provider package via `overrides`.
"""

from collections.abc import Callable
from typing import Any

from fastapi import FastAPI

from provider_lib.config import ProviderSettings


class BackendOverrides:
    """Hook bundle a provider package supplies to customize the generic app.

    Subclass or instantiate with callables; unset hooks fall back to lib
    defaults. Phase 4 fleshes these out against a real backend manager.
    """

    def __init__(
        self,
        *,
        provider_type: str,
        version: str,
        calculate_usage: Callable[[Any], dict[str, int]] | None = None,
    ) -> None:
        self.provider_type = provider_type
        self.version = version
        self.calculate_usage = calculate_usage


def create_provider_app(
    settings: ProviderSettings, overrides: BackendOverrides
) -> FastAPI:
    app = FastAPI(
        title=f"Inference Matrix provider ({overrides.provider_type})",
        version=overrides.version,
    )

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "provider_type": overrides.provider_type,
            "version": overrides.version,
            "machine_uid": settings.MACHINE_UID,
        }

    return app
