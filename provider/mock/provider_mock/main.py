"""Mock provider package.

A hardware-free provider instance used for local development and to exercise
the full admin -> scheduler -> litellm -> SSE path. Registration and backend
lifecycle wiring land in Phases 3-4; this entrypoint starts the provider app
so the container is runnable now.
"""

import uvicorn
from provider_lib.app_factory import BackendOverrides, create_provider_app
from provider_lib.config import ProviderSettings

PROVIDER_TYPE = "mock"
VERSION = "dev"


def build_app():
    settings = ProviderSettings()
    overrides = BackendOverrides(provider_type=PROVIDER_TYPE, version=VERSION)
    return create_provider_app(settings, overrides)


def main() -> None:
    settings = ProviderSettings()
    uvicorn.run(build_app(), host="0.0.0.0", port=settings.PROVIDER_PORT)


if __name__ == "__main__":
    main()
