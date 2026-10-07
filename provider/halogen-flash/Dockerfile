# halogen-flash Inference Matrix provider.
#
# Build context = repo root, e.g.:
#   docker build -f provider/halogen-flash/Dockerfile -t matrix-provider-halogen-flash .
#
# The Halogen image owns the engine (`flash_serve`, `halogen-npu`), its
# /usr/local/bin/entrypoint.sh launcher, and its Python 3.12 + ROCm runtime.
# That runtime is the engine's and stays exactly as it sits: the provider never
# installs into it, never re-points /usr/local/bin/python3, and never prepends
# its own venv to PATH (the entrypoint resolves `python3` from PATH).
#
# The provider itself needs Python 3.14 (workspace requires-python), so a
# standalone 3.14 interpreter is copied in beside the base one and the venv is
# built IN THIS IMAGE (like the legacy recipes/halogen-flash build). A venv
# copied from a build stage is not portable: its interpreter symlink and its
# editable-install paths are absolute, so /agent/.venv resolved to the base
# image's 3.12 and `import provider_halogen_flash` failed at runtime.
FROM python:3.14-slim AS provider-runtime

FROM ghcr.io/peonist-ai/halogen-flash-server:0.16.2

USER root

# Standalone CPython 3.14 (interpreter, stdlib, shared lib, bundled pip) under
# its own name only — the base image's python/python3 symlinks keep pointing at
# 3.12 so /usr/local/bin/entrypoint.sh and the ROCm site-packages stay intact.
COPY --from=provider-runtime /usr/local/bin/python3.14 /usr/local/bin/python3.14
COPY --from=provider-runtime /usr/local/lib/python3.14 /usr/local/lib/python3.14
COPY --from=provider-runtime /usr/local/lib/libpython3.14.so* /usr/local/lib/

# Workspace root manifest + lock, and only the packages this provider needs
# (never the admin or the sibling provider packages).
COPY pyproject.toml uv.lock /app/
COPY provider/lib /app/provider/lib
COPY provider/halogen-flash /app/provider/halogen-flash

WORKDIR /app

# uv is a build-time tool: it is installed into the copied 3.14 tree (not the
# engine's interpreter) and invoked as a module, so no console-script shebang of
# either interpreter is relied upon. The venv lands at /app/.venv, in place, so
# its interpreter link and the editable installs of matrix-provider-lib /
# matrix-provider-halogen-flash resolve inside this image.
RUN python3.14 -m pip install --no-cache-dir uv \
    && python3.14 -m uv sync --frozen --no-dev \
        --package matrix-provider-halogen-flash \
        --python /usr/local/bin/python3.14 \
    && rm -rf /root/.cache/uv \
    && mkdir -p /models /cache

# HF_HUB_OFFLINE stays exactly as the Halogen image bakes it (1): it is what
# stops the front-end reaching for the Hub at request time. Downloads are the
# engine's own job at start (HALOGEN_DOWNLOAD, which overrides the flag for
# its `hf download` calls) and — for an operator-picked HuggingFace artifact —
# the provider's, which clears huggingface_hub's in-process offline switch in
# provider_lib.downloader without touching this environment.
ENV PROVIDER_PORT=8081 \
    HALOGEN_FLASH_SERVER_PATH=/usr/local/bin/entrypoint.sh \
    HF_HUB_DISABLE_XET=0

EXPOSE 8081

# Override the base image's ENTRYPOINT: halogen-flash-server enters straight
# into its engine launcher, which would read the provider command as its mode
# word ("mode sh" -> usage error -> exit 2) and never start the provider.
# PATH is deliberately left alone; the provider is invoked by absolute path.
ENTRYPOINT ["sh", "-c", "exec /app/.venv/bin/python -m provider_halogen_flash.main"]
