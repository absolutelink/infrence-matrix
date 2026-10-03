# Inference Matrix Agent with Gufo (Strix Halo inference engine) support.
#
# The official gufo-runtime image is a minimal Nix closure (bash, coreutils,
# curl) with no Python interpreter, so this recipe inverts the usual layering:
# the Matrix agent provides the Python runtime and the Gufo engine plus its
# Nix store closure are copied in from the runtime image.
ARG AGENT_IMAGE=ghcr.io/absolutelink/agent:main
ARG GUFO_RUNTIME_IMAGE=ghcr.io/gufo-org/toolboxes/gufo-runtime:latest

FROM ${GUFO_RUNTIME_IMAGE} AS gufo-runtime

FROM ${AGENT_IMAGE} AS agent

USER root

COPY --from=gufo-runtime /nix/store /nix/store
COPY --from=gufo-runtime /bin/gufo /usr/local/bin/gufo

WORKDIR /app
RUN chmod +x /usr/local/bin/entrypoint.sh \
    && uv sync --frozen --no-dev

ENV VIRTUAL_ENV=/app/.venv \
    AGENT_PLATFORM=gufo \
    AGENT_TYPE=gufo \
    AGENT_PORT=8080 \
    GPU_BACKEND=rocm \
    MODELS_PATH=/models \
    CACHE_PATH=/cache \
    GUFO_SERVER_PATH=/usr/local/bin/gufo \
    GUFO_MAX_INSTANCES=2

EXPOSE 8080

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["sh", "-c", ". ${VIRTUAL_ENV:-/app/.venv}/bin/activate && exec uvicorn app.main:app --host 0.0.0.0 --port \"${AGENT_PORT:-8080}\" --log-level info --ws-ping-interval 30 --ws-ping-timeout 60"]
