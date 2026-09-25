# Inference Matrix Agent with one Halogen process per server instance.
ARG AGENT_IMAGE

FROM ${AGENT_IMAGE} AS agent

FROM ghcr.io/peonist-ai/halogen:0.1.4

USER root

COPY --from=agent /app/pyproject.toml /agent/pyproject.toml
COPY --from=agent /app/uv.lock /agent/uv.lock
COPY --from=agent /app/app /agent/app
COPY recipes/halogen-rocm/docker/matrix-entrypoint.sh /usr/local/bin/matrix-entrypoint.sh

RUN pip3 install --no-cache-dir --break-system-packages uv \
    && chmod +x /usr/local/bin/matrix-entrypoint.sh \
    && mkdir -p /agent /models /cache

WORKDIR /agent
RUN uv sync --frozen --no-dev

ENV PATH="/agent/.venv/bin:${PATH}" \
    AGENT_PLATFORM=halogen \
    AGENT_TYPE=rocm \
    AGENT_PORT=8080 \
    GPU_BACKEND=rocm \
    MODELS_PATH=/models \
    CACHE_PATH=/cache \
    HF_HUB_OFFLINE=0 \
    HF_HUB_DISABLE_XET=1 \
    HALOGEN_ENTRYPOINT=/usr/local/bin/entrypoint.sh \
    HALOGEN_MAX_INSTANCES=2

EXPOSE 8080

ENTRYPOINT ["/usr/local/bin/matrix-entrypoint.sh"]
CMD ["sh", "-c", "exec uvicorn app.main:app --host 0.0.0.0 --port \"${AGENT_PORT:-8080}\" --log-level info"]
