# Inference Matrix Agent with the q38rocm ROCmFP4 engine.
# AGENT_IMAGE must point at an image built by the agent job.
ARG AGENT_IMAGE=ghcr.io/absolutelink/agent:main

FROM ${AGENT_IMAGE} AS agent

FROM ghcr.io/julianmb/q38rocm:latest

USER root

COPY --from=agent /app/pyproject.toml /agent/pyproject.toml
COPY --from=agent /app/uv.lock /agent/uv.lock
COPY --from=agent /app/app /agent/app

# q38rocm provides pip but not uv. Install the agent dependencies using the
# lockfile rather than reusing the Python environment from the base image.
RUN pip3 install --no-cache-dir --break-system-packages uv
WORKDIR /agent
RUN uv sync --frozen --no-dev

ENV PATH="/agent/.venv/bin:${PATH}"

COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh \
    && mkdir -p /etc/entrypoint.d /opt/rootfs /models /cache

ENV LLAMA_SERVER_PATH=/app/engine/bin/llama-server \
    LLAMA_BENCH_PATH=/app/engine/bin/llama-bench \
    AGENT_PLATFORM=llamacpp \
    AGENT_TYPE=ROCmFP4_FAST \
    AGENT_PORT=8080 \
    GPU_BACKEND=rocm \
    MODELS_PATH=/models \
    CACHE_PATH=/cache

EXPOSE 8080

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["sh", "-c", "exec uvicorn app.main:app --host 0.0.0.0 --port \"${AGENT_PORT:-8080}\" --log-level info"]
