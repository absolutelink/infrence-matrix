# Inference Matrix Agent with llama.cpp Vulkan support
#
# Instead of compiling llama.cpp, this recipe layers the agent application
# on top of the official llama.cpp Vulkan image. AGENT_IMAGE must point at
# an agent image built by the build-agent job (e.g. ghcr.io/<owner>/agent:main).
ARG AGENT_IMAGE=ghcr.io/absolutelink/agent:main

FROM ${AGENT_IMAGE} AS agent

FROM ghcr.io/ggml-org/llama.cpp:full-vulkan

USER root

# Copy the agent application from the agent image
COPY --from=agent /app/pyproject.toml /agent/pyproject.toml
COPY --from=agent /app/uv.lock /agent/uv.lock
COPY --from=agent /app/app /agent/app

# Install the agent's Python dependencies (uv ships with the llama.cpp image)
WORKDIR /agent
RUN uv sync --frozen --no-dev

ENV PATH="/agent/.venv/bin:${PATH}"

# Entrypoint script (shared with matrix-app) and entrypoint.d folder for
# user-added scripts
COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh \
    && mkdir -p /etc/entrypoint.d /opt/rootfs /models /cache

# Agent runtime configuration
ENV LLAMA_SERVER_PATH=/app/llama-server \
    LLAMA_BENCH_PATH=/app/llama-bench \
    AGENT_PLATFORM=llamacpp \
    AGENT_TYPE=vulkan \
    AGENT_PORT=8080 \
    GPU_BACKEND=vulkan \
    MODELS_PATH=/models \
    CACHE_PATH=/cache

EXPOSE 8080

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]

# Run the agent
CMD ["sh", "-c", "exec uvicorn app.main:app --host 0.0.0.0 --port \"${AGENT_PORT:-8080}\" --log-level info"]
