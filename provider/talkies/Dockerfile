# talkies Inference Matrix provider (Phase 24).
#
# Build context = repo root, e.g.:
#   docker build -f provider/talkies/Dockerfile -t matrix-provider-talkies .
#
# Two Python stacks live side by side:
#   * /opt/venv          — the talkies speech server (python 3.12, baked in
#                          the upstream image; env TALKIES_PYTHON). The engine
#                          subprocess is launched from it.
#   * /opt/provider-venv — the Inference Matrix provider stack (python 3.14,
#                          required by provider_lib), installed with uv from a
#                          copied workspace subset. The container entrypoint.
#
# The heavy talkies ML layers are NEVER rebuilt here: this image only adds the
# provider venv on top of psyb0t/talkies:latest-cuda.
#
# The base image (Dockerfile.cuda) ends as `USER talkies` (uid 1000) with a
# root-owned /opt, an ENTRYPOINT that ignores "$@" and execs stock talkies,
# ENV TALKIES_DATA_DIR=/data + HF_HOME=/data/hf, and a HEALTHCHECK probing
# :8000 — every one of those is wrong for the agent, and each is overridden
# below deliberately (kept adjacent to the override so they stay auditable).
ARG BASE_IMAGE=psyb0t/talkies:latest-cuda
FROM ${BASE_IMAGE}

# The base's final USER is talkies (uid 1000); /opt is root-owned, so the
# venv/COPY layers below must run as root and hand a world-readable tree to
# the talkies user at the end.
USER root

# uv binary (managed python 3.14 download happens in the build).
COPY --from=ghcr.io/astral-sh/uv /uv /usr/local/bin/uv

# Provider sources (workspace subset: lib + this package only).
COPY provider/lib /opt/matrix/provider/lib
COPY provider/talkies /opt/matrix/provider/talkies

RUN uv venv --python 3.14 /opt/provider-venv \
    && uv pip install --python /opt/provider-venv/bin/python \
        /opt/matrix/provider/lib /opt/matrix/provider/talkies \
    && chmod -R a+rX /opt/provider-venv /opt/matrix

# The agent runs as talkies (uid 1000) and writes under CACHE_DIR (/cache:
# provider_config.json) and MODELS_DIR (/models: prefetched snapshots,
# staged files, custom voices). The base image never creates these paths,
# so a FRESH named volume would be seeded root-owned by Docker and every
# write would fail — pre-create them owned by talkies. NOTE: volumes that
# already exist from a pre-fix deploy keep their old ownership; operators
# must `chown -R 1000:1000` on the host (or remove the volume) once.
RUN mkdir -p /models /cache && chown -R talkies:talkies /models /cache

# Agent wiring. The agent derives everything else from the standard provider
# env (MACHINE_UID / MACHINE_SECRET / AGENT_ID / ADMIN_BASE_URL / CACHE_DIR /
# MODELS_DIR).
#   TALKIES_PYTHON      — talkies venv interpreter (engine subprocess).
#   TALKIES_MODELS_FILE — bundled registry JSON baked by the talkies image.
#   TALKIES_DATA_DIR    — BLANKED on purpose: the base exports /data, which
#                         is an anonymous container path the matrix compose
#                         never mounts. Empty => the driver derives the data
#                         root from MODELS_DIR (the mounted models volume),
#                         so model snapshots, staged files, and custom
#                         voices persist across recreations. The driver also
#                         re-points HF_HOME to $DATA/hf for the engine.
ENV PATH="/opt/provider-venv/bin:$PATH" \
    PROVIDER_PORT=8081 \
    TALKIES_PYTHON=/opt/venv/bin/python \
    TALKIES_MODELS_FILE=/app/models.json \
    TALKIES_DATA_DIR=

# The base HEALTHCHECK probes http://127.0.0.1:8000/healthz (stock talkies).
# The agent never serves :8000 — its liveness is /health on PROVIDER_PORT,
# which the admin polls over the WS control plane. Disable the inherited
# probe so the container is not permanently "unhealthy".
HEALTHCHECK NONE

USER talkies

# The base ENTRYPOINT (talkies-entrypoint) ignores "$@" and execs stock
# talkies on 0.0.0.0:8000 — replace it with the provider agent outright.
ENTRYPOINT ["/opt/provider-venv/bin/python", "-m", "provider_talkies.main"]
