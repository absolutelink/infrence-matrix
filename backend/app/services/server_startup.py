"""Shared server startup logic: make cold starts invisible to inference clients.

Given a ServerInstance (or a model), ensure a llama-server is running and
healthy before the request is proxied:

- running/healthy  -> return immediately
- starting         -> wait until the running server becomes healthy (dedup:
                      parallel requests share one startup)
- stopped/errored  -> dispatch start to the agent and wait
- no instance      -> caller creates one (v1_chat_completions._get_or_create_server)

The client connection is held the whole time (await inside the request
handler / streaming generator), so the user sees nothing except a slightly
longer first-token latency.
"""

import asyncio
import logging
import uuid as uuid_module
from datetime import UTC, datetime
from typing import Any

from sqlmodel import col, select

from app.db.session import AsyncSessionMaker
from app.models import Agent, Model, ServerInstance
from app.services.agent_manager import agent_manager
from app.services.benchmark import is_benchmark_blocking

logger = logging.getLogger(__name__)

# Cold start includes a possible model download (multi-GB GGUF), so the
# dispatch must allow far more than the default 30s agent timeout.
START_DISPATCH_TIMEOUT = 900.0
# How long we wait for an in-flight (or newly dispatched) start to become
# healthy before giving up on the request.
READY_TIMEOUT = 900.0
READY_POLL_INTERVAL = 1.0


def build_start_payload(instance: ServerInstance, model: Model) -> dict[str, Any]:
    """Build the agent /servers/start payload from instance + model.

    The agent allocates the port (random free one) when none is sent.
    When the instance has an mmproj projector selected it is attached so
    llama-server loads it (--mmproj); with none selected the flag is
    omitted entirely.
    """
    filename = model.source_file or model.path.rsplit("/", 1)[-1]
    mmproj = _find_mmproj(instance, model)
    payload: dict[str, Any] = {
        "config": {
            "id": str(instance.id),
            "model_path": model.path,
            "gpu_layers": instance.gpu_layers,
            "context_size": instance.context_size,
            "batch_size": 512,
            "cache_prompt": True,
            "flash_attn": instance.flash_attn,
            "mtp_draft_max": instance.mtp_draft_max,
        },
        # The agent downloads the model file first if it is missing.
        "source": {
            "source": model.source,
            "repo_id": model.source_repo_id or "",
            "filename": filename,
            "job_id": f"server-{instance.id}",
        }
        if model.source_repo_id
        else None,
    }
    if mmproj is not None:
        mmproj_model, mmproj_filename = mmproj
        payload["config"]["mmproj_path"] = mmproj_model.path
        # Download the projector alongside the main model when missing.
        # Send the source whenever the projector has a repo — even for
        # repo-style "/models/{repo}/{file}" paths: without it the agent
        # falls back to the main model's source and "downloads" the main
        # GGUF as the projector (llama-server then fails to load it as a
        # CLIP model).
        if mmproj_model.source_repo_id:
            payload["mmproj_source"] = {
                "source": mmproj_model.source,
                "repo_id": mmproj_model.source_repo_id,
                "filename": mmproj_filename,
                "job_id": f"server-{instance.id}-mmproj",
            }
    return payload


def _find_mmproj(instance: ServerInstance, model: Model) -> tuple[Model, str] | None:
    """mmproj projector for a server instance, else None.

    The instance row is the source of truth: whatever projector was
    selected in the UI (start/edit dialog) is used, None means no
    --mmproj flag.
    """
    mmproj_model = instance.mmproj_model
    if not mmproj_model:
        return None
    # Guard against a mis-registered projector that points at the main
    # GGUF (llama-server fails to load it as a CLIP model).
    if mmproj_model.path == model.path or mmproj_model.id == model.id:
        logger.warning(
            f"Server {instance.id}: mmproj selection {mmproj_model.name} "
            "points at the main model file; ignoring it"
        )
        return None
    filename = mmproj_model.source_file or mmproj_model.path.rsplit("/", 1)[-1]
    return mmproj_model, filename


async def dispatch_start(
    agent_id: str, server_id: str, payload: dict[str, Any]
) -> None:
    """Send the start command to the agent, marking failure on the row."""
    try:
        response = await agent_manager.send_to_agent(
            agent_id,
            "POST",
            "/servers/start",
            payload,
            timeout=START_DISPATCH_TIMEOUT,
        )
        # The agent returned success (llama-server healthy). Mark running in
        # case the server.started event was lost (e.g. backend restart). The
        # agent-allocated port is echoed back and kept in the JSON config
        # for display only (not a column).
        allocated_port = response.get("port") if isinstance(response, dict) else None
        async with AsyncSessionMaker() as session:
            server = await session.get(ServerInstance, uuid_module.UUID(server_id))
            if server and server.status != "running":
                server.status = "running"
                server.health_status = "healthy"
                server.started_at = server.started_at or datetime.now(UTC)
                if allocated_port:
                    server.config = {
                        **(server.config or {}),
                        "port": str(allocated_port),
                    }
                session.add(server)
                await session.commit()
                logger.info(f"Server {server_id} marked as running (dispatch ack)")
    except Exception as e:
        logger.error(f"Failed to start server {server_id} on agent {agent_id}: {e}")
        async with AsyncSessionMaker() as session:
            server = await session.get(ServerInstance, uuid_module.UUID(server_id))
            if server:
                server.status = "error"
                server.error_message = str(e)
                session.add(server)
                await session.commit()


async def _reload_instance(server_id: str) -> ServerInstance | None:
    async with AsyncSessionMaker() as session:
        return await session.get(ServerInstance, uuid_module.UUID(server_id))


async def wait_until_ready(
    server_id: str,
    *,
    timeout: float = READY_TIMEOUT,
    poll_interval: float = READY_POLL_INTERVAL,
) -> ServerInstance:
    """Wait until the instance row reports running/healthy (or fail).

    Raises RuntimeError when the server errored during startup or the
    timeout elapsed.
    """
    deadline = asyncio.get_event_loop().time() + timeout
    while True:
        instance = await _reload_instance(server_id)
        if instance is None:
            raise RuntimeError(f"Server instance {server_id} no longer exists")
        if instance.status == "running":
            return instance
        if instance.status == "error":
            raise RuntimeError(
                instance.error_message or f"Server {server_id} failed to start"
            )
        if asyncio.get_event_loop().time() > deadline:
            raise RuntimeError(
                f"Server {server_id} did not become healthy within {int(timeout)}s"
            )
        await asyncio.sleep(poll_interval)


class ServerStartupError(Exception):
    """Raised when a server cannot be made ready for a request."""


async def ensure_server_ready(
    server: ServerInstance,
    *,
    start_timeout: float = START_DISPATCH_TIMEOUT,
    poll_interval: float = READY_POLL_INTERVAL,
) -> ServerInstance:
    """Ensure the server is running and healthy before proxying.

    Returns the (possibly refreshed) ServerInstance to use.
    Raises ServerStartupError with a user-presentable message on failure.
    """
    if await is_benchmark_blocking():
        raise ServerStartupError("Servers are unavailable while a benchmark is running")
    instance = await _reload_instance(str(server.id))
    if instance is None:
        raise ServerStartupError(f"Server {server.id} no longer exists")

    if instance.status == "running":
        return instance

    if instance.status == "starting":
        # Someone else's startup is in flight; wait for it.
        try:
            return await wait_until_ready(
                str(instance.id),
                timeout=start_timeout,
                poll_interval=poll_interval,
            )
        except RuntimeError as e:
            raise ServerStartupError(str(e)) from e

    # stopped or errored: (re)start it and wait for health.
    async with AsyncSessionMaker() as session:
        fresh = await session.get(ServerInstance, instance.id)
        agent = await session.get(Agent, fresh.agent_id) if fresh else None
        model = await session.get(Model, fresh.model_id) if fresh else None
        if fresh is None or agent is None or model is None:
            raise ServerStartupError(
                "Server instance, agent, or model no longer exists"
            )
        if agent.status != "online":
            raise ServerStartupError(f"Agent {agent.name} is {agent.status}")

        fresh.status = "starting"
        fresh.error_message = None
        session.add(fresh)
        await session.commit()

        payload = build_start_payload(fresh, model)
        agent_id = str(agent.id)
        server_id = str(fresh.id)

    logger.info(f"Auto-starting server {server_id} (was {instance.status})")
    asyncio.create_task(dispatch_start(agent_id, server_id, payload))

    try:
        return await wait_until_ready(
            server_id, timeout=start_timeout, poll_interval=poll_interval
        )
    except RuntimeError as e:
        raise ServerStartupError(str(e)) from e


async def find_alias_instance(
    alias: str, *, active_only: bool = False
) -> ServerInstance | None:
    """Find a server instance by alias.

    active_only=True keeps the old behaviour (starting/running only);
    the default includes stopped/errored so requests can auto-start them.
    """
    stmt = select(ServerInstance).where(ServerInstance.alias == alias)
    if active_only:
        stmt = stmt.where(col(ServerInstance.status).in_(["starting", "running"]))
    async with AsyncSessionMaker() as session:
        result = await session.execute(stmt)
        return result.scalars().first()


async def ensure_server_ready_by_id(
    server_id: str,
    *,
    start_timeout: float = START_DISPATCH_TIMEOUT,
    poll_interval: float = READY_POLL_INTERVAL,
) -> ServerInstance:
    """ensure_server_ready for callers that only hold the instance id
    (e.g. streaming generators)."""
    instance = await _reload_instance(server_id)
    if instance is None:
        raise ServerStartupError(f"Server {server_id} no longer exists")
    return await ensure_server_ready(
        instance, start_timeout=start_timeout, poll_interval=poll_interval
    )
