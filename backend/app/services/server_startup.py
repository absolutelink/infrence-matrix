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
from app.services.server_options import validate_server_options

logger = logging.getLogger(__name__)

# Cold start includes a possible model download (multi-GB GGUF), so the
# dispatch must allow far more than the default 30s agent timeout.
START_DISPATCH_TIMEOUT = 900.0
# How long we wait for an in-flight (or newly dispatched) start to become
# healthy before giving up on the request.
READY_TIMEOUT = 900.0
# A process that was healthy and became unhealthy should get a short recovery
# window, but must not hold an inference request for the full cold-start budget.
UNHEALTHY_RECOVERY_TIMEOUT = 30.0
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
    dflash = _find_dflash(instance, model)
    payload: dict[str, Any] = {
        "config": {
            "id": str(instance.id),
            "engine": instance.engine,
            "engine_options": instance.engine_options or {},
            "model_path": model.path,
            "gpu_layers": instance.gpu_layers,
            "context_size": instance.context_size,
            "batch_size": 512,
            "cache_prompt": True,
            "flash_attn": instance.flash_attn,
            "mtp_draft_max": instance.mtp_draft_max,
            "options": validate_server_options(instance.server_options or {}),
        },
        # The agent downloads the model file first if it is missing.
        "source": {
            "source": model.source,
            "repo_id": model.source_repo_id or "",
            "filename": filename,
            "job_id": f"server-{instance.id}",
        }
        if instance.engine != "halogen" and model.source_repo_id
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
    if dflash is not None:
        dflash_model, dflash_filename = dflash
        payload["config"]["draft_model_path"] = dflash_model.path
        if payload["config"]["options"].get("strict_mtp_qwen"):
            # Strict Qwen MTP is only valid with draft-MTP. A dflash model
            # selects draft-dflash below, so forwarding both flags makes
            # llama-server reject the model during startup.
            logger.warning(
                "Server %s: disabling strict_mtp_qwen for draft-dflash configuration",
                instance.id,
            )
            payload["config"]["options"] = {
                **payload["config"]["options"],
                "strict_mtp_qwen": False,
            }
        if dflash_model.source_repo_id:
            payload["draft_source"] = {
                "source": dflash_model.source,
                "repo_id": dflash_model.source_repo_id,
                "filename": dflash_filename,
                "job_id": f"server-{instance.id}-dflash",
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


def _find_dflash(instance: ServerInstance, model: Model) -> tuple[Model, str] | None:
    """Return a valid dflash draft model for the instance, if selected."""
    dflash_model = instance.dflash_model
    if not dflash_model:
        return None
    if dflash_model.model_type != "dflash" or dflash_model.path == model.path:
        return None
    filename = dflash_model.source_file or dflash_model.path.rsplit("/", 1)[-1]
    return dflash_model, filename


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


async def dispatch_prepare(
    agent_id: str, server_id: str, payload: dict[str, Any]
) -> None:
    """Prepare model files without changing the stopped server state."""
    try:
        await agent_manager.send_to_agent(
            agent_id,
            "POST",
            "/servers/prepare",
            payload,
            timeout=START_DISPATCH_TIMEOUT,
        )
        logger.info(f"Prepared files for server {server_id} on agent {agent_id}")
    except Exception as e:
        logger.warning(f"Failed to prepare files for server {server_id}: {e}")


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
        if instance.status == "running" and instance.health_status == "healthy":
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

    if instance.status == "running" and instance.health_status == "healthy":
        return instance
    if instance.status == "error":
        raise ServerStartupError(
            instance.error_message or f"Server {server.id} failed to start"
        )

    if instance.status in {"starting", "running"}:
        # A process may exist before its health check completes; wait for it
        # instead of dispatching a duplicate start.
        try:
            return await wait_until_ready(
                str(instance.id),
                timeout=(
                    UNHEALTHY_RECOVERY_TIMEOUT
                    if instance.health_status == "unhealthy"
                    else start_timeout
                ),
                poll_interval=poll_interval,
            )
        except RuntimeError as e:
            raise ServerStartupError(str(e)) from e

    # stopped or errored: (re)start it and wait for health.
    async with AsyncSessionMaker() as session:
        fresh = (
            await session.execute(
                select(ServerInstance)
                .where(ServerInstance.id == instance.id)
                .with_for_update()
            )
        ).scalar_one_or_none()
        if fresh is None:
            raise ServerStartupError("Server instance no longer exists")

        # Another request may have won the startup race while this request
        # was loading the row. Join that startup instead of dispatching twice.
        if fresh.status in {"starting", "running"}:
            server_id = str(fresh.id)
            should_start = False
        else:
            should_start = True
        agent = await session.get(Agent, fresh.agent_id) if fresh else None
        model = await session.get(Model, fresh.model_id) if fresh else None
        if fresh is None or agent is None or model is None:
            raise ServerStartupError(
                "Server instance, agent, or model no longer exists"
            )
        if not should_start:
            payload = None
        elif agent.status != "online":
            raise ServerStartupError(f"Agent {agent.name} is {agent.status}")

        if should_start:
            fresh.status = "starting"
            fresh.health_status = "unknown"
            fresh.error_message = None
            session.add(fresh)
            await session.commit()

            payload = build_start_payload(fresh, model)
            agent_id = str(agent.id)
            server_id = str(fresh.id)

    if not should_start:
        try:
            return await wait_until_ready(
                server_id, timeout=start_timeout, poll_interval=poll_interval
            )
        except RuntimeError as e:
            raise ServerStartupError(str(e)) from e

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
