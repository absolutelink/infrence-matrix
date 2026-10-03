"""Shared OpenAI model and server target resolution."""

from dataclasses import dataclass
from uuid import UUID

from sqlmodel import Session, select

from app.models import Model, ServerInstance
from app.services.npu_aliases import NPU_SUFFIXES


@dataclass(frozen=True)
class InferenceTarget:
    """Resolved model plus optional scheduler restrictions."""

    model: Model
    preferred_server_id: UUID | None = None
    required_agent_id: UUID | None = None
    server: ServerInstance | None = None
    # When set, the request targets one of the flash instance's NPU small
    # models; the proxied ``model`` field must be rewritten to this upstream
    # id so halogen routes to the NPU instead of the Flash engine.
    npu_model: str | None = None


def _npu_virtual_alias(
    session: Session, model_ref: str
) -> tuple[ServerInstance, str] | None:
    """Resolve ``<flash-alias>-<suffix>`` to (instance, upstream NPU id).

    Instance aliases are checked first by the caller, so a real alias always
    wins over a virtual NPU name. Ordering prefers a running instance, then
    starting, then stopped, so two instances enabling the same NPU model
    resolve deterministically.
    """
    for upstream_id, suffix in NPU_SUFFIXES.items():
        if not model_ref.endswith(f"-{suffix}"):
            continue
        alias = model_ref[: -(len(suffix) + 1)]
        if not alias:
            continue
        statement = select(ServerInstance).where(
            ServerInstance.alias == alias,
            ServerInstance.engine == "halogen-flash",
            ServerInstance.status.in_(["running", "starting", "stopped"]),
        )
        priority = {"running": 0, "starting": 1, "stopped": 2}
        candidates = sorted(
            session.exec(statement),
            key=lambda inst: priority.get(inst.status, 3),
        )
        for instance in candidates:
            enabled = (instance.engine_options or {}).get("npu_models") or []
            if upstream_id in enabled:
                return instance, upstream_id
        return None
    return None


def resolve_inference_target(
    session: Session,
    model_ref: str,
    agent_id: str | None = None,
) -> InferenceTarget:
    """Resolve an alias, model name, or model ID without pinning replicas."""
    server = session.exec(
        select(ServerInstance).where(
            ServerInstance.alias == model_ref,
            ServerInstance.status.in_(["starting", "running", "stopped"]),
        )
    ).first()
    if server is not None:
        model = session.get(Model, server.model_id)
        if model is None:
            raise LookupError(f"Model for server alias {model_ref} not found")
        return InferenceTarget(
            model=model,
            preferred_server_id=server.id,
            required_agent_id=UUID(agent_id) if agent_id else None,
            server=server,
        )

    npu_hit = _npu_virtual_alias(session, model_ref)
    if npu_hit is not None:
        flash_instance, upstream_id = npu_hit
        model = session.get(Model, flash_instance.model_id)
        if model is None:
            raise LookupError(f"Model for NPU host {flash_instance.alias} not found")
        return InferenceTarget(
            model=model,
            preferred_server_id=flash_instance.id,
            required_agent_id=UUID(agent_id) if agent_id else None,
            server=flash_instance,
            npu_model=upstream_id,
        )

    model = session.exec(select(Model).where(Model.name == model_ref)).first()
    if model is None:
        try:
            model = session.get(Model, UUID(model_ref))
        except ValueError:
            model = None
    if model is None:
        raise LookupError(f"Model {model_ref} not found")
    return InferenceTarget(
        model=model,
        required_agent_id=UUID(agent_id) if agent_id else None,
    )
