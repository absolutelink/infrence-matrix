"""Shared OpenAI model and server target resolution."""

from dataclasses import dataclass
from uuid import UUID

from sqlmodel import Session, select

from app.models import Model, ServerInstance


@dataclass(frozen=True)
class InferenceTarget:
    """Resolved model plus optional scheduler restrictions."""

    model: Model
    preferred_server_id: UUID | None = None
    required_agent_id: UUID | None = None
    server: ServerInstance | None = None


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
