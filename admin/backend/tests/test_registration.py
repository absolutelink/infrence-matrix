"""Agent registration endpoint validation and upsert behavior (Phase 16)."""

import hashlib
import json
import uuid

import pytest
from fastapi.testclient import TestClient
from sqlmodel import Session, select

from app.api.admin.providers import compute_config_fingerprint
from app.core.config import settings
from app.models import Machine, ProviderAgent, ProviderDefinition, ProviderInstance
from app.services import redis_keys


def _machine(
    session: Session, uid: str = "mach-1", secret: str = "mach-secret"
) -> Machine:
    m = Machine(
        uid=uid,
        name=f"name-{uid}",
        host="10.0.0.5",
        dns="box.local",
        registration_secret=secret,
    )
    session.add(m)
    session.commit()
    session.refresh(m)
    return m


def _definition(
    session: Session,
    *,
    provider_type: str = "mock",
    enabled: bool = True,
    backend_config: dict | None = None,
    alias: str | None = None,
) -> ProviderDefinition:
    d = ProviderDefinition(
        alias=alias or f"alias-{uuid.uuid4().hex[:8]}",
        provider_type=provider_type,
        backend_config=backend_config
        or {"model": {"file": "m.gguf"}, "args": {"ctx": 4096}},
        vram_required_bytes=8 * 1024**3,
        idle_timeout_seconds=123,
        capacity=2,
        enabled=enabled,
    )
    session.add(d)
    session.commit()
    session.refresh(d)
    return d


def _body(**overrides) -> dict:
    body = {
        "machine_uid": "mach-1",
        "machine_secret": "mach-secret",
        "agent_id": "agent-1",
        "provider_type": "mock",
        "schema": {"type": "object"},
        "version": settings.VERSION,
        "base_port": 8081,
        "hardware": {
            "gpus": [{"uuid": "gpu-1", "total_vram_bytes": 24 * 1024**3}],
            "total_vram_bytes": 24 * 1024**3,
        },
        "metrics_categories": ["vram"],
        "registered_at": "2026-01-01T00:00:00+00:00",
    }
    body.update(overrides)
    return body


@pytest.fixture
def client() -> TestClient:
    from app.main import app

    with TestClient(app) as c:
        yield c


def test_register_success(client: TestClient, session: Session, clean_redis) -> None:
    machine = _machine(session)
    definition = _definition(session)

    resp = client.post("/admin/api/providers/register", json=_body())
    assert resp.status_code == 200, resp.text
    data = resp.json()

    assert data["agent_id"]
    assert len(data["agent_secret"]) >= 32
    assert data["machine"]["uid"] == machine.uid
    assert len(data["backends"]) == 1
    backend = data["backends"][0]
    assert backend["definition"]["alias"] == definition.alias
    assert backend["definition"]["provider_type"] == "mock"
    assert backend["definition"]["modality"] == "llm"  # Phase 18 slice 3
    assert backend["definition"]["idle_timeout_seconds"] == 123
    assert backend["definition"]["capacity"] == 2
    assert backend["definition"]["vram_required_bytes"] == 8 * 1024**3
    assert "port" not in backend  # port model overhaul: no per-backend port

    expected_fp = hashlib.sha256(
        json.dumps(
            definition.backend_config, sort_keys=True, separators=(",", ":")
        ).encode()
    ).hexdigest()
    assert backend["definition"]["config_fingerprint"] == expected_fp
    assert compute_config_fingerprint(definition.backend_config) == expected_fp

    agent = session.exec(select(ProviderAgent)).one()
    assert str(agent.id) == data["agent_id"]
    assert agent.machine_id == machine.id
    assert agent.provider_type == "mock"
    assert agent.agent_id == "agent-1"
    assert agent.base_port == 8081
    assert agent.version == settings.VERSION
    assert agent.agent_status == "registering"

    inst = session.exec(select(ProviderInstance)).one()
    assert str(inst.id) == backend["instance_id"]
    assert inst.agent_id == agent.id
    assert inst.provider_definition_id == definition.id
    assert inst.config_fingerprint == expected_fp

    # Secret lives in Redis (keyed by agent id), not Postgres.
    stored = clean_redis.get(redis_keys.secret_key(data["agent_id"]))
    assert stored == data["agent_secret"]

    session.refresh(machine)
    assert machine.hardware == _body()["hardware"]
    assert machine.total_vram_bytes == 24 * 1024**3


def test_register_definition_dict_carries_modality(
    client: TestClient, session: Session
) -> None:
    """Phase 18 slice 3: the registration ``backends[].definition`` carries the
    definition's modality so the provider learns the endpoint kind at boot."""
    _machine(session)
    emb = ProviderDefinition(
        alias="emb-reg",
        provider_type="mock",
        backend_config={"model": {"file": "e.gguf"}},
        modality="embedding",
    )
    session.add(emb)
    session.commit()

    resp = client.post("/admin/api/providers/register", json=_body())
    assert resp.status_code == 200, resp.text
    entry = next(
        b for b in resp.json()["backends"] if b["definition"]["alias"] == "emb-reg"
    )
    assert entry["definition"]["modality"] == "embedding"


def test_register_reregister_reuses_rows(client: TestClient, session: Session) -> None:
    _machine(session)
    _definition(session)

    first = client.post(
        "/admin/api/providers/register", json=_body(base_port=9000)
    ).json()
    second = client.post(
        "/admin/api/providers/register", json=_body(base_port=9100)
    ).json()

    assert first["agent_id"] == second["agent_id"]
    assert first["agent_secret"] != second["agent_secret"]
    assert len(session.exec(select(ProviderAgent)).all()) == 1
    assert len(session.exec(select(ProviderInstance)).all()) == 1
    # The single agent row is reused and its env base_port updated to the latest
    # registration (backends no longer carry their own port).
    agent = session.exec(select(ProviderAgent)).one()
    assert agent.base_port == 9100


def test_register_hosts_all_enabled_definitions_of_type(
    client: TestClient, session: Session
) -> None:
    _machine(session)
    _definition(session, alias="a1")
    _definition(session, alias="a2")
    _definition(session, alias="a3", enabled=False)
    _definition(session, alias="other", provider_type="llama-cpp")

    resp = client.post("/admin/api/providers/register", json=_body())
    assert resp.status_code == 200
    data = resp.json()
    aliases = sorted(b["definition"]["alias"] for b in data["backends"])
    assert aliases == ["a1", "a2"]
    # Port model overhaul: backends carry no per-backend port; the agent
    # publishes a single env base_port and routes by model.
    assert all("port" not in b for b in data["backends"])


def test_register_mixed_placement_union(
    client: TestClient, session: Session
) -> None:
    """B2: an agent's placed set = (all enabled any_of_type of its type) ∪
    (enabled specific defs linked to THIS agent). A def placed specifically
    on agent A must not leak onto agent B."""
    from app.models import DefinitionAgent

    _machine(session)
    any_def = _definition(session, alias="any-of-type")
    spec_def = _definition(session, alias="specific-on-a")

    # Agent A registers first (creates its ProviderAgent row).
    a = client.post(
        "/admin/api/providers/register", json=_body(agent_id="agent-A")
    ).json()
    agent_a_id = uuid.UUID(a["agent_id"])
    # Place spec_def specifically on agent A.
    session.add(
        DefinitionAgent(provider_definition_id=spec_def.id, agent_id=agent_a_id)
    )
    spec_def.agent_placement = "specific"
    session.add(spec_def)
    session.commit()

    # Re-register A: gets BOTH the any_of_type def and its specific def.
    a2 = client.post(
        "/admin/api/providers/register", json=_body(agent_id="agent-A")
    ).json()
    a_aliases = sorted(b["definition"]["alias"] for b in a2["backends"])
    assert a_aliases == ["any-of-type", "specific-on-a"]

    # Agent B (same type, no links): gets ONLY the any_of_type def — the
    # def placed specifically on A must not leak here.
    b = client.post(
        "/admin/api/providers/register", json=_body(agent_id="agent-B")
    ).json()
    b_aliases = [b_["definition"]["alias"] for b_ in b["backends"]]
    assert b_aliases == ["any-of-type"]
    assert "specific-on-a" not in b_aliases


def test_register_prunes_deplaced_backend(
    client: TestClient, session: Session
) -> None:
    """M2: when a definition is no longer placed on an agent, re-registration
    deletes the ghost ProviderInstance row so it can never be scheduled."""
    _machine(session)
    d = _definition(session, alias="ghost-me")

    first = client.post(
        "/admin/api/providers/register", json=_body(agent_id="agent-A")
    ).json()
    assert len(first["backends"]) == 1
    inst_id = uuid.UUID(first["backends"][0]["instance_id"])
    assert session.get(ProviderInstance, inst_id) is not None

    # Disable the definition → it is no longer placed anywhere.
    d.enabled = False
    session.add(d)
    session.commit()

    second = client.post(
        "/admin/api/providers/register", json=_body(agent_id="agent-A")
    ).json()
    assert second["backends"] == []
    session.expire_all()
    assert session.get(ProviderInstance, inst_id) is None


def test_register_prune_skips_running_backend(
    client: TestClient, session: Session
) -> None:
    """MEDIUM (round 2): a de-placed backend that is still running/in_use is
    NOT pruned (its engine holds VRAM + the scheduler's per-booted hold is
    keyed by its id); once stopped it is pruned on the next registration."""
    _machine(session)
    d = _definition(session, alias="running-ghost")

    first = client.post(
        "/admin/api/providers/register", json=_body(agent_id="agent-A")
    ).json()
    inst_id = uuid.UUID(first["backends"][0]["instance_id"])

    # Mark it running, then de-place it.
    inst = session.get(ProviderInstance, inst_id)
    inst.backend_status = "running"
    session.add(inst)
    session.commit()
    d.enabled = False
    session.add(d)
    session.commit()

    client.post("/admin/api/providers/register", json=_body(agent_id="agent-A"))
    session.expire_all()
    # Still there — running backends are never pruned out from under the engine.
    assert session.get(ProviderInstance, inst_id) is not None

    # Now stop it and re-register: the ghost is finally removed.
    inst = session.get(ProviderInstance, inst_id)
    inst.backend_status = "stopped"
    session.add(inst)
    session.commit()
    client.post("/admin/api/providers/register", json=_body(agent_id="agent-A"))
    session.expire_all()
    assert session.get(ProviderInstance, inst_id) is None


def test_register_unknown_machine(client: TestClient, session: Session) -> None:
    _definition(session)
    resp = client.post(
        "/admin/api/providers/register", json=_body(machine_uid="ghost-machine")
    )
    assert resp.status_code == 404


def test_register_wrong_machine_secret(client: TestClient, session: Session) -> None:
    _machine(session, secret="correct")
    _definition(session)
    resp = client.post(
        "/admin/api/providers/register", json=_body(machine_secret="wrong")
    )
    assert resp.status_code == 401


def test_register_version_mismatch(client: TestClient, session: Session) -> None:
    _machine(session)
    _definition(session)
    resp = client.post(
        "/admin/api/providers/register",
        json=_body(version="definitely-not-" + settings.VERSION),
    )
    assert resp.status_code == 409
    assert "version mismatch" in resp.json()["detail"]


def test_register_hardware_total_is_auto_sum(
    client: TestClient, session: Session
) -> None:
    """The union sum is authoritative — a mismatched reported total is ignored."""
    machine = _machine(session)
    machine.total_vram_bytes = 99 * 1024**3
    session.add(machine)
    session.commit()
    _definition(session)

    body = _body(
        hardware={
            "gpus": [{"uuid": "gpu-1", "total_vram_bytes": 24 * 1024**3}],
            "total_vram_bytes": 1,  # bogus: must be overridden by the GPU sum
        }
    )
    resp = client.post("/admin/api/providers/register", json=body)
    assert resp.status_code == 200
    session.refresh(machine)
    assert machine.total_vram_bytes == 24 * 1024**3
    assert machine.hardware["total_vram_bytes"] == 24 * 1024**3


def test_register_two_agents_union_gpus(client: TestClient, session: Session) -> None:
    """Two device-isolated agents each see one GPU → the machine shows the union."""
    machine = _machine(session)
    _definition(session)

    r1 = client.post(
        "/admin/api/providers/register",
        json=_body(
            agent_id="a1",
            hardware={
                "gpus": [{"uuid": "gpu-1", "total_vram_bytes": 24 * 1024**3}],
                "total_vram_bytes": 24 * 1024**3,
            },
        ),
    )
    assert r1.status_code == 200
    r2 = client.post(
        "/admin/api/providers/register",
        json=_body(
            agent_id="a2",
            hardware={
                "gpus": [{"uuid": "gpu-2", "total_vram_bytes": 16 * 1024**3}],
                "total_vram_bytes": 16 * 1024**3,
            },
        ),
    )
    assert r2.status_code == 200

    session.refresh(machine)
    assert {g["uuid"] for g in machine.hardware["gpus"]} == {"gpu-1", "gpu-2"}
    assert machine.total_vram_bytes == (24 + 16) * 1024**3
    assert machine.hardware["total_vram_bytes"] == (24 + 16) * 1024**3


def test_register_reregister_drops_stale_gpu_keeps_other_agent(
    client: TestClient, session: Session
) -> None:
    """Re-registering an agent with a changed GPU set drops only ITS stale GPUs."""
    machine = _machine(session)
    _definition(session)
    client.post(
        "/admin/api/providers/register",
        json=_body(
            agent_id="a1",
            hardware={
                "gpus": [{"uuid": "gpu-1", "total_vram_bytes": 24 * 1024**3}],
                "total_vram_bytes": 24 * 1024**3,
            },
        ),
    )
    client.post(
        "/admin/api/providers/register",
        json=_body(
            agent_id="a2",
            hardware={
                "gpus": [{"uuid": "gpu-2", "total_vram_bytes": 16 * 1024**3}],
                "total_vram_bytes": 16 * 1024**3,
            },
        ),
    )
    # a1 now sees gpu-3 instead of gpu-1 (e.g. the container was re-pinned).
    client.post(
        "/admin/api/providers/register",
        json=_body(
            agent_id="a1",
            hardware={
                "gpus": [{"uuid": "gpu-3", "total_vram_bytes": 8 * 1024**3}],
                "total_vram_bytes": 8 * 1024**3,
            },
        ),
    )

    session.refresh(machine)
    assert {g["uuid"] for g in machine.hardware["gpus"]} == {"gpu-2", "gpu-3"}
    assert machine.total_vram_bytes == (16 + 8) * 1024**3


def test_register_assigned_gpus_are_uuid_strings(
    client: TestClient, session: Session
) -> None:
    """assigned_gpus is normalized to a list of uuid strings, not GPU dicts."""
    _machine(session)
    _definition(session)
    resp = client.post("/admin/api/providers/register", json=_body())
    assert resp.status_code == 200
    agent = session.exec(select(ProviderAgent)).one()
    assert agent.assigned_gpus == ["gpu-1"]
    assert all(isinstance(x, str) for x in agent.assigned_gpus)


def test_register_missing_gpus_key_preserves_existing_union(
    client: TestClient, session: Session
) -> None:
    """A report with NO gpus key never erases GPUs another agent contributed."""
    machine = _machine(session)
    _definition(session)
    client.post(
        "/admin/api/providers/register",
        json=_body(
            agent_id="a1",
            hardware={
                "gpus": [{"uuid": "gpu-1", "total_vram_bytes": 24 * 1024**3}],
                "total_vram_bytes": 24 * 1024**3,
            },
        ),
    )
    # a2 reports no gpus key at all, only a cpu section.
    r2 = client.post(
        "/admin/api/providers/register",
        json=_body(agent_id="a2", hardware={"cpu": {"cores": 8}}),
    )
    assert r2.status_code == 200

    session.refresh(machine)
    assert [g["uuid"] for g in machine.hardware["gpus"]] == ["gpu-1"]
    assert machine.total_vram_bytes == 24 * 1024**3
    assert machine.hardware["cpu"] == {"cores": 8}
    a2 = session.exec(
        select(ProviderAgent).where(ProviderAgent.agent_id == "a2")
    ).one()
    assert a2.assigned_gpus == []


def test_register_empty_gpus_list_recomputes_surviving_union(
    client: TestClient, session: Session
) -> None:
    """A report WITH an empty gpus list against an existing union recomputes to
    the surviving sum (the union base survives; total is the surviving auto-sum).
    """
    machine = _machine(session)
    _definition(session)
    client.post(
        "/admin/api/providers/register",
        json=_body(
            agent_id="a1",
            hardware={
                "gpus": [{"uuid": "gpu-1", "total_vram_bytes": 24 * 1024**3}],
                "total_vram_bytes": 24 * 1024**3,
            },
        ),
    )
    r2 = client.post(
        "/admin/api/providers/register",
        json=_body(agent_id="a2", hardware={"gpus": [], "total_vram_bytes": 0}),
    )
    assert r2.status_code == 200

    session.refresh(machine)
    assert [g["uuid"] for g in machine.hardware["gpus"]] == ["gpu-1"]
    assert machine.total_vram_bytes == 24 * 1024**3
    assert machine.hardware["total_vram_bytes"] == 24 * 1024**3


def test_register_no_gpus_report_preserves_manual_budget(
    client: TestClient, session: Session
) -> None:
    """L1: a report with no gpus key on a machine with a manually-set budget and
    no existing gpus list must PRESERVE that budget (not clobber it with 0)."""
    machine = _machine(session)
    machine.total_vram_bytes = 99 * 1024**3
    session.add(machine)
    session.commit()
    _definition(session)

    resp = client.post(
        "/admin/api/providers/register",
        json=_body(hardware={"cpu": {"cores": 8}}),
    )
    assert resp.status_code == 200
    session.refresh(machine)
    assert machine.total_vram_bytes == 99 * 1024**3
    assert "gpus" not in machine.hardware
    assert machine.hardware["cpu"] == {"cores": 8}


def test_register_reregister_keeps_gpu_still_claimed_by_other_agent(
    client: TestClient, session: Session
) -> None:
    """M1: a GPU another agent still claims survives this agent dropping it."""
    machine = _machine(session)
    _definition(session)
    # Overlapping visibility: a1 sees gpu-1 + gpu-2, a2 also sees gpu-1.
    client.post(
        "/admin/api/providers/register",
        json=_body(
            agent_id="a1",
            hardware={
                "gpus": [
                    {"uuid": "gpu-1", "total_vram_bytes": 24 * 1024**3},
                    {"uuid": "gpu-2", "total_vram_bytes": 8 * 1024**3},
                ],
                "total_vram_bytes": 32 * 1024**3,
            },
        ),
    )
    client.post(
        "/admin/api/providers/register",
        json=_body(
            agent_id="a2",
            hardware={
                "gpus": [{"uuid": "gpu-1", "total_vram_bytes": 24 * 1024**3}],
                "total_vram_bytes": 24 * 1024**3,
            },
        ),
    )
    # a1 re-registers now seeing only gpu-3 (dropped both gpu-1 and gpu-2).
    client.post(
        "/admin/api/providers/register",
        json=_body(
            agent_id="a1",
            hardware={
                "gpus": [{"uuid": "gpu-3", "total_vram_bytes": 16 * 1024**3}],
                "total_vram_bytes": 16 * 1024**3,
            },
        ),
    )

    session.refresh(machine)
    uuids = {g["uuid"] for g in machine.hardware["gpus"]}
    # gpu-1 kept (a2 still claims it), gpu-2 dropped (only a1 had it), gpu-3 added.
    assert uuids == {"gpu-1", "gpu-3"}
    assert machine.total_vram_bytes == (24 + 16) * 1024**3


# ============================================================================
# Schema consensus gate (Phase 12) — voters are agents now
# ============================================================================


def test_register_bootstraps_provider_type(
    client: TestClient, session: Session
) -> None:
    from app.models import ProviderType

    _machine(session)
    _definition(session)
    resp = client.post("/admin/api/providers/register", json=_body())
    assert resp.status_code == 200
    ptype = session.exec(
        select(ProviderType).where(ProviderType.name == "mock")
    ).first()
    assert ptype is not None
    assert ptype.status == "active"


def test_register_second_agent_same_schema_commits(
    client: TestClient, session: Session
) -> None:
    """Two agents of one type presenting the same schema: both succeed."""
    _machine(session)
    _definition(session)
    r1 = client.post("/admin/api/providers/register", json=_body(agent_id="a1"))
    r2 = client.post("/admin/api/providers/register", json=_body(agent_id="a2"))
    assert r1.status_code == 200
    assert r2.status_code == 200
