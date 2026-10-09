"""Daemon-free Dockerfile semantics checks (review round 1: B1/B2/H1/M1).

The real base (psyb0t/talkies:latest-cuda) is multi-GB and the dev sandbox
has no container runtime, so the provider Dockerfile is pinned
*structurally* here: every base-image hazard must be overridden in the
right order. Dockerfile.stubtest reproduces the same hazards on a
pull-cheap base for a real `docker build` + `docker run` smoke where a
daemon exists (see its header).
"""

import re
from pathlib import Path

DOCKERFILE = Path(__file__).resolve().parents[1] / "Dockerfile"
STUB_BASE = Path(__file__).resolve().parents[1] / "Dockerfile.stubtest"


def _instructions(text: str) -> list[tuple[str, str]]:
    """Flat (INSTRUCTION, argument) pairs, line continuations joined."""
    lines = []
    buf = ""
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        buf += (" " if buf else "") + line.rstrip("\\").strip()
        if line.endswith("\\"):
            continue
        lines.append(buf)
        buf = ""
    out = []
    for entry in lines:
        m = re.match(r"^([A-Z]+)\s+(.*)$", entry)
        if m:
            out.append((m.group(1), m.group(2)))
    return out


def test_stub_base_reproduces_the_hazards() -> None:
    ins = _instructions(STUB_BASE.read_text())
    assert ("ENTRYPOINT", '["/usr/local/bin/talkies-entrypoint"]') in ins
    assert ("USER", "talkies") in ins
    env_text = " ".join(a for i, a in ins if i == "ENV")
    assert "TALKIES_DATA_DIR=/data" in env_text and "HF_HOME=/data/hf" in env_text
    assert any(i == "HEALTHCHECK" for i, _ in ins)


def test_provider_dockerfile_overrides_every_base_hazard() -> None:
    text = DOCKERFILE.read_text()
    ins = _instructions(text)
    kinds = [i for i, _ in ins]

    # B1: explicit exec-form ENTRYPOINT for the provider agent (the base's
    # talkies-entrypoint ignores "$@" and execs stock talkies).
    entrypoints = [a for i, a in ins if i == "ENTRYPOINT"]
    assert entrypoints == [
        '["/opt/provider-venv/bin/python", "-m", "provider_talkies.main"]'
    ]

    # B2: the venv/COPY layers run as root; the FINAL user is talkies.
    users = [a for i, a in ins if i == "USER"]
    assert users == ["root", "talkies"]
    assert kinds.index("USER") < kinds.index("RUN")  # root before the venv RUN
    assert users[-1] == "talkies"
    # The venv RUN makes the tree world-readable for the talkies user.
    run_text = " ".join(a for i, a in ins if i == "RUN")
    assert "chmod -R a+rX /opt/provider-venv" in run_text

    # NEW-2: /models + /cache are pre-created root-side and chowned to
    # talkies BEFORE the final USER switch, so fresh named volumes are
    # writable by the agent (the base image never creates these paths).
    volume_run_idx = next(
        i
        for i, (kind, arg) in enumerate(ins)
        if kind == "RUN"
        and "mkdir -p /models /cache" in arg
        and "chown -R talkies:talkies /models /cache" in arg
    )
    final_user_idx = max(i for i, (kind, _) in enumerate(ins) if kind == "USER")
    assert ins[final_user_idx][1] == "talkies"
    assert volume_run_idx < final_user_idx

    # H1: the inherited TALKIES_DATA_DIR=/data is BLANKED so the driver's
    # MODELS_DIR derivation (the mounted volume) governs.
    env_text = " ".join(a for i, a in ins if i == "ENV")
    assert "TALKIES_DATA_DIR=" in env_text
    assert "TALKIES_DATA_DIR=/data" not in env_text

    # M1: the inherited :8000 HEALTHCHECK is disabled.
    health = [a for i, a in ins if i == "HEALTHCHECK"]
    assert health == ["NONE"]

    # Parameterized base so Dockerfile.stubtest can stand in for the CUDA
    # image in daemon-equipped validation runs.
    assert "ARG BASE_IMAGE=psyb0t/talkies:latest-cuda" in text
    assert re.search(r"FROM\s+\$\{BASE_IMAGE\}", text)
