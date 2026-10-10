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

import pytest

DOCKERFILE = Path(__file__).resolve().parents[1] / "Dockerfile"
DOCKERFILE_CUDA = Path(__file__).resolve().parents[1] / "Dockerfile.cuda"
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


DOCKERFILE_PATHS = [DOCKERFILE, DOCKERFILE_CUDA]


@pytest.mark.parametrize("dockerfile", DOCKERFILE_PATHS, ids=lambda p: p.name)
def test_provider_dockerfile_overrides_every_base_hazard(dockerfile: Path) -> None:
    text = dockerfile.read_text()
    ins = _instructions(text)

    # B1: explicit exec-form ENTRYPOINT for the provider agent (the base's
    # talkies-entrypoint ignores "$@" and execs stock talkies).
    entrypoints = [a for i, a in ins if i == "ENTRYPOINT"]
    assert entrypoints == [
        '["/opt/provider-venv/bin/python", "-m", "provider_talkies.main"]'
    ]

    # B2: the venv/COPY layers run as root; the FINAL user is talkies.
    users = [a for i, a in ins if i == "USER"]
    assert users == ["root", "talkies"]
    # (stage-aware: the cuda variant has driver-stage RUNs before the
    # provider stage's USER root — pin the venv RUN after USER root instead)
    venv_run_idx = next(
        i for i, (kind, arg) in enumerate(ins) if kind == "RUN" and "uv venv" in arg
    )
    root_user_idx = next(i for i, (kind, _) in enumerate(ins) if kind == "USER")
    assert root_user_idx < venv_run_idx
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

    # CI failure pin (build-provider-talkies): provider_talkies declares
    # matrix-provider-lib = { workspace = true }, so the workspace root
    # manifests must be copied and the install must go through
    # `uv sync --frozen --package` (uv pip install cannot resolve workspace
    # sources without the root) — same pattern as llama-cpp Dockerfile.cuda12.
    copy_text = " ".join(a for i, a in ins if i == "COPY")
    assert "pyproject.toml uv.lock /opt/matrix/" in copy_text
    run_text_all = " ".join(a for i, a in ins if i == "RUN")
    assert "uv sync --frozen --package matrix-provider-talkies" in run_text_all
    assert "UV_PROJECT_ENVIRONMENT=/opt/provider-venv" in run_text_all
    assert "uv pip install" not in run_text_all

    # Deploy failure pin: `uv venv --python 3.14` symlinks the venv python to
    # the managed interpreter under $HOME/.local/share/uv — under /root the
    # uid-1000 talkies user cannot traverse it and the container dies with
    # "exec /opt/provider-venv/bin/python: Permission denied". The managed
    # install dir must be world-readable and covered by the chmod.
    env_text_all = " ".join(a for i, a in ins if i == "ENV")
    assert "UV_PYTHON_INSTALL_DIR=/opt/uv-python" in env_text_all
    assert "chmod -R a+rX /opt/provider-venv /opt/uv-python" in run_text_all


def test_cuda_dockerfile_base_arg_is_global_scoped() -> None:
    """CI failure pin (build-provider-talkies-cuda): a stage-scoped ARG (one
    declared after a FROM) is invisible to the NEXT stage's FROM —
    'base name (${BASE_IMAGE}) should not be blank'. The ARG must precede the
    first FROM (global scope) to be usable in a later FROM line."""
    lines = DOCKERFILE_CUDA.read_text().splitlines()
    first_from = next(
        i for i, line in enumerate(lines) if line.strip().upper().startswith("FROM ")
    )
    arg_line = next(
        i
        for i, line in enumerate(lines)
        if line.strip().startswith("ARG BASE_IMAGE=psyb0t/talkies:latest-cuda")
    )
    assert arg_line < first_from


def test_cuda_dockerfile_bakes_the_nvidia_userspace_driver() -> None:
    """Dockerfile.cuda mirrors provider/llama-cpp/Dockerfile.cuda12: the RPM
    Fusion driver stage + baked libs, so the image runs on hosts WITHOUT the
    NVIDIA Container Toolkit (plain/rootful Podman + /dev/nvidia* devices)."""
    text = DOCKERFILE_CUDA.read_text()
    ins = _instructions(text)

    # Stage 1 is the fedora RPM Fusion driver source with the host-matched
    # version args (same defaults as the llama-cpp cuda12 job).
    assert re.search(
        r"FROM\s+registry\.fedoraproject\.org/fedora:44\s+AS\s+nvidia", text
    )
    assert "ARG NVIDIA_DRIVER_BRANCH=580" in text
    assert "ARG NVIDIA_VERSION=580.178.04" in text
    assert "rpmfusion" in text

    # The driver libs + nvidia-smi are copied from the stage into
    # /usr/local/nvidia (libcuda = driver API for torch in BOTH venvs,
    # NVML + nvidia-smi = the provider's vram/gpu_usage metrics).
    copy_text = " ".join(a for i, a in ins if i == "COPY")
    for artifact in (
        "libcuda.so*",
        "libnvidia-ml.so*",
        "libnvidia-ptxjitcompiler.so*",
        "nvidia-smi",
    ):
        assert artifact in copy_text

    # LD_LIBRARY_PATH PREPENDS the baked libs (base's cudnn/cublas entries
    # must survive appended) and PATH gains /usr/local/nvidia/bin so the
    # agent's nvidia-smi sampler resolves.
    env_text = " ".join(a for i, a in ins if i == "ENV")
    assert "LD_LIBRARY_PATH=/usr/local/nvidia/lib64:${LD_LIBRARY_PATH}" in env_text
    assert "/usr/local/nvidia/bin" in env_text
