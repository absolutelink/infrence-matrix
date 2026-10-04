"""NPU small-model knowledge + host probe for halogen-flash.

Ported from the legacy agent's `npu_models` + `npu_probe` modules.

The Flash engine only recognises upstream model ids in a request's
``model`` field for its NPU small models. The broker exposes each
enabled NPU model as ``<alias>-<suffix>``; ``options.npu_models`` stores
the upstream ids and the suffix map turns them into client-facing names
and back.

``probe_npu`` decides whether *this* host can actually run the NPU
(device node, XRT with the NPU plugin, the engine binary). Everything is
best-effort: on a machine with no NPU (CI, dev laptops) each check fails
with a reason and the caller simply omits the NPU env — never an error.
"""

from __future__ import annotations

import glob
import os
import subprocess
from dataclasses import dataclass, field
from typing import Any

# upstream model id -> client-facing suffix
NPU_SUFFIXES: dict[str, str] = {
    "qwen3-embedding-0.6b": "embed",
    "qwen3-reranker-0.6b": "rerank",
    "qwen3.5-2b": "nano",
    "decider-0.8b": "decide",
    "qwen3guard-gen-0.6b": "guard",
}

# suffix -> upstream model id (inverse of NPU_SUFFIXES)
UPSTREAM_BY_SUFFIX: dict[str, str] = {v: k for k, v in NPU_SUFFIXES.items()}

STOCK_NPU_MODEL_IDS: tuple[str, ...] = tuple(NPU_SUFFIXES)

# upstream model id -> capability key used by broker routes
NPU_CAPABILITIES: dict[str, str] = {
    "qwen3-embedding-0.6b": "embeddings",
    "qwen3-reranker-0.6b": "rerank",
    "qwen3.5-2b": "chat",
    "decider-0.8b": "decisions",
    "qwen3guard-gen-0.6b": "moderations",
}


def client_name(alias: str, upstream_id: str) -> str:
    """Return the ``<alias>-<suffix>`` public name for an enabled NPU model."""
    suffix = NPU_SUFFIXES[upstream_id]
    return f"{alias}-{suffix}"


def split_client_name(alias: str, model_ref: str) -> str | None:
    """If ``model_ref`` is ``<alias>-<suffix>`` for a known suffix, return
    the upstream model id, else ``None``."""
    prefix = f"{alias}-"
    if not model_ref.startswith(prefix):
        return None
    suffix = model_ref[len(prefix) :]
    return UPSTREAM_BY_SUFFIX.get(suffix)


@dataclass
class NpuPinRecord:
    """One NPU model's download record parsed from the image's pins file."""

    model_id: str
    repo: str | None = None
    revision: str | None = None
    devices_of: str = ""
    # list of (relative_path, size, sha256)
    files: list[tuple[str, int, str]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.devices_of:
            self.devices_of = self.model_id

    @property
    def devices_files(self) -> list[tuple[str, int, str]]:
        return [f for f in self.files if f[0].startswith("devices/")]

    @property
    def own_files(self) -> list[tuple[str, int, str]]:
        return [f for f in self.files if not f[0].startswith("devices/")]


def parse_npu_pins(text: str) -> dict[str, NpuPinRecord]:
    """Parse ``/opt/halogen/npu/models.txt``.

    Mirrors the awk readers in the engine's entrypoint:

    - ``model <id> repo=<r> revision=<rev> devices=<other> ...``
    - ``file <id> <path> <size> <sha256>``
    """
    records: dict[str, NpuPinRecord] = {}

    def record(mid: str) -> NpuPinRecord:
        return records.setdefault(mid, NpuPinRecord(mid))

    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        parts = line.split()
        kind = parts[0]
        if kind == "model" and len(parts) >= 2:
            mid = parts[1]
            rec = record(mid)
            for token in parts[2:]:
                if "=" not in token:
                    continue
                key, value = token.split("=", 1)
                if key == "repo":
                    rec.repo = value
                elif key == "revision":
                    rec.revision = value
                elif key == "devices":
                    rec.devices_of = value
        elif kind == "file" and len(parts) >= 5:
            mid, path, size, sha = parts[1], parts[2], parts[3], parts[4]
            try:
                size_int = int(size)
            except ValueError:
                continue
            record(mid).files.append((path, size_int, sha))
    return records


def resolve_npu_download_set(
    records: dict[str, NpuPinRecord], upstream_ids: list[str]
) -> dict[str, tuple[str, str | None, list[tuple[str, int, str]]]]:
    """Map each target ``MODELS_DIR/npu/<id>/`` directory to its download spec.

    A model that runs on another's ``devices/`` program contributes only
    its own files; the shared ``devices/`` files come from the
    devices-owner's directory. Returns
    ``{dir_id: (repo_id, revision, [(path, size, sha256), ...])}``.
    """
    plan_files: dict[str, list[tuple[str, int, str]]] = {}
    plan_source: dict[str, tuple[str, str | None]] = {}
    seen: dict[str, set[str]] = {}

    def add(rec: NpuPinRecord, files: list[tuple[str, int, str]]) -> None:
        bucket = plan_files.setdefault(rec.model_id, [])
        plan_source.setdefault(rec.model_id, (rec.repo, rec.revision))
        paths = seen.setdefault(rec.model_id, set())
        for entry in files:
            if entry[0] in paths:
                continue
            paths.add(entry[0])
            bucket.append(entry)

    for uid in upstream_ids:
        rec = records.get(uid)
        if rec is None:
            raise ValueError(f"no NPU pin record for '{uid}'")
        if not rec.repo:
            raise ValueError(f"NPU pin record for '{uid}' names no download repo")
        add(rec, rec.files)
        owner = rec.devices_of
        if owner != uid:
            owner_rec = records.get(owner)
            if owner_rec is None:
                raise ValueError(f"no NPU pin record for devices owner '{owner}'")
            if not owner_rec.repo:
                raise ValueError(f"NPU pin record for '{owner}' names no download repo")
            add(owner_rec, owner_rec.devices_files)
    return {
        dir_id: (plan_source[dir_id][0], plan_source[dir_id][1], files)
        for dir_id, files in plan_files.items()
    }


def load_npu_pins(pins_file: str) -> dict[str, NpuPinRecord]:
    """Read and parse the image's NPU pins file. Raises OSError if unreadable."""
    with open(pins_file) as fh:
        return parse_npu_pins(fh.read())


# ----------------------------------------------------------------------
# Host probe
# ----------------------------------------------------------------------

XRT_LIBS = (
    "libxrt_coreutil.so.2",
    "libxrt_core.so.2",
    "libxrt_driver_xdna.so.2",
)

# Roots under which the host's /sys may be visible: -v /sys:/host/sys puts
# it at /host/sys; a privileged container reads its own /sys.
_SYS_ROOTS = ("/host/sys", "/sys")


def _check_device(reasons: list[str], device_path: str) -> bool:
    node = device_path
    if not os.path.exists(node):
        if not os.path.isdir("/sys/module/amdxdna"):
            reasons.append("the NPU driver (amdxdna) is not loaded on this host")
        else:
            reasons.append(f"{node} is not present (pass --device {node})")
        return False
    if not os.access(node, os.R_OK | os.W_OK):
        reasons.append(f"{node} is not openable by this process")
        return False
    return True


def _check_xrt(reasons: list[str], lib_dir: str) -> bool:
    for name in XRT_LIBS:
        path = os.path.join(lib_dir, name)
        if os.path.islink(path) and not os.path.exists(path):
            target = os.readlink(path)
            reasons.append(
                f"{path} is a dangling link to {target}: the mount carries "
                "the host's links, not the files they point at"
            )
            return False
    coreutil = os.path.join(lib_dir, "libxrt_coreutil.so.2")
    if not os.path.exists(coreutil):
        reasons.append(f"no XRT found at {lib_dir}")
        return False
    plugin = os.path.join(lib_dir, "libxrt_driver_xdna.so.2")
    if not os.path.exists(plugin):
        reasons.append("the host's XRT has no NPU plugin (libxrt_driver_xdna.so.2)")
        return False
    return True


def _check_binary(reasons: list[str], binary: str) -> bool:
    if not os.path.exists(binary):
        reasons.append(f"{binary} is not present")
        return False
    try:
        proc = subprocess.run(  # noqa: S603
            [binary], capture_output=True, text=True, timeout=30
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        reasons.append(f"the NPU engine could not run: {exc}")
        return False
    output = (proc.stdout or "") + (proc.stderr or "")
    if proc.returncode != 1 or not output.startswith("usage: halogen-npu"):
        first = output.splitlines()[:1]
        reasons.append(
            "the NPU engine does not load against the host's XRT "
            f"(exit {proc.returncode}{': ' + first[0] if first else ''})"
        )
        return False
    return True


def _fabric_clock_state() -> tuple[bool | None, str]:
    """Read the GPU's fabric clock over every amdgpu device with the control.

    Returns ``(held, detail)``. ``held`` is ``None`` when no control is
    readable at all (the probe cannot tell).
    """
    for root in _SYS_ROOTS:
        devices = sorted(glob.glob(os.path.join(root, "class/drm/card*/device")))
        seen = False
        all_held = True
        detail = ""
        for d in devices:
            fclk = os.path.join(d, "pp_dpm_fclk")
            perf = os.path.join(d, "power_dpm_force_performance_level")
            if not (os.access(fclk, os.R_OK) and os.access(perf, os.R_OK)):
                continue
            seen = True
            try:
                with open(perf) as fh:
                    level = fh.read().strip()
                with open(fclk) as fh:
                    lines = [ln for ln in fh.read().splitlines() if ln.strip()]
            except OSError:
                all_held = False
                continue
            if not lines:
                all_held = False
                continue
            top = lines[-1]
            top_is_max = "*" in top
            if level == "high":
                continue
            if level == "manual" and sum("*" in ln for ln in lines) == 1 and top_is_max:
                continue
            all_held = False
            detail = f"{level} at {top}"
            break
        if seen:
            if all_held:
                return (True, "held")
            return (False, detail)
    return (None, "no fabric clock control readable from this container")


def probe_npu(
    *,
    device_path: str,
    xrt_lib_dir: str,
    binary_path: str,
) -> dict[str, Any]:
    """Probe NPU availability. Safe to call on any platform.

    The fabric clock is a start-time requirement, not a capability: a
    root container with /host/sys mounted holds it itself, and the host
    unit holds it at boot. Report it so the UI can warn, but gate only on
    the device, XRT and engine checks.
    """
    reasons: list[str] = []
    checks = {
        "device": _check_device(reasons, device_path),
        "xrt": _check_xrt(reasons, xrt_lib_dir),
    }
    if checks["device"] and checks["xrt"]:
        checks["engine"] = _check_binary(reasons, binary_path)
    else:
        checks["engine"] = False
    clock_held, clock_detail = _fabric_clock_state()
    if clock_held is False:
        reasons.append(
            f"the GPU's fabric clock is not held at its top speed ({clock_detail})"
        )
    available = all(checks.values())
    return {
        "available": available,
        "checks": checks,
        "fabric_clock_held": clock_held,
        "fabric_clock_detail": clock_detail,
        "reasons": reasons,
    }
