"""Machine-level metrics collectors + assignment-driven emitter.

Ported from the legacy agent's gpu_monitor plus cheap OS collectors. All
blocking work (subprocess calls, /proc reads, disk stats) runs through
`asyncio.to_thread` so the provider event loop never stalls, and every
collector is failure-tolerant: a missing tool or unreadable file simply
omits its category rather than raising.

Categories (see ProviderSettings.METRICS_CATEGORIES) split into two
ownership classes (Phase 17):

  GPU_CATEGORIES (per-agent, always-on when declared)
    gpu_usage  GPU utilization percent (nvidia-smi or AMD sysfs)
    vram       GPU memory totals/used/free (same sampler as gpu_usage)
  MACHINE_WIDE_CATEGORIES (owner-gated, single agent per machine)
    os_ram     /proc/meminfo totals
    cpu        load average + core count
    storage    shutil.disk_usage of MODELS_DIR and CACHE_DIR

On a device-isolated box each container sees only its own GPU, so the GPU
categories are emitted by EVERY agent that declares them, filtered to that
agent's assigned GPUs (`ASSIGNED_GPU_UUIDS`; empty = implicit = every GPU
the container sees). The machine-wide categories stay owner-gated: the admin
assigns ownership of the machine-wide snapshot to exactly one agent via the
`metrics.assign` / `metrics.unassign` WS commands; the emitter loop runs
whenever it is started (on connect) and `set_owned()` only toggles whether
the machine-wide categories are included in each frame.
"""

import asyncio
import logging
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

logger = logging.getLogger("provider.metrics")

_NVIDIA_QUERY = (
    "index,uuid,name,memory.total,memory.used,utilization.gpu,temperature.gpu"
)

# Phase 17 category split. GPU categories are emitted by every agent that
# declares them (filtered to its assigned GPUs); machine-wide categories stay
# owner-gated (exactly one agent per machine reports them).
GPU_CATEGORIES = frozenset({"vram", "gpu_usage"})
MACHINE_WIDE_CATEGORIES = frozenset({"os_ram", "cpu", "storage"})


def parse_gpu_assignment(tokens: list[str]) -> set[str]:
    """Normalize ``ASSIGNED_GPU_UUIDS`` tokens into a match set.

    Each token is either a full GPU ``uuid`` (matched case-insensitively) or a
    decimal GPU index (matched against the sample's ``id``). An empty result
    means *implicit* — report every GPU the container sees.
    """
    return {t.strip().lower() for t in tokens if t.strip()}


def gpu_uuid(gpu: dict[str, Any]) -> str:
    """Stable identifier for a GPU sample.

    Returns the real ``uuid`` when present (NVIDIA NVML uuid or the AMD
    PCI-slot-derived uuid), else a synthesized ``gpu-<id>`` fallback. This is
    the single canonical uuid space: providers report it and operators pass it
    back as an ``ASSIGNED_GPU_UUIDS`` token, so the reported value and the
    accepted token always agree.
    """
    return str(gpu.get("uuid") or f"gpu-{gpu.get('id')}")


def filter_gpus(
    sample: list[dict[str, Any]], assignment: set[str]
) -> list[dict[str, Any]]:
    """Keep only GPUs the agent is assigned.

    An empty ``assignment`` is implicit (pass-through). Otherwise a GPU is kept
    when its canonical uuid (``gpu_uuid`` — real uuid or the synthesized
    ``gpu-<id>`` form, case-insensitive) or its decimal ``id`` is in the set.
    """
    if not assignment:
        return sample
    kept: list[dict[str, Any]] = []
    for gpu in sample:
        if gpu_uuid(gpu).lower() in assignment or str(gpu.get("id", "")) in assignment:
            kept.append(gpu)
    return kept


# ----------------------------------------------------------------------
# GPU sampling (blocking)
# ----------------------------------------------------------------------


def _query_nvidia_smi(query: str) -> str | None:
    try:
        result = subprocess.run(
            ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):  # fmt: skip
        return None
    if result.returncode == 0:
        return result.stdout.strip()
    return None


def _parse_nvidia_output(output: str) -> list[dict[str, Any]]:
    gpus: list[dict[str, Any]] = []
    for line in output.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 7:
            continue
        try:
            total = int(float(parts[3])) * 1024 * 1024
            used = int(float(parts[4])) * 1024 * 1024
            gpus.append(
                {
                    "id": int(parts[0]),
                    "uuid": parts[1],
                    "name": parts[2],
                    "vendor": "nvidia",
                    "backend": "cuda",
                    "vram_total": total,
                    "vram_used": used,
                    "vram_free": max(total - used, 0),
                    "utilization": float(parts[5]),
                    "temperature": float(parts[6]),
                }
            )
        except (ValueError, IndexError):  # fmt: skip
            continue
    return gpus


def _read_int(path: Path) -> int | None:
    try:
        return int(path.read_text().strip())
    except (OSError, ValueError):  # fmt: skip
        return None


def _amd_uuid(device: Path) -> str:
    """Stable, host-unique uuid for an AMD card.

    AMD sysfs exposes no NVML-style uuid, so derive one from the PCI slot
    (``PCI_SLOT_NAME`` in the device ``uevent``, e.g. ``amd-0000:03:00.0``).
    When unavailable, fall back to the ``cardN`` directory name (the device's
    parent) — never the container-local enumeration index, which is ``0`` for
    every device-isolated container and would collide across agents in the
    Phase-17 union-by-uuid merge.
    """
    try:
        uevent = (device / "uevent").read_text()
    except OSError:  # fmt: skip
        uevent = ""
    for line in uevent.splitlines():
        key, _, value = line.partition("=")
        if key.strip() == "PCI_SLOT_NAME" and value.strip():
            return f"amd-{value.strip()}"
    return f"amd-{device.parent.name}"


def _sample_amd_sysfs() -> list[dict[str, Any]]:
    """Read AMDGPU memory/busy telemetry from the kernel driver sysfs."""
    gpus: list[dict[str, Any]] = []
    for device in sorted(Path("/sys/class/drm").glob("card*/device")):
        try:
            if device.joinpath("vendor").read_text().strip().lower() != "0x1002":
                continue
        except OSError:
            continue

        vram_total = _read_int(device / "mem_info_vram_total") or 0
        vram_used = _read_int(device / "mem_info_vram_used") or 0
        gtt_total = _read_int(device / "mem_info_gtt_total") or 0
        gtt_used = _read_int(device / "mem_info_gtt_used") or 0
        shared_memory = gtt_total > vram_total * 2
        total = gtt_total if shared_memory else vram_total
        used = gtt_used if shared_memory else vram_used
        if total <= 0:
            continue

        busy = _read_int(device / "gpu_busy_percent") or 0
        gpus.append(
            {
                "id": len(gpus),
                "uuid": _amd_uuid(device),
                "name": "AMD GPU",
                "vendor": "amd",
                "backend": "amd",
                "vram_total": total,
                "vram_used": used,
                "vram_free": max(total - used, 0),
                "utilization": float(busy),
                "temperature": None,
                "memory_type": "shared" if shared_memory else "dedicated",
            }
        )
    return gpus


def _sample_gpus_blocking() -> list[dict[str, Any]]:
    output = _query_nvidia_smi(_NVIDIA_QUERY)
    if output:
        return _parse_nvidia_output(output)
    return _sample_amd_sysfs()


# ----------------------------------------------------------------------
# Category collectors (async; never raise — omit the key on failure)
# ----------------------------------------------------------------------


def _vram_from(gpus: list[dict[str, Any]]) -> dict[str, Any]:
    total = sum(int(g.get("vram_total", 0)) for g in gpus)
    used = sum(int(g.get("vram_used", 0)) for g in gpus)
    return {
        "total_bytes": total,
        "used_bytes": used,
        "free_bytes": max(total - used, 0),
        "gpu_count": len(gpus),
        "gpus": gpus,
    }


def _usage_from(gpus: list[dict[str, Any]]) -> dict[str, Any]:
    util = (
        sum(float(g.get("utilization", 0.0)) for g in gpus) / len(gpus) if gpus else 0.0
    )
    return {
        "utilization_percent": round(util, 2),
        "gpus": [
            {"id": g.get("id"), "utilization": g.get("utilization")} for g in gpus
        ],
    }


async def collect_gpu_sample() -> list[dict[str, Any]]:
    """Shared blocking-free GPU sample (nvidia-smi CSV or AMD sysfs)."""
    try:
        return await asyncio.to_thread(_sample_gpus_blocking)
    except Exception:  # noqa: BLE001
        logger.debug("gpu sample failed", exc_info=True)
        return []


async def collect_vram(gpu_assignment: set[str] | None = None) -> dict[str, Any]:
    gpus = await collect_gpu_sample()
    if gpu_assignment is not None:
        gpus = filter_gpus(gpus, gpu_assignment)
    if not gpus:
        raise LookupError("no GPU telemetry source available")
    return _vram_from(gpus)


async def collect_gpu_usage(
    gpu_assignment: set[str] | None = None,
) -> dict[str, Any]:
    gpus = await collect_gpu_sample()
    if gpu_assignment is not None:
        gpus = filter_gpus(gpus, gpu_assignment)
    if not gpus:
        raise LookupError("no GPU telemetry source available")
    return _usage_from(gpus)


def _parse_meminfo(text: str) -> dict[str, int]:
    values: dict[str, int] = {}
    for line in text.splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[1].isdigit():
            values[parts[0].rstrip(":")] = int(parts[1]) * 1024
    return values


async def collect_os_ram() -> dict[str, Any]:
    def blocking() -> dict[str, Any]:
        info = _parse_meminfo(Path("/proc/meminfo").read_text())
        total = info.get("MemTotal", 0)
        available = info.get("MemAvailable", info.get("MemFree", 0))
        return {
            "total_bytes": total,
            "available_bytes": available,
            "used_bytes": max(total - available, 0),
        }

    return await asyncio.to_thread(blocking)


async def collect_cpu() -> dict[str, Any]:
    def blocking() -> dict[str, Any]:
        load1, load5, load15 = os.getloadavg()
        return {
            "cores": os.cpu_count() or 0,
            "load_1m": round(load1, 2),
            "load_5m": round(load5, 2),
            "load_15m": round(load15, 2),
        }

    return await asyncio.to_thread(blocking)


async def collect_storage(models_dir: Path, cache_dir: Path) -> dict[str, Any]:
    def blocking(path: Path) -> dict[str, int]:
        usage = shutil.disk_usage(path)
        return {
            "total_bytes": usage.total,
            "used_bytes": usage.used,
            "free_bytes": usage.free,
        }

    async def safe(path: Path) -> dict[str, Any] | None:
        try:
            return await asyncio.to_thread(blocking, path)
        except Exception:  # noqa: BLE001
            return None

    result: dict[str, Any] = {}
    models = await safe(Path(models_dir))
    if models is not None:
        result["models"] = models
    cache = await safe(Path(cache_dir))
    if cache is not None:
        result["cache"] = cache
    if not result:
        raise LookupError("no storage paths readable")
    return result


_COLLECTORS: dict[str, Any] = {
    "vram": collect_vram,
    "gpu_usage": collect_gpu_usage,
    "os_ram": collect_os_ram,
    "cpu": collect_cpu,
    # storage needs paths; handled specially below.
}


async def collect_machine_snapshot(
    categories: set[str],
    *,
    models_dir: Path | None = None,
    cache_dir: Path | None = None,
    gpu_assignment: set[str] | None = None,
) -> dict[str, Any]:
    """Collect only the requested categories; never raises.

    A category whose collector fails (missing tool, unreadable path) is
    simply omitted from the returned dict. When ``gpu_assignment`` is given
    the shared GPU sample is filtered to the agent's assigned GPUs before
    deriving ``vram`` / ``gpu_usage`` (empty set = implicit = every GPU).
    """
    snapshot: dict[str, Any] = {}
    # vram and gpu_usage share one sampler: sample once, derive both.
    if {"vram", "gpu_usage"} & set(categories):
        try:
            gpus = await collect_gpu_sample()
        except Exception:  # noqa: BLE001
            logger.debug("gpu sampler failed; omitting vram/gpu_usage")
            gpus = []
        if gpu_assignment is not None:
            gpus = filter_gpus(gpus, gpu_assignment)
        if "vram" in categories and gpus:
            snapshot["vram"] = _vram_from(gpus)
        if "gpu_usage" in categories and gpus:
            snapshot["gpu_usage"] = _usage_from(gpus)
    for category in sorted(categories - {"vram", "gpu_usage"}):
        try:
            if category == "storage":
                if models_dir is None and cache_dir is None:
                    continue
                snapshot[category] = await collect_storage(
                    models_dir or Path("/"), cache_dir or Path("/")
                )
                continue
            collector = _COLLECTORS.get(category)
            if collector is None:
                logger.debug("unknown metrics category %r; skipping", category)
                continue
            snapshot[category] = await collector()
        except Exception:  # noqa: BLE001
            logger.debug("metrics collector %r failed; omitting", category)
    return snapshot


# ----------------------------------------------------------------------
# Assignment-driven emitter
# ----------------------------------------------------------------------


class MachineMetricsEmitter:
    """Periodically sends `metrics.machine` events while started.

    Phase 17 split: the emitter loop runs whenever `start()`ed (on connect),
    independent of ownership. GPU categories (`vram` / `gpu_usage`) are emitted
    by EVERY agent that declares them, filtered to its assigned GPUs
    (`ASSIGNED_GPU_UUIDS`; empty = implicit). Machine-wide categories
    (`os_ram` / `cpu` / `storage`) are included only while the admin has
    assigned machine-metrics ownership via `metrics.assign`
    (`set_owned(True)`); `metrics.unassign` (`set_owned(False)`) drops them
    but never stops the loop, so GPU telemetry keeps flowing. A frame whose
    composed snapshot is empty is skipped (no empty-frame spam).
    """

    def __init__(self, lifecycle: Any, client: Any, settings: Any) -> None:
        self._lifecycle = lifecycle
        self._client = client
        self._settings = settings
        self._task: asyncio.Task[None] | None = None
        self._owned: bool = False

    @property
    def running(self) -> bool:
        return self._task is not None and not self._task.done()

    @property
    def owned(self) -> bool:
        return self._owned

    def set_owned(self, value: bool) -> None:
        """Toggle whether machine-wide categories are included in each frame."""
        self._owned = value

    def start(self) -> None:
        if self.running:
            return
        self._task = asyncio.create_task(self._loop())

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    async def emit_once(self) -> dict[str, Any] | None:
        categories = self._settings.metrics_categories
        gpu_cats = categories & GPU_CATEGORIES
        machine_cats = categories & MACHINE_WIDE_CATEGORIES
        assignment = parse_gpu_assignment(self._settings.assigned_gpu_tokens)

        snapshot: dict[str, Any] = {}
        assigned_gpus: list[str] = []

        # GPU categories: emitted by every agent that declares them, filtered
        # to this agent's assignment, regardless of ownership.
        if gpu_cats:
            try:
                gpus = await collect_gpu_sample()
            except Exception:  # noqa: BLE001
                logger.debug("gpu sampler failed; omitting vram/gpu_usage")
                gpus = []
            gpus = filter_gpus(gpus, assignment)
            assigned_gpus = [gpu_uuid(g) for g in gpus]
            if "vram" in gpu_cats and gpus:
                snapshot["vram"] = _vram_from(gpus)
            if "gpu_usage" in gpu_cats and gpus:
                snapshot["gpu_usage"] = _usage_from(gpus)

        # Machine-wide categories: owner-gated single snapshot.
        if machine_cats and self._owned:
            machine_snapshot = await collect_machine_snapshot(
                machine_cats,
                models_dir=self._settings.MODELS_DIR,
                cache_dir=self._settings.CACHE_DIR,
            )
            snapshot.update(machine_snapshot)

        if not snapshot:
            # Nothing to report (e.g. only machine-wide cats declared while
            # not owned) — skip the frame instead of spamming empties.
            return None

        payload: dict[str, Any] = {
            **snapshot,
            "backend_status": self._lifecycle.backend_status,
            "in_flight": self._lifecycle.in_flight,
            "capacity": self._lifecycle.capacity,
            "machine_uid": self._settings.MACHINE_UID,
            "assigned_gpus": assigned_gpus,
        }
        await self._client.send_event("metrics.machine", payload)
        return payload

    async def _loop(self) -> None:
        interval = float(self._settings.MACHINE_METRICS_INTERVAL)
        try:
            while True:
                try:
                    await self.emit_once()
                except Exception:  # noqa: BLE001
                    logger.exception("machine metrics emit failed; continuing")
                await asyncio.sleep(interval)
        except asyncio.CancelledError:
            raise
