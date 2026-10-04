"""Machine-level metrics collectors + assignment-driven emitter.

Ported from the legacy agent's gpu_monitor plus cheap OS collectors. All
blocking work (subprocess calls, /proc reads, disk stats) runs through
`asyncio.to_thread` so the provider event loop never stalls, and every
collector is failure-tolerant: a missing tool or unreadable file simply
omits its category rather than raising.

Categories (see ProviderSettings.METRICS_CATEGORIES):
  gpu_usage  GPU utilization percent (nvidia-smi or AMD sysfs)
  vram       GPU memory totals/used/free (same sampler as gpu_usage)
  os_ram     /proc/meminfo totals
  cpu        load average + core count
  storage    shutil.disk_usage of MODELS_DIR and CACHE_DIR

Machine metrics ownership (exactly one instance per machine reports) is
assigned by the admin via the `metrics.assign` WS command; the emitter
runs only between assign and unassign.
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
    except OSError, subprocess.SubprocessError:
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
        except ValueError, IndexError:
            continue
    return gpus


def _read_int(path: Path) -> int | None:
    try:
        return int(path.read_text().strip())
    except OSError, ValueError:
        return None


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


async def collect_vram() -> dict[str, Any]:
    gpus = await collect_gpu_sample()
    if not gpus:
        raise LookupError("no GPU telemetry source available")
    return _vram_from(gpus)


async def collect_gpu_usage() -> dict[str, Any]:
    gpus = await collect_gpu_sample()
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
) -> dict[str, Any]:
    """Collect only the requested categories; never raises.

    A category whose collector fails (missing tool, unreadable path) is
    simply omitted from the returned dict.
    """
    snapshot: dict[str, Any] = {}
    # vram and gpu_usage share one sampler: sample once, derive both.
    if {"vram", "gpu_usage"} & set(categories):
        try:
            gpus = await collect_gpu_sample()
        except Exception:  # noqa: BLE001
            logger.debug("gpu sampler failed; omitting vram/gpu_usage")
            gpus = []
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
    """Periodically sends `metrics.machine` events while assigned.

    The admin assigns machine-metrics ownership with the `metrics.assign`
    WS command (exactly one instance per machine) and revokes it with
    `metrics.unassign` / disconnect. This emitter only runs between
    start() and stop(); call those from the command handlers.
    """

    def __init__(self, lifecycle: Any, client: Any, settings: Any) -> None:
        self._lifecycle = lifecycle
        self._client = client
        self._settings = settings
        self._task: asyncio.Task[None] | None = None

    @property
    def running(self) -> bool:
        return self._task is not None and not self._task.done()

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

    async def emit_once(self) -> dict[str, Any]:
        categories = self._settings.metrics_categories
        snapshot = await collect_machine_snapshot(
            categories,
            models_dir=self._settings.MODELS_DIR,
            cache_dir=self._settings.CACHE_DIR,
        )
        payload: dict[str, Any] = {
            **snapshot,
            "backend_status": self._lifecycle.backend_status,
            "in_flight": self._lifecycle.in_flight,
            "capacity": self._lifecycle.capacity,
            "machine_uid": self._settings.MACHINE_UID,
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
