"""Synthetic machine-metrics emitter for the mock provider (Phase 17 slice 5).

Mirrors :class:`provider_lib.metrics.MachineMetricsEmitter`'s lifecycle API
(``start`` / ``stop`` / ``set_owned`` / ``running``) and the exact
``metrics.machine`` payload shape the admin merges in
``app/services/metrics_service.py``, but derives every value from the mock's
fake inventory instead of real hardware. This lets a device-isolated
two-agent GPU split (compose ``--profile two-gpu``) be exercised end-to-end
with no GPU present.

Category split (identical to the real emitter):

* GPU categories (``vram`` / ``gpu_usage``) are emitted by EVERY agent that
  declares them, filtered to its ``ASSIGNED_GPU_UUIDS``, regardless of
  machine-metrics ownership.
* Machine-wide categories (``os_ram`` / ``cpu`` / ``storage``) are included
  only while the admin has assigned ownership via ``metrics.assign``
  (``set_owned(True)``).

A frame whose composed snapshot is empty is skipped (no empty-frame spam).
"""

import asyncio
import logging
from typing import Any

from provider_lib.metrics import (
    GPU_CATEGORIES,
    MACHINE_WIDE_CATEGORIES,
    filter_gpus,
    gpu_uuid,
    parse_gpu_assignment,
)

# NOTE: ``FAKE_HARDWARE`` lives in ``provider_mock.main``. Importing it at
# module load is safe because ``main`` imports this module lazily (inside
# ``run_async``), so there is no import cycle at package-import time.
from provider_mock.main import FAKE_HARDWARE

logger = logging.getLogger("provider.mock.metrics")

_GIB = 1024**3
# Deterministic synthetic VRAM load: each GPU's total divided by this, so the
# union's used/free are stable and easy to assert on.
_VRAM_USED_DIVISOR = 8
# Deterministic synthetic utilization, varied by GPU id (percent).
_UTILIZATION_PER_INDEX = 10.0


def _synthetic_gpu_samples(assignment: set[str]) -> list[dict[str, Any]]:
    """Build metrics-shaped GPU samples from the assignment-filtered inventory.

    Each entry carries the fields the admin's ``read_machine_metrics`` unions
    per-GPU on (``uuid`` / ``vram_total`` / ``vram_used`` / ``vram_free`` /
    ``utilization``) plus ``name`` / ``vendor`` for display. The per-GPU ``id``
    comes from the inventory's explicit ``id`` (falling back to its index in the
    FULL inventory, never the filtered position) so a device-isolated agent
    reporting only ``mock-gpu-2`` still stamps ``id=1`` — matching what a real
    agent reports for the same uuid.
    """
    inventory = filter_gpus(FAKE_HARDWARE["gpus"], assignment)
    full_index = {gpu_uuid(g): i for i, g in enumerate(FAKE_HARDWARE["gpus"])}
    samples: list[dict[str, Any]] = []
    for gpu in inventory:
        uid = gpu_uuid(gpu)
        gpu_id = gpu.get("id", full_index.get(uid, 0))
        total = int(gpu.get("total_vram_bytes", 0))
        used = total // _VRAM_USED_DIVISOR
        samples.append(
            {
                "id": gpu_id,
                "uuid": uid,
                "name": gpu.get("name", "Mock GPU"),
                "vendor": gpu.get("vendor", "mock"),
                "vram_total": total,
                "vram_used": used,
                "vram_free": max(total - used, 0),
                "utilization": _UTILIZATION_PER_INDEX * (gpu_id + 1),
            }
        )
    return samples


def _vram_section(gpus: list[dict[str, Any]]) -> dict[str, Any]:
    total = sum(int(g["vram_total"]) for g in gpus)
    used = sum(int(g["vram_used"]) for g in gpus)
    return {
        "total_bytes": total,
        "used_bytes": used,
        "free_bytes": max(total - used, 0),
        "gpu_count": len(gpus),
        "gpus": gpus,
    }


def _usage_section(gpus: list[dict[str, Any]]) -> dict[str, Any]:
    util = sum(float(g["utilization"]) for g in gpus) / len(gpus) if gpus else 0.0
    return {
        "utilization_percent": round(util, 2),
        "gpus": [{"id": g["id"], "utilization": g["utilization"]} for g in gpus],
    }


def _machine_wide_section(categories: set[str]) -> dict[str, Any]:
    """Synthetic machine-wide snapshot (owner-gated), mirroring the real
    collectors' shapes so the admin/UI read paths are exercised."""
    out: dict[str, Any] = {}
    if "os_ram" in categories:
        total = int(FAKE_HARDWARE["ram"]["total_bytes"])
        used = total // 4
        out["os_ram"] = {
            "total_bytes": total,
            "available_bytes": total - used,
            "used_bytes": used,
        }
    if "cpu" in categories:
        cpu = FAKE_HARDWARE["cpu"]
        out["cpu"] = {
            "cores": int(cpu.get("cores", 0)),
            "model": cpu.get("model", "mock-cpu"),
            "load_1m": 0.5,
            "load_5m": 0.4,
            "load_15m": 0.3,
        }
    if "storage" in categories:
        total = 1000 * _GIB
        used = 400 * _GIB
        out["storage"] = {
            "models": {
                "total_bytes": total,
                "used_bytes": used,
                "free_bytes": total - used,
            },
            "cache": {
                "total_bytes": total,
                "used_bytes": used // 10,
                "free_bytes": total - used // 10,
            },
        }
    return out


class MockMachineMetricsEmitter:
    """Periodically sends synthetic ``metrics.machine`` events while started.

    Lifecycle + payload shape mirror
    :class:`provider_lib.metrics.MachineMetricsEmitter`; values are derived
    from the mock's fake inventory filtered to ``ASSIGNED_GPU_UUIDS``.
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
            gpus = _synthetic_gpu_samples(assignment)
            assigned_gpus = [g["uuid"] for g in gpus]
            if "vram" in gpu_cats and gpus:
                snapshot["vram"] = _vram_section(gpus)
            if "gpu_usage" in gpu_cats and gpus:
                snapshot["gpu_usage"] = _usage_section(gpus)

        # Machine-wide categories: owner-gated single snapshot.
        if machine_cats and self._owned:
            snapshot.update(_machine_wide_section(machine_cats))

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
                    logger.exception("mock machine metrics emit failed; continuing")
                await asyncio.sleep(interval)
        except asyncio.CancelledError:
            raise
