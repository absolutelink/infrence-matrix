"""In-memory lifecycle tracking for agent-owned inference connections."""

import asyncio
import time
from typing import Any

MAX_RETAINED_OPERATIONS = 4096


class DuplicateInferenceOperation(Exception):
    """A request ID already identifies an active or completed operation."""


def _operations(manager) -> dict[str, dict[str, Any]]:
    operations = getattr(manager, "_inference_operations", None)
    if operations is None:
        operations = manager._inference_operations = {}
    return operations


def start_operation(manager, request_id: str, server_id: str, generation: int) -> None:
    """Record a request ID before dispatch so retries cannot duplicate work."""
    operations = _operations(manager)
    if request_id in operations:
        raise DuplicateInferenceOperation(request_id)
    if len(operations) >= MAX_RETAINED_OPERATIONS:
        completed = sorted(
            (
                (operation.get("finished_at", 0.0), key)
                for key, operation in operations.items()
                if operation["status"]
                not in {"queued", "reserved", "active", "cancelling"}
            )
        )
        for _finished_at, key in completed[
            : len(operations) - MAX_RETAINED_OPERATIONS + 1
        ]:
            del operations[key]
    operations[request_id] = {
        "request_id": request_id,
        "server_id": server_id,
        "slot_generation": generation,
        "status": "queued",
        "started_at": time.time(),
        "task": asyncio.current_task(),
    }


def operation_exists(manager, request_id: str) -> bool:
    return request_id in _operations(manager)


def finish_operation(manager, request_id: str | None, status: str) -> None:
    if request_id is None:
        return
    operation = _operations(manager).get(request_id)
    if operation is not None and operation["status"] in {
        "queued",
        "reserved",
        "active",
        "cancelling",
    }:
        timer = operation.pop("expiry_task", None)
        if timer is not None and timer is not asyncio.current_task():
            timer.cancel()
        operation["status"] = status
        operation["finished_at"] = time.time()
        operation["task"] = None


def activate_operation(manager, request_id: str | None) -> None:
    if request_id is None:
        return
    operation = _operations(manager).get(request_id)
    if operation is not None and operation["status"] == "queued":
        operation["status"] = "active"


def bind_operation_task(manager, request_id: str | None) -> None:
    if request_id is None:
        return
    operation = _operations(manager).get(request_id)
    if operation is not None and operation["status"] == "active":
        timer = operation.pop("expiry_task", None)
        if timer is not None:
            timer.cancel()
        operation["task"] = asyncio.current_task()


def reserve_operation(
    manager, request_id: str, server_id: str, generation: int, token: str | None = None
) -> None:
    """Record an already acquired slot, before any LLM request is sent."""
    operation = _operations(manager).get(request_id)
    if operation is not None:
        if operation["status"] != "cancelled" or operation.get("dispatched"):
            raise DuplicateInferenceOperation(request_id)
        del _operations(manager)[request_id]
    start_operation(manager, request_id, server_id, generation)
    _operations(manager)[request_id]["status"] = "reserved"
    _operations(manager)[request_id]["dispatched"] = False
    _operations(manager)[request_id]["reservation_id"] = token


def consume_reservation(
    manager, request_id: str, server_id: str, generation: int, token: str | None = None
) -> bool:
    """A reservation can dispatch exactly once, and only on its own generation."""
    operation = _operations(manager).get(request_id)
    if (
        operation is None
        or operation["server_id"] != server_id
        or operation["slot_generation"] != generation
        or operation.get("reservation_id") != token
        or operation["status"] != "reserved"
    ):
        return False
    operation["status"] = "active"
    operation["dispatched"] = True
    operation["task"] = asyncio.current_task()
    return True


def get_operation(manager, request_id: str, server_id: str) -> dict[str, Any] | None:
    operation = _operations(manager).get(request_id)
    if operation is None or operation["server_id"] != server_id:
        return None
    return {
        key: value
        for key, value in operation.items()
        if key not in {"task", "expiry_task"}
    }


def list_operations(manager, server_id: str) -> list[dict[str, Any]]:
    return [
        {
            key: value
            for key, value in operation.items()
            if key not in {"task", "expiry_task"}
        }
        for operation in _operations(manager).values()
        if operation["server_id"] == server_id
    ]


def cancel_operation(manager, request_id: str, server_id: str) -> bool:
    operation = _operations(manager).get(request_id)
    if (
        operation is None
        or operation["server_id"] != server_id
        or operation["status"] not in {"queued", "active"}
    ):
        return False
    task = operation.get("task")
    if task is None or task is asyncio.current_task():
        return False
    operation["status"] = "cancelling"
    task.cancel()
    return True


async def stop_server_operations(manager, server_id: str) -> None:
    """Fence reservations and wake waiters when a server stops or restarts."""
    for operation in list(_operations(manager).values()):
        if operation["server_id"] != server_id or operation["status"] not in {
            "queued",
            "reserved",
            "active",
            "cancelling",
        }:
            continue
        task = operation.get("task")
        timer = operation.pop("expiry_task", None)
        if timer is not None:
            timer.cancel()
        operation["status"] = "failed"
        operation["finished_at"] = time.time()
        operation["task"] = None
        if task is not None and task is not asyncio.current_task():
            task.cancel()
    getattr(manager, "_active_inference_requests", {}).pop(server_id, None)
    condition = getattr(manager, "_inference_slot_conditions", {}).get(server_id)
    if condition is not None:
        async with condition:
            condition.notify_all()
