"""Pure hardware-inventory helpers (Phase 17 B).

Device-isolated provider containers each see only a subset of a machine's GPUs,
so the machine's ``hardware["gpus"]`` inventory is the **union across every
agent on the box**, keyed by ``uuid``. These helpers are the single source of
truth for that union and for the ``ProviderAgent.assigned_gpus`` contract
(ARCHITECTURE.md §4: a list of GPU uuid strings). They are deliberately free of
DB/Redis/FastAPI dependencies so both the registration handler
(``app/api/admin/providers.py``) and the agent-delete handler
(``app/api/admin/agents.py``) share identical semantics.
"""

from typing import Any


def normalize_gpu_uuids(gpus: Any) -> list[str]:
    """Extract GPU uuid strings from a report or an ``assigned_gpus`` value.

    Accepts the structured descriptor form (a list of dicts each carrying a
    ``uuid``) and the already-normalized form (a list of uuid strings); anything
    else yields an empty list. Duplicate uuids are collapsed while preserving
    first-seen order. This is the single source of truth for the
    ``ProviderAgent.assigned_gpus`` contract (ARCHITECTURE.md §4: a list of
    uuid strings), so legacy rows that stored full dicts self-heal on read.
    """
    out: list[str] = []
    if isinstance(gpus, list):
        for g in gpus:
            if isinstance(g, dict):
                uid = g.get("uuid")
                if uid:
                    out.append(str(uid))
            elif isinstance(g, str):
                out.append(g)
    return list(dict.fromkeys(out))


def _coerce_vram(value: Any) -> int | None:
    """Coerce a per-GPU ``total_vram_bytes`` to an int, or ``None`` to ignore.

    Accepts ``int`` and ``float`` (truncated to int) but explicitly excludes
    ``bool`` (which is an ``int`` subclass and would otherwise count as 1) and
    any other type.
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    return None


def sum_gpu_vram(gpus: Any) -> int:
    """Auto-sum the ``total_vram_bytes`` over a list of GPU descriptors.

    Non-dict entries and entries whose ``total_vram_bytes`` is not a numeric
    (int/float, excluding bool) are ignored (see :func:`_coerce_vram`).
    """
    total = 0
    if isinstance(gpus, list):
        for g in gpus:
            if isinstance(g, dict):
                vram = _coerce_vram(g.get("total_vram_bytes"))
                if vram is not None:
                    total += vram
    return total


def merge_hardware_union(
    existing_hardware: dict[str, Any] | None,
    report: dict[str, Any],
    prior_assigned_uuids: list[str],
    other_agent_uuids: list[str],
) -> tuple[dict[str, Any], int | None]:
    """Union one agent's hardware report into the machine inventory (Phase 17 B).

    The machine's ``hardware["gpus"]`` is the union across every agent on the
    box, keyed by ``uuid``:

    * GPUs previously attributed to *this* agent (``prior_assigned_uuids``) but
      absent from the new report are dropped — but only if no *other* agent on
      the machine still claims them (``other_agent_uuids``), mirroring the
      survivor-aware delete path so an overlapping-visibility GPU is never
      dropped by one agent's re-registration while another still reports it;
    * reported GPUs are upserted by ``uuid`` (latest report wins per uuid),
      preserving existing order and appending unseen uuids;
    * an empty/missing ``gpus`` report never erases GPUs another agent
      contributed (the union base survives).

    ``total_vram_bytes`` is the **auto-sum** over the union and is authoritative
    whenever a GPU list is in play (the report carried a ``gpus`` key OR the
    existing hardware already has a ``gpus`` list) — including a legitimately
    empty union, which sums to 0. When no GPU list is in play the total is left
    untouched (returned as ``None``) so a manually-set admission budget survives,
    consistent with the delete path. Every other top-level key
    (``cpu``/``ram``/...) keeps last-writer-wins but is only overwritten when the
    report actually carries it.

    Returns the merged hardware dict and the recomputed VRAM total, or ``None``
    when the total should be left unchanged.
    """
    merged = dict(existing_hardware or {})

    # Last-writer-wins for non-GPU top-level keys the report actually carries.
    for key, value in report.items():
        if key in ("gpus", "total_vram_bytes"):
            continue
        merged[key] = value

    gpus_in_play = ("gpus" in report) or isinstance(merged.get("gpus"), list)
    if not gpus_in_play:
        # Neither side contributed a GPU list: leave the union and the auto-sum
        # untouched so a directly-set budget is not clobbered with 0.
        return merged, None

    existing_gpus = merged.get("gpus")
    if not isinstance(existing_gpus, list):
        existing_gpus = []
    reported_gpus = report.get("gpus")
    if not isinstance(reported_gpus, list):
        reported_gpus = []

    reported_uuids = set(normalize_gpu_uuids(reported_gpus))
    stale = set(prior_assigned_uuids) - reported_uuids - set(other_agent_uuids)

    # Drop GPUs this agent stopped reporting *and* no survivor claims; keep the
    # rest of the union intact.
    base = [
        g for g in existing_gpus if not (isinstance(g, dict) and g.get("uuid") in stale)
    ]

    # Upsert reported GPUs by uuid (latest wins), appending unseen uuids.
    index_by_uuid = {
        g["uuid"]: i
        for i, g in enumerate(base)
        if isinstance(g, dict) and g.get("uuid")
    }
    unioned: list[Any] = list(base)
    for g in reported_gpus:
        if not (isinstance(g, dict) and g.get("uuid")):
            continue
        uid = g["uuid"]
        if uid in index_by_uuid:
            unioned[index_by_uuid[uid]] = g
        else:
            index_by_uuid[uid] = len(unioned)
            unioned.append(g)

    total = sum_gpu_vram(unioned)
    merged["gpus"] = unioned
    merged["total_vram_bytes"] = total
    return merged, total
