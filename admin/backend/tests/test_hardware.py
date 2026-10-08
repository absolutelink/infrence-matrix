"""Unit tests for the pure hardware-inventory helpers (Phase 17 B).

These target ``app.services.hardware`` directly so each behavior fix has a
discriminating test that fails if the fix is reverted.
"""

from app.services.hardware import (
    merge_hardware_union,
    normalize_gpu_uuids,
    sum_gpu_vram,
)


def test_normalize_gpu_uuids_dedups_preserving_order() -> None:
    # L3: duplicate uuids collapse to first-seen order.
    assert normalize_gpu_uuids(["a", "b", "a", "c", "b"]) == ["a", "b", "c"]
    assert normalize_gpu_uuids(
        [{"uuid": "x"}, {"uuid": "x"}, {"uuid": "y"}]
    ) == ["x", "y"]
    # mixed dict/str/junk still dedups and drops non-uuid entries.
    assert normalize_gpu_uuids([{"uuid": "x"}, "x", None, {}, 5]) == ["x"]


def test_sum_gpu_vram_excludes_bool_and_coerces_float() -> None:
    # L2: bool must NOT count as 1; float truncates to int; junk ignored.
    assert sum_gpu_vram([{"total_vram_bytes": True}]) == 0
    assert sum_gpu_vram([{"total_vram_bytes": 1.9}]) == 1
    assert sum_gpu_vram([{"total_vram_bytes": 8}, {"total_vram_bytes": 4.0}]) == 12
    assert (
        sum_gpu_vram([{"total_vram_bytes": "8"}, {}, None, {"total_vram_bytes": 2}])
        == 2
    )
    assert sum_gpu_vram("not a list") == 0


def test_merge_union_survivor_aware_stale_drop() -> None:
    # M1: a GPU another agent still claims is NOT dropped when this agent stops
    # reporting it.
    existing = {
        "gpus": [
            {"uuid": "g1", "total_vram_bytes": 10},
            {"uuid": "g2", "total_vram_bytes": 20},
        ]
    }
    merged, total = merge_hardware_union(
        existing,
        {"gpus": [{"uuid": "g3", "total_vram_bytes": 30}]},
        prior_assigned_uuids=["g1"],
        other_agent_uuids=["g1"],
    )
    assert {g["uuid"] for g in merged["gpus"]} == {"g1", "g2", "g3"}
    assert total == 60


def test_merge_union_drops_stale_when_no_survivor() -> None:
    # Control for M1: with no other claimant, the stale GPU IS dropped.
    existing = {
        "gpus": [
            {"uuid": "g1", "total_vram_bytes": 10},
            {"uuid": "g2", "total_vram_bytes": 20},
        ]
    }
    merged, total = merge_hardware_union(
        existing,
        {"gpus": [{"uuid": "g3", "total_vram_bytes": 30}]},
        prior_assigned_uuids=["g1"],
        other_agent_uuids=[],
    )
    assert {g["uuid"] for g in merged["gpus"]} == {"g2", "g3"}
    assert total == 50


def test_merge_union_no_gpus_in_play_preserves_total() -> None:
    # L1: neither side has a gpus list → total returned as None (no change).
    merged, total = merge_hardware_union(
        {},
        {"cpu": {"cores": 8}},
        prior_assigned_uuids=[],
        other_agent_uuids=[],
    )
    assert total is None
    assert "gpus" not in merged
    assert "total_vram_bytes" not in merged
    assert merged["cpu"] == {"cores": 8}


def test_merge_union_empty_reported_gpus_recomputes_surviving_sum() -> None:
    # L1/N2: a report WITH an empty gpus list against an existing union
    # recomputes to the surviving sum (not "no change").
    existing = {
        "gpus": [{"uuid": "g1", "total_vram_bytes": 10}],
        "total_vram_bytes": 10,
    }
    merged, total = merge_hardware_union(
        existing,
        {"gpus": []},
        prior_assigned_uuids=[],
        other_agent_uuids=[],
    )
    assert total == 10
    assert merged["total_vram_bytes"] == 10
    assert [g["uuid"] for g in merged["gpus"]] == ["g1"]
