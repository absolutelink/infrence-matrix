"""Phase 25 S1: ``ModelSpec`` + ``models_from_entry`` parsing.

The agent-side parser prefers ``served_models`` (skipping invalid entries with a
warning) and falls back to the legacy single-model trio
(``alias``/``modality``/``backend_config``) for old admins.
"""

from __future__ import annotations

from provider_lib.models import DEFAULT_MODALITY, ModelSpec, models_from_entry


def test_legacy_trio_single_entry() -> None:
    specs = models_from_entry(
        {"alias": "solo", "modality": "embedding", "backend_config": {"x": 1}}
    )
    assert specs == [
        ModelSpec(
            name="solo", modality="embedding", backend_config={"x": 1}, enabled=True
        )
    ]


def test_legacy_trio_defaults() -> None:
    # Only an alias: modality defaults to llm, config to {}, enabled True.
    specs = models_from_entry({"alias": "solo"})
    assert specs == [
        ModelSpec(
            name="solo", modality=DEFAULT_MODALITY, backend_config={}, enabled=True
        )
    ]


def test_served_models_preferred_over_legacy() -> None:
    specs = models_from_entry(
        {
            "alias": "stale-alias",
            "modality": "llm",
            "served_models": [
                {"name": "a", "modality": "tts", "backend_config": {"slug": "A"}},
                {"name": "b", "modality": "asr", "enabled": False},
            ],
        }
    )
    assert specs == [
        ModelSpec(name="a", modality="tts", backend_config={"slug": "A"}, enabled=True),
        ModelSpec(name="b", modality="asr", backend_config={}, enabled=False),
    ]


def test_served_models_defaults_per_item() -> None:
    specs = models_from_entry({"served_models": [{"name": "only"}]})
    assert specs == [
        ModelSpec(
            name="only", modality=DEFAULT_MODALITY, backend_config={}, enabled=True
        )
    ]


def test_invalid_entries_skipped_with_valid_kept() -> None:
    specs = models_from_entry(
        {
            "alias": "legacy",
            "served_models": [
                "not-an-object",
                {"modality": "llm"},  # missing name
                {"name": ""},  # empty name
                {"name": None},  # null name
                {"name": "good", "modality": "embedding"},
            ],
        }
    )
    # Only the one valid entry survives; the rest are skipped (warned).
    assert specs == [
        ModelSpec(name="good", modality="embedding", backend_config={}, enabled=True)
    ]


def test_all_invalid_served_models_falls_back_to_legacy() -> None:
    specs = models_from_entry(
        {"alias": "legacy", "modality": "llm", "served_models": [{"nope": 1}, 5, None]}
    )
    assert specs == [
        ModelSpec(name="legacy", modality="llm", backend_config={}, enabled=True)
    ]


def test_empty_served_models_list_falls_back_to_legacy() -> None:
    specs = models_from_entry({"alias": "legacy", "served_models": []})
    assert specs == [
        ModelSpec(
            name="legacy", modality=DEFAULT_MODALITY, backend_config={}, enabled=True
        )
    ]


def test_no_alias_and_no_served_models_is_empty() -> None:
    assert models_from_entry({}) == []
    assert models_from_entry({"modality": "llm"}) == []


def test_non_dict_entry_is_empty() -> None:
    assert models_from_entry(None) == []  # type: ignore[arg-type]
    assert models_from_entry(["x"]) == []  # type: ignore[arg-type]


def test_bad_backend_config_type_degrades_to_empty_dict() -> None:
    specs = models_from_entry(
        {"served_models": [{"name": "a", "backend_config": "not-a-dict"}]}
    )
    assert specs == [
        ModelSpec(name="a", modality=DEFAULT_MODALITY, backend_config={}, enabled=True)
    ]
