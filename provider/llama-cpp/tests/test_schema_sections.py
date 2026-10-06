"""Phase 12 llama-cpp schema tests: the shipped schema.json is valid
2020-12, a canonical sectioned backend_config validates against it, the
frozen fingerprint pins the consensus contract, and the driver + command
builder read the new nested shape."""

import json
from typing import Any

from jsonschema import Draft202012Validator
from provider_lib.schema import load_schema, schema_fingerprint, validate_backend_config

from provider_llama_cpp.command import build_llama_command, flatten_args
from provider_llama_cpp.driver import SCHEMA
from provider_llama_cpp.main import PROVIDER_TYPE

# Canonical example backend_config for llama-cpp (sectioned, Phase 12).
CANONICAL_CONFIG: dict[str, Any] = {
    "artifacts": {
        "model": {
            "source": "hf",
            "repo": "ggml-org/models",
            "file": "gemma/ggml-model.gguf",
        },
        "mmproj": {"path": "/models/gemma/mmproj.gguf"},
    },
    "context": {"ctx": 8192, "predict": -1, "keep": 0},
    "kv_cache": {
        "cache_prompt": True,
        "cache_reuse": 4,
        "cache_type_k": "q8_0",
        "cache_type_v": "f16",
        "kv_offload": True,
        "ctx_checkpoints": 32,
        "checkpoint_min_step": 8192,
        "cache_ram": 8192,
        "cache_idle_slots": True,
    },
    "loading": {"load_mode": "mmap"},
    "gpu": {
        "gpu_layers": "auto",
        "device": "CUDA0",
        "tensor_split": "1,1",
        "main_gpu": 0,
        "fit": "on",
        "fit_ctx": 4096,
    },
    "speculative": {
        "spec_type": "draft-mtp",
        "spec_draft_n_max": 3,
        "spec_draft_p_min": 0.0,
    },
    "sampling": {
        "temperature": 0.8,
        "top_k": 40,
        "top_p": 0.95,
        "min_p": 0.05,
        "seed": -1,
        "repeat_penalty": 1.0,
    },
    "reasoning": {"reasoning": "on", "reasoning_budget": -1, "jinja": True},
    "vision": {"image_max_tokens": 1024},
    "server": {
        "backend_port": 9999,
        "threads": -1,
        "batch_size": 2048,
        "ubatch_size": 512,
        "parallel": 1,
        "cont_batching": True,
        "warmup": True,
        "flash_attn": "auto",
    },
    "endpoints": {"host": "127.0.0.1", "sse_ping_interval": 30},
    "logging": {"verbosity": 3},
}


def test_shipped_schema_loads_and_is_valid_2020_12() -> None:
    schema = load_schema("provider_llama_cpp")
    assert schema == SCHEMA
    Draft202012Validator.check_schema(schema)
    assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    assert schema["additionalProperties"] is False


def test_schema_has_the_planned_sections() -> None:
    sections = schema_keys()
    assert sections == [
        "artifacts",
        "context",
        "kv_cache",
        "loading",
        "gpu",
        "speculative",
        "sampling",
        "reasoning",
        "vision",
        "server",
        "endpoints",
        "logging",
    ]
    for name, section in SCHEMA["properties"].items():
        assert section["title"], name
        assert isinstance(section.get("x-order"), int), name
    # x-orders are unique and match the declaration order.
    orders = [SCHEMA["properties"][k]["x-order"] for k in sections]
    assert orders == sorted(orders) == list(range(1, len(sections) + 1))


def schema_keys() -> list[str]:
    return list(SCHEMA["properties"])


# Frozen fingerprint of the shipped provider_llama_cpp/schema.json — the
# fleet consensus contract (docs/ws-protocol.md §2). If you change
# schema.json, every known llama-cpp instance must re-register with the
# new file (or the operator force-commits); update this pin deliberately.
SHIPPED_SCHEMA_FP = "0af969e61cb3c535aadef910f78df51c408e5b880e315ad42aa2152f79e45cb6"


def test_schema_fingerprint_is_stable() -> None:
    fp = schema_fingerprint(SCHEMA)
    assert fp == SHIPPED_SCHEMA_FP
    # Round-trip determinism (serialization-independent digest).
    assert fp == schema_fingerprint(json.loads(json.dumps(SCHEMA)))
    assert len(fp) == 64


def test_canonical_config_validates() -> None:
    assert validate_backend_config(SCHEMA, CANONICAL_CONFIG) == []


def test_artifacts_required_model() -> None:
    cfg = json.loads(json.dumps(CANONICAL_CONFIG))
    del cfg["artifacts"]["model"]
    errors = validate_backend_config(SCHEMA, cfg)
    assert any("model" in e for e in errors)


def test_empty_config_validates() -> None:
    # Every section is optional at the top level; only artifacts.model is
    # required when the artifacts section is present.
    assert validate_backend_config(SCHEMA, {}) == []
    assert validate_backend_config(SCHEMA, {"context": {"ctx": 128}}) == []


def test_schema_rejects_old_flat_shape() -> None:
    # The pre-Phase-12 flat shape (top-level model/args/backend_port)
    # must be rejected by additionalProperties: false.
    old = {
        "model": {"source": "hf", "repo": "r", "file": "f.gguf"},
        "args": {"ctx": 8192, "gpu_layers": 35},
        "backend_port": 9999,
    }
    errors = validate_backend_config(SCHEMA, old)
    assert errors
    joined = " ".join(errors)
    assert "model" in joined and "args" in joined and "backend_port" in joined


def test_schema_rejects_unknown_keys_in_sections() -> None:
    cfg = json.loads(json.dumps(CANONICAL_CONFIG))
    cfg["server"]["nonsense"] = 1
    cfg["sampling"]["prompt_cache"] = True
    errors = validate_backend_config(SCHEMA, cfg)
    assert any("nonsense" in e for e in errors)
    assert any("prompt_cache" in e for e in errors)


def test_schema_rejects_out_of_range_and_bad_enum() -> None:
    cfg = json.loads(json.dumps(CANONICAL_CONFIG))
    cfg["context"]["ctx"] = 0
    cfg["sampling"]["temperature"] = 3.0
    cfg["server"]["flash_attn"] = "sometimes"
    cfg["gpu"]["gpu_layers"] = "lots"
    errors = validate_backend_config(SCHEMA, cfg)
    assert any("ctx" in e for e in errors)
    assert any("temperature" in e for e in errors)
    assert any("flash_attn" in e for e in errors)
    assert any("gpu_layers" in e for e in errors)


def test_gpu_layers_accepts_int_auto_all() -> None:
    for value in (0, 35, 1024, "auto", "all"):
        cfg = json.loads(json.dumps(CANONICAL_CONFIG))
        cfg["gpu"]["gpu_layers"] = value
        assert validate_backend_config(SCHEMA, cfg) == [], value


def test_obsolete_flags_absent_from_schema() -> None:
    # No schema field may map to an obsolete llama.cpp flag (descriptions
    # mentioning them as removed history are fine; x-flag is the contract).
    x_flags = {
        f.get("x-flag")
        for section in SCHEMA["properties"].values()
        for f in section["properties"].values()
    }
    for obsolete in ("--prompt-cache", "--mmap", "--no-mmap", "--defrag-thold"):
        assert obsolete not in x_flags


def test_every_field_has_description_and_flag_or_supported() -> None:
    for section_name, section in SCHEMA["properties"].items():
        for key, field in section["properties"].items():
            label = f"{section_name}.{key}"
            assert field.get("description"), label
            # backend_port has no schema default (env-derived PROVIDER_PORT+1).
            assert "default" in field or "$ref" in field or key == "backend_port", label
            if section_name != "artifacts":
                assert field.get("x-flag"), label


def test_flatten_args_merges_sections_and_prefers_them() -> None:
    flat = flatten_args(
        {
            "args": {"ctx": 1024, "threads": 2},
            "context": {"ctx": 8192},
            "server": {"threads": 8},
        }
    )
    assert flat["ctx"] == 8192
    assert flat["threads"] == 8
    # artifacts never flatten into CLI args.
    assert "artifacts" not in flatten_args(CANONICAL_CONFIG)


def test_command_builder_from_canonical_sectioned_config() -> None:
    cmd = build_llama_command(
        CANONICAL_CONFIG,
        model_path="/models/ggml-model.gguf",
        port=9999,
        mmproj_path="/models/mmproj.gguf",
    )
    joined = " ".join(cmd)
    assert "--prompt-cache" not in joined
    assert cmd[cmd.index("--ctx-size") + 1] == "8192"
    assert cmd[cmd.index("--n-gpu-layers") + 1] == "auto"
    assert cmd[cmd.index("--flash-attn") + 1] == "auto"
    assert cmd[cmd.index("--cache-type-k") + 1] == "q8_0"
    assert cmd[cmd.index("--batch-size") + 1] == "2048"
    assert cmd[cmd.index("--temperature") + 1] == "0.8"
    assert cmd[cmd.index("--reasoning") + 1] == "on"
    assert cmd[cmd.index("--load-mode") + 1] == "mmap"
    assert cmd[cmd.index("--mmproj") + 1] == "/models/mmproj.gguf"
    assert "--jinja" in cmd
    # draft-mtp explicit path (no draft artifact).
    assert cmd[cmd.index("--spec-type") + 1] == "draft-mtp"
    assert cmd[cmd.index("--spec-draft-n-max") + 1] == "3"


def test_driver_reads_nested_sections(tmp_path) -> None:
    from conftest import make_settings

    settings = make_settings(tmp_path, "llama-server")
    driver = _driver_with_schema(settings)
    driver.apply_config(CANONICAL_CONFIG)
    assert driver._config == CANONICAL_CONFIG
    # backend_port relocated under `server`.
    assert driver.backend_port == 9999
    driver.apply_config({"context": {"ctx": 512}})
    assert driver.backend_port == settings.PROVIDER_PORT + 1
    # Legacy top-level backend_port still honored.
    driver.apply_config({"backend_port": 7777})
    assert driver.backend_port == 7777


def _driver_with_schema(settings: Any) -> Any:
    from provider_llama_cpp.driver import LlamaCppBackend

    return LlamaCppBackend(settings, {})


def test_apply_config_resets_resolved_artifacts(tmp_path) -> None:
    from conftest import make_settings

    driver = _driver_with_schema(make_settings(tmp_path, "llama-server"))
    driver.resolved_artifacts = ["/models/old.gguf"]
    driver.apply_config(CANONICAL_CONFIG)
    assert driver.resolved_artifacts == []


async def test_revision_descriptor_reaches_downloader(tmp_path, monkeypatch) -> None:
    """H2: the schema's optional `revision` on hfFile descriptors is
    passed through to provider_lib.downloader.ensure_artifact for
    model/mmproj/draft uniformly."""
    from conftest import make_settings

    from provider_llama_cpp import driver as driver_mod

    calls: list[dict[str, Any]] = []

    async def fake_ensure_artifact(descriptor, **kwargs):
        calls.append({"descriptor": descriptor, **kwargs})
        return f"/models/{descriptor.get('file', 'x.gguf')}"

    monkeypatch.setattr(driver_mod, "ensure_artifact", fake_ensure_artifact)

    driver = driver_mod.LlamaCppBackend(make_settings(tmp_path, "llama-server"), {})
    cfg = json.loads(json.dumps(CANONICAL_CONFIG))
    cfg["artifacts"] = {
        "model": {
            "source": "hf",
            "repo": "ggml-org/models",
            "file": "m.gguf",
            "revision": "v2",
        },
        "mmproj": {
            "source": "hf",
            "repo": "ggml-org/models",
            "file": "mm.gguf",
            "revision": "abc123",
        },
        "draft": {
            "source": "hf",
            "repo": "ggml-org/models",
            "file": "d.gguf",
        },
    }
    driver.apply_config(cfg)
    model_path, mmproj_path, draft_path = await driver._resolve_artifacts()
    assert (model_path, mmproj_path, draft_path) == (
        "/models/m.gguf",
        "/models/mm.gguf",
        "/models/d.gguf",
    )
    assert len(calls) == 3
    by_file = {c["descriptor"]["file"]: c for c in calls}
    assert by_file["m.gguf"]["revision"] == "v2"
    assert by_file["mm.gguf"]["revision"] == "abc123"
    # Absent revision → None (downloader default), never "" or missing kwarg.
    assert by_file["d.gguf"]["revision"] is None


async def test_blank_revision_normalized_to_none(tmp_path, monkeypatch) -> None:
    from conftest import make_settings

    from provider_llama_cpp import driver as driver_mod

    calls: list[dict[str, Any]] = []

    async def fake_ensure_artifact(_descriptor, **kwargs):
        calls.append(kwargs)
        return "/models/x.gguf"

    monkeypatch.setattr(driver_mod, "ensure_artifact", fake_ensure_artifact)
    driver = driver_mod.LlamaCppBackend(
        make_settings(tmp_path, "llama-server"),
        {"artifacts": {"model": {"path": "/models/x.gguf", "revision": "   "}}},
    )
    await driver._resolve_artifacts()
    assert calls[0]["revision"] is None


async def test_legacy_top_level_artifact_fallback(tmp_path, monkeypatch) -> None:
    """L5: hand-rolled configs with top-level model/mmproj/draft (no
    `artifacts` section) still resolve via the driver's fallback."""
    from conftest import make_settings

    from provider_llama_cpp import driver as driver_mod

    seen: list[Any] = []

    async def fake_ensure_artifact(descriptor, **_kwargs):
        seen.append(descriptor)
        return str(descriptor["path"])

    monkeypatch.setattr(driver_mod, "ensure_artifact", fake_ensure_artifact)
    legacy = {
        "model": {"path": "/m/main.gguf"},
        "mmproj": {"path": "/m/mm.gguf"},
        "draft": {"path": "/m/draft.gguf"},
    }
    driver = driver_mod.LlamaCppBackend(make_settings(tmp_path, "llama-server"), legacy)
    model_path, mmproj_path, draft_path = await driver._resolve_artifacts()
    assert (model_path, mmproj_path, draft_path) == (
        "/m/main.gguf",
        "/m/mm.gguf",
        "/m/draft.gguf",
    )
    assert seen == [legacy["model"], legacy["mmproj"], legacy["draft"]]
    assert driver.resolved_artifacts == ["/m/main.gguf", "/m/mm.gguf", "/m/draft.gguf"]


def test_provider_type_constant_matches_schema_context() -> None:
    assert PROVIDER_TYPE == "llama-cpp"
