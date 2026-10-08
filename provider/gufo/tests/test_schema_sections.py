"""Phase 12 gufo schema tests: the shipped schema.json is valid 2020-12,
a canonical sectioned backend_config validates against it, the frozen
fingerprint pins the consensus contract, the sections→options adapter +
argv build honor every wired field (including the bool-only emit rule
and aux-descriptor resolution), and the old flat shape (top-level
model/backend_port + options blob) is rejected at the top level."""

import json
from typing import Any

import pytest
from jsonschema import Draft202012Validator
from provider_lib.schema import load_schema, schema_fingerprint, validate_backend_config

from provider_gufo.command import (
    BOOL_FLAGS,
    CONFIG_SECTIONS,
    VALUE_FLAGS,
    build_command,
    flatten_args,
)
from provider_gufo.driver import AUX_KEYS, SCHEMA
from provider_gufo.main import PROVIDER_TYPE

# Canonical example backend_config for gufo (sectioned, Phase 12).
# Exercises every section with realistic values.
CANONICAL_CONFIG: dict[str, Any] = {
    "artifacts": {
        "model": {"source": "hf", "repo": "team/models", "file": "main.gguf"},
        "mmproj": {"source": "hf", "repo": "team/models", "file": "vision.gguf"},
        "dflash_model": {"source": "hf", "repo": "team/models", "file": "dflash.gguf"},
    },
    "context": {"context": 131072, "served_model_name": "my-alias"},
    "disk_cache": {
        "cache_disk": True,
        "cache_disk_bytes": 1073741824,
        "cache_disk_staging_bytes": 67108864,
    },
    "speculative": {
        "speculative": "dflash2",
        "draft_policy": "adaptive",
        "draft_tokens": 4,
        "min_draft_tokens": 1,
    },
    "sampling": {
        "temperature": 0.7,
        "top_k": 40,
        "top_p": 0.9,
        "min_p": 0.05,
        "min_keep": 1,
        "seed": 42,
        "repeat_penalty": 1.1,
        "repeat_last_n": 64,
        "frequency_penalty": 0.1,
        "presence_penalty": 0.0,
        "max_tokens": 2048,
    },
    "reasoning": {
        "think": "on",
        "reasoning_effort": "high",
        "preserve_thinking": "true",
    },
    "limits": {
        "prefill_chunk": 2048,
        "max_pending": 64,
        "max_pending_per_client": 8,
        "request_timeout_ms": 600000,
        "max_output_bytes": 33554432,
        "max_buffered_output_bytes": 8388608,
        "max_buffered_output_total": 67108864,
    },
    "server": {
        "sessions": 4,
        "max_connections": 128,
        "max_request_bytes": 16777216,
        "api_key": "s3cret",
        "verbose": True,
        "log_progress": False,
    },
}


def test_shipped_schema_loads_and_is_valid_2020_12() -> None:
    schema = load_schema("provider_gufo")
    assert schema == SCHEMA
    Draft202012Validator.check_schema(schema)
    assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    assert schema["additionalProperties"] is False


def test_schema_has_the_planned_sections() -> None:
    sections = list(SCHEMA["properties"])
    assert sections == [
        "artifacts",
        "context",
        "disk_cache",
        "speculative",
        "sampling",
        "reasoning",
        "limits",
        "server",
    ]
    for name, section in SCHEMA["properties"].items():
        assert section["title"], name
        assert isinstance(section.get("x-order"), int), name
    orders = [SCHEMA["properties"][k]["x-order"] for k in sections]
    assert orders == sorted(orders) == list(range(1, len(sections) + 1))
    # The adapter flattens exactly the non-artifacts sections.
    assert CONFIG_SECTIONS == tuple(s for s in sections if s != "artifacts")


# Frozen fingerprint of the shipped provider_gufo/schema.json — the
# fleet consensus contract (docs/ws-protocol.md §2). If you change
# schema.json, every known gufo instance must re-register with the new
# file (or the operator force-commits); update this pin deliberately.
# Port model overhaul: the operator-facing `server.backend_port` field was
# removed (the engine now binds an OS-assigned private port), bumping the fp.
SHIPPED_SCHEMA_FP = "cbe7abf16b55f505eaf8118489f61ce31b005802382b1e2694e7ac580ea28429"


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
    assert validate_backend_config(SCHEMA, {"context": {"context": 4096}}) == []


def test_schema_rejects_old_flat_shape() -> None:
    # The pre-Phase-12 flat shape (top-level model/backend_port +
    # options blob) must be rejected by additionalProperties: false.
    old = {
        "model": {"source": "hf", "repo": "r", "file": "f.gguf"},
        "backend_port": 8123,
        "options": {"context": 131072, "sessions": 4},
    }
    errors = validate_backend_config(SCHEMA, old)
    joined = " ".join(errors)
    for key in ("model", "backend_port", "options"):
        assert key in joined, key


def test_schema_rejects_unknown_keys_in_sections() -> None:
    cfg = json.loads(json.dumps(CANONICAL_CONFIG))
    cfg["context"]["nonsense"] = 1
    cfg["sampling"]["top_m"] = 5
    errors = validate_backend_config(SCHEMA, cfg)
    assert any("nonsense" in e for e in errors)
    assert any("top_m" in e for e in errors)


def test_schema_rejects_out_of_range_and_bad_types() -> None:
    cfg = json.loads(json.dumps(CANONICAL_CONFIG))
    cfg["context"]["context"] = 0
    cfg["server"]["sessions"] = 0
    cfg["disk_cache"]["cache_disk"] = "yes"  # must be boolean
    errors = validate_backend_config(SCHEMA, cfg)
    joined = " ".join(errors)
    for token in ("context", "sessions", "cache_disk"):
        assert token in joined, token


def test_schema_rejects_server_backend_port() -> None:
    """Port model overhaul: `server.backend_port` is no longer an accepted
    schema property (the engine binds an OS-assigned private port). The `server`
    section's additionalProperties:false rejects it."""
    cfg = json.loads(json.dumps(CANONICAL_CONFIG))
    cfg["server"]["backend_port"] = 8123
    errors = validate_backend_config(SCHEMA, cfg)
    assert any("backend_port" in e for e in errors)
    assert "backend_port" not in SCHEMA["properties"]["server"]["properties"]


def test_bool_only_flags_are_boolean_with_false_default() -> None:
    for key in BOOL_FLAGS:
        field = SCHEMA["properties"]["server"]["properties"][key]
        assert field["type"] == "boolean", key
        assert field["default"] is False, key


# Every schema field must be reachable through the executable mapping:
# VALUE_FLAGS, BOOL_FLAGS, the fixed base argv, the driver-resolved
# artifacts, or the port reader.
FIXED_ARGV_FLAGS = {"--host", "--port", "--model", "--cache-disk"}


def _field_flags() -> tuple[set[str], set[str]]:
    wired: set[str] = set()
    unwired: set[str] = set()
    for section in SCHEMA["properties"].values():
        for field in section["properties"].values():
            flag = field.get("x-flag")
            if not flag:
                continue
            (unwired if field.get("x-supported") is False else wired).add(flag)
    return wired, unwired


def test_every_field_has_description_and_flag() -> None:
    for section_name, section in SCHEMA["properties"].items():
        for key, field in section["properties"].items():
            label = f"{section_name}.{key}"
            assert field.get("description"), label
            assert field.get("title"), label
            # Artifact fields are $ref-based (descriptor objects carry no
            # scalar default).
            no_scalar_default = "$ref" in field
            assert "default" in field or no_scalar_default, label
            assert field.get("x-flag"), label
            assert str(field["x-flag"]).startswith("--"), label
            if field.get("x-supported") is not False:
                reachable = (
                    field["x-flag"] in VALUE_FLAGS.values()
                    or field["x-flag"] in BOOL_FLAGS.values()
                    or field["x-flag"] in FIXED_ARGV_FLAGS
                )
                assert reachable, label


def test_all_command_mapped_keys_are_wired_schema_fields() -> None:
    wired, unwired = _field_flags()
    assert unwired == set()
    # Every VALUE_FLAGS / BOOL_FLAGS target is a schema field flag.
    assert set(VALUE_FLAGS.values()) <= wired
    assert set(BOOL_FLAGS.values()) <= wired
    # The aux descriptor keys map through VALUE_FLAGS with driver-
    # resolved local paths.
    for key in AUX_KEYS:
        assert key in VALUE_FLAGS, key
    total_fields = sum(len(sec["properties"]) for sec in SCHEMA["properties"].values())
    assert len(wired) == total_fields


def test_api_key_is_marked_secret() -> None:
    api_key = SCHEMA["properties"]["server"]["properties"]["api_key"]
    assert api_key.get("x-secret") is True


# --- argv spot-checks per section ---------------------------------------


def _cmd(cfg: dict[str, Any], **kw: Any) -> list[str]:
    defaults: dict[str, Any] = {
        "model_path": "/models/main.gguf",
        "port": 8123,
    }
    defaults.update(kw)
    return build_command(flatten_args(cfg), **defaults)  # type: ignore[arg-type]


def test_argv_spotcheck_context_section() -> None:
    joined = " ".join(_cmd({"context": {"context": 131072, "served_model_name": "a"}}))
    assert "--context 131072" in joined
    assert "--served-model-name a" in joined


def test_argv_spotcheck_disk_cache_creates_per_instance_dir(tmp_path) -> None:
    joined = " ".join(
        _cmd(
            {"disk_cache": {"cache_disk": True, "cache_disk_bytes": 1024}},
            cache_dir=tmp_path,
            instance_id="srv-7",
        )
    )
    assert f"--cache-disk {tmp_path / 'srv-7'}" in joined
    assert "--cache-disk-bytes 1024" in joined
    assert (tmp_path / "srv-7").is_dir()
    # cache_disk false → no flag, no dir.
    joined = " ".join(
        _cmd(
            {"disk_cache": {"cache_disk": False}},
            cache_dir=tmp_path,
            instance_id="srv-off",
        )
    )
    assert "--cache-disk" not in joined
    assert not (tmp_path / "srv-off").exists()


def test_argv_spotcheck_speculative_section() -> None:
    joined = " ".join(
        _cmd(
            {
                "speculative": {
                    "speculative": "dflash2",
                    "draft_policy": "adaptive",
                    "draft_tokens": 4,
                    "min_draft_tokens": 1,
                }
            }
        )
    )
    assert "--speculative dflash2" in joined
    assert "--draft-policy adaptive" in joined
    assert "--draft-tokens 4" in joined
    assert "--min-draft-tokens 1" in joined


def test_argv_spotcheck_sampling_and_reasoning_sections() -> None:
    joined = " ".join(
        _cmd(
            {
                "sampling": {"temperature": 0.3, "top_k": 20, "max_tokens": 512},
                "reasoning": {"think": "on", "reasoning_effort": "low"},
            }
        )
    )
    assert "--temperature 0.3" in joined
    assert "--top-k 20" in joined
    assert "--max-tokens 512" in joined
    assert "--think on" in joined
    assert "--reasoning-effort low" in joined


def test_argv_spotcheck_limits_and_server_sections() -> None:
    joined = " ".join(
        _cmd(
            {
                "limits": {"prefill_chunk": 1024, "max_pending": 16},
                "server": {
                    "sessions": 3,
                    "max_connections": 64,
                    "max_request_bytes": 1048576,
                    "api_key": "k9",
                },
            }
        )
    )
    assert "--prefill-chunk 1024" in joined
    assert "--max-pending 16" in joined
    assert "--sessions 3" in joined
    assert "--max-connections 64" in joined
    assert "--max-request-bytes 1048576" in joined
    assert "--api-key k9" in joined


def test_argv_bool_only_emit_rule() -> None:
    # verbose/log_progress emit ONLY when true (false/None omitted so gufo
    # keeps its default) — the schema default is false, matching.
    assert "--verbose" in _cmd({"server": {"verbose": True}})
    assert "--verbose" not in _cmd({"server": {"verbose": False}})
    assert "--verbose" not in _cmd({"server": {"verbose": None}})
    assert "--log-progress" in _cmd({"server": {"log_progress": True}})
    assert "--log-progress" not in _cmd({"server": {"log_progress": False}})


def test_canonical_config_flattens_into_argv_completely() -> None:
    """Every wired (x-supported != false) non-artifacts schema field with
    a value in the canonical config emits its exact CLI flag pair (or
    bare flag for bool-only)."""
    joined = " ".join(_cmd(CANONICAL_CONFIG))
    for section_name, section in SCHEMA["properties"].items():
        if section_name == "artifacts":
            continue
        cfg_section = CANONICAL_CONFIG.get(section_name) or {}
        for key, field in section["properties"].items():
            if field.get("x-supported") is False or key not in cfg_section:
                continue
            flag = field["x-flag"]
            value = cfg_section[key]
            if key == "cache_disk":
                # Special-cased: --cache-disk takes the per-instance dir,
                # not the raw value (covered by its own spot-check).
                continue
            if flag in BOOL_FLAGS.values():
                if value is True:
                    assert flag in joined, f"{section_name}.{key} -> {flag}"
                else:
                    assert flag not in joined, f"{section_name}.{key} -> {flag}"
            else:
                assert f"{flag} {value}" in joined, f"{section_name}.{key} -> {flag}"


def test_flatten_args_merges_sections_and_prefers_them() -> None:
    flat = flatten_args(
        {
            "options": {"context": 1024, "sessions": 1},
            "context": {"context": 8192},
            "server": {"sessions": 4},
        }
    )
    assert flat["context"] == 8192
    assert flat["sessions"] == 4
    # artifacts never flatten into CLI options.
    assert "artifacts" not in flatten_args(CANONICAL_CONFIG)


# --- driver: artifacts section + aux resolution + port ------------


def test_driver_reads_nested_sections(tmp_path) -> None:
    from conftest import make_settings

    from provider_gufo.driver import GufoBackend

    settings = make_settings(tmp_path, "gufo")
    driver = GufoBackend(settings, {})
    driver.apply_config(CANONICAL_CONFIG)
    assert driver._config == CANONICAL_CONFIG
    # Port model overhaul: the engine port is NOT config-derived anymore — it is
    # allocated (OS-assigned free port) at start, so it stays None until then.
    assert driver.backend_port is None
    driver.apply_config({"context": {"context": 512}})
    assert driver.backend_port is None
    # effective_capacity from the flattened `server` section.
    driver.apply_config({"server": {"sessions": 6}})
    assert driver.effective_capacity == 6
    driver.apply_config({"options": {"sessions": 2}})
    assert driver.effective_capacity == 2


def test_apply_config_resets_resolved_artifacts(tmp_path) -> None:
    from conftest import make_settings

    from provider_gufo.driver import GufoBackend

    driver = GufoBackend(make_settings(tmp_path, "gufo"), {})
    driver.resolved_artifacts = ["/models/old.gguf"]
    driver.apply_config(CANONICAL_CONFIG)
    assert driver.resolved_artifacts == []


async def test_revision_descriptor_reaches_downloader(tmp_path, monkeypatch) -> None:
    """The schema's optional `revision` on hfFile descriptors is passed
    through to provider_lib.downloader.ensure_artifact for the main
    model and every aux descriptor."""
    from conftest import make_settings

    from provider_gufo import driver as driver_mod

    calls: list[dict[str, Any]] = []

    async def fake_ensure_artifact(descriptor, **kwargs):
        calls.append({"descriptor": descriptor, **kwargs})
        return f"/models/{descriptor.get('file', 'x')}"

    monkeypatch.setattr(driver_mod, "ensure_artifact", fake_ensure_artifact)

    driver = driver_mod.GufoBackend(make_settings(tmp_path, "gufo"), {})
    cfg = json.loads(json.dumps(CANONICAL_CONFIG))
    cfg["artifacts"] = {
        "model": {"source": "hf", "repo": "r", "file": "m.gguf", "revision": "v2"},
        "mmproj": {"source": "hf", "repo": "r", "file": "mm.gguf"},
        "dflash_model": {
            "source": "hf",
            "repo": "r",
            "file": "df.gguf",
            "revision": "  ",
        },
        "dspark_model": {"source": "hf", "repo": "r", "file": "ds.gguf"},
        "mtp_model": {"source": "hf", "repo": "r", "file": "mtp.gguf"},
    }
    driver.apply_config(cfg)
    model_path = await driver._resolve_main_model()
    options = await driver._resolve_aux()
    assert model_path == "/models/m.gguf"
    assert options["mmproj"] == "/models/mm.gguf"
    assert options["dflash_model"] == "/models/df.gguf"
    assert options["dspark_model"] == "/models/ds.gguf"
    assert options["mtp_model"] == "/models/mtp.gguf"
    by_file = {c["descriptor"]["file"]: c for c in calls}
    assert by_file["m.gguf"]["revision"] == "v2"
    # Absent / blank revision → None (downloader default), never "" or "  ".
    assert by_file["mm.gguf"]["revision"] is None
    assert by_file["df.gguf"]["revision"] is None


async def test_legacy_options_aux_descriptor_fallback(tmp_path, monkeypatch) -> None:
    """L-shape: aux descriptors still under the legacy `options.*`
    positions resolve via the driver's fallback and are rewritten to
    local paths in the flattened options."""
    from conftest import make_settings

    from provider_gufo import driver as driver_mod

    seen: list[Any] = []

    async def fake_ensure_artifact(descriptor, **_kwargs):
        seen.append(descriptor)
        return str(descriptor["path"])

    monkeypatch.setattr(driver_mod, "ensure_artifact", fake_ensure_artifact)
    aux = tmp_path / "models" / "vision.gguf"
    aux.parent.mkdir(parents=True, exist_ok=True)
    aux.write_bytes(b"mmproj-fake")
    legacy = {
        "model": {"path": str(tmp_path / "models" / "m.gguf")},
        "options": {"mmproj": {"path": str(aux)}, "dspark_model": "/preplaced.gguf"},
    }
    (tmp_path / "models" / "m.gguf").write_bytes(b"gguf")
    driver = driver_mod.GufoBackend(make_settings(tmp_path, "gufo"), legacy)
    options = await driver._resolve_aux()
    assert options["mmproj"] == str(aux)
    # Plain strings pass through untouched (pre-placed file semantics).
    assert options["dspark_model"] == "/preplaced.gguf"
    assert seen == [{"path": str(aux)}]


async def test_sectioned_aux_resolution_start_shape(tmp_path, monkeypatch) -> None:
    """Phase 12: aux descriptors from `artifacts` land in the flattened
    options consumed by build_command (end-to-end argv check)."""
    from conftest import make_settings

    from provider_gufo import driver as driver_mod

    async def fake_ensure_artifact(descriptor, **_kwargs):
        return f"/models/{descriptor.get('file', 'x')}"

    monkeypatch.setattr(driver_mod, "ensure_artifact", fake_ensure_artifact)
    driver = driver_mod.GufoBackend(make_settings(tmp_path, "gufo"), {})
    driver.apply_config(CANONICAL_CONFIG)
    options = await driver._resolve_aux()
    cmd = build_command(options, model_path="/models/main.gguf", port=8123)
    joined = " ".join(cmd)
    assert "--mmproj /models/vision.gguf" in joined
    assert "--dflash-model /models/dflash.gguf" in joined
    assert "--served-model-name my-alias" in joined
    assert "--sessions 4" in joined
    assert "--verbose" in joined  # canonical has verbose: true
    assert "--log-progress" not in joined  # canonical has log_progress: false


async def test_cache_disk_uses_real_instance_id_after_registration(
    tmp_path, monkeypatch
) -> None:
    """M2: --cache-disk is keyed by the REAL registered instance id, so
    two gufo definitions on one machine never share a disk-cache dir."""
    from conftest import make_settings

    from provider_gufo import driver as driver_mod

    captured: dict[str, Any] = {}

    class _NoProc:
        stdout = None
        stderr = None
        pid = 4242

        def poll(self):
            return 0

        def wait(self, timeout=None):
            return 0

    def fake_popen(cmd, **_kwargs):
        captured["cmd"] = cmd
        return _NoProc()

    monkeypatch.setattr(driver_mod.subprocess, "Popen", fake_popen)

    # Never actually poll health: fail fast after argv capture.
    async def fake_wait_health(_self):
        raise RuntimeError("stop-after-argv")

    monkeypatch.setattr(driver_mod.GufoBackend, "_wait_for_health", fake_wait_health)

    settings = make_settings(tmp_path, "gufo")
    model = tmp_path / "models" / "m.gguf"
    model.write_bytes(b"gguf")
    driver = driver_mod.GufoBackend(
        settings,
        {
            "artifacts": {"model": {"path": str(model)}},
            "disk_cache": {"cache_disk": True},
        },
    )
    driver.set_instance_id("11111111-2222-3333-4444-555555555555")
    with pytest.raises(RuntimeError, match="stop-after-argv"):
        await driver.start()
    joined = " ".join(captured["cmd"])
    assert (
        f"--cache-disk {settings.CACHE_DIR / '11111111-2222-3333-4444-555555555555'}"
        in joined
    )
    assert str(settings.MACHINE_UID) not in joined.split("--cache-disk")[1].split()[0]


async def test_cache_disk_falls_back_to_machine_uid_pre_registration(
    tmp_path, monkeypatch
) -> None:
    """Pre-registration (no set_instance_id call), the disk cache dir is
    MACHINE_UID-scoped — the documented fallback."""
    from conftest import make_settings

    from provider_gufo import driver as driver_mod

    captured: dict[str, Any] = {}

    class _NoProc:
        stdout = None
        stderr = None
        pid = 4243

        def poll(self):
            return 0

        def wait(self, timeout=None):
            return 0

    def fake_popen(cmd, **_kwargs):
        captured["cmd"] = cmd
        return _NoProc()

    monkeypatch.setattr(driver_mod.subprocess, "Popen", fake_popen)

    async def fake_wait_health(_self):
        raise RuntimeError("stop-after-argv")

    monkeypatch.setattr(driver_mod.GufoBackend, "_wait_for_health", fake_wait_health)

    settings = make_settings(tmp_path, "gufo")
    model = tmp_path / "models" / "m.gguf"
    model.write_bytes(b"gguf")
    driver = driver_mod.GufoBackend(
        settings,
        {
            "artifacts": {"model": {"path": str(model)}},
            "disk_cache": {"cache_disk": True},
        },
    )
    with pytest.raises(RuntimeError, match="stop-after-argv"):
        await driver.start()
    joined = " ".join(captured["cmd"])
    assert f"--cache-disk {settings.CACHE_DIR / settings.MACHINE_UID}" in joined


def test_apply_registration_sets_instance_id(tmp_path) -> None:
    """apply_registration() records the registration response's
    instance_id on the driver (M2 wiring)."""
    from conftest import make_settings
    from provider_lib.admin_client import RegistrationResult

    from provider_gufo.main import apply_registration, make_lifecycle

    settings = make_settings(tmp_path, "gufo")
    lifecycle = make_lifecycle(_StubClient(settings), {})
    result = RegistrationResult(
        {
            "agent_id": "aaaaaaaa-2222-3333-4444-555555555555",
            "agent_secret": "s",
            "machine": {"uid": "u"},
            "backends": [
                {
                    "instance_id": "cafe-uuid",
                    "port": 8081,
                    "definition": {"capacity": 1},
                }
            ],
        }
    )
    apply_registration(lifecycle, result)
    assert lifecycle.driver.instance_id == "cafe-uuid"


class _StubClient:
    def __init__(self, settings):
        self.settings = settings

    async def send_event(self, *_a, **_k):
        pass


def test_provider_type_constant_matches_schema_context() -> None:
    assert PROVIDER_TYPE == "gufo"
