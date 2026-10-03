"""Tests for the halogen-flash NPU small-model integration in the agent."""

import hashlib
import os
from unittest.mock import AsyncMock, Mock, patch
from uuid import uuid4

os.environ.setdefault("AGENT_ID", "test-agent")
os.environ.setdefault("FRONTEND_URL", "http://test:8000")
os.environ.setdefault("MODELS_PATH", "/tmp/models")

import pytest

from app.services.halogen_flash_server import (
    HALOGEN_FLASH_CHECKPOINT,
    HalogenFlashServerConfig,
    HalogenFlashServerManager,
)
from app.services.npu_models import (
    NPU_SUFFIXES,
    client_name,
    parse_npu_pins,
    resolve_npu_download_set,
)

SAMPLE_PINS = """
model decider-0.8b repo=peonist-ai/halogen-npu revision=v3
file decider-0.8b decider-0.8b.hnpw 1000 aaa111
file decider-0.8b tokenizer/tokenizer.json 200 bbb222
file decider-0.8b devices/devices.hnpm 300 ccc333
file decider-0.8b devices/kernel.elf 400 ddd444
model qwen3-embedding-0.6b repo=peonist-ai/halogen-npu revision=v3
file qwen3-embedding-0.6b qwen3-embedding-0.6b.hnpw 5000 eee555
file qwen3-embedding-0.6b tokenizer/tokenizer.json 210 fff666
file qwen3-embedding-0.6b devices/devices.hnpm 310 eee310
file qwen3-embedding-0.6b devices/kernel.elf 410 eee410
model qwen3-reranker-0.6b repo=peonist-ai/halogen-npu revision=v3 devices=qwen3-embedding-0.6b
file qwen3-reranker-0.6b qwen3-reranker-0.6b.hnpw 6000 ggg777
file qwen3-reranker-0.6b tokenizer/tokenizer.json 220 hhh888
"""


def test_client_names_are_stable() -> None:
    assert client_name("myflash", "qwen3-embedding-0.6b") == "myflash-embed"
    assert client_name("myflash", "qwen3-reranker-0.6b") == "myflash-rerank"
    assert client_name("myflash", "qwen3.5-2b") == "myflash-nano"
    assert client_name("myflash", "decider-0.8b") == "myflash-decide"
    assert client_name("myflash", "qwen3guard-gen-0.6b") == "myflash-guard"
    assert set(NPU_SUFFIXES.values()) == {
        "embed",
        "rerank",
        "nano",
        "decide",
        "guard",
    }


def test_parse_pins_model_and_file_lines() -> None:
    records = parse_npu_pins(SAMPLE_PINS)
    assert set(records) >= {
        "decider-0.8b",
        "qwen3-embedding-0.6b",
        "qwen3-reranker-0.6b",
    }
    decider = records["decider-0.8b"]
    assert decider.repo == "peonist-ai/halogen-npu"
    assert decider.revision == "v3"
    assert decider.devices_of == "decider-0.8b"
    assert ("decider-0.8b.hnpw", 1000, "aaa111") in decider.files
    assert decider.devices_files == [
        ("devices/devices.hnpm", 300, "ccc333"),
        ("devices/kernel.elf", 400, "ddd444"),
    ]
    reranker = records["qwen3-reranker-0.6b"]
    assert reranker.devices_of == "qwen3-embedding-0.6b"
    assert reranker.own_files == [
        ("qwen3-reranker-0.6b.hnpw", 6000, "ggg777"),
        ("tokenizer/tokenizer.json", 220, "hhh888"),
    ]


def test_resolve_download_set_pulls_shared_devices_from_owner() -> None:
    records = parse_npu_pins(SAMPLE_PINS)
    plan = resolve_npu_download_set(
        records, ["qwen3-embedding-0.6b", "qwen3-reranker-0.6b"]
    )
    # The reranker runs on the embedder's devices program.
    assert set(plan) == {"qwen3-embedding-0.6b", "qwen3-reranker-0.6b"}
    embed_repo, embed_rev, embed_files = plan["qwen3-embedding-0.6b"]
    assert embed_repo == "peonist-ai/halogen-npu"
    assert embed_rev == "v3"
    embed_paths = [p for p, _, _ in embed_files]
    assert "qwen3-embedding-0.6b.hnpw" in embed_paths
    assert "devices/devices.hnpm" in embed_paths
    assert "devices/kernel.elf" in embed_paths
    rerank_paths = [p for p, _, _ in plan["qwen3-reranker-0.6b"][2]]
    assert "qwen3-reranker-0.6b.hnpw" in rerank_paths
    assert not any(p.startswith("devices/") for p in rerank_paths)


def test_resolve_download_set_rejects_unknown_model() -> None:
    records = parse_npu_pins(SAMPLE_PINS)
    with pytest.raises(ValueError, match="no NPU pin record"):
        resolve_npu_download_set(records, ["does-not-exist"])


@pytest.mark.asyncio
@patch("app.services.halogen_flash_server.subprocess.Popen")
async def test_npu_models_env_is_comma_joined(mock_popen):
    manager = HalogenFlashServerManager()
    mock_popen.return_value = Mock()
    config = HalogenFlashServerConfig(
        model_path=HALOGEN_FLASH_CHECKPOINT,
        port=8091,
        api_port=8091,
        engine_port=8092,
        options={
            "npu_models": [
                "decider-0.8b",
                "qwen3-embedding-0.6b",
                "qwen3-reranker-0.6b",
            ]
        },
    )
    with (
        patch(
            "app.services.halogen_flash_server.model_manager.download_repository",
            new=AsyncMock(),
        ),
        patch.object(manager, "_wait_for_health", new=AsyncMock()),
    ):
        assert await manager.start_server("flash-npu", config) is True

    env = mock_popen.call_args.kwargs["env"]
    assert (
        env["HALOGEN_NPU_MODELS"]
        == "decider-0.8b,qwen3-embedding-0.6b,qwen3-reranker-0.6b"
    )
    # Never a Python list repr.
    assert "[" not in env["HALOGEN_NPU_MODELS"]
    # The engine keeps its default fabric-clock requirement.
    assert "HALOGEN_NPU_WITH_GPU" not in env


@pytest.mark.asyncio
@patch("app.services.halogen_flash_server.subprocess.Popen")
async def test_no_npu_models_sets_no_env(mock_popen):
    manager = HalogenFlashServerManager()
    mock_popen.return_value = Mock()
    config = HalogenFlashServerConfig(
        model_path=HALOGEN_FLASH_CHECKPOINT,
        port=8091,
        api_port=8091,
        engine_port=8092,
        options={},
    )
    with (
        patch(
            "app.services.halogen_flash_server.model_manager.download_repository",
            new=AsyncMock(),
        ),
        patch.object(manager, "_wait_for_health", new=AsyncMock()),
    ):
        assert await manager.start_server("flash-plain", config) is True
    env = mock_popen.call_args.kwargs["env"]
    assert "HALOGEN_NPU_MODELS" not in env


@pytest.mark.asyncio
async def test_prepare_downloads_npu_files(monkeypatch, tmp_path):
    from app.api.routes import servers as servers_route
    from app.services.model_manager import model_manager

    pins = tmp_path / "models.txt"
    pins.write_text(SAMPLE_PINS)
    monkeypatch.setattr(
        "app.api.routes.servers.settings.NPU_PINS_FILE", str(pins)
    )
    monkeypatch.setattr(
        "app.api.routes.servers.settings.AGENT_PLATFORM", "halogen-flash"
    )

    calls: list[tuple] = []

    async def fake_download(dir_id, files, repo_id, revision=None, **kwargs):  # noqa: ARG001
        calls.append((dir_id, [p for p, _, _ in files], repo_id, revision))
        return f"/models/npu/{dir_id}"

    with patch.object(model_manager, "download_npu_files", new=fake_download):
        result = await servers_route._prepare_npu_models(
            {"npu_models": ["qwen3-embedding-0.6b", "qwen3-reranker-0.6b"]},
            f"srv-{uuid4().hex[:6]}",
        )

    assert set(result) == {"qwen3-embedding-0.6b", "qwen3-reranker-0.6b"}
    dirs = {c[0] for c in calls}
    assert dirs == {"qwen3-embedding-0.6b", "qwen3-reranker-0.6b"}
    embed_call = next(c for c in calls if c[0] == "qwen3-embedding-0.6b")
    assert "devices/devices.hnpm" in embed_call[1]


@pytest.mark.asyncio
async def test_prepare_no_npu_is_noop() -> None:
    from app.api.routes import servers as servers_route

    assert await servers_route._prepare_npu_models({}, "srv-1") == []
    assert await servers_route._prepare_npu_models({"npu_models": []}, "srv-1") == []


@pytest.mark.asyncio
async def test_prepare_rejects_unknown_npu_model(monkeypatch, tmp_path) -> None:
    from fastapi import HTTPException

    from app.api.routes import servers as servers_route

    pins = tmp_path / "models.txt"
    pins.write_text(SAMPLE_PINS)
    monkeypatch.setattr(
        "app.api.routes.servers.settings.NPU_PINS_FILE", str(pins)
    )
    with pytest.raises(HTTPException) as exc:
        await servers_route._prepare_npu_models(
            {"npu_models": ["nope-1b"]}, "srv-1"
        )
    assert exc.value.status_code == 422


class TestNpuDownloadIntegrity:
    def test_matching_pin_is_recognised(self, monkeypatch, tmp_path) -> None:
        from app.services.model_manager import ModelManager

        monkeypatch.setattr(
            "app.services.model_manager.settings.MODELS_PATH", str(tmp_path)
        )
        mgr = ModelManager()
        body = b"hello-npu-weights"
        sha = hashlib.sha256(body).hexdigest()
        npu_dir = tmp_path / "npu" / "m"
        npu_dir.mkdir(parents=True)
        f = npu_dir / "m.hnpw"
        f.write_bytes(body)
        assert mgr._matches_pin(f, len(body), sha)

    def test_size_match_hash_mismatch_is_not_recognised(
        self, monkeypatch, tmp_path
    ) -> None:
        from app.services.model_manager import ModelManager

        monkeypatch.setattr(
            "app.services.model_manager.settings.MODELS_PATH", str(tmp_path)
        )
        mgr = ModelManager()
        npu_dir = tmp_path / "npu" / "m"
        npu_dir.mkdir(parents=True)
        f = npu_dir / "m.hnpw"
        f.write_bytes(b"wrong-content-but-same-size")
        assert not mgr._matches_pin(f, f.stat().st_size, "0" * 64)

    async def test_escapes_local_dir_rejected(self, monkeypatch, tmp_path) -> None:
        from app.services.model_manager import ModelManager

        monkeypatch.setattr(
            "app.services.model_manager.settings.MODELS_PATH", str(tmp_path)
        )
        mgr = ModelManager()
        with pytest.raises(ValueError, match="escapes"):
            await mgr.download_npu_files("m", [("../evil.bin", 5, "x")], "repo")

    async def test_stale_file_redownloaded(self, monkeypatch, tmp_path) -> None:
        from app.services.model_manager import ModelManager

        monkeypatch.setattr(
            "app.services.model_manager.settings.MODELS_PATH", str(tmp_path)
        )
        mgr = ModelManager()
        npu_dir = tmp_path / "npu" / "m"
        npu_dir.mkdir(parents=True)
        f = npu_dir / "m.hnpw"
        f.write_bytes(b"stale")

        downloaded: list = []

        async def fake_dl(*args):  # noqa: ARG001
            rel, target = args[1], args[3]
            downloaded.append((rel, str(target)))
            target.write_bytes(b"fresh")

        with patch.object(mgr, "_download_npu_file", new=fake_dl):
            await mgr.download_npu_files(
                "m",
                [("m.hnpw", 5, "0" * 64)],  # hash won't match "stale"
                "repo",
            )
        assert downloaded == [("m.hnpw", str(f))]
        assert f.read_bytes() == b"fresh"

