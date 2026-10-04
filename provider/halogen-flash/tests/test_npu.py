"""NPU small-model helpers + probe tests (no real NPU: fake paths)."""

import pytest

from provider_halogen_flash.npu import (
    NPU_CAPABILITIES,
    NPU_SUFFIXES,
    STOCK_NPU_MODEL_IDS,
    UPSTREAM_BY_SUFFIX,
    client_name,
    parse_npu_pins,
    probe_npu,
    resolve_npu_download_set,
    split_client_name,
)

PINS = """
model qwen3-embedding-0.6b repo=acme/npu-embed revision=v1 devices=qwen3-embedding-0.6b
file qwen3-embedding-0.6b weights.bin 1000 abc
file qwen3-embedding-0.6b devices/prog.bin 2000 def
model qwen3.5-2b repo=acme/npu-nano revision=v2 devices=qwen3-embedding-0.6b
file qwen3.5-2b nano.bin 3000 ghi
model orphan devices=missing-owner
"""


def test_suffix_maps_are_inverses() -> None:
    assert len(UPSTREAM_BY_SUFFIX) == len(NPU_SUFFIXES)
    for upstream, suffix in NPU_SUFFIXES.items():
        assert UPSTREAM_BY_SUFFIX[suffix] == upstream
    assert STOCK_NPU_MODEL_IDS == tuple(NPU_SUFFIXES)
    assert set(NPU_CAPABILITIES) == set(NPU_SUFFIXES)


def test_client_and_split_roundtrip() -> None:
    assert client_name("flash", "qwen3-embedding-0.6b") == "flash-embed"
    assert split_client_name("flash", "flash-embed") == "qwen3-embedding-0.6b"
    assert split_client_name("flash", "flash-rerank") == "qwen3-reranker-0.6b"
    # Unknown suffix / other alias / bare name → None.
    assert split_client_name("flash", "flash-unknown") is None
    assert split_client_name("other", "flash-embed") is None
    assert split_client_name("flash", "qwen3.5-2b") is None


def test_parse_npu_pins() -> None:
    records = parse_npu_pins(PINS)
    embed = records["qwen3-embedding-0.6b"]
    assert embed.repo == "acme/npu-embed"
    assert embed.revision == "v1"
    assert embed.devices_of == "qwen3-embedding-0.6b"
    assert len(embed.files) == 2
    assert embed.devices_files == [("devices/prog.bin", 2000, "def")]
    assert embed.own_files == [("weights.bin", 1000, "abc")]
    nano = records["qwen3.5-2b"]
    assert nano.devices_of == "qwen3-embedding-0.6b"
    assert records["orphan"].repo is None


def test_parse_skips_bad_size_lines() -> None:
    records = parse_npu_pins("file m path.bin notanumber sha256")
    assert records == {}


def test_resolve_download_set_pulls_devices_owner_program() -> None:
    records = parse_npu_pins(PINS)
    plan = resolve_npu_download_set(records, ["qwen3.5-2b"])
    # The nano's own files land in its dir; the shared devices/ program is
    # fetched into the devices-owner's dir (not duplicated into nano's).
    assert plan["qwen3.5-2b"] == ("acme/npu-nano", "v2", [("nano.bin", 3000, "ghi")])
    assert plan["qwen3-embedding-0.6b"][0] == "acme/npu-embed"
    assert ("devices/prog.bin", 2000, "def") in plan["qwen3-embedding-0.6b"][2]


def test_resolve_download_set_missing_record_raises() -> None:
    records = parse_npu_pins(PINS)
    with pytest.raises(ValueError, match="no NPU pin record"):
        resolve_npu_download_set(records, ["ghost-model"])


def test_resolve_download_set_no_repo_raises() -> None:
    records = parse_npu_pins(PINS)
    with pytest.raises(ValueError, match="no download repo"):
        resolve_npu_download_set(records, ["orphan"])


def test_probe_npu_unavailable_on_dev_machine(tmp_path) -> None:
    """No NPU hardware: the probe reports unavailable with reasons and
    never raises (tests run fake / no /dev/accel)."""
    result = probe_npu(
        device_path=str(tmp_path / "no-such-device"),
        xrt_lib_dir=str(tmp_path / "no-xrt"),
        binary_path=str(tmp_path / "no-binary"),
    )
    assert result["available"] is False
    assert result["checks"]["device"] is False
    assert result["reasons"]


def test_probe_npu_available_with_fakes(tmp_path) -> None:
    import sys

    dev = tmp_path / "accel0"
    dev.write_bytes(b"")
    dev.chmod(0o666)
    lib = tmp_path / "xrt"
    lib.mkdir()
    for name in ("libxrt_coreutil.so.2", "libxrt_core.so.2", "libxrt_driver_xdna.so.2"):
        (lib / name).write_text("")
    binary = tmp_path / "halogen-npu"
    # exit 1 + "usage: halogen-npu..." is the engine's no-arg contract.
    binary.write_text(
        f"#!{sys.executable}\n"
        "import sys\n"
        "print('usage: halogen-npu <args>', file=sys.stderr)\n"
        "sys.exit(1)\n"
    )
    binary.chmod(0o755)
    result = probe_npu(
        device_path=str(dev), xrt_lib_dir=str(lib), binary_path=str(binary)
    )
    assert result["available"] is True
    assert result["checks"] == {"device": True, "xrt": True, "engine": True}
