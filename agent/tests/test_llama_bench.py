"""Focused tests for llama-bench command construction and parsing."""

import os

os.environ["AGENT_ID"] = "test-agent"
os.environ["FRONTEND_URL"] = "http://test:8000"

from types import SimpleNamespace

from app.services.llama_bench import _parse_cases, _summary, build_command


def test_parse_markdown_output() -> None:
    output = """
    | test | t/s |
    | pp512 | 123.45 |
    | tg128 | 67.8 |
    """

    cases = _parse_cases(output)

    assert cases[0]["test"] == "pp512"
    assert cases[0]["tokens_per_second"] == 123.45
    assert _summary(cases)["peak_tokens_per_second"] == 123.45


def test_build_command_uses_bench_options() -> None:
    request = SimpleNamespace(
        prompt_sizes=[128, 512],
        generation_sizes=[64],
        repetitions=2,
        batch_size=256,
        ubatch_size=128,
        context_size=4096,
        gpu_layers=20,
        flash_attn=True,
    )

    command = build_command(request, "/models/test.gguf")

    assert command[0] == "llama-bench"
    assert command[1:] == [
        "-m",
        "/models/test.gguf",
        "-p",
        "128,512",
        "-n",
        "64",
        "-r",
        "2",
        "-b",
        "256",
        "-c",
        "4096",
        "-ub",
        "128",
        "-ngl",
        "20",
        "-fa",
        "on",
    ]
