"""Focused tests for llama-bench command construction and parsing."""

import os

os.environ["AGENT_ID"] = "test-agent"
os.environ["FRONTEND_URL"] = "http://test:8000"

from types import SimpleNamespace

from app.services.llama_bench import (
    LlamaBenchManager,
    _parse_cases,
    _summary,
    build_command,
)
from app.services.log_buffers import CursorLogRing


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
        "-ub",
        "128",
        "-ngl",
        "20",
        "-fa",
        "on",
    ]


def test_benchmark_logs_are_cursor_addressable() -> None:
    import asyncio

    async def case() -> None:
        manager = LlamaBenchManager()

        class FakeStream:
            def __init__(self, lines: list[bytes]) -> None:
                self._lines = lines

            def __aiter__(self):
                return self._next()

            async def _next(self):
                for line in self._lines:
                    yield line

        run = {"run_id": "run-1", "status": "running", "stdout": [], "stderr": []}
        manager.runs["run-1"] = run
        manager._log_rings["run-1"] = CursorLogRing(10)

        await manager._read_stream(run, FakeStream([b"one\n", b"two\n"]), "stdout")
        await manager._read_stream(run, FakeStream([b"boom\n"]), "stderr")

        tail = manager.get_logs("run-1")
        assert tail["next_cursor"] == 3
        assert [entry["line"] for entry in tail["lines"]] == ["one", "two", "boom"]

        incremental = manager.get_logs("run-1", after=2)
        assert [entry["line"] for entry in incremental["lines"]] == ["boom"]
        assert incremental["next_cursor"] == 3
        assert incremental["gap"] is False

    asyncio.run(case())
