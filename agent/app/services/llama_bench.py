"""Run and parse llama.cpp benchmark processes."""

import asyncio
import csv
import os
import re
import time
import uuid
from pathlib import Path
from typing import Any

from app.core.config import settings
from app.services.event_bus import publish_event

_NUMBER = re.compile(r"[-+]?\d+(?:\.\d+)?")


def _parse_cases(output: str) -> list[dict[str, Any]]:
    """Parse the markdown, CSV, and simple table output used by llama-bench."""
    cases: list[dict[str, Any]] = []
    header: list[str] | None = None

    for raw_line in output.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if len(cells) > 1 and any("test" in cell.lower() for cell in cells):
            header = [cell.lower() for cell in cells]
            continue

        if len(cells) <= 1:
            if "," not in line:
                continue
            cells = next(csv.reader([line]))
            cells = [cell.strip() for cell in cells]

        if header and len(cells) >= len(header):
            test_index = next(
                (i for i, cell in enumerate(header) if "test" in cell), None
            )
            speed_index = next(
                (
                    i
                    for i, cell in enumerate(header)
                    if "t/s" in cell or "tokens/s" in cell or "speed" in cell
                ),
                None,
            )
            if test_index is not None and speed_index is not None:
                speed = _NUMBER.search(cells[speed_index])
                if speed:
                    cases.append(
                        {
                            "test": cells[test_index],
                            "tokens_per_second": float(speed.group()),
                            "raw": raw_line,
                        }
                    )
                    continue

            if test_index is not None:
                # Some versions omit the speed column name and print values
                # such as ``pp512 123.4 +/- 0.1`` instead.
                speed = next(
                    (_NUMBER.search(cell) for cell in cells[test_index + 1 :]), None
                )
                if speed:
                    cases.append(
                        {
                            "test": cells[test_index],
                            "tokens_per_second": float(speed.group()),
                            "raw": raw_line,
                        }
                    )
                    continue

        test_match = next(
            (re.search(r"\b(pp|tg)\s*\d+", cell, re.I) for cell in cells), None
        )
        numbers = [_NUMBER.search(cell) for cell in cells]
        if test_match and numbers:
            numeric = [match for match in numbers if match]
            if numeric:
                cases.append(
                    {
                        "test": test_match.group(0),
                        "tokens_per_second": float(numeric[-1].group()),
                        "raw": raw_line,
                    }
                )
    return cases


def _summary(cases: list[dict[str, Any]]) -> dict[str, Any]:
    speeds = [case["tokens_per_second"] for case in cases]
    by_test: dict[str, float] = {}
    for case in cases:
        test = str(case["test"]).lower()
        by_test.setdefault(test, case["tokens_per_second"])
    return {
        "case_count": len(cases),
        "average_tokens_per_second": sum(speeds) / len(speeds) if speeds else None,
        "peak_tokens_per_second": max(speeds) if speeds else None,
        "by_test": by_test,
    }


class LlamaBenchManager:
    """Own the single llama-bench process allowed on an agent."""

    def __init__(self) -> None:
        self.runs: dict[str, dict[str, Any]] = {}
        self._process: asyncio.subprocess.Process | None = None
        self._task: asyncio.Task | None = None
        self._stop_requested = False

    @property
    def active_run_id(self) -> str | None:
        if self._task and not self._task.done():
            return next(
                (
                    run_id
                    for run_id, run in self.runs.items()
                    if run["status"] == "running"
                ),
                None,
            )
        return None

    async def start(self, run_id: str, command: list[str]) -> dict[str, Any]:
        if self.active_run_id:
            raise RuntimeError(f"Benchmark {self.active_run_id} is already running")
        run = {
            "run_id": run_id,
            "status": "running",
            "command": command,
            "started_at": time.time(),
            "cases": [],
            "summary": _summary([]),
            "stdout": [],
            "stderr": [],
        }
        self.runs[run_id] = run
        self._stop_requested = False
        self._task = asyncio.create_task(self._execute(run))
        publish_event("benchmark.started", {"run_id": run_id, "command": command})
        return self.status(run_id)

    async def _read_stream(self, run: dict[str, Any], stream: Any, name: str) -> None:
        async for raw_line in stream:
            line = raw_line.decode(errors="replace").rstrip("\r\n")
            run[name].append(line)
            publish_event(
                "benchmark.log",
                {"run_id": run["run_id"], "stream": name, "line": line},
            )

    async def _execute(self, run: dict[str, Any]) -> None:
        try:
            env = dict(os.environ)
            executable = settings.LLAMA_BENCH_PATH
            if "/" in executable:
                env.setdefault("LD_LIBRARY_PATH", str(Path(executable).parent))
            self._process = await asyncio.create_subprocess_exec(
                *run["command"],
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
            )
            await asyncio.gather(
                self._read_stream(run, self._process.stdout, "stdout"),
                self._read_stream(run, self._process.stderr, "stderr"),
                self._process.wait(),
            )
            output = "\n".join(run["stdout"] + run["stderr"])
            run["raw_output"] = output
            run["cases"] = _parse_cases(output)
            run["summary"] = _summary(run["cases"])
            run["status"] = (
                "stopped"
                if self._stop_requested
                else ("completed" if self._process.returncode == 0 else "failed")
            )
            run["return_code"] = self._process.returncode
            publish_event(
                f"benchmark.{run['status']}",
                {
                    "run_id": run["run_id"],
                    "summary": run["summary"],
                    "cases": run["cases"],
                    "raw_output": run.get("raw_output", ""),
                    "return_code": run["return_code"],
                },
            )
        except Exception as exc:
            run["status"] = "failed"
            run["error"] = str(exc)
            publish_event(
                "benchmark.error", {"run_id": run["run_id"], "error": str(exc)}
            )
        finally:
            self._process = None

    async def stop(self, run_id: str) -> dict[str, Any]:
        run = self.runs.get(run_id)
        if not run:
            raise KeyError(run_id)
        if run["status"] == "running" and self._process:
            self._stop_requested = True
            self._process.terminate()
            if self._task:
                await self._task
        return self.status(run_id)

    def status(self, run_id: str) -> dict[str, Any]:
        if run_id not in self.runs:
            raise KeyError(run_id)
        run = self.runs[run_id]
        return {
            key: value for key, value in run.items() if key not in {"stdout", "stderr"}
        }


llama_bench_manager = LlamaBenchManager()


def build_command(request: Any, model_path: str) -> list[str]:
    """Build llama-bench arguments from the API request."""
    command = [settings.LLAMA_BENCH_PATH, "-m", model_path]
    for flag, values in (
        ("-p", request.prompt_sizes),
        ("-n", request.generation_sizes),
    ):
        if values:
            command.extend([flag, ",".join(str(value) for value in values)])
    for flag, value in (
        ("-r", request.repetitions),
        ("-b", request.batch_size),
        ("-c", request.context_size),
    ):
        if value is not None:
            command.extend([flag, str(value)])
    if request.ubatch_size is not None:
        command.extend(["-ub", str(request.ubatch_size)])
    if request.gpu_layers is not None:
        command.extend(["-ngl", str(request.gpu_layers)])
    if request.flash_attn is not None:
        command.extend(["-fa", "on" if request.flash_attn else "off"])
    return command


def new_run_id() -> str:
    return str(uuid.uuid4())
