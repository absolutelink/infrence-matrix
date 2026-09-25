"""llama-bench execution endpoints."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.api.routes.servers import _ensure_model
from app.services.llama_bench import build_command, llama_bench_manager, new_run_id

router = APIRouter(prefix="/benchmarks", tags=["benchmarks"])


class BenchmarkRequest(BaseModel):
    run_id: str | None = None
    model_path: str
    source: dict[str, str] | None = None
    prompt_sizes: list[int] = Field(default_factory=lambda: [512])
    generation_sizes: list[int] = Field(default_factory=lambda: [128])
    repetitions: int = 3
    batch_size: int | None = 512
    ubatch_size: int | None = None
    context_size: int | None = None
    gpu_layers: int | None = -1
    flash_attn: bool | None = None
    draft_model_path: str | None = None
    draft_source: dict[str, str] | None = None


class BenchmarkStopRequest(BaseModel):
    run_id: str


@router.post("/run")
async def run_benchmark(request: BenchmarkRequest) -> dict:
    """Resolve/download a model and start llama-bench asynchronously."""
    try:
        model_path = await _ensure_model(request.model_path, request.source)
        draft_model_path = None
        if request.draft_model_path:
            draft_model_path = await _ensure_model(
                request.draft_model_path, request.draft_source
            )
        run_id = request.run_id or new_run_id()
        return await llama_bench_manager.start(
            run_id, build_command(request, model_path, draft_model_path)
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/stop")
async def stop_benchmark(request: BenchmarkStopRequest) -> dict:
    try:
        return await llama_bench_manager.stop(request.run_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Benchmark run not found") from exc


@router.get("/status/{run_id}")
async def benchmark_status(run_id: str) -> dict:
    try:
        return llama_bench_manager.status(run_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Benchmark run not found") from exc


@router.get("/status")
async def active_benchmark_status() -> dict:
    run_id = llama_bench_manager.active_run_id
    return {"run_id": run_id, "status": "running" if run_id else "idle"}
