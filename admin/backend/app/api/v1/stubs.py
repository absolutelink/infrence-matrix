"""501 stubs for out-of-scope public endpoints (Phase 7).

These existed pre-overhaul and are consciously dropped (ARCHITECTURE.md
§7/§13): embeddings, legacy text completions, rerank, moderations,
decisions, audio, files, and batches. Every stub answers with the OpenAI
error envelope::

    {"error": {"message": "...", "type": "not_supported_error",
              "code": "endpoint_not_supported", "param": "<endpoint>"}}

and HTTP 501, so clients get a clear "not supported here" instead of a
confusing 404. Do NOT restore old implementations from ``legacy/`` —
re-implement against the provider/scheduler model when a need arises.
"""

from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

router = APIRouter(tags=["stubs"])

# (method, path) pairs stubbed with 501. Item routes for files/batches are
# included so DELETE/GET/cancel on a resource id also answer "not
# supported" rather than 404.
STUBBED_ENDPOINTS: tuple[tuple[str, str], ...] = (
    ("POST", "/v1/embeddings"),
    ("POST", "/v1/completions"),
    ("POST", "/v1/rerank"),
    ("POST", "/v1/moderations"),
    ("POST", "/v1/decisions"),
    ("POST", "/v1/audio/speech"),
    ("POST", "/v1/audio/transcriptions"),
    ("POST", "/v1/audio/translations"),
    ("POST", "/v1/files"),
    ("GET", "/v1/files"),
    ("GET", "/v1/files/{file_id}"),
    ("GET", "/v1/files/{file_id}/content"),
    ("DELETE", "/v1/files/{file_id}"),
    ("POST", "/v1/batches"),
    ("GET", "/v1/batches"),
    ("GET", "/v1/batches/{batch_id}"),
    ("POST", "/v1/batches/{batch_id}/cancel"),
)


def _not_supported(request: Request) -> JSONResponse:
    return JSONResponse(
        status_code=501,
        content={
            "error": {
                "message": (
                    f"{request.method} {request.url.path} is not supported "
                    "by Inference Matrix"
                ),
                "type": "not_supported_error",
                "code": "endpoint_not_supported",
                "param": request.url.path,
            }
        },
    )


def _register_stub(method: str, path: str) -> None:
    async def handler(request: Request) -> Any:
        return _not_supported(request)

    handler.__name__ = f"stub_{method.lower()}_{path.strip('/').replace('/', '_')}"
    router.add_api_route(
        path, handler, methods=[method], include_in_schema=False, name=handler.__name__
    )


for _method, _path in STUBBED_ENDPOINTS:
    _register_stub(_method, _path)
