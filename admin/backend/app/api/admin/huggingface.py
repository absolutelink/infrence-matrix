"""Admin HuggingFace proxy endpoints (Phase 12).

``/admin/api/huggingface`` — unauthenticated trusted-LAN read proxy
over the public HuggingFace API, backing the ``hf-file`` picker widget
(provider/README.md, ARCHITECTURE.md §3/§4). Ported from the legacy
``huggingface.py`` routes and generalized beyond GGUF-only so halogen
``.hgn``/``.hnpw`` repos and tokenizer artifacts are discoverable
(the legacy hardcoded ``config=gguf`` + a ``.gguf``-only sibling
filter, which made non-GGUF repos invisible).

Security posture:
- Only the fixed ``https://huggingface.co/api`` base is ever contacted;
  user input is confined to query params and a validated ``repo_id``.
- ``repo_id`` must match the canonical ``{org}/{name}`` shape — no
  scheme, host, ``..``, or extra path segments (SSRF / path-trick
  guard; the value is interpolated into the request path).
- HF errors/timeouts map to clean 502/503 ``{"detail": ...}`` bodies;
  raw exception tracebacks never reach the client.
"""

import logging
import re
from typing import Any, Literal

import httpx
from fastapi import APIRouter, HTTPException, Query

from app.services.hf_classify import (
    classify_file,
    extract_quantization,
    guess_model_type,
    is_aux_type,
)

logger = logging.getLogger("admin.huggingface")

router = APIRouter(prefix="/admin/api/huggingface", tags=["admin"])

HUGGINGFACE_API_BASE = "https://huggingface.co/api"
HF_TIMEOUT_SECONDS = 30.0

# Canonical HF model repo id: "{org}/{name}", one slash, standard HF
# charset. Deliberately strict — the value is interpolated into the
# outbound request path.
_REPO_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9._-]+$")

# Kinds the picker can produce an hf-file descriptor for.
_RECOGNIZED_KINDS = frozenset({"gguf", "hgn", "hnpw", "tokenizer"})
# Kinds for which quantization / model_type classification is meaningful.
_WEIGHT_KINDS = frozenset({"gguf", "hgn", "hnpw"})


def validate_repo_id(repo_id: str) -> str:
    """Validate an untrusted repo id before it reaches the outbound URL.

    Rejects empty/whitespace, anything with a scheme or host, ``..``
    segments, and non ``{org}/{name}`` shapes. Raises 422 on violation.
    """
    repo_id = repo_id.strip()
    if ".." in repo_id:
        raise HTTPException(
            status_code=422, detail="repo_id must not contain '..' path segments"
        )
    if "://" in repo_id or repo_id.startswith("//"):
        raise HTTPException(
            status_code=422, detail="repo_id must not contain a URL scheme or authority"
        )
    if repo_id.count("/") != 1:
        raise HTTPException(
            status_code=422, detail="repo_id must have the form '{org}/{name}'"
        )
    if not _REPO_ID_RE.fullmatch(repo_id):
        raise HTTPException(
            status_code=422,
            detail=(
                "repo_id contains characters outside the allowed set "
                "[A-Za-z0-9._-] (org must start alphanumeric)"
            ),
        )
    return repo_id


def _hf_get(client: httpx.Client, url: str, *, what: str, **params: Any) -> Any:
    """GET with error mapping: 404 on HF-not-found, 502 on HF error,
    503 on timeout/connection failure. Never leaks the raw exception."""
    try:
        response = client.get(url, params=params)
        response.raise_for_status()
        return response.json()
    except httpx.HTTPStatusError as exc:
        status = exc.response.status_code
        if status == 404:
            raise HTTPException(status_code=404, detail=f"{what} not found") from None
        logger.warning("huggingface %s returned HTTP %s", what, status)
        raise HTTPException(
            status_code=502,
            detail=f"huggingface.co returned an error while trying to {what}",
        ) from None
    except (httpx.TimeoutException, httpx.TransportError) as exc:
        logger.warning("huggingface %s failed: %s", what, exc)
        raise HTTPException(
            status_code=503,
            detail=f"could not reach huggingface.co while trying to {what}",
        ) from None
    except ValueError:
        raise HTTPException(
            status_code=502,
            detail=f"huggingface.co returned invalid JSON while trying to {what}",
        ) from None


def _coerce_int(value: Any) -> int:
    """Best-effort int for HF counters (non-numeric/None → 0)."""
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return 0
    return 0


def _normalize_search_result(entry: dict[str, Any]) -> dict[str, Any]:
    """Normalize one HF /models list entry into the exact Phase E
    contract: ``{id: str, author: str, downloads: int, likes: int,
    tags: list[str], lastModified: str}`` (empty string where unknown).
    Never trusts key presence or types."""
    raw_tags = entry.get("tags")
    tags = (
        [t for t in raw_tags if isinstance(t, str)]
        if isinstance(raw_tags, list)
        else []
    )
    last_modified = entry.get("lastModified") or entry.get("createdAt")
    return {
        "id": str(entry.get("id") or entry.get("_id") or ""),
        "author": str(entry.get("author") or ""),
        "downloads": _coerce_int(entry.get("downloads")),
        "likes": _coerce_int(entry.get("likes")),
        "tags": tags,
        "lastModified": str(last_modified) if last_modified is not None else "",
    }


@router.get("/search")
def search_models(
    search: str = Query(..., min_length=1, description="Search query"),
    limit: int = Query(20, ge=1, le=100, description="Max results"),
    full: bool = Query(False, description="Include full model info"),
    filter: Literal["all", "gguf"] = Query(
        "all",
        description=(
            "File-type filter: 'gguf' applies the legacy HF 'config=gguf' "
            "repo filter; 'all' is a general model search (halogen .hgn "
            "repos are not GGUF-configured and must stay discoverable)."
        ),
    ),
) -> list[dict[str, Any]]:
    """Search HuggingFace model repos for the ``hf-file`` picker.

    General search by default; ``filter=gguf`` re-adds the legacy
    ``config=gguf`` filter.
    """
    params: dict[str, Any] = {
        "search": search,
        "limit": limit,
        "full": str(full).lower(),
    }
    if filter == "gguf":
        params["config"] = "gguf"
    with httpx.Client(timeout=HF_TIMEOUT_SECONDS) as client:
        data = _hf_get(
            client,
            f"{HUGGINGFACE_API_BASE}/models",
            what="search HuggingFace",
            **params,
        )
    if not isinstance(data, list):
        raise HTTPException(
            status_code=502, detail="huggingface.co returned an unexpected search shape"
        )
    return [
        _normalize_search_result(entry) for entry in data if isinstance(entry, dict)
    ]


@router.get("/files")
def list_repo_files(
    repo_id: str = Query(
        ...,
        min_length=3,
        description="HuggingFace repository ID (e.g. 'unsloth/Qwen3-27B-GGUF')",
    ),
    include_all: bool = Query(
        False,
        description="Include unclassified/irrelevant files (default: recognized kinds only)",
    ),
) -> dict[str, Any]:
    """List a repo's selectable artifact files with sizes and kinds.

    Recognized kinds: ``gguf``, ``hgn`` (halogen checkpoints),
    ``hnpw`` (NPU weights), ``tokenizer`` (tokenizer dirs/files).
    Every file entry carries the full key set so the picker never
    special-cases presence: ``rfilename: str``, ``size: int | null``
    (null = unknown, distinct from a real 0), ``kind: str``,
    ``quantization: str | null``, ``model_type: str | null`` and
    ``is_aux: bool`` — the last three are meaningful only for weight
    kinds (gguf/hgn/hnpw) and null/false elsewhere.

    The tree-endpoint size fallback is best-effort: if it fails (missing
    ``main`` branch, rate limit, timeout) the successfully fetched
    sibling listing still returns 200 with ``size: null`` for the
    unknown entries.
    """
    repo_id = validate_repo_id(repo_id)
    with httpx.Client(timeout=HF_TIMEOUT_SECONDS) as client:
        model_data = _hf_get(
            client,
            f"{HUGGINGFACE_API_BASE}/models/{repo_id}",
            what=f"list files of repo '{repo_id}'",
            full="true",
        )
        if not isinstance(model_data, dict):
            raise HTTPException(
                status_code=502,
                detail="huggingface.co returned an unexpected repo shape",
            )
        siblings = model_data.get("siblings") or []

        # cardData.gguf quantization mapping (defensive; inconsistent
        # across repos).
        card_gguf: Any = None
        try:
            card_gguf = (model_data.get("cardData") or {}).get("gguf")
        except AttributeError:
            card_gguf = None

        files: list[dict[str, Any]] = []
        needs_sizes = False
        for sib in siblings:
            if not isinstance(sib, dict):
                continue
            rfilename = sib.get("rfilename")
            if not isinstance(rfilename, str):
                continue
            kind = classify_file(rfilename)
            if not include_all and kind not in _RECOGNIZED_KINDS:
                continue
            size = sib.get("size")
            entry_size: int | None = (
                int(size)
                if isinstance(size, (int, float)) and not isinstance(size, bool)
                else None
            )
            if entry_size is None:
                needs_sizes = True
            entry: dict[str, Any] = {
                "rfilename": rfilename,
                "size": entry_size,
                "kind": kind,
                "quantization": None,
                "model_type": None,
                "is_aux": False,
            }
            if kind in _WEIGHT_KINDS:
                entry["quantization"] = extract_quantization(rfilename, card_gguf)
                if kind == "gguf":
                    entry["model_type"] = guess_model_type(rfilename)
                    entry["is_aux"] = is_aux_type(entry["model_type"])
            files.append(entry)

        # HF's model API omits sizes on some repos; the tree endpoint
        # carries them (legacy fallback). Best-effort: a tree failure
        # (missing main branch, rate limit, timeout) must not sink the
        # already-successful sibling listing — unknown sizes stay null.
        if needs_sizes and files:
            try:
                tree_data = _hf_get(
                    client,
                    f"{HUGGINGFACE_API_BASE}/models/{repo_id}/tree/main",
                    what=f"fetch the file tree of repo '{repo_id}'",
                    recursive="true",
                )
            except HTTPException as exc:
                logger.warning(
                    "size fallback for repo '%s' failed (HTTP %s); "
                    "returning files with unknown sizes",
                    repo_id,
                    exc.status_code,
                )
            else:
                size_map: dict[str, int] = {}
                if isinstance(tree_data, list):
                    for node in tree_data:
                        if isinstance(node, dict) and node.get("type") == "file":
                            path = node.get("path")
                            node_size = node.get("size")
                            if (
                                isinstance(path, str)
                                and isinstance(node_size, (int, float))
                                and not isinstance(node_size, bool)
                            ):
                                size_map[path] = int(node_size)
                for entry in files:
                    if entry["size"] is None:
                        entry["size"] = size_map.get(entry["rfilename"])

    return {"repo_id": repo_id, "files": files}
