from typing import Any

import httpx
from fastapi import APIRouter, HTTPException, Query

router = APIRouter(prefix="/huggingface", tags=["huggingface"])

HUGGINGFACE_API_BASE = "https://huggingface.co/api"


@router.get("/search")
def search_models(
    search: str = Query(..., min_length=1, description="Search query"),
    limit: int = Query(20, ge=1, le=100, description="Max results"),
    full: bool = Query(False, description="Include full model info"),
) -> list[dict[str, Any]]:
    """
    Search for GGUF models on HuggingFace.
    """
    try:
        with httpx.Client(timeout=30.0) as client:
            response = client.get(
                f"{HUGGINGFACE_API_BASE}/models",
                params={
                    "search": search,
                    "limit": limit,
                    "full": str(full).lower(),
                    "config": "gguf",
                },
            )
            response.raise_for_status()
            return response.json()
    except httpx.HTTPError as e:
        raise HTTPException(
            status_code=503,
            detail=f"Failed to search HuggingFace: {str(e)}"
        )


@router.get("/models/files")
def list_model_files(
    repo_id: str = Query(..., description="HuggingFace repository ID (e.g., 'unsloth/Qwen3.8-27B-GGUF')"),
) -> list[dict[str, Any]]:
    """
    List GGUF files in a HuggingFace repository with their sizes.
    """
    try:
        with httpx.Client(timeout=60.0) as client:
            # First get the list of files
            response = client.get(
                f"{HUGGINGFACE_API_BASE}/models/{repo_id}",
                params={"full": "true"},
            )
            response.raise_for_status()
            model_data = response.json()

            # Get the siblings (files) from the response
            siblings = model_data.get("siblings", [])

            # Filter for GGUF files only
            gguf_files = [
                file for file in siblings
                if file.get("rfilename", "").endswith(".gguf")
            ]

            # If sizes are not included, fetch them from the tree endpoint
            if gguf_files and not gguf_files[0].get("size"):
                # Fetch file tree with sizes
                tree_response = client.get(
                    f"{HUGGINGFACE_API_BASE}/models/{repo_id}/tree/main",
                    params={"recursive": "true"},
                )
                if tree_response.status_code == 200:
                    tree_data = tree_response.json()
                    # Create a mapping of filename -> size
                    size_map = {
                        file.get("path"): file.get("size", 0)
                        for file in tree_data
                        if file.get("type") == "file"
                    }
                    # Add sizes to GGUF files
                    gguf_files = [
                        {
                            "path": file.get("rfilename"),
                            "size": size_map.get(file.get("rfilename"), 0)
                        }
                        for file in gguf_files
                    ]
            else:
                # Sizes are already included
                gguf_files = [
                    {"path": file.get("rfilename"), "size": file.get("size", 0)}
                    for file in gguf_files
                ]

            return gguf_files
    except httpx.HTTPError as e:
        raise HTTPException(
            status_code=503,
            detail=f"Failed to fetch model files: {str(e)}"
        )


@router.get("/models/info")
def get_model_info(
    repo_id: str = Query(..., description="HuggingFace repository ID"),
) -> dict[str, Any]:
    """
    Get detailed information about a HuggingFace model.
    """
    try:
        with httpx.Client(timeout=30.0) as client:
            response = client.get(
                f"{HUGGINGFACE_API_BASE}/models/{repo_id}",
                params={"full": "true"},
            )
            response.raise_for_status()
            model_info = response.json()

            # Extract GGUF specific info if available
            gguf_info = {}
            if "gguf" in model_info.get("cardData", {}):
                gguf_info = model_info["cardData"]["gguf"]

            return {
                "id": model_info.get("id"),
                "modelId": model_info.get("modelId"),
                "author": model_info.get("author"),
                "sha": model_info.get("sha"),
                "downloads": model_info.get("downloads", 0),
                "likes": model_info.get("likes", 0),
                "private": model_info.get("private", False),
                "config": model_info.get("config", {}),
                "gguf": gguf_info,
                "tags": model_info.get("tags", []),
                "createdAt": model_info.get("createdAt"),
            }
    except httpx.HTTPError as e:
        raise HTTPException(
            status_code=503,
            detail=f"Failed to fetch model info: {str(e)}"
        )


@router.get("/models/params")
def get_parameter_count(
    repo_id: str = Query(..., description="HuggingFace repository ID"),
) -> dict[str, Any]:
    """
    Get the actual parameter count from a model's config.json.
    Fetches from HuggingFace's raw file API.
    """
    try:
        with httpx.Client(timeout=30.0) as client:
            # Try to fetch config.json from the repo
            response = client.get(
                f"https://huggingface.co/{repo_id}/raw/main/config.json",
                follow_redirects=True,
            )

            if response.status_code == 404:
                # Try master branch if main doesn't exist
                response = client.get(
                    f"https://huggingface.co/{repo_id}/raw/master/config.json",
                    follow_redirects=True,
                )

            response.raise_for_status()
            config = response.json()

            # Extract parameter count from various possible fields
            param_count = None

            # Check common fields for parameter count
            if "num_params" in config:
                param_count = config["num_params"]
            elif "total_params" in config:
                param_count = config["total_params"]
            elif "architectures" in config and "num_hidden_layers" in config:
                # Estimate from architecture
                hidden_layers = config.get("num_hidden_layers", 0)
                hidden_size = config.get("hidden_size", 0)
                vocab_size = config.get("vocab_size", 0)
                # Rough estimate: layers * hidden^2 + vocab * hidden
                if hidden_layers and hidden_size:
                    param_count = hidden_layers * (hidden_size ** 2) + (vocab_size * hidden_size)

            # Also check gguf specific fields
            if "gguf" in config:
                gguf = config["gguf"]
                if "params" in gguf:
                    param_count = gguf["params"]
                elif "parameter_count" in gguf:
                    param_count = gguf["parameter_count"]

            return {
                "repo_id": repo_id,
                "parameter_count": param_count,
                "config": config,
            }
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            return {
                "repo_id": repo_id,
                "parameter_count": None,
                "error": "config.json not found",
            }
        raise HTTPException(
            status_code=503,
            detail=f"Failed to fetch config: {str(e)}"
        )
    except httpx.HTTPError as e:
        raise HTTPException(
            status_code=503,
            detail=f"Failed to fetch parameter count: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail=f"Failed to parse config: {str(e)}"
        )
