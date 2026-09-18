import httpx
from typing import Any
from fastapi import APIRouter, Depends, HTTPException, Query
from app.api.deps import CurrentUser

router = APIRouter(prefix="/huggingface", tags=["huggingface"])

HUGGINGFACE_API_BASE = "https://huggingface.co/api"


@router.get("/search")
def search_models(
    current_user: CurrentUser,
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
    current_user: CurrentUser,
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
    current_user: CurrentUser,
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
