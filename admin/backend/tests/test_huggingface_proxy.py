"""Phase 12 HF proxy endpoints (``/admin/api/huggingface``).

The HuggingFace API is mocked with respx — no real network in tests.
Covers: search normalization + ``filter=gguf`` (``config=gguf``)
on/off, files classification (``.gguf``/``.hgn``/``.hnpw``/tokenizer),
``cardData.gguf`` quantization mapping, tree-endpoint size fallback
(best-effort: tree failure degrades to ``size: null``, never sinks the
listing), unknown-vs-zero size distinction, irrelevant-file filtering,
SSRF/bad ``repo_id`` rejection, and clean 5xx error mapping on HF
failure/timeout/invalid-JSON with no exception-message leakage.
"""

import httpx
import pytest
import respx
from fastapi.testclient import TestClient

HF_BASE = "https://huggingface.co/api"


@pytest.fixture
def client() -> TestClient:
    from app.main import app

    with TestClient(app) as c:
        yield c


# ---------------------------------------------------------------- search


@respx.mock
def test_search_normalizes_results(client: TestClient) -> None:
    route = respx.get(f"{HF_BASE}/models").mock(
        return_value=httpx.Response(
            200,
            json=[
                {
                    "id": "theBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",
                    "author": "theBloke",
                    "downloads": 1234,
                    "likes": 56,
                    "tags": ["gguf", "llama"],
                    "lastModified": "2024-01-02T03:04:05.000Z",
                },
                # Missing most keys — normalization must not blow up.
                {"id": "org/sparse"},
            ],
        )
    )
    resp = client.get("/admin/api/huggingface/search", params={"search": "tinyllama"})
    assert resp.status_code == 200, resp.text
    results = resp.json()
    assert results[0] == {
        "id": "theBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",
        "author": "theBloke",
        "downloads": 1234,
        "likes": 56,
        "tags": ["gguf", "llama"],
        "lastModified": "2024-01-02T03:04:05.000Z",
    }
    assert results[1] == {
        "id": "org/sparse",
        "author": "",
        "downloads": 0,
        "likes": 0,
        "tags": [],
        "lastModified": "",
    }
    # Default: general search, no forced GGUF config.
    assert "config" not in route.calls.last.request.url.params


@respx.mock
def test_search_coerces_weird_field_types(client: TestClient) -> None:
    respx.get(f"{HF_BASE}/models").mock(
        return_value=httpx.Response(
            200,
            json=[
                {
                    "id": "org/str-counts",
                    "downloads": "1234",
                    "likes": None,
                    "tags": {"gguf": 1},  # dict, not list -> []
                },
                {
                    "id": "org/junk-counts",
                    "downloads": "many",
                    "likes": True,
                    "tags": ["a", 3, "b"],  # non-str items dropped
                },
            ],
        )
    )
    resp = client.get("/admin/api/huggingface/search", params={"search": "x"})
    assert resp.status_code == 200, resp.text
    a, b = resp.json()
    assert a["downloads"] == 1234
    assert a["likes"] == 0
    assert a["tags"] == []
    assert a["lastModified"] == ""
    assert b["downloads"] == 0
    assert b["likes"] == 0
    assert b["tags"] == ["a", "b"]


@respx.mock
def test_search_non_list_response_is_clean_502(client: TestClient) -> None:
    respx.get(f"{HF_BASE}/models").mock(
        return_value=httpx.Response(200, json={"error": "unexpected"})
    )
    resp = client.get("/admin/api/huggingface/search", params={"search": "x"})
    assert resp.status_code == 502
    assert "unexpected" in resp.json()["detail"]


@respx.mock
def test_search_invalid_json_is_502(client: TestClient) -> None:
    respx.get(f"{HF_BASE}/models").mock(
        return_value=httpx.Response(200, text="<html>not json {{{")
    )
    resp = client.get("/admin/api/huggingface/search", params={"search": "x"})
    assert resp.status_code == 502
    assert "invalid JSON" in resp.json()["detail"]


@respx.mock
def test_search_gguf_only_adds_config_param(client: TestClient) -> None:
    route = respx.get(f"{HF_BASE}/models").mock(
        return_value=httpx.Response(200, json=[])
    )
    resp = client.get(
        "/admin/api/huggingface/search",
        params={"search": "qwen", "filter": "gguf", "limit": 5, "full": "true"},
    )
    assert resp.status_code == 200, resp.text
    params = route.calls.last.request.url.params
    assert params["config"] == "gguf"
    assert params["limit"] == "5"
    assert params["full"] == "true"


@respx.mock
def test_search_hf_error_maps_to_502(client: TestClient) -> None:
    respx.get(f"{HF_BASE}/models").mock(return_value=httpx.Response(500, json={}))
    resp = client.get("/admin/api/huggingface/search", params={"search": "boom"})
    assert resp.status_code == 502
    assert "detail" in resp.json()
    # No raw traceback / exception class leakage.
    assert "Traceback" not in resp.text


@respx.mock
def test_search_timeout_maps_to_503(client: TestClient) -> None:
    respx.get(f"{HF_BASE}/models").mock(
        side_effect=httpx.ConnectTimeout(
            "timed out", request=httpx.Request("GET", HF_BASE)
        )
    )
    resp = client.get("/admin/api/huggingface/search", params={"search": "slow"})
    assert resp.status_code == 503
    assert "huggingface.co" in resp.json()["detail"]


def test_search_requires_query(client: TestClient) -> None:
    resp = client.get("/admin/api/huggingface/search", params={"search": ""})
    assert resp.status_code == 422


def test_search_rejects_unknown_filter(client: TestClient) -> None:
    resp = client.get(
        "/admin/api/huggingface/search", params={"search": "x", "filter": "hgn"}
    )
    assert resp.status_code == 422


# ----------------------------------------------------------------- files


@respx.mock
def test_files_classifies_and_surfaces_quantization(client: TestClient) -> None:
    respx.get(f"{HF_BASE}/models/org/halogen-repo").mock(
        return_value=httpx.Response(
            200,
            json={
                "siblings": [
                    {"rfilename": "model-Q4_K_M.hgn", "size": 100},
                    {"rfilename": "weights/llm.hnpw", "size": 200},
                    {"rfilename": "tokenizer/vocab.txt", "size": 30},
                    {"rfilename": "README.md"},
                    {"rfilename": "setup.py"},
                ]
            },
        )
    )
    resp = client.get(
        "/admin/api/huggingface/files", params={"repo_id": "org/halogen-repo"}
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["repo_id"] == "org/halogen-repo"
    names = [f["rfilename"] for f in body["files"]]
    kinds = {f["rfilename"]: f["kind"] for f in body["files"]}
    assert "model-Q4_K_M.hgn" in names
    assert "weights/llm.hnpw" in names
    assert "tokenizer/vocab.txt" in names
    assert kinds["model-Q4_K_M.hgn"] == "hgn"
    assert kinds["weights/llm.hnpw"] == "hnpw"
    assert kinds["tokenizer/vocab.txt"] == "tokenizer"
    # Irrelevant files filtered from the primary result.
    assert "README.md" not in names
    assert "setup.py" not in names
    # hgn carries a quantization guess from the filename.
    hgn = next(f for f in body["files"] if f["rfilename"] == "model-Q4_K_M.hgn")
    assert hgn["quantization"] == "Q4_K_M"
    assert hgn["size"] == 100
    # L3: every entry carries the full key set (null/false for non-weights).
    for f in body["files"]:
        assert set(f) == {
            "rfilename",
            "size",
            "kind",
            "quantization",
            "model_type",
            "is_aux",
        }
    tok = next(f for f in body["files"] if f["kind"] == "tokenizer")
    assert tok["quantization"] is None
    assert tok["model_type"] is None
    assert tok["is_aux"] is False


@respx.mock
def test_files_gguf_quantization_from_card_data(client: TestClient) -> None:
    respx.get(f"{HF_BASE}/models/org/gguf-repo").mock(
        return_value=httpx.Response(
            200,
            json={
                "cardData": {
                    "gguf": {
                        "repo_file.gguf": {
                            "filename": "repo_file.gguf",
                            "quantization": "Q8_0",
                        },
                        "mystery.gguf": "UD-Q4_K_XL",
                    }
                },
                "siblings": [
                    {"rfilename": "repo_file.gguf", "size": 42},
                    {"rfilename": "mystery.gguf", "size": 45},
                    {"rfilename": "model-IQ2_XXS.gguf", "size": 43},
                    {"rfilename": "mmproj-BF16.gguf", "size": 44},
                ],
            },
        )
    )
    resp = client.get(
        "/admin/api/huggingface/files", params={"repo_id": "org/gguf-repo"}
    )
    assert resp.status_code == 200, resp.text
    by_name = {f["rfilename"]: f for f in resp.json()["files"]}
    # "repo_file.gguf" has no quant hint in its name — only cardData.gguf supplies it.
    assert by_name["repo_file.gguf"]["kind"] == "gguf"
    assert by_name["repo_file.gguf"]["quantization"] == "Q8_0"
    # List-style entry: filename -> quant label via direct dict key.
    assert by_name["mystery.gguf"]["quantization"] == "UD-Q4_K_XL"
    assert by_name["model-IQ2_XXS.gguf"]["kind"] == "gguf"
    assert by_name["model-IQ2_XXS.gguf"]["quantization"] == "IQ2_XXS"
    assert by_name["model-IQ2_XXS.gguf"]["model_type"] == "llm"
    assert by_name["model-IQ2_XXS.gguf"]["is_aux"] is False
    mmproj = by_name["mmproj-BF16.gguf"]
    assert mmproj["model_type"] == "mmproj"
    assert mmproj["is_aux"] is True
    assert mmproj["quantization"] == "BF16"


@respx.mock
def test_files_size_fallback_via_tree(client: TestClient) -> None:
    respx.get(f"{HF_BASE}/models/org/nosizes").mock(
        return_value=httpx.Response(
            200,
            json={
                "siblings": [
                    {"rfilename": "a-Q4_K_M.gguf"},
                    {"rfilename": "b.hgn"},
                ]
            },
        )
    )
    respx.get(f"{HF_BASE}/models/org/nosizes/tree/main").mock(
        return_value=httpx.Response(
            200,
            json=[
                {"path": "a-Q4_K_M.gguf", "type": "file", "size": 111},
                {"path": "b.hgn", "type": "file", "size": 222},
                {"path": "sub", "type": "directory", "size": 0},
            ],
        )
    )
    resp = client.get("/admin/api/huggingface/files", params={"repo_id": "org/nosizes"})
    assert resp.status_code == 200, resp.text
    sizes = {f["rfilename"]: f["size"] for f in resp.json()["files"]}
    assert sizes == {"a-Q4_K_M.gguf": 111, "b.hgn": 222}


@respx.mock
def test_files_tree_failure_degrades_to_null_sizes(client: TestClient) -> None:
    # M1: siblings succeed but the tree fallback errors -> 200 with null sizes.
    respx.get(f"{HF_BASE}/models/org/master-branch").mock(
        return_value=httpx.Response(
            200,
            json={
                "siblings": [
                    {"rfilename": "model-Q4_K_M.gguf"},
                    {"rfilename": "helper.hgn", "size": 55},
                ]
            },
        )
    )
    respx.get(f"{HF_BASE}/models/org/master-branch/tree/main").mock(
        return_value=httpx.Response(404, json={"error": "bump not found"})
    )
    resp = client.get(
        "/admin/api/huggingface/files", params={"repo_id": "org/master-branch"}
    )
    assert resp.status_code == 200, resp.text
    by_name = {f["rfilename"]: f for f in resp.json()["files"]}
    # Unknown size -> null (M2), not 0; known size survives untouched.
    assert by_name["model-Q4_K_M.gguf"]["size"] is None
    assert by_name["helper.hgn"]["size"] == 55
    assert "bump not found" not in resp.text


@respx.mock
def test_files_tree_timeout_skipped_when_sizes_known(client: TestClient) -> None:
    respx.get(f"{HF_BASE}/models/org/slowtree").mock(
        return_value=httpx.Response(
            200, json={"siblings": [{"rfilename": "m.gguf", "size": 10}]}
        )
    )
    tree_route = respx.get(f"{HF_BASE}/models/org/slowtree/tree/main").mock(
        side_effect=httpx.ReadTimeout("read timed out")
    )
    resp = client.get(
        "/admin/api/huggingface/files", params={"repo_id": "org/slowtree"}
    )
    assert resp.status_code == 200, resp.text
    # All sizes known from siblings -> the tree fallback is never called.
    assert resp.json()["files"][0]["size"] == 10
    assert not tree_route.called


@respx.mock
def test_files_unknown_size_not_in_tree_stays_null(client: TestClient) -> None:
    # Tree succeeds but the file isn't in it -> size stays null (unknown),
    # while a real 0-byte file from the tree is reported as 0 (M2: the
    # two are distinguishable).
    respx.get(f"{HF_BASE}/models/org/partial-tree").mock(
        return_value=httpx.Response(
            200,
            json={
                "siblings": [
                    {"rfilename": "ghost.gguf"},
                    {"rfilename": "empty.gguf"},
                ]
            },
        )
    )
    respx.get(f"{HF_BASE}/models/org/partial-tree/tree/main").mock(
        return_value=httpx.Response(
            200,
            json=[{"path": "empty.gguf", "type": "file", "size": 0}],
        )
    )
    resp = client.get(
        "/admin/api/huggingface/files", params={"repo_id": "org/partial-tree"}
    )
    assert resp.status_code == 200, resp.text
    by_name = {f["rfilename"]: f for f in resp.json()["files"]}
    assert by_name["ghost.gguf"]["size"] is None
    assert by_name["empty.gguf"]["size"] == 0


@respx.mock
def test_files_include_all_shows_irrelevant(client: TestClient) -> None:
    respx.get(f"{HF_BASE}/models/org/mixed").mock(
        return_value=httpx.Response(
            200,
            json={
                "siblings": [
                    {"rfilename": "model.gguf", "size": 1},
                    {"rfilename": "README.md", "size": 2},
                ]
            },
        )
    )
    resp = client.get(
        "/admin/api/huggingface/files",
        params={"repo_id": "org/mixed", "include_all": "true"},
    )
    assert resp.status_code == 200, resp.text
    names = {f["rfilename"] for f in resp.json()["files"]}
    assert "README.md" in names
    readme = next(f for f in resp.json()["files"] if f["rfilename"] == "README.md")
    assert readme["kind"] == "irrelevant"


@respx.mock
def test_files_repo_not_found_is_404(client: TestClient) -> None:
    respx.get(f"{HF_BASE}/models/org/ghost").mock(return_value=httpx.Response(404))
    resp = client.get("/admin/api/huggingface/files", params={"repo_id": "org/ghost"})
    assert resp.status_code == 404
    assert "not found" in resp.json()["detail"]


@respx.mock
def test_files_hf_error_is_clean_502(client: TestClient) -> None:
    respx.get(f"{HF_BASE}/models/org/bad").mock(
        return_value=httpx.Response(503, json={"error": "upstream blew up"})
    )
    resp = client.get("/admin/api/huggingface/files", params={"repo_id": "org/bad"})
    assert resp.status_code == 502
    assert "Traceback" not in resp.text
    # Raw HF error payload / exception message must not leak into detail.
    assert "blew up" not in resp.text


@respx.mock
def test_files_invalid_json_is_502(client: TestClient) -> None:
    respx.get(f"{HF_BASE}/models/org/notjson").mock(
        return_value=httpx.Response(200, text="}{ not json")
    )
    resp = client.get("/admin/api/huggingface/files", params={"repo_id": "org/notjson"})
    assert resp.status_code == 502
    assert "invalid JSON" in resp.json()["detail"]
    assert "not json" not in resp.json()["detail"]
    assert "Traceback" not in resp.text


@respx.mock
def test_files_timeout_is_503(client: TestClient) -> None:
    respx.get(f"{HF_BASE}/models/org/slow").mock(
        side_effect=httpx.ReadTimeout("secret internal timeout detail")
    )
    resp = client.get("/admin/api/huggingface/files", params={"repo_id": "org/slow"})
    assert resp.status_code == 503
    assert "secret internal timeout detail" not in resp.text


# ------------------------------------------------------- repo_id validation


@pytest.mark.parametrize(
    "bad_repo_id",
    [
        "evil.com/org/repo",  # host-like, two slashes
        "http://evil.com/x",  # scheme
        "//evil.com/x",  # protocol-relative
        "../../etc/passwd",  # traversal
        "org/../repo",  # traversal mid-id
        "justname",  # no org
        "a/b/c",  # too many segments
        "org/re po",  # space
        "org/re?x=1",  # query injection
        "org/re#frag",  # fragment
        "org/repo:latest",  # colon (not in HF repo charset)
    ],
)
def test_files_rejects_bad_repo_id(client: TestClient, bad_repo_id: str) -> None:
    resp = client.get("/admin/api/huggingface/files", params={"repo_id": bad_repo_id})
    assert resp.status_code == 422, f"{bad_repo_id!r} -> {resp.status_code} {resp.text}"


@respx.mock
def test_files_accepts_valid_hf_repo_ids(client: TestClient) -> None:
    for repo_id in ("unsloth/Qwen3-27B-GGUF", "org/name.with.dots", "a1/b2"):
        respx.get(f"{HF_BASE}/models/{repo_id}").mock(
            return_value=httpx.Response(200, json={"siblings": []})
        )
        resp = client.get("/admin/api/huggingface/files", params={"repo_id": repo_id})
        assert resp.status_code == 200, f"{repo_id}: {resp.text}"
