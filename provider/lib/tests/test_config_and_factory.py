from provider_lib.config import ProviderSettings


def _settings(**overrides: object) -> ProviderSettings:
    base = {
        "MACHINE_UID": "m-test",
        "MACHINE_SECRET": "tok",
        "AGENT_ID": "m-test-agent",
        "ADMIN_BASE_URL": "http://admin:8000",
        "CACHE_DIR": "/tmp/prov_test_cache",
        "MODELS_DIR": "/tmp/prov_test_models",
        "METRICS_CATEGORIES": "gpu_usage vram os_ram cpu storage",
    }
    base.update(overrides)
    return ProviderSettings(**base)  # type: ignore[call-arg]


def test_phase16_required_fields() -> None:
    """Phase 16: MACHINE_SECRET + AGENT_ID are required fields; the old
    PROVIDER_REGISTRATION_TOKEN field is gone (extra='ignore' drops it)."""
    fields = ProviderSettings.model_fields
    assert "MACHINE_SECRET" in fields
    assert "AGENT_ID" in fields
    assert "PROVIDER_REGISTRATION_TOKEN" not in fields
    # Required (no default) on the mandatory wiring.
    assert fields["MACHINE_SECRET"].is_required()
    assert fields["AGENT_ID"].is_required()
    s = _settings()
    assert s.MACHINE_SECRET == "tok"
    assert s.AGENT_ID == "m-test-agent"


def test_metrics_categories_excludes_inference() -> None:
    s = _settings()
    assert "inference" not in s.metrics_categories
    assert "gpu_usage" in s.metrics_categories


def test_provider_config_path() -> None:
    s = _settings(CACHE_DIR="/tmp/prov_test_cache")
    assert s.provider_config_path.name == "provider_config.json"


def test_persist_config_carries_modality(tmp_path: object) -> None:
    """Phase 18 slice 3: the persisted provider_config.json keeps each backend's
    definition.modality (parity/inspection — the authoritative re-apply is the
    fresh registration, not this cached file)."""
    import json

    from provider_lib.admin_client import AdminClient, RegistrationResult

    s = _settings(
        CACHE_DIR=str(tmp_path) + "/cache", MODELS_DIR=str(tmp_path) + "/models"
    )
    client = AdminClient(s)
    result = RegistrationResult(
        {
            "agent_id": "a1",
            "agent_secret": "secret",
            "machine": {},
            "backends": [
                {
                    "instance_id": "i1",
                    "port": 8081,
                    "definition": {
                        "alias": "emb",
                        "modality": "embedding",
                        "backend_config": {},
                        "config_fingerprint": "fp",
                    },
                }
            ],
        }
    )
    client._persist_config(result)
    saved = json.loads(s.provider_config_path.read_text())
    assert saved["backends"][0]["definition"]["modality"] == "embedding"


def test_app_factory_health() -> None:
    from fastapi.testclient import TestClient

    from provider_lib.app_factory import BackendOverrides, create_provider_app

    app = create_provider_app(
        _settings(), BackendOverrides(provider_type="mock", version="v1")
    )
    client = TestClient(app)
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["provider_type"] == "mock"
    assert body["machine_uid"] == "m-test"
