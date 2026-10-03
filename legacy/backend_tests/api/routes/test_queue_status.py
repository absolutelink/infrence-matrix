import uuid

from app.models import InferenceLease, Model


def test_request_status_reports_queue_then_terminal_state(client, db):
    model = Model(
        name=f"queue-status-{uuid.uuid4()}",
        path="/models/status.gguf",
        size_bytes=1,
        architecture="llama",
        quantization="Q4_K_M",
        source="huggingface",
    )
    db.add(model)
    db.commit()
    db.refresh(model)
    request_id = f"chatcmpl-{uuid.uuid4()}"
    lease = InferenceLease(request_id=request_id, model_id=model.id)
    db.add(lease)
    db.commit()

    response = client.get(f"/api/v1/queue/{request_id}")
    assert response.status_code == 200
    assert response.json() == {
        "request_id": request_id,
        "status": "queued",
        "terminal_reason": None,
    }
    lease.status = "failed"
    lease.terminal_reason = "upstream_error"
    db.add(lease)
    db.commit()
    assert (
        client.get(f"/api/v1/queue/{request_id}").json()["terminal_reason"]
        == "upstream_error"
    )
    assert client.get(f"/api/v1/queue/chatcmpl-{uuid.uuid4()}").status_code == 404

    payload = {"model": model.name, "messages": [{"role": "user", "content": "hi"}]}
    assert (
        client.post(
            "/v1/chat/completions",
            json=payload,
            headers={"X-Inference-Request-ID": request_id},
        ).status_code
        == 409
    )
    assert (
        client.post(
            "/v1/chat/completions",
            json=payload,
            headers={"X-Inference-Request-ID": "chatcmpl-invalid"},
        ).status_code
        == 400
    )
