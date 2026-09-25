from app.services.inference_scheduler import normalize_slot_telemetry


def test_normalize_wrapped_slots():
    telemetry = normalize_slot_telemetry(
        {"slots": [{"id": 0, "state": 0}, {"id": 1, "state": 1}]}
    )

    assert telemetry.capacity == 4
    assert telemetry.active == 1
    assert telemetry.available == 3
    assert telemetry.known


def test_normalize_unavailable_payload_is_safe():
    telemetry = normalize_slot_telemetry({"unexpected": True})

    assert (telemetry.capacity, telemetry.active, telemetry.available) == (4, 0, 4)
    assert not telemetry.known


def test_normalize_uses_configured_parallel_capacity():
    telemetry = normalize_slot_telemetry(
        {"slots": [{"state": 1}, {"state": 0}]}, capacity=8
    )

    assert telemetry.capacity == 8
    assert telemetry.active == 1
    assert telemetry.available == 7
