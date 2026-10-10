"""Phase 25 S2 talkies multi-model tests: ONE talkies process serves SEVERAL
client-facing names, each mapped to a talkies registry slug (spec §7).

Covers the four behaviours the spec locks for the first multi-model provider:

* **start** prefetches EVERY served slug and spawns one uvicorn with the whole
  slug set in ``TALKIES_ENABLED_MODELS`` / ``TALKIES_PRELOAD`` and a merged
  per-slug ``TALKIES_MODEL_CONCURRENCY`` map;
* **health** pins the whole set (``/api/ps`` must contain every slug — one
  missing is not healthy);
* **routing** rewrites the inbound request ``model`` (a served NAME) to its
  slug on speech / transcribe / voices / the live-ASR WS;
* **set_models** adds a served name live (no restart) and an unknown slug fails
  the start (registry pre-validation).

A legacy single-slug definition (config ``model.slug``, no served list) stays
byte-identical — exercised by the existing Phase 24 suite.
"""

import asyncio
import json
from typing import Any

import pytest
from conftest import (
    ASR_SLUG,
    TTS_SLUG,
    make_settings,
    make_stub_engine,
    serve_stub,
)
from fastapi import HTTPException
from provider_lib.app_factory import BackendOverrides, create_provider_app
from provider_lib.backend import BackendLifecycle
from provider_lib.config_update import ConfigState
from provider_lib.models import ModelSpec
from provider_lib.registry import BackendHandle, BackendRegistry

import provider_talkies.driver as driver_mod
from provider_talkies.driver import TalkiesBackend


def _multi_models() -> list[ModelSpec]:
    return [
        ModelSpec(
            name="a",
            modality="tts",
            backend_config={"slug": TTS_SLUG, "limits": {"model_concurrency": 1}},
        ),
        ModelSpec(
            name="b",
            modality="asr",
            backend_config={"slug": ASR_SLUG, "limits": {"model_concurrency": 2}},
        ),
    ]


def _connected_multi(
    tmp_path: Any, port: int, token: str = "stub-token"
) -> TalkiesBackend:
    """A multi-model driver wired to the stub engine without spawning."""
    driver = TalkiesBackend(make_settings(tmp_path), {})
    driver.set_models(_multi_models())
    driver.backend_port = port
    driver.slug = TTS_SLUG
    driver._auth_token = token
    driver._data_dir = tmp_path / "data"
    return driver


async def _collect(stream: Any) -> bytes:
    return b"".join([chunk async for chunk in stream])


class _AliveProc:
    """Minimal stand-in for a live talkies Popen so ``health()`` sees a running
    process without spawning one. ``poll()`` reports alive; ``pid`` is a value
    beyond any real process so ``stop()``'s ``killpg`` hits ``ProcessLookupError``
    (suppressed) and ``wait()`` returns immediately."""

    pid = 0x7FFFFFFF

    def poll(self) -> None:
        return None

    def wait(self, timeout: float | None = None) -> int:
        return 0


# ---------------------------------------------------------------------------
# start: prefetch every slug + spawn the whole set
# ---------------------------------------------------------------------------


async def test_multi_start_prefetches_all_and_spawns_all_slugs(
    tmp_path,
    fake_talkies_python,
    talkies_registry,
    model_dirs,
    spawn_state,
    monkeypatch,
) -> None:
    settings = make_settings(
        tmp_path, fake_talkies_python, models_file=talkies_registry
    )
    driver = TalkiesBackend(settings, {})
    driver.set_models(_multi_models())

    prefetched: list[str] = []
    real_prefetch = driver_mod.prefetch_model

    async def spy(slug: str, entry: dict, data_dir, **kw):
        prefetched.append(slug)
        return await real_prefetch(slug, entry, data_dir, **kw)

    monkeypatch.setattr(driver_mod, "prefetch_model", spy)
    try:
        await driver.start()
        # BOTH served slugs prefetched (no network: model_dirs pre-populated).
        assert set(prefetched) == {TTS_SLUG, ASR_SLUG}
        assert await driver.health() is True
        # resolved_artifacts records every prefetched model dir + voices store.
        assert str(model_dirs[TTS_SLUG]) in driver.resolved_artifacts
        assert str(model_dirs[ASR_SLUG]) in driver.resolved_artifacts

        spawn = json.loads((spawn_state / "spawn.json").read_text())
        env = spawn["env"]
        joined = f"{TTS_SLUG},{ASR_SLUG}"
        assert env["TALKIES_ENABLED_MODELS"] == joined
        assert env["TALKIES_PRELOAD"] == joined
        # per-slug concurrency map merged (a=1, b=2).
        assert env["TALKIES_MODEL_CONCURRENCY"] == f"{TTS_SLUG}=1,{ASR_SLUG}=2"
    finally:
        await driver.stop()
        await driver.aclose()


async def test_multi_health_requires_all_slugs(
    tmp_path, fake_talkies_python, talkies_registry, model_dirs, monkeypatch
) -> None:
    """All-slug pin: /api/ps with only ONE of the two served slugs is NOT
    healthy (the boot times out and tears the process down)."""
    assert model_dirs  # both slugs cached -> prefetch never touches network
    monkeypatch.setenv("FAKE_TALKIES_PS_SLUGS", TTS_SLUG)  # only one resident
    settings = make_settings(
        tmp_path,
        fake_talkies_python,
        models_file=talkies_registry,
        TALKIES_BOOT_TIMEOUT=2,
    )
    driver = TalkiesBackend(settings, {})
    driver.set_models(_multi_models())
    with pytest.raises((RuntimeError, TimeoutError), match="health"):
        await driver.start()
    assert driver.process is None


async def test_multi_unknown_slug_fails_start(
    tmp_path, fake_talkies_python, talkies_registry
) -> None:
    """Registry pre-validation: an unknown slug anywhere in the set fails the
    start BEFORE spawn."""
    settings = make_settings(
        tmp_path, fake_talkies_python, models_file=talkies_registry
    )
    driver = TalkiesBackend(settings, {})
    driver.set_models(
        [
            ModelSpec(name="a", modality="tts", backend_config={"slug": TTS_SLUG}),
            ModelSpec(name="x", modality="tts", backend_config={"slug": "nope"}),
        ]
    )
    with pytest.raises(RuntimeError, match="not in registry"):
        await driver.start()


# ---------------------------------------------------------------------------
# routing: served NAME -> slug at the engine boundary
# ---------------------------------------------------------------------------


async def test_multi_speech_routes_name_to_slug(tmp_path) -> None:
    engine, stub = make_stub_engine(slugs=[TTS_SLUG, ASR_SLUG])
    async with serve_stub(engine) as (_url, port):
        driver = _connected_multi(tmp_path, port)
        try:
            stream = driver.speech(
                {"model": "a", "input": "hi", "response_format": "wav"}
            )
            await _collect(stream.chunks)
            assert stub["last_speech"]["model"] == TTS_SLUG
        finally:
            await driver.aclose()


async def test_multi_transcribe_routes_name_to_slug(tmp_path) -> None:
    engine, stub = make_stub_engine(slugs=[TTS_SLUG, ASR_SLUG])
    async with serve_stub(engine) as (_url, port):
        driver = _connected_multi(tmp_path, port)
        try:
            status, _headers, body = await driver.transcribe(
                {"model": "b", "language": "en"},
                [("file", "clip.wav", b"RIFFfake", "audio/wav")],
            )
            assert status == 200
            assert stub["last_transcription"]["model"] == ASR_SLUG
        finally:
            await driver.aclose()


async def test_multi_voices_routes_primary_slug(tmp_path) -> None:
    engine, stub = make_stub_engine(slugs=[TTS_SLUG, ASR_SLUG])
    async with serve_stub(engine) as (_url, port):
        driver = _connected_multi(tmp_path, port)
        try:
            await driver.voices()
            # No name (legacy / direct call) -> primary slug.
            assert stub["last_voices_query"] == TTS_SLUG
        finally:
            await driver.aclose()


async def test_multi_voices_routes_name_to_slug(tmp_path) -> None:
    """The requested served NAME (not always the primary) maps to its slug."""
    engine, stub = make_stub_engine(slugs=[TTS_SLUG, ASR_SLUG])
    async with serve_stub(engine) as (_url, port):
        driver = _connected_multi(tmp_path, port)
        try:
            await driver.voices("b")  # served name b -> asr slug (NOT primary)
            assert stub["last_voices_query"] == ASR_SLUG
            await driver.voices("a")  # served name a -> tts slug (primary)
            assert stub["last_voices_query"] == TTS_SLUG
        finally:
            await driver.aclose()


async def test_multi_voices_filtered_to_routed_slug(tmp_path) -> None:
    """Upstream talkies ignores its model filter and returns every loaded
    slug's catalog; the driver must narrow the response to the routed slug
    and rewrite entries to the served NAME the client asked with."""
    engine, _stub = make_stub_engine(slugs=[TTS_SLUG, ASR_SLUG])
    async with serve_stub(engine) as (_url, port):
        driver = _connected_multi(tmp_path, port)
        try:
            for name, _slug in (("a", TTS_SLUG), ("b", ASR_SLUG)):
                catalog = await driver.voices(name)
                models = {v["model"] for v in catalog["voices"]}
                assert models == {name}, f"name {name}: {catalog}"
                assert {v["voice"] for v in catalog["voices"]} == {"Vivian", "clone1"}
            # No name (legacy / direct) -> primary slug's voices, tagged slug.
            catalog = await driver.voices()
            assert {v["model"] for v in catalog["voices"]} == {TTS_SLUG}
        finally:
            await driver.aclose()


async def test_multi_voice_put_routes_name_to_slug(tmp_path) -> None:
    engine, _stub = make_stub_engine(slugs=[TTS_SLUG, ASR_SLUG])
    async with serve_stub(engine) as (_url, port):
        driver = _connected_multi(tmp_path, port)
        try:
            summary = await driver.voice_put("clone", b"RIFF", "ref", "en", model="b")
            assert summary["model"] == ASR_SLUG  # enrollment attributed to b's slug
            summary = await driver.voice_put("clone", b"RIFF", "ref", "en", model="a")
            assert summary["model"] == TTS_SLUG
            summary = await driver.voice_put("clone", b"RIFF", None, None)
            assert summary["model"] == TTS_SLUG  # no name -> primary
            dele = await driver.voice_delete("clone", model="b")
            assert dele["model"] == ASR_SLUG
        finally:
            await driver.aclose()


async def test_multi_unknown_name_404s(tmp_path) -> None:
    engine, _stub = make_stub_engine(slugs=[TTS_SLUG, ASR_SLUG])
    async with serve_stub(engine) as (_url, port):
        driver = _connected_multi(tmp_path, port)
        try:
            with pytest.raises(HTTPException) as exc_info:
                driver.speech({"model": "zzz", "input": "hi"})
            assert exc_info.value.status_code == 404
        finally:
            await driver.aclose()


async def test_multi_per_model_defaults_merge(tmp_path) -> None:
    """A served model's per-model ``defaults`` override the shared config."""
    engine, stub = make_stub_engine(slugs=[TTS_SLUG, ASR_SLUG])
    async with serve_stub(engine) as (_url, port):
        driver = TalkiesBackend(make_settings(tmp_path), {"defaults": {"speed": 1.0}})
        driver.set_models(
            [
                ModelSpec(
                    name="a",
                    modality="tts",
                    backend_config={"slug": TTS_SLUG, "defaults": {"voice": "Vivian"}},
                ),
            ]
        )
        driver.backend_port = port
        driver.slug = TTS_SLUG
        driver._auth_token = "stub-token"
        try:
            stream = driver.speech({"model": "a", "input": "hi"})
            await _collect(stream.chunks)
            body = stub["last_speech"]
            assert body["voice"] == "Vivian"  # per-model default applied
            assert body["speed"] == 1.0  # shared default still present
        finally:
            await driver.aclose()


# ---------------------------------------------------------------------------
# set_models: live add without restart
# ---------------------------------------------------------------------------


async def test_set_models_live_add_updates_map(tmp_path) -> None:
    engine, stub = make_stub_engine(slugs=[TTS_SLUG, ASR_SLUG])
    async with serve_stub(engine) as (_url, port):
        driver = _connected_multi(tmp_path, port)
        try:
            # "c" is not yet a served name -> unknown (multi set, map miss).
            with pytest.raises(HTTPException):
                driver.speech({"model": "c", "input": "hi"})
            # Live add of a third name (no restart) -> routes to its slug.
            driver.set_models(
                [
                    *_multi_models(),
                    ModelSpec(
                        name="c", modality="tts", backend_config={"slug": ASR_SLUG}
                    ),
                ]
            )
            stream = driver.speech(
                {"model": "c", "input": "hi", "response_format": "wav"}
            )
            await _collect(stream.chunks)
            assert stub["last_speech"]["model"] == ASR_SLUG
        finally:
            await driver.aclose()


async def test_live_add_resident_slug_keeps_health_and_routes(tmp_path) -> None:
    """(a) A live set_models add whose slug is ALREADY resident keeps health
    green and routes the new name (no restart needed)."""
    engine, stub = make_stub_engine(slugs=[TTS_SLUG, ASR_SLUG])
    async with serve_stub(engine) as (_url, port):
        driver = _connected_multi(tmp_path, port)
        driver._proc = _AliveProc()
        driver._spawned_slugs = [TTS_SLUG, ASR_SLUG]  # simulate a started proc
        try:
            assert await driver.health() is True
            driver.set_models(
                [
                    *_multi_models(),
                    ModelSpec(
                        name="c", modality="tts", backend_config={"slug": ASR_SLUG}
                    ),
                ]
            )
            # c -> ASR is resident: health stays green, routing works.
            assert await driver.health() is True
            stream = driver.speech(
                {"model": "c", "input": "hi", "response_format": "wav"}
            )
            await _collect(stream.chunks)
            assert stub["last_speech"]["model"] == ASR_SLUG
        finally:
            await driver.aclose()


async def test_live_add_new_slug_does_not_flip_health(tmp_path) -> None:
    """(b) MEDIUM regression: the health pin tracks the SPAWNED slug set, so a
    served name added live whose slug was never loaded here does NOT flip an
    otherwise-healthy running backend unhealthy (that change must go through
    the config.update restart path)."""
    engine, _stub = make_stub_engine(slugs=[TTS_SLUG])  # only TTS resident
    async with serve_stub(engine) as (_url, port):
        driver = TalkiesBackend(make_settings(tmp_path), {})
        driver.set_models(
            [ModelSpec(name="a", modality="tts", backend_config={"slug": TTS_SLUG})]
        )
        driver.backend_port = port
        driver.slug = TTS_SLUG
        driver._auth_token = "stub-token"
        driver._proc = _AliveProc()
        driver._spawned_slugs = [TTS_SLUG]  # only TTS was spawned
        try:
            assert await driver.health() is True
            # Live add b -> ASR, a slug NOT resident in this process.
            driver.set_models(
                [
                    ModelSpec(
                        name="a", modality="tts", backend_config={"slug": TTS_SLUG}
                    ),
                    ModelSpec(
                        name="b", modality="asr", backend_config={"slug": ASR_SLUG}
                    ),
                ]
            )
            # The live slug set grew, but the pin stays on what is loaded.
            assert driver.slugs == [TTS_SLUG, ASR_SLUG]
            assert await driver.health() is True
        finally:
            await driver.aclose()


async def test_spawned_slugs_cleared_on_stop(
    tmp_path, fake_talkies_python, talkies_registry, model_dirs
) -> None:
    """stop() clears the health-pin set so a stale spawned slug can never leak
    into the next boot's readiness gate."""
    assert model_dirs  # both slugs cached -> prefetch never touches network
    settings = make_settings(
        tmp_path, fake_talkies_python, models_file=talkies_registry
    )
    driver = TalkiesBackend(settings, {})
    driver.set_models(_multi_models())
    await driver.start()
    assert driver._spawned_slugs == [TTS_SLUG, ASR_SLUG]
    await driver.stop()
    assert driver._spawned_slugs == []
    await driver.aclose()


# ---------------------------------------------------------------------------
# WS bridge: served NAME -> slug (end-to-end through the agent /v1 surface)
# ---------------------------------------------------------------------------


async def test_multi_ws_bridge_routes_name_to_slug(tmp_path) -> None:
    import uvicorn
    import websockets

    engine, _stub = make_stub_engine(slugs=[TTS_SLUG, ASR_SLUG])
    driver = TalkiesBackend(make_settings(tmp_path), {})
    driver.set_models(_multi_models())
    driver.slug = TTS_SLUG
    driver._auth_token = "stub-token"

    async def fake_start() -> None:
        return None

    async def fake_health() -> bool:
        return True

    driver.start = fake_start  # type: ignore[method-assign]
    driver.health = fake_health  # type: ignore[method-assign]

    async with serve_stub(engine) as (_url, engine_port):
        driver.backend_port = engine_port
        lifecycle = BackendLifecycle(driver, capacity=1, instance_id="inst-mm")
        await lifecycle.start()
        registry = BackendRegistry()
        registry.add(
            BackendHandle(
                "inst-mm",
                lifecycle,
                ConfigState(),
                alias="a",
                models=list(driver._models),
            )
        )
        app = create_provider_app(
            make_settings(tmp_path),
            BackendOverrides(
                provider_type="talkies", version="test", registry=registry
            ),
        )
        config = uvicorn.Config(app, host="127.0.0.1", port=0, log_level="warning")
        server = uvicorn.Server(config)
        task = asyncio.create_task(server.serve())
        while not server.started:
            await asyncio.sleep(0.02)
        agent_port = int(server.servers[0].sockets[0].getsockname()[1])
        try:
            # Connect for served name "b" (the ASR model): the bridge must map
            # the query + start-frame name to its slug before the engine sees it.
            async with websockets.connect(
                f"ws://127.0.0.1:{agent_port}/v1/audio/transcriptions/stream?model=b"
            ) as ws:
                await ws.send(
                    json.dumps(
                        {
                            "type": "start",
                            "model": "b",
                            "encoding": "pcm_s16le",
                            "sample_rate": 16000,
                            "channels": 1,
                        }
                    )
                )
                ready = json.loads(await asyncio.wait_for(ws.recv(), 5))
                assert ready["type"] == "ready"
                assert ready["model"] == ASR_SLUG  # name b -> asr slug
                await ws.send(json.dumps({"type": "end"}))
                finals = [
                    json.loads(await asyncio.wait_for(ws.recv(), 5)) for _ in range(2)
                ]
                assert [m["type"] for m in finals] == ["final", "stats"]
        finally:
            server.should_exit = True
            await task
            await lifecycle.stop()
            await driver.aclose()


# ---------------------------------------------------------------------------
# env: multi-slug map construction (pure)
# ---------------------------------------------------------------------------


def test_env_multi_slug_map() -> None:
    from provider_talkies import env as tenv

    models = _multi_models()
    assert tenv.resolve_slugs({}, models) == [TTS_SLUG, ASR_SLUG]
    assert tenv.resolve_model_concurrency_map({}, models) == {TTS_SLUG: 1, ASR_SLUG: 2}
    # legacy (no models) collapses to the single shared slug.
    assert tenv.resolve_slugs({"model": {"slug": TTS_SLUG}}, []) == [TTS_SLUG]
    assert tenv.resolve_model_concurrency_map({"model": {"slug": TTS_SLUG}}, []) == {
        TTS_SLUG: 1
    }


def test_env_resolve_slugs_multi_model_error_is_accurate() -> None:
    """LOW-1: a multi-model served model missing backend_config.slug raises an
    accurate error, NOT the misleading legacy 'requires model.slug'."""
    from provider_talkies import env as tenv

    slugless = [ModelSpec(name="a", modality="tts", backend_config={})]
    with pytest.raises(RuntimeError, match="missing backend_config.slug"):
        tenv.resolve_slugs({}, slugless)
    # An all-disabled served list is also an accurate multi-model error.
    disabled = [
        ModelSpec(
            name="a",
            modality="tts",
            backend_config={"slug": TTS_SLUG},
            enabled=False,
        )
    ]
    with pytest.raises(RuntimeError, match="no enabled served models"):
        tenv.resolve_slugs({}, disabled)


def test_env_concurrency_map_max_per_slug() -> None:
    """LOW-2: two served names sharing one slug (one engine pool) take the MAX
    concurrency, not last-write-wins (order-independent)."""
    from provider_talkies import env as tenv

    models = [
        ModelSpec(
            name="a",
            modality="tts",
            backend_config={"slug": TTS_SLUG, "limits": {"model_concurrency": 2}},
        ),
        ModelSpec(
            name="b",
            modality="tts",
            backend_config={"slug": TTS_SLUG, "limits": {"model_concurrency": 5}},
        ),
    ]
    assert tenv.resolve_model_concurrency_map({}, models) == {TTS_SLUG: 5}
    assert tenv.resolve_model_concurrency_map({}, list(reversed(models))) == {
        TTS_SLUG: 5
    }
    # A name with no explicit concurrency falls back to the shared default,
    # and the max still wins against an explicit higher value.
    mixed = [
        ModelSpec(name="a", modality="tts", backend_config={"slug": TTS_SLUG}),
        ModelSpec(
            name="b",
            modality="tts",
            backend_config={"slug": TTS_SLUG, "limits": {"model_concurrency": 4}},
        ),
    ]
    assert tenv.resolve_model_concurrency_map(
        {"limits": {"model_concurrency": 1}}, mixed
    ) == {TTS_SLUG: 4}


# ---------------------------------------------------------------------------
# list_models: voice-catalog embedding (backend.metadata scrape)
# ---------------------------------------------------------------------------


async def test_list_models_embeds_voices_per_tts_entry(tmp_path) -> None:
    """Each tts entry carries its live catalog (fetched via the driver's voices
    proxy, retagged to the served name); asr entries carry no voices key."""
    engine, _stub = make_stub_engine(slugs=[TTS_SLUG, ASR_SLUG])
    async with serve_stub(engine) as (_url, port):
        driver = _connected_multi(tmp_path, port)
        try:
            models = await driver.list_models()
            by_id = {m["id"]: m for m in models}
            assert "voices" in by_id["a"]
            assert {v["voice"] for v in by_id["a"]["voices"]} == {"Vivian", "clone1"}
            assert {v["model"] for v in by_id["a"]["voices"]} == {"a"}
            assert "voices" not in by_id["b"]  # asr entry: no voices
        finally:
            await driver.aclose()


async def test_list_models_legacy_embeds_voices(tmp_path) -> None:
    """Legacy single-slug tts definition embeds its voices list under the alias."""
    engine, _stub = make_stub_engine()
    async with serve_stub(engine) as (_url, port):
        driver = TalkiesBackend(make_settings(tmp_path), {"model": {"slug": TTS_SLUG}})
        driver.apply_config({"model": {"slug": TTS_SLUG}})
        driver.backend_port = port
        driver.slug = TTS_SLUG
        driver._auth_token = "stub-token"
        driver._data_dir = tmp_path / "data"
        driver.set_alias("my-tts")
        driver.modality = "tts"
        try:
            models = await driver.list_models()
            assert models[0]["id"] == "my-tts"
            assert "voices" in models[0]
            assert {v["voice"] for v in models[0]["voices"]} == {"Vivian", "clone1"}
        finally:
            await driver.aclose()


async def test_list_models_voices_failure_omits_key(tmp_path) -> None:
    """A per-entry voices-fetch failure omits only that entry's voices key —
    the other entries still publish (a scrape must never fail a boot)."""
    driver = TalkiesBackend(make_settings(tmp_path), {})
    driver.set_models(
        [
            ModelSpec(name="a", modality="tts", backend_config={"slug": TTS_SLUG}),
            ModelSpec(name="b", modality="tts", backend_config={"slug": ASR_SLUG}),
            ModelSpec(name="c", modality="asr", backend_config={"slug": ASR_SLUG}),
        ]
    )

    async def flaky_voices(model: str | None = None, *, timeout: float | None = None):  # noqa: ARG001
        if model == "a":
            raise RuntimeError("engine down")
        return {"voices": [{"voice": "Vivian", "model": model}]}

    driver.voices = flaky_voices  # type: ignore[method-assign]
    models = await driver.list_models()
    by_id = {m["id"]: m for m in models}
    assert set(by_id) == {"a", "b", "c"}  # every entry still returned
    assert "voices" not in by_id["a"]  # failed fetch -> key omitted
    assert by_id["b"]["voices"] == [{"voice": "Vivian", "model": "b"}]
    assert "voices" not in by_id["c"]  # asr never fetched


async def test_list_models_uses_short_voices_timeout(tmp_path) -> None:
    """The boot-ack scrape passes the short per-entry timeout (not the 15 s
    client default) so a hung engine cannot stall the boot ack."""
    driver = TalkiesBackend(make_settings(tmp_path), {})
    driver.set_models(
        [
            ModelSpec(name="a", modality="tts", backend_config={"slug": TTS_SLUG}),
            ModelSpec(name="b", modality="tts", backend_config={"slug": ASR_SLUG}),
        ]
    )
    seen: list[tuple[str | None, float | None]] = []

    async def spy_voices(model: str | None = None, *, timeout: float | None = None):
        seen.append((model, timeout))
        return {"voices": [{"voice": "Vivian", "model": model}]}

    driver.voices = spy_voices  # type: ignore[method-assign]
    await driver.list_models()
    assert {m for m, _ in seen} == {"a", "b"}
    assert all(t == driver_mod._VOICES_SCRAPE_TIMEOUT for _, t in seen)
