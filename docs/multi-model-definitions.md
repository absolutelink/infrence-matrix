# Multi-Model Definitions (Phase 25)

Status: **spec — locked decisions below are canonical.** First consumer:
`talkies`. Follow-ups: `gufo`, `halogen-flash`.

## 1. Problem

Some provider engines natively serve **several models from one process**
(talkies: multiple TTS/ASR slugs in one server with one shared model cache;
gufo: multi-model GGUF serving; halogen-flash: main LLM + NPU satellite
models — embed/rerank/decision). The Phase 24 shape — one definition per
slug — forces N definitions and N processes per machine (N× CUDA contexts,
N× boot cost, N placements to manage). Operators want **one definition =
one backend process = many client-facing model names.**

## 2. Concept

A `ProviderDefinition` may expose a **set of served models**. Each served
model has its own client-facing **name** (usable in any `model` field),
its own **modality**, its own per-model **backend_config**, and an
**enabled** flag. Exactly one `ProviderInstance` (one backend process)
serves them all on one agent.

Division of responsibility (locked):

- **The provider type declares the capability**: top-level
  `x-multi-model: true` in its shipped `schema.json` →
  `ProviderType.multi_model`. The type may also declare
  `x-served-model-config-schema` (a JSON Schema validating each served
  model's per-model `backend_config`; absent = any object).
- **The admin owns the names**: the operator names each served model and
  picks its modality; the admin pushes names + per-model config to the
  agent on the assignment/registration payloads ("the admin will tell the
  agent what its name is").
- **The provider routes**: the driver maps the inbound request's `model`
  (a served name) to the engine's internal model (talkies: slug; gufo:
  served_model_name; halogen-flash: which sub-model to invoke).

## 3. Data model

### ProviderType (new column)

| Field | Notes |
| --- | --- |
| `multi_model` | bool, from schema `x-multi-model` (default false). Read at every schema-commit point like `max_running_backends`. |

### ProviderModel (new table)

Used **only by multi-model definitions**; single-model definitions keep
using `ProviderDefinition.alias/modality/backend_config` unchanged
(backward compatible — no backfill).

| Field | Notes |
| --- | --- |
| `id` | PK uuid. |
| `definition_id` | FK → ProviderDefinition, CASCADE. |
| `name` | Client-facing model name. **Globally unique** (DB unique index) across all `ProviderModel.name` AND all `ProviderDefinition.alias` (app-enforced cross-check on create/PATCH). Immutable while the definition has instances attached (same gate as `modality`). |
| `modality` | `llm` \| `embedding` \| `tts` \| `asr`. Must be in the type's `serves_modalities`. **This is the routing key for endpoint gating** (per-model modality). |
| `backend_config` | Per-model engine config (validated against the type's `x-served-model-config-schema` when declared). talkies: `{slug, revision?, defaults?}`. |
| `enabled` | Disabled names 404 like a disabled definition. |

Canonical accessor `resolve_models(definition) -> list[ModelSpec]`
(`name`, `modality`, `backend_config`, `enabled`): the definition's
`ProviderModel` rows when any exist, else the synthesized single entry
`[{alias, modality, backend_config, enabled}]`. **Every consumer routes
through this accessor** — scheduler, /v1/models, assignment payloads,
v1 endpoint gates, alias registration.

### ProviderDefinition (semantics)

- `alias` on a multi-model definition = the **first enabled** served
  model's name (kept in sync by the API; display/back-compat only).
- `modality`/`backend_config` on a multi-model definition = the
  **shared engine-level** config (talkies: `engine`/`limits`/`security`
  sections); per-model knobs live in `ProviderModel.backend_config`.
- `vram_required_bytes` = the **sum** for the one process (operator-set;
  UI may auto-suggest the sum). One boot, one admission, one idle
  timeout, one capacity pool shared by all served models.

## 4. Control plane

- **Registration response** `backends[].definition` and
  **`agent.assignments.update` entries** gain
  `served_models: [{name, modality, backend_config, enabled}]` (always
  the canonical list — single-model definitions send one entry).
  Additive: old agents ignore the key; new agents must prefer it when
  present and fall back to `alias`/`modality`/`backend_config` otherwise.
- **`provider.config.update`** (Phase 9/15) pushes per-model config
  changes to the running backend; fingerprint = hash of
  `(shared backend_config, served_models list)` so a per-model edit
  triggers the update flow.

## 5. Data plane routing

- **Admin**: every `/v1` endpoint resolves `request.model` →
  `ModelSpec` → owning definition → instance. The endpoint's required
  modality is checked against the **served model's** modality (e.g.
  `/v1/audio/speech` accepts any name whose spec modality is `tts`,
  whether it came from a single- or multi-model definition). Scheduler
  `acquire/release` key stays the **requested name** (per-name FIFO
  fairness), while boot/VRAM admission operates on the single owning
  instance.
- **Agent**: `BackendHandle` carries the model list;
  `registry.resolve_by_model` matches any enabled name. The driver
  receives the request unchanged and routes internally by `model`
  (talkies rewrites name→slug on the forwarded request; the name→slug
  map comes from the served-model configs).
- **`/v1/models`** (admin): lists every enabled served name with its
  `modality` marker and `owned_by` = the owning definition's provider
  type.
- **litellm**: registration is per-name with that name's modality
  (llm/embedding names only; tts/asr names bypass litellm as of P24).

## 6. Validation rules (admin API)

1. `served_models` may be set only when the type has `multi_model`;
   non-multi-model types sending it → 422.
2. ≥1 entry; names globally unique (vs aliases + other served names);
   each modality ∈ type's `serves_modalities`; each per-model config
   validates against `x-served-model-config-schema` when declared.
3. While instances are attached: names are immutable (409), entries may
   only be toggled `enabled` or have their config updated (config change
   flows through `provider.config.update`).
4. Single-model definitions are untouched by all of the above (no rows
   in `provider_models`).

## 7. talkies mapping (first implementation)

- Schema: `x-multi-model: true`; `x-served-model-config-schema` = the
  current `model` section object (`{slug, revision?}` + per-model
  `defaults`); the top-level `model` section becomes **optional**
  (single-slug defs keep working unchanged).
- One definition (e.g. `talkies-speech`) with 4 served models:
  `qwen3-tts-1.7b-custom`/`-1.7b`/`-1.7b-design` (tts) +
  `whisper-large-v3-turbo` (asr); `vram_required_bytes` = sum (~22 GiB).
- Driver: one engine process with
  `TALKIES_ENABLED_MODELS=TALKIES_PRELOAD=<all enabled slugs>`;
  `speech/transcribe/voices` map `request.model` (name) → slug before
  forwarding; `custom-voices` enrollment stays per-definition (shared
  data dir). Health pin: `/api/ps` must contain **all** enabled slugs.
- The 4 deployed single-slug definitions remain valid (no forced
  cutover); the operator collapses them to one definition when ready.

## 8. Slices

- **S0** (this doc + law docs).
- **S1** `provider_lib`: `ModelSpec`; handle carries model list;
  `resolve_by_model` matches any enabled name; assignment/registration
  parsing prefers `served_models`; mock multi-name backend for tests.
- **S2** `provider/talkies`: multi-slug process (env from served-model
  configs), name→slug rewrite in speech/transcribe/voices/WS, health
  pin over all slugs, schema `x-multi-model` + per-model config schema.
- **S3** admin: `ProviderModel` table + migration; `resolve_models`
  accessor; scheduler name→(definition, spec) lookup + per-name
  acquire/instance boot; v1 route gates via spec modality; `/v1/models`
  expansion; definition create/PATCH/GET + assignment payload;
  alias_registry per-name; fingerprint includes served_models.
- **S4** frontend + client regen: multi-model definition form (repeatable
  name/modality/enabled/per-model config entries), badges listing served
  names.
- **S5** docs polish (provider/README.md authoring section, ws-protocol
  payload rows) + optional deploy cutover of the 4 talkies defs into one.
