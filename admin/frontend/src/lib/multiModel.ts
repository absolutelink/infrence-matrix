// Phase 25 (S4): framework-free helpers for the multi-model definition editor.
//
// A multi-model provider type (schema `x-multi-model: true`) lets one
// definition expose a repeatable list of served models, each with its own
// client-facing name, modality, per-model backend_config, and enabled flag
// (docs/multi-model-definitions.md §3). The admin owns the names; the alias is
// DERIVED server-side from the first enabled entry, and the top-level
// backend_config/modality become the shared engine-level config.
//
// Everything here is pure (no React, no generated client) so the served_models
// <-> form-state round-trip, the §6 client-side validation, the primary-alias
// derivation, and the add/remove/toggle list ops can be locked with light Node
// tests (see tests/multiModel.spec.ts), mirroring the @/lib/audio pattern.

import type { RJSFSchema } from "@rjsf/utils"

import { isModality, type Modality } from "./audio"

/** One editable served-model row in the form (mirrors the wire ServedModelIn,
 *  plus a stable client-side `id` used as the React list key so per-row local
 *  state (e.g. the raw-JSON config textarea) survives add/remove reordering. */
export interface ServedModelForm {
  id: string
  name: string
  modality: Modality
  backend_config: Record<string, unknown>
  enabled: boolean
}

/** The wire shape sent to the admin (mirrors ServedModelIn / ModelSpec). */
export interface ServedModelIn {
  name: string
  modality: string
  backend_config: Record<string, unknown>
  enabled: boolean
}

/** Max served-model name length — mirrors the admin's ServedModelIn.name
 *  (Field(max_length=255)) so an over-long name surfaces inline, not as a 422. */
export const SERVED_NAME_MAX_LENGTH = 255

/** A stable unique id for a served row. Uses crypto.randomUUID where available
 *  (browsers in a secure context + Node ≥ 19.6 / Bun) and falls back to a
 *  time+random token so the editor never collides keys in an insecure context. */
function newRowId(): string {
  const c: Crypto | undefined = globalThis.crypto
  if (c && typeof c.randomUUID === "function") return c.randomUUID()
  return `sm-${Date.now().toString(36)}-${Math.random().toString(36).slice(2)}`
}

type TypeDetailLike =
  | {
      multi_model?: boolean
      schema?: Record<string, unknown> | null
      serves_modalities?: string[] | null
    }
  | null
  | undefined

/**
 * True when the selected provider type declares multi-model capability.
 *
 * The canonical source is the committed schema's top-level `x-multi-model`
 * (docs/multi-model-definitions.md §2) — the provider-types endpoint returns
 * the full `schema`, so we read the flag from there. A `multi_model` field on
 * the type detail (if the API ever surfaces it directly) is honored too.
 * Returns false while the detail is still loading (undefined) so a cold cache
 * never flips a single-model form into the multi-model layout.
 */
export function isMultiModelType(typeDetail: TypeDetailLike): boolean {
  if (!typeDetail) return false
  if (typeDetail.multi_model === true) return true
  const schema = typeDetail.schema
  return !!schema && schema["x-multi-model"] === true
}

/**
 * The type's `x-served-model-config-schema` (validates each served model's
 * per-model backend_config), or null when the type declares none (any object
 * accepted — the editor falls back to a raw JSON textarea).
 */
export function servedModelConfigSchema(
  typeDetail: TypeDetailLike,
): RJSFSchema | null {
  const schema = typeDetail?.schema
  const sub = schema?.["x-served-model-config-schema"]
  if (sub && typeof sub === "object" && !Array.isArray(sub)) {
    return sub as RJSFSchema
  }
  return null
}

/** A blank row to seed the editor / append on "add model". */
export function emptyServedModel(modality: Modality = "llm"): ServedModelForm {
  return {
    id: newRowId(),
    name: "",
    modality,
    backend_config: {},
    enabled: true,
  }
}

/**
 * Normalize the definition GET's `served_models` into editor rows, filling
 * defaults for anything the API omitted (modality → "llm", enabled → true,
 * backend_config → {}) and assigning a stable client-side `id` per row.
 * Non-array input yields an empty list.
 */
export function servedModelsToForm(rows: unknown): ServedModelForm[] {
  if (!Array.isArray(rows)) return []
  return rows.map((raw) => {
    const o = (raw ?? {}) as Record<string, unknown>
    const cfg = o.backend_config
    return {
      id: newRowId(),
      name: typeof o.name === "string" ? o.name : "",
      modality: isModality(o.modality) ? o.modality : "llm",
      backend_config:
        cfg && typeof cfg === "object" && !Array.isArray(cfg)
          ? (cfg as Record<string, unknown>)
          : {},
      enabled: o.enabled !== false,
    }
  })
}

/** Map editor rows to the wire list sent on create/PATCH (names trimmed). */
export function formToServedModels(rows: ServedModelForm[]): ServedModelIn[] {
  return rows.map((r) => ({
    name: r.name.trim(),
    modality: r.modality,
    backend_config: r.backend_config ?? {},
    enabled: r.enabled,
  }))
}

/**
 * The primary alias = the first ENABLED row's trimmed name (the admin syncs
 * `definition.alias` to exactly this — §3). Empty when no enabled row has a
 * name (the caller falls back to the first row's name for the create body).
 */
export function derivePrimaryAlias(rows: ServedModelForm[]): string {
  const first = rows.find((r) => r.enabled && r.name.trim())
  return first ? first.name.trim() : ""
}

export interface ServedModelRowErrors {
  name?: string
  modality?: string
}

export interface ServedModelsValidation {
  ok: boolean
  /** Top-level message (e.g. the empty-list case). */
  message?: string
  /** Per-row errors, index-parallel to the input rows. */
  rows: ServedModelRowErrors[]
}

/**
 * Client-side mirror of the admin's §6 served_models validation (the subset
 * checkable without the DB): ≥1 entry; names non-empty, ≤255 chars, and unique
 * within the list; each modality in the type's serves_modalities. Global
 * uniqueness (vs other definitions' aliases/names) and per-model config schema
 * validation still require the server — those surface as 422s the caller maps
 * inline.
 */
export function validateServedModels(
  rows: ServedModelForm[],
  servesModalities: readonly string[],
): ServedModelsValidation {
  const out: ServedModelsValidation = { ok: true, rows: rows.map(() => ({})) }
  if (rows.length === 0) {
    out.ok = false
    out.message = "add at least one served model"
    return out
  }
  const seen = new Set<string>()
  rows.forEach((r, i) => {
    const name = r.name.trim()
    if (!name) {
      out.rows[i].name = "name is required"
      out.ok = false
    } else if (name.length > SERVED_NAME_MAX_LENGTH) {
      out.rows[i].name = `name is too long (max ${SERVED_NAME_MAX_LENGTH})`
      out.ok = false
    } else if (seen.has(name)) {
      out.rows[i].name = "duplicate name in the list"
      out.ok = false
    } else {
      seen.add(name)
    }
    if (!servesModalities.includes(r.modality)) {
      out.rows[i].modality = "modality is not served by this provider type"
      out.ok = false
    }
  })
  return out
}

// --- Pure list operations (each returns a new array) -----------------------

export function addServedModel(
  rows: ServedModelForm[],
  modality: Modality = "llm",
): ServedModelForm[] {
  return [...rows, emptyServedModel(modality)]
}

export function removeServedModel(
  rows: ServedModelForm[],
  index: number,
): ServedModelForm[] {
  return rows.filter((_, i) => i !== index)
}

export function toggleServedModel(
  rows: ServedModelForm[],
  index: number,
): ServedModelForm[] {
  return rows.map((r, i) => (i === index ? { ...r, enabled: !r.enabled } : r))
}

export function updateServedModel(
  rows: ServedModelForm[],
  index: number,
  patch: Partial<ServedModelForm>,
): ServedModelForm[] {
  return rows.map((r, i) => (i === index ? { ...r, ...patch } : r))
}
