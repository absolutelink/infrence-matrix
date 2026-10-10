import { expect, test } from "@playwright/test"

import {
  addServedModel,
  derivePrimaryAlias,
  emptyServedModel,
  formToServedModels,
  isMultiModelType,
  removeServedModel,
  type ServedModelForm,
  servedModelConfigSchema,
  servedModelsToForm,
  toggleServedModel,
  updateServedModel,
  validateServedModels,
} from "../src/lib/multiModel"

// Phase 25 (S4) locks the framework-free multi-model definition logic the
// definitions editor depends on: served_models <-> form-state round-trip, the
// §6 client-side validation, the primary-alias derivation, and the add/remove/
// toggle/update list ops. These run in Node (no browser, no live backend) —
// mirroring tests/audio.spec.ts, since the repo has no component/unit harness
// and the generated client is not mockable without a DOM runner.

function row(over: Partial<ServedModelForm> = {}): ServedModelForm {
  return {
    id: `id-${Math.random().toString(36).slice(2)}`,
    name: "m",
    modality: "llm",
    backend_config: {},
    enabled: true,
    ...over,
  }
}

test("isMultiModelType reads x-multi-model from the committed schema", () => {
  // The provider-types endpoint returns the full schema; the flag lives there.
  expect(isMultiModelType({ schema: { "x-multi-model": true } })).toBe(true)
  expect(isMultiModelType({ schema: { "x-multi-model": false } })).toBe(false)
  expect(isMultiModelType({ schema: {} })).toBe(false)
  // A direct multi_model field (if the API ever surfaces it) is honored too.
  expect(isMultiModelType({ multi_model: true })).toBe(true)
  expect(isMultiModelType({ multi_model: false })).toBe(false)
})

test("isMultiModelType is false while the type detail is loading (single-model default)", () => {
  // A cold cache must never flip a single-model form into the multi-model
  // layout (llama-cpp / halogen-flash / mock render exactly as before).
  expect(isMultiModelType(undefined)).toBe(false)
  expect(isMultiModelType(null)).toBe(false)
  expect(isMultiModelType({})).toBe(false)
})

test("servedModelConfigSchema extracts the per-model subschema or null", () => {
  const sub = { type: "object", required: ["slug"] }
  expect(
    servedModelConfigSchema({
      schema: { "x-served-model-config-schema": sub },
    }),
  ).toEqual(sub)
  expect(servedModelConfigSchema({ schema: {} })).toBeNull()
  expect(servedModelConfigSchema({ schema: null })).toBeNull()
  expect(servedModelConfigSchema(undefined)).toBeNull()
  // A non-object (array) value is ignored.
  expect(
    servedModelConfigSchema({ schema: { "x-served-model-config-schema": [] } }),
  ).toBeNull()
})

test("served_models round-trips through the form and back to the wire", () => {
  const api = [
    {
      name: "qwen3-tts",
      modality: "tts",
      backend_config: { slug: "qwen3-tts-1.7b" },
      enabled: true,
    },
    {
      name: "whisper",
      modality: "asr",
      backend_config: { slug: "whisper-large-v3-turbo" },
      enabled: false,
    },
  ]
  const form = servedModelsToForm(api)
  expect(form).toHaveLength(2)
  // Each row gets a stable client-side id (used as the React list key).
  expect(typeof form[0].id).toBe("string")
  expect(form[0].id).not.toBe(form[1].id)
  const { id: _id0, ...form0 } = form[0]
  expect(form0).toEqual({
    name: "qwen3-tts",
    modality: "tts",
    backend_config: { slug: "qwen3-tts-1.7b" },
    enabled: true,
  })
  expect(form[1].enabled).toBe(false)
  // Back to the wire shape preserves every field (names trimmed, id dropped).
  expect(formToServedModels(form)).toEqual(api)
})

test("servedModelsToForm fills defaults for omitted fields", () => {
  const form = servedModelsToForm([{ name: "x" }])
  const { id: _id, ...rest } = form[0]
  expect(typeof _id).toBe("string")
  expect(rest).toEqual({
    name: "x",
    modality: "llm",
    backend_config: {},
    enabled: true,
  })
  // Non-array / junk input yields an empty list.
  expect(servedModelsToForm(null)).toEqual([])
  expect(servedModelsToForm(undefined)).toEqual([])
  expect(servedModelsToForm("nope")).toEqual([])
  // A bad modality falls back to llm; a non-object config to {}.
  const messy = servedModelsToForm([
    { name: "y", modality: "bogus", backend_config: [1, 2] },
  ])
  expect(messy[0].modality).toBe("llm")
  expect(messy[0].backend_config).toEqual({})
})

test("derivePrimaryAlias is the first ENABLED named row", () => {
  expect(
    derivePrimaryAlias([
      row({ name: "a", enabled: false }),
      row({ name: "b", enabled: true }),
      row({ name: "c", enabled: true }),
    ]),
  ).toBe("b")
  // Leading whitespace is trimmed.
  expect(derivePrimaryAlias([row({ name: "  primary  " })])).toBe("primary")
  // No enabled row → empty (caller falls back to the first row's name).
  expect(derivePrimaryAlias([row({ name: "x", enabled: false })])).toBe("")
  expect(derivePrimaryAlias([])).toBe("")
})

test("validateServedModels requires at least one entry", () => {
  const v = validateServedModels([], ["llm"])
  expect(v.ok).toBe(false)
  expect(v.message).toBeTruthy()
})

test("validateServedModels rejects empty and duplicate names", () => {
  const v = validateServedModels(
    [row({ name: "  " }), row({ name: "dup" }), row({ name: "dup" })],
    ["llm", "tts"],
  )
  expect(v.ok).toBe(false)
  expect(v.rows[0].name).toBeTruthy()
  // The second "dup" is flagged; the first is fine.
  expect(v.rows[1].name).toBeUndefined()
  expect(v.rows[2].name).toBeTruthy()
})

test("validateServedModels rejects a modality the type does not serve", () => {
  const v = validateServedModels([row({ name: "a", modality: "asr" })], ["tts"])
  expect(v.ok).toBe(false)
  expect(v.rows[0].modality).toBeTruthy()
})

test("validateServedModels accepts a well-formed list", () => {
  const v = validateServedModels(
    [
      row({ name: "tts-a", modality: "tts" }),
      row({ name: "asr-b", modality: "asr" }),
    ],
    ["tts", "asr"],
  )
  expect(v.ok).toBe(true)
  expect(v.message).toBeUndefined()
  expect(v.rows.every((r) => !r.name && !r.modality)).toBe(true)
})

test("list ops are pure and return new arrays", () => {
  const base = [row({ name: "a" }), row({ name: "b" })]

  const added = addServedModel(base, "tts")
  expect(added).toHaveLength(3)
  // The appended row is a fresh blank (its id is generated, so compare fields).
  const { id: _addedId, ...addedBlank } = added[2]
  expect(typeof _addedId).toBe("string")
  expect(addedBlank).toEqual({
    name: "",
    modality: "tts",
    backend_config: {},
    enabled: true,
  })
  expect(base).toHaveLength(2) // original untouched

  const removed = removeServedModel(base, 0)
  expect(removed.map((r) => r.name)).toEqual(["b"])
  expect(base).toHaveLength(2)

  const toggled = toggleServedModel(base, 1)
  expect(toggled[1].enabled).toBe(false)
  expect(toggled[0].enabled).toBe(true)
  expect(base[1].enabled).toBe(true)

  const updated = updateServedModel(base, 0, { name: "renamed" })
  expect(updated[0].name).toBe("renamed")
  expect(base[0].name).toBe("a")
})

test("emptyServedModel defaults to an enabled llm row with a fresh id", () => {
  const r = emptyServedModel()
  expect(typeof r.id).toBe("string")
  expect(r.id).not.toBe("")
  const { id: _id, ...rest } = r
  expect(rest).toEqual({
    name: "",
    modality: "llm",
    backend_config: {},
    enabled: true,
  })
  // Two blanks never share an id.
  expect(emptyServedModel().id).not.toBe(emptyServedModel().id)
})

test("served row ids are stable across add/remove/toggle/update (MEDIUM-2)", () => {
  const a = row({ name: "a" })
  const b = row({ name: "b" })
  const c = row({ name: "c" })
  const base = [a, b, c]

  // Removing the middle row keeps the surviving rows' ids intact (so their
  // per-row local state — e.g. the raw-JSON config textarea — is not reused
  // for a different row).
  const removed = removeServedModel(base, 1)
  expect(removed.map((r) => r.id)).toEqual([a.id, c.id])

  // add appends a brand-new id; toggle/update preserve the target's id.
  const added = addServedModel(base, "tts")
  expect(added[3].id).not.toBe(a.id)
  expect(added.slice(0, 3).map((r) => r.id)).toEqual([a.id, b.id, c.id])

  expect(toggleServedModel(base, 0)[0].id).toBe(a.id)
  expect(updateServedModel(base, 2, { name: "c2" })[2].id).toBe(c.id)
})

test("validateServedModels rejects an over-long name (LOW-2)", () => {
  const long = "x".repeat(256)
  const v = validateServedModels([row({ name: long })], ["llm"])
  expect(v.ok).toBe(false)
  expect(v.rows[0].name).toContain("too long")
  // Exactly 255 is accepted.
  const ok = validateServedModels([row({ name: "y".repeat(255) })], ["llm"])
  expect(ok.ok).toBe(true)
})
