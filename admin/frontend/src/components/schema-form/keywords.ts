// Custom `x-` keyword readers for provider schema.json (provider/README.md
// "Authoring schema.json", Phase 12). The admin validates these
// server-side; the form honors them for rendering only:
//   x-flag            → tooltip (CLI flag / env var mapping)
//   x-widget: hf-file → HuggingFace artifact picker
//   x-numeric-effect  → "bitwise" | "numeric" (numeric warns "changes outputs")
//   x-secret          → masked write-only input
//   x-supported: false→ rendered disabled ("not wired")
//   title / x-order   → section labels + ordering

import type { RJSFSchema, UiSchema } from "@rjsf/utils"

export const HF_FILE_FIELD = "hfFile"

type Rec = Record<string, unknown>

export function asRecord(value: unknown): Rec | undefined {
  return typeof value === "object" && value !== null && !Array.isArray(value)
    ? (value as Rec)
    : undefined
}

function kw(value: unknown, key: string): unknown {
  return asRecord(value)?.[key]
}

export function xFlag(schema: unknown): string | undefined {
  const v = kw(schema, "x-flag")
  return typeof v === "string" ? v : undefined
}

export function xSecret(schema: unknown): boolean {
  return kw(schema, "x-secret") === true
}

export function xNotSupported(schema: unknown): boolean {
  return kw(schema, "x-supported") === false
}

export function xNumericEffect(
  schema: unknown,
): "bitwise" | "numeric" | undefined {
  const v = kw(schema, "x-numeric-effect")
  return v === "bitwise" || v === "numeric" ? v : undefined
}

export function xWidget(schema: unknown): string | undefined {
  const v = kw(schema, "x-widget")
  return typeof v === "string" ? v : undefined
}

/** A property is an hf-file artifact descriptor when it $refs the
 *  canonical `#/$defs/hfFile` def or carries `x-widget: hf-file`. */
export function isHfFileSchema(schema: unknown): boolean {
  const ref = kw(schema, "$ref")
  if (typeof ref === "string" && ref.endsWith("/hfFile")) return true
  return xWidget(schema) === "hf-file"
}

/** Order property names by their numeric `x-order` (missing → end,
 *  stable on name). */
export function orderPropertyNames(
  properties: Record<string, unknown>,
  names: string[],
): string[] {
  const rank = (name: string): number => {
    const v = kw(properties[name], "x-order")
    return typeof v === "number" ? v : Number.MAX_SAFE_INTEGER
  }
  return [...names].sort((a, b) => rank(a) - rank(b) || (a < b ? -1 : 1))
}

/**
 * Build the rjsf uiSchema tree from a provider schema, translating the
 * custom `x-` keywords into rjsf directives:
 *  - `x-order`  → `ui:order` on every object level
 *  - hf-file   → `ui:field: hfFile` (also per-branch inside oneOf/anyOf)
 *  - `x-secret` → `ui:widget: secret`
 *  - `x-supported: false` → `ui:disabled: true`
 */
export function buildUiSchema(schema: RJSFSchema): UiSchema {
  const root: UiSchema = {
    "ui:submitButtonOptions": { norender: true },
  }
  fillLevel(root, schema)
  return root
}

function fillLevel(ui: UiSchema, schema: RJSFSchema): void {
  const props = (asRecord(schema.properties) ?? {}) as Record<
    string,
    RJSFSchema
  >
  const names = Object.keys(props)
  if (names.length > 0) {
    ui["ui:order"] = [...orderPropertyNames(props, names), "*"]
  }
  for (const name of names) {
    const sub = props[name]
    const subRec = asRecord(sub)
    if (!subRec) continue
    const entry: UiSchema = {}
    if (isHfFileSchema(subRec)) {
      // Direct hfFile property: the custom field fully replaces the
      // union selector rjsf would otherwise build for the resolved
      // oneOf descriptor.
      entry["ui:field"] = HF_FILE_FIELD
      entry["ui:fieldReplacesAnyOrOneOf"] = true
    }
    // Union branches (oneOf/anyOf) may contain an hfFile option — point
    // that branch at the custom field so the picker shows up inside
    // MultiSchemaField too (e.g. halogen-flash vision_tower).
    // NOTE (verified against installed @rjsf/core 6.11.0):
    // MultiSchemaField reads per-branch overrides as the BARE schema
    // keywords on the uiSchema (`uiSchema[ONE_OF_KEY]` with
    // ONE_OF_KEY === 'oneOf', see MultiSchemaField.js:120-132), not
    // `ui:oneOf`. We write the bare key (what rjsf actually reads) and
    // mirror it under `ui:oneOf`/`ui:anyOf` for forward-compat in case
    // a future version switches to the prefixed spelling.
    for (const key of ["oneOf", "anyOf"] as const) {
      const branches = subRec[key]
      if (Array.isArray(branches)) {
        const perOption: UiSchema[] = branches.map((b) =>
          isHfFileSchema(b)
            ? { "ui:field": HF_FILE_FIELD, "ui:fieldReplacesAnyOrOneOf": true }
            : {},
        )
        if (perOption.some((o) => Object.keys(o).length > 0)) {
          ;(entry as Rec)[key] = perOption
          ;(entry as Rec)[`ui:${key}`] = perOption
        }
      }
    }
    if (xSecret(subRec)) {
      entry["ui:widget"] = "secret"
    }
    if (xNotSupported(subRec)) {
      entry["ui:disabled"] = true
    }
    // Recurse into plain object sections (hf-file descriptors render
    // through the custom field, not the generic object template).
    if (
      subRec.type === "object" &&
      subRec.properties &&
      entry["ui:field"] === undefined
    ) {
      const child: UiSchema = {}
      fillLevel(child, sub)
      if (Object.keys(entry).length === 0) {
        ui[name] = child
      } else {
        for (const [k, v] of Object.entries(child)) {
          if (!(k in entry)) (entry as Rec)[k] = v
        }
        ui[name] = entry
      }
    } else if (Object.keys(entry).length > 0) {
      ui[name] = entry
    }
  }
}

/** Dotted paths (e.g. "endpoints.api_key") of every x-secret leaf field
 *  in the schema. Used to strip stored secrets out of the form's initial
 *  formData and to merge them back on submit (write-only semantics:
 *  empty input = keep the stored value). */
export function collectSecretPaths(schema: unknown): string[] {
  const out: string[] = []
  const walk = (node: unknown, prefix: string) => {
    const props = asRecord(asRecord(node)?.properties)
    if (!props) return
    for (const [name, sub] of Object.entries(props)) {
      const path = prefix ? `${prefix}.${name}` : name
      if (xSecret(sub)) out.push(path)
      walk(sub, path)
    }
  }
  walk(schema, "")
  return out
}

export function getByDotPath(obj: unknown, path: string): unknown {
  let cur: unknown = obj
  for (const part of path.split(".")) {
    cur = asRecord(cur)?.[part]
  }
  return cur
}

/**
 * Immutably set (value !== undefined) or delete (value === undefined) a
 * dotted path.
 *
 * L3 fix: on delete, if the parent chain doesn't exist we return the
 * object UNTOUCHED — the old code materialized empty intermediate
 * objects before deleting the leaf, so `stripSecrets` on a config
 * without a `server` section produced `"server": {}` and churned the
 * config fingerprint. On set, intermediates are created as needed.
 */
export function setByDotPath<T>(obj: T, path: string, value: unknown): T {
  const parts = path.split(".")
  const root = { ...(asRecord(obj) ?? {}) } as Rec
  let cur = root
  for (const part of parts.slice(0, -1)) {
    const next = asRecord(cur[part])
    if (value === undefined && next === undefined) {
      // Nothing to delete along this chain — return the (already
      // shallow-copied) root with no intermediate objects created.
      return root as T
    }
    cur[part] = next ? { ...next } : {}
    cur = cur[part] as Rec
  }
  const last = parts[parts.length - 1]
  if (value === undefined) {
    delete cur[last]
  } else {
    cur[last] = value
  }
  return root as T
}

/** Compact one-line summary of an hf-file descriptor value for the UI. */
export function describeHfDescriptor(value: unknown): string | null {
  const rec = asRecord(value)
  if (!rec) return null
  if (typeof rec.path === "string") return rec.path
  const repo = rec.repo
  const file = rec.file
  if (typeof repo === "string" && typeof file === "string") {
    const rev =
      typeof rec.revision === "string" ? `@${rec.revision.slice(0, 8)}` : ""
    return `${repo}/${file}${rev}`
  }
  return null
}
