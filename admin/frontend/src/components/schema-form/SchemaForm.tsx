// SchemaForm (Phase 12 E2): the reusable rjsf wrapper that renders a
// provider type's committed schema.json as the backend_config editor.
//
// - Controlled: parent owns `value` (the sectioned backend_config
//   object) and receives every change via `onChange`.
// - Collapsible sections via SectionedObjectFieldTemplate (artifacts
//   open, others closed).
// - x-secret fields never display the stored value: the initial data is
//   scrubbed with `stripSecrets` and the stored values are merged back on
//   submit via `restoreSecrets` (empty input = keep current; there is
//   deliberately no UI to REMOVE a stored secret — rotate by typing a
//   new value).
// - Server 422 per-field errors map into rjsf `extraErrors` with
//   `serverErrorsToErrorSchema`.
// - `pruneUntouchedDefaults` keeps an untouched save byte-identical to
//   the stored config (no fingerprint churn from rjsf materializing
//   schema defaults).

import Form from "@rjsf/core"
import type { ErrorSchema, RJSFSchema } from "@rjsf/utils"
import { createSchemaUtils, deepEquals, ErrorSchemaBuilder } from "@rjsf/utils"
import validator from "@rjsf/validator-ajv8"
import { useMemo } from "react"

import { HfFileField } from "./HfFileWidget"
import {
  asRecord,
  buildUiSchema,
  collectSecretPaths,
  getByDotPath,
  setByDotPath,
} from "./keywords"
import { HintFieldTemplate, SectionedObjectFieldTemplate } from "./templates"
import {
  SecretWidget,
  ThemedCheckboxWidget,
  ThemedInputWidget,
  ThemedSelectWidget,
  ThemedTextareaWidget,
} from "./widgets"

export type BackendConfig = Record<string, unknown>

/** The value the form should start from: the stored config minus every
 *  x-secret leaf (write-only fields never round-trip into the UI). */
export function stripSecrets(
  schema: RJSFSchema,
  config: BackendConfig,
): BackendConfig {
  let out = config
  for (const path of collectSecretPaths(schema)) {
    out = setByDotPath(out, path, undefined)
  }
  return out
}

/** Merge the form's (possibly empty) secret values with the stored ones:
 *  a non-empty typed value wins, an empty/missing value keeps `stored`. */
export function restoreSecrets(
  schema: RJSFSchema,
  formValue: BackendConfig,
  stored: BackendConfig,
): BackendConfig {
  let out = formValue
  for (const path of collectSecretPaths(schema)) {
    const typed = getByDotPath(formValue, path)
    if (typed === undefined || typed === null || typed === "") {
      const current = getByDotPath(stored, path)
      out = setByDotPath(out, path, current)
    }
  }
  return out
}

type Rec = Record<string, unknown>

/** Run rjsf's real default materialization (getDefaultFormState) over a
 *  config — exactly what the controlled <Form> produces as formData on
 *  first render from `value`. Captured as the diff baseline so we know
 *  which keys rjsf added vs what the operator actually typed. */
export function materializedDefaults(
  schema: RJSFSchema,
  value: BackendConfig,
): BackendConfig {
  const schemaUtils = createSchemaUtils(validator, schema)
  return (schemaUtils.getDefaultFormState(schema, value) ?? {}) as BackendConfig
}

/**
 * H2 fix: rjsf's `getDefaultFormState` materializes every schema
 * `default`/`const` into formData (e.g. an untouched `mmproj` becomes
 * `{"source":"hf"}`), so a controlled submit of an UNTOUCHED form
 * would newly write all of that explicitly, change the
 * `config_fingerprint`, and push a pointless `provider.config.update`.
 *
 * Diff rule ("emit only operator-changed keys"), documented:
 *   `initial`   = secret-stripped stored config (the truth to preserve)
 *   `populated` = materializedDefaults(schema, initial) — what the
 *                 form started showing (initial + everything rjsf
 *                 auto-filled)
 *   `current`   = the live form data
 *
 *   For every key in `populated ∪ current`:
 *     - both sides plain objects → recurse; drop the node if the
 *       recursion comes back empty and `initial` had no node there
 *       (no `"server": {}` churn);
 *     - current value deep-equals the populated baseline → the
 *       operator didn't touch it → keep whatever `initial` had
 *       (usually absent — rjsf-materialized defaults never ship);
 *     - current missing where populated had a value → the operator
 *       cleared it → delete from the result;
 *     - otherwise → the operator set/changed it → carry the new value.
 *
 *   Result: an untouched save returns `initial` verbatim (byte-identical
 *   modulo the secret restore, which re-adds the stored secret → equal
 *   to the stored config, no fingerprint churn). Only real edits ship.
 *
 * Known trade-off: if the operator explicitly sets a previously-absent
 * field to exactly what the populated baseline already shows (a schema
 * default), the key is not written — semantically equivalent, since
 * provider/README.md requires schema defaults to mirror the upstream
 * engine defaults.
 */
export function pruneUntouchedDefaults(
  initial: BackendConfig,
  populated: BackendConfig,
  current: BackendConfig,
): BackendConfig {
  const out: Rec = { ...initial }
  const keys = new Set([...Object.keys(populated), ...Object.keys(current)])
  for (const key of keys) {
    const pVal = populated[key]
    const cVal = current[key]
    const pObj = asRecord(pVal)
    const cObj = asRecord(cVal)
    if (pObj && cObj) {
      const merged = pruneUntouchedDefaults(
        asRecord(out[key]) ?? {},
        pObj,
        cObj,
      )
      // An empty section after pruning means the operator cleared every
      // leaf in it — drop the node entirely (never ship `{"server": {}}`
      // churn; a required section would be rejected server-side anyway).
      if (Object.keys(merged).length === 0) {
        delete out[key]
      } else {
        out[key] = merged
      }
      continue
    }
    if (deepEquals(cVal, pVal)) {
      // Untouched relative to the baseline — keep initial's state.
      continue
    }
    if (cVal === undefined) {
      delete out[key]
      continue
    }
    out[key] = cVal
  }
  return out
}

/** Convert the admin's 422 detail
 *  `{error: "backend_config_schema_validation_failed", errors: [{path:
 *  "$.context.ctx", message}]}` into an rjsf ErrorSchema keyed by field
 *  path. Non-schema 422s (plain string detail) go to the form root.
 *
 *  M2: jsonschema reports missing-required errors on the PARENT object
 *  (`$.artifacts` + "'model' is a required property"). When a `schema`
 *  is provided, the offending child property name is parsed out of the
 *  message and appended to the path so the specific field highlights
 *  instead of a bare object-level note. */
export function serverErrorsToErrorSchema(
  detail: unknown,
  schema?: RJSFSchema | null,
): ErrorSchema<BackendConfig> | undefined {
  const builder = new ErrorSchemaBuilder<BackendConfig>()
  let added = false
  const d = detail as Record<string, unknown> | null
  if (d && Array.isArray(d.errors)) {
    for (const e of d.errors as Array<{ path?: string; message?: string }>) {
      const raw = String(e.path ?? "$")
      const message = String(e.message ?? "invalid value")
      // "$.a.b[0].c" -> ["a", "b", 0, "c"]
      const segments: (string | number)[] = raw
        .replace(/^\$\.?/, "")
        .split(".")
        .filter((s) => s !== "")
        .flatMap((s) => {
          const m = /^([^[\]]+)((?:\[\d+\])*)$/.exec(s)
          if (!m) return [s]
          const idxs = [...m[2].matchAll(/\[(\d+)\]/g)].map((x) => Number(x[1]))
          return [m[1], ...idxs]
        })
      // Required-property error on an object node → retarget the child
      // property so its field renders the error inline.
      const reqMatch = /^'([^']+)' is a required property$/.exec(message)
      if (reqMatch && schema) {
        // Walk the schema property tree along the error path.
        let node: Rec | undefined = asRecord(schema.properties)
        for (const seg of segments) {
          node = asRecord(node?.[String(seg)])
          if (!node) break
        }
        const childProps = asRecord(node?.properties)
        if (childProps && reqMatch[1] in childProps) {
          segments.push(reqMatch[1])
        }
      }
      builder.addErrors(message, segments)
      added = true
    }
  } else if (typeof detail === "string" && detail) {
    builder.addErrors(detail)
    added = true
  }
  return added ? builder.ErrorSchema : undefined
}

export interface SchemaFormProps {
  schema: RJSFSchema
  /** The (already secret-stripped) backend_config to render. */
  value: BackendConfig
  onChange: (next: BackendConfig) => void
  /** Server-side per-field errors surfaced into the form. */
  extraErrors?: ErrorSchema<BackendConfig>
  disabled?: boolean
}

/**
 * Renders the sectioned backend_config form. The rjsf Form uses
 * `tagName="div"` so it can live inside the surrounding
 * react-hook-form <form> without nested-form HTML violations; the
 * parent's submit button reads the controlled `value`.
 *
 * M3 note: `extraErrorsBlockSubmit` was removed. In the installed
 * @rjsf/core 6.11.0 the prop DOES exist (Form.js gates its own
 * onSubmit on it), but it is irrelevant here: this form renders with
 * `tagName="div"`, rjsf's own submit never fires, and the parent
 * react-hook-form submit path owns blocking (server 422s are surfaced
 * via `extraErrors` for display and re-submission is the operator's
 * choice). Dropping it avoids depending on rjsf's internal submit
 * semantics.
 */
export function SchemaForm({
  schema,
  value,
  onChange,
  extraErrors,
  disabled,
}: SchemaFormProps) {
  const uiSchema = useMemo(() => buildUiSchema(schema), [schema])
  return (
    <Form<BackendConfig, RJSFSchema>
      schema={schema}
      uiSchema={uiSchema}
      validator={validator}
      formData={value}
      disabled={disabled}
      liveValidate
      showErrorList="top"
      extraErrors={extraErrors}
      tagName="div"
      onChange={(e) => onChange((e.formData ?? {}) as BackendConfig)}
      templates={{
        FieldTemplate: HintFieldTemplate,
        ObjectFieldTemplate: SectionedObjectFieldTemplate,
        // M2: root/object-level errors (plain 422 details, required
        // errors that can't be retargeted to a child) are invisible
        // with showErrorList={false} — render them as a compact alert.
        // Leaf-field errors are already shown inline by
        // HintFieldTemplate; the list keeps them too as a summary with
        // their property path so nothing is hidden.
        ErrorListTemplate: ({ errors }: { errors: unknown[] }) =>
          errors.length === 0 ? null : (
            <div className="mb-2 rounded-md border border-destructive/40 bg-destructive/5 px-3 py-2">
              <p className="mb-1 text-xs font-semibold text-destructive">
                Validation errors
              </p>
              <ul className="space-y-0.5">
                {errors.map((raw, i) => {
                  const e = (raw ?? {}) as {
                    property?: string
                    message?: string
                  }
                  return (
                    <li key={i} className="text-xs text-destructive/90">
                      {e.property && e.property !== "." ? (
                        <code className="mr-1 rounded bg-muted px-1 font-mono text-[10px]">
                          {e.property}
                        </code>
                      ) : null}
                      {e.message}
                    </li>
                  )
                })}
              </ul>
            </div>
          ),
        MultiSchemaFieldTemplate: ({ selector, optionSchemaField }) => (
          <div className="flex flex-col gap-1">
            <div className="max-w-xs">{selector}</div>
            {optionSchemaField}
          </div>
        ),
        ArrayFieldItemTemplate: ({ children, buttons, className }) => (
          <div className={`flex items-center gap-2 ${className ?? ""}`}>
            <div className="min-w-0 flex-1">{children}</div>
            <div className="flex shrink-0 items-center gap-1">{buttons}</div>
          </div>
        ),
        TitleFieldTemplate: () => null,
        FieldHelpTemplate: () => null,
        UnsupportedFieldTemplate: ({ schema: s }) => (
          <p className="text-xs text-destructive">
            Unsupported schema: {JSON.stringify(s).slice(0, 120)}
          </p>
        ),
      }}
      fields={{
        hfFile: HfFileField as never,
      }}
      widgets={{
        secret: SecretWidget as never,
        text: ThemedInputWidget as never,
        select: ThemedSelectWidget as never,
        checkbox: ThemedCheckboxWidget as never,
        textarea: ThemedTextareaWidget as never,
        updown: ThemedInputWidget as never,
      }}
    />
  )
}
