// Custom Tailwind/Radix rjsf widgets (Phase 12 E2) so the schema form
// matches the existing admin UI. Registered in SchemaForm's `widgets` /
// `fields` overrides.

import type { WidgetProps } from "@rjsf/utils"
import { asNumber, getDecimalSeparator } from "@rjsf/utils"

import { Checkbox } from "@/components/ui/checkbox"
import { Input } from "@/components/ui/input"
import { PasswordInput } from "@/components/ui/password-input"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"
import { Textarea } from "@/components/ui/textarea"

import { HfFileField } from "./HfFileWidget"

const NUMERIC_TYPES = new Set(["number", "integer"])

/** Text/number/integer input styled with the app's Radix <Input>. */
export function ThemedInputWidget(props: WidgetProps) {
  const {
    id,
    value,
    disabled,
    readonly,
    placeholder,
    rawErrors,
    schema,
    onChange,
    onBlur,
    onFocus,
    autofocus,
  } = props
  const type = (schema.type as string) ?? "string"
  const isNumeric = NUMERIC_TYPES.has(type)
  const separator = getDecimalSeparator()
  let display: string
  if (isNumeric) {
    display =
      value === undefined || value === null || value === ("" as unknown)
        ? ""
        : String(value).replace(".", separator)
  } else {
    display = value == null ? "" : String(value)
  }

  return (
    <Input
      id={id}
      type={isNumeric ? "number" : "text"}
      step={type === "number" ? "any" : 1}
      min={typeof schema.minimum === "number" ? schema.minimum : undefined}
      max={typeof schema.maximum === "number" ? schema.maximum : undefined}
      placeholder={placeholder}
      disabled={disabled}
      readOnly={readonly}
      autoFocus={autofocus}
      value={display}
      aria-invalid={!!(rawErrors && rawErrors.length > 0)}
      onChange={(e) => {
        const raw = e.target.value
        if (!isNumeric) {
          onChange(raw === "" ? undefined : raw)
          return
        }
        const normalized = raw.replace(separator, ".")
        if (normalized === "" || normalized === "-") {
          onChange(undefined)
          return
        }
        onChange(asNumber(normalized))
      }}
      onBlur={(e) => {
        if (isNumeric && e.target.value !== "") {
          const normalized = e.target.value.replace(separator, ".")
          onBlur(id, asNumber(normalized))
        } else {
          onBlur(id, e.target.value)
        }
      }}
      onFocus={(e) => onFocus(id, e.target.value)}
      className="font-mono text-sm"
    />
  )
}

/** Boolean → Radix switch-style checkbox (kept as a checkbox to match
 *  the density of the form; the row label comes from FieldTemplate). */
export function ThemedCheckboxWidget(props: WidgetProps) {
  const { id, value, disabled, readonly, onChange, onBlur } = props
  return (
    <label
      htmlFor={id}
      className="inline-flex cursor-pointer items-center gap-2"
    >
      <Checkbox
        id={id}
        checked={value === true}
        disabled={disabled || readonly}
        onCheckedChange={(checked) => onChange(checked === true)}
        onBlur={() => onBlur(id, value)}
      />
      <span className="text-xs text-muted-foreground">
        {value === true ? "true" : "false"}
      </span>
    </label>
  )
}

/** Enum (and oneOf-constants) → Radix Select. rjsf passes `options.enumOptions`
 *  with the real values; Radix Select needs string values, so we map by index. */
export function ThemedSelectWidget(props: WidgetProps) {
  const {
    id,
    value,
    disabled,
    readonly,
    placeholder,
    required,
    options,
    onChange,
    onBlur,
  } = props
  const enumOptions = options.enumOptions ?? []
  const stringValue =
    value === undefined || value === null
      ? undefined
      : String(
          enumOptions.findIndex((o) =>
            typeof o.value === "number"
              ? o.value === Number(value)
              : o.value === value,
          ),
        )
  const hasValue = stringValue !== undefined && stringValue !== "-1"
  return (
    <Select
      value={hasValue ? stringValue : undefined}
      disabled={disabled || readonly}
      onValueChange={(idx) => {
        const opt = enumOptions[Number(idx)]
        onChange(opt ? opt.value : undefined)
      }}
      onOpenChange={(open) => {
        if (!open) onBlur(id, value)
      }}
    >
      <SelectTrigger id={id} className="w-full max-w-xs">
        <SelectValue
          placeholder={placeholder ?? (required ? "select…" : "(unset)")}
        />
      </SelectTrigger>
      <SelectContent>
        {enumOptions.map((opt, i) => (
          <SelectItem key={`${String(opt.value)}-${i}`} value={String(i)}>
            <span className="font-mono text-sm">{opt.label}</span>
          </SelectItem>
        ))}
      </SelectContent>
    </Select>
  )
}

/** Multi-line string (ui:widget textarea — used by the raw-JSON hatch
 *  and any future long-text schema field). */
export function ThemedTextareaWidget(props: WidgetProps) {
  const { id, value, disabled, readonly, placeholder, onChange, onBlur, rows } =
    props
  return (
    <Textarea
      id={id}
      value={value == null ? "" : String(value)}
      disabled={disabled}
      readOnly={readonly}
      placeholder={placeholder}
      rows={(rows as number) ?? 8}
      className="font-mono text-xs"
      onChange={(e) => onChange(e.target.value)}
      onBlur={(e) => onBlur(id, e.target.value)}
    />
  )
}

/** x-secret widget: password-style input. The committed config never
 *  round-trips the stored value in the form — an empty input means
 *  "unchanged" and the field is dropped from formData by SchemaForm's
 *  secret scrubbing (see scrubSecrets). */
export function SecretWidget(props: WidgetProps) {
  const { id, value, disabled, readonly, placeholder, onChange, onBlur } = props
  return (
    <PasswordInput
      id={id}
      value={value == null ? "" : String(value)}
      disabled={disabled}
      readOnly={readonly}
      placeholder={placeholder ?? "•••••••• (write-only — empty keeps current)"}
      className="font-mono text-sm"
      onChange={(e) =>
        onChange(e.target.value === "" ? undefined : e.target.value)
      }
      onBlur={(e) => onBlur(id, e.target.value)}
    />
  )
}

export { HfFileField }
