// Custom Tailwind/Radix rjsf templates (Phase 12 E2).
//
// - SectionedObjectFieldTemplate: each top-level schema section renders as
//   a collapsible fieldset (chevron + `title`, ordered by `x-order`).
//   `artifacts` starts expanded; everything else starts collapsed.
// - HintFieldTemplate: label + description + x-flag tooltip (monospace,
//   "maps to --ngl / HALOGEN_*"), x-numeric-effect: numeric warning,
//   x-supported: false disabled note.
//
// These plug into @rjsf/core via the `templates` registry in SchemaForm.

import type { FieldTemplateProps, ObjectFieldTemplateProps } from "@rjsf/utils"
import { AlertTriangle, ChevronDown, ChevronRight, PlugZap } from "lucide-react"
import { type ReactNode, useState } from "react"

import { Badge } from "@/components/ui/badge"
import { Label } from "@/components/ui/label"
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip"
import { cn } from "@/lib/utils"

import { xFlag, xNotSupported, xNumericEffect, xSecret } from "./keywords"

/** The little monospace badge + tooltip that documents the CLI flag / env
 *  var a schema field maps to (x-flag), per provider/README.md. */
export function FlagHint({ schema }: { schema: unknown }) {
  const flag = xFlag(schema)
  if (!flag) return null
  const isEnv = /^[A-Z0-9_]+$/.test(flag)
  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <code className="rounded bg-muted px-1 py-0.5 font-mono text-[10px] text-muted-foreground hover:bg-muted/80">
          {flag}
        </code>
      </TooltipTrigger>
      <TooltipContent className="max-w-xs font-mono text-xs">
        {isEnv
          ? `maps to env ${flag} (HALOGEN_*/provider env var)`
          : `maps to ${flag} (upstream CLI flag)`}
      </TooltipContent>
    </Tooltip>
  )
}

function NumericEffectBadge({ schema }: { schema: unknown }) {
  const effect = xNumericEffect(schema)
  if (!effect) return null
  if (effect === "bitwise") {
    return (
      <Tooltip>
        <TooltipTrigger asChild>
          <Badge
            variant="outline"
            className="h-4 border-sky-500/40 px-1 text-[9px] text-sky-600 dark:text-sky-400"
          >
            bitwise
          </Badge>
        </TooltipTrigger>
        <TooltipContent className="max-w-xs text-xs">
          bitwise effect — output stays byte-identical (speed/policy only).
        </TooltipContent>
      </Tooltip>
    )
  }
  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <Badge
          variant="outline"
          className="h-4 gap-1 border-amber-500/50 px-1 text-[9px] text-amber-600 dark:text-amber-400"
        >
          <AlertTriangle className="size-3" />
          numeric
        </Badge>
      </TooltipTrigger>
      <TooltipContent className="max-w-xs text-xs">
        numeric effect — changes outputs. Verify parity before rolling out
        fleet-wide.
      </TooltipContent>
    </Tooltip>
  )
}

/**
 * Field chrome shared by every leaf field: label, x-flag tooltip,
 * numeric-effect warning, secret marker, description, errors.
 */
export function HintFieldTemplate(props: FieldTemplateProps) {
  const {
    id,
    label,
    required,
    description,
    rawErrors,
    children,
    hidden,
    schema,
    displayLabel = true,
  } = props
  if (hidden) return <div className="hidden">{children}</div>
  const notWired = xNotSupported(schema)
  const secret = xSecret(schema)
  const hasHeader = displayLabel && label
  return (
    <div className={cn("flex flex-col gap-1 py-1.5", notWired && "opacity-70")}>
      {hasHeader && (
        <div className="flex flex-wrap items-center gap-2">
          <Label
            htmlFor={id}
            className={cn(
              "text-sm font-medium",
              notWired && "text-muted-foreground",
            )}
          >
            {label}
            {required && <span className="ml-0.5 text-destructive">*</span>}
          </Label>
          <FlagHint schema={schema} />
          <NumericEffectBadge schema={schema} />
          {secret && (
            <Badge
              variant="outline"
              className="h-4 px-1 text-[9px] text-violet-600 dark:text-violet-400"
            >
              secret
            </Badge>
          )}
          {notWired && (
            <Badge
              variant="outline"
              className="h-4 gap-1 border-zinc-500/40 px-1 text-[9px] text-zinc-500"
            >
              <PlugZap className="size-3" />
              not wired
            </Badge>
          )}
        </div>
      )}
      <div className={cn(hasHeader && "sm:pl-0")}>{children}</div>
      {description && (
        <p className="text-xs leading-snug text-muted-foreground">
          {description}
        </p>
      )}
      {rawErrors && rawErrors.length > 0 && (
        <p className="text-xs text-destructive">{rawErrors.join("; ")}</p>
      )}
    </div>
  )
}

/**
 * Object template: the root renders as a plain stack; every top-level
 * section renders as a collapsible fieldset labeled by its `title`,
 * ordered by `x-order` (ordering already applied by the uiSchema
 * `ui:order` builder). `artifacts` starts expanded, the rest collapsed.
 */
export function SectionedObjectFieldTemplate(props: ObjectFieldTemplateProps) {
  const { title, description, properties, fieldPathId, schema } = props
  const depth = fieldPathId.path.length
  const isSection = depth >= 1
  const sectionName = String(fieldPathId.path[0] ?? "")
  const isArtifacts = depth === 1 && sectionName === "artifacts"
  const [open, setOpen] = useState(isArtifacts)

  const contents: ReactNode = (
    <>
      {properties.map((p) => (
        <div key={p.name}>{p.content}</div>
      ))}
    </>
  )

  if (!isSection) {
    // Root object: no collapse, just the stacked sections.
    return <div className="flex flex-col gap-1">{contents}</div>
  }

  // Nested objects inside a section (rare) render inline without the
  // collapse chrome to keep the tree two levels deep visually.
  if (depth > 1) {
    return (
      <div className="flex flex-col gap-1">
        {title && (
          <p className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
            {title}
          </p>
        )}
        {contents}
      </div>
    )
  }

  const notWired = xNotSupported(schema)
  return (
    <div className="rounded-lg border">
      <button
        type="button"
        onClick={() => setOpen(!open)}
        className="flex w-full items-center gap-2 px-3 py-2 text-left transition-colors hover:bg-accent/50"
        aria-expanded={open}
      >
        {open ? (
          <ChevronDown className="size-4 shrink-0 text-muted-foreground" />
        ) : (
          <ChevronRight className="size-4 shrink-0 text-muted-foreground" />
        )}
        <span className="text-sm font-semibold">{title}</span>
        {notWired && (
          <Badge
            variant="outline"
            className="h-4 px-1 text-[9px] text-zinc-500"
          >
            not wired
          </Badge>
        )}
        {description && !open && (
          <span className="ml-2 hidden truncate text-xs text-muted-foreground md:block">
            {description}
          </span>
        )}
      </button>
      {open && <div className="border-t px-3 py-2">{contents}</div>}
    </div>
  )
}
