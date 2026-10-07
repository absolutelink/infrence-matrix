import { zodResolver } from "@hookform/resolvers/zod"
import type { RJSFSchema } from "@rjsf/utils"
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query"
import { createFileRoute, Link } from "@tanstack/react-router"
import {
  AlertTriangle,
  Check,
  ChevronDown,
  ChevronRight,
  Code2,
  Copy,
  Eye,
  EyeOff,
  FormInput,
  Pencil,
  Plus,
  Trash2,
} from "lucide-react"
import { useMemo, useState } from "react"
import { useForm } from "react-hook-form"
import { z } from "zod"

import type { DefinitionCreate, DefinitionPatch } from "@/client"
import { AdminService } from "@/client"
import { EmptyNudge } from "@/components/Common/EmptyNudge"
import { StatusBadge } from "@/components/Common/StatusBadge"
import {
  collectSecretPaths,
  getByDotPath,
  setByDotPath,
} from "@/components/schema-form/keywords"
import {
  materializedDefaults,
  pruneUntouchedDefaults,
  restoreSecrets,
  SchemaForm,
  serverErrorsToErrorSchema,
  stripSecrets,
} from "@/components/schema-form/SchemaForm"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog"
import {
  Form,
  FormControl,
  FormDescription,
  FormField,
  FormItem,
  FormLabel,
  FormMessage,
} from "@/components/ui/form"
import { Input } from "@/components/ui/input"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"
import { Switch } from "@/components/ui/switch"
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table"
import { Textarea } from "@/components/ui/textarea"
import {
  definitionKeys,
  instanceKeys,
  providerTypeKeys,
  useDefinitions,
  useProviderType,
} from "@/hooks/useAdminData"
import useCustomToast from "@/hooks/useCustomToast"
import { extractError } from "@/lib/errors"
import type {
  ConfigUpdateResult,
  ProviderDefinition,
  ProviderTypeSummary,
} from "@/types/admin"

export const Route = createFileRoute("/_layout/definitions")({
  component: DefinitionsPage,
  head: () => ({ meta: [{ title: "Definitions - Inference Matrix" }] }),
})

const definitionSchema = z.object({
  alias: z.string().min(1, "alias is required").max(255),
  // Phase 14: empty = shell definition (type adopted at registration,
  // backend_config authored afterwards). Editing an existing shell's type
  // is the operator repair path.
  provider_type: z.string().max(64),
  capacity: z
    .string()
    .refine((t) => Number.isInteger(Number(t)) && Number(t) >= 1, {
      message: "capacity must be an integer ≥ 1",
    }),
  vram_required_bytes: z
    .string()
    .refine((t) => Number.isInteger(Number(t)) && Number(t) >= 0, {
      message: "must be an integer ≥ 0",
    }),
  idle_timeout_seconds: z
    .string()
    .refine((t) => Number.isInteger(Number(t)) && Number(t) >= 0, {
      message: "must be an integer ≥ 0",
    }),
  registration_token: z.string().max(255).optional(),
  enabled: z.boolean(),
})

type DefinitionFormValues = z.infer<typeof definitionSchema>

function summarizeResults(results: ConfigUpdateResult[] | undefined): {
  ok: number
  noop: number
  failed: ConfigUpdateResult[]
} {
  const acc = { ok: 0, noop: 0, failed: [] as ConfigUpdateResult[] }
  for (const r of results ?? []) {
    if (!r.ok) acc.failed.push(r)
    else if (r.noop) acc.noop += 1
    else acc.ok += 1
  }
  return acc
}

function DefinitionsPage() {
  const { data: definitions = [], isLoading } = useDefinitions()
  const [editing, setEditing] = useState<ProviderDefinition | null>(null)
  const [creating, setCreating] = useState(false)
  const [deleting, setDeleting] = useState<ProviderDefinition | null>(null)

  return (
    <div className="flex flex-col gap-6">
      <div className="flex items-start justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold tracking-tight">
            Provider Definitions
          </h1>
          <p className="text-muted-foreground">
            Client-facing model aliases: how to boot a backend and how to
            schedule it. backend_config is rendered from each provider type's
            committed{" "}
            <Link to="/provider-types" className="text-primary hover:underline">
              schema
            </Link>
            .
          </p>
        </div>
        <Button onClick={() => setCreating(true)}>
          <Plus /> Add definition
        </Button>
      </div>

      {isLoading ? (
        <p className="text-muted-foreground">Loading definitions…</p>
      ) : definitions.length === 0 ? (
        <EmptyNudge
          text="No provider definitions"
          actionLabel="Create your first definition"
          onAction={() => setCreating(true)}
        />
      ) : (
        <div className="rounded-md border">
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Alias</TableHead>
                <TableHead>Type</TableHead>
                <TableHead>Enabled</TableHead>
                <TableHead>Capacity</TableHead>
                <TableHead>VRAM req.</TableHead>
                <TableHead>Idle (s)</TableHead>
                <TableHead>Instances</TableHead>
                <TableHead className="text-right">Actions</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {definitions.map((d) => (
                <DefinitionRow
                  key={d.id}
                  definition={d}
                  onEdit={() => setEditing(d)}
                  onDelete={() => setDeleting(d)}
                />
              ))}
            </TableBody>
          </Table>
        </div>
      )}

      <DefinitionFormDialog
        // Remount per target: defaultValues are read only at first mount.
        key={editing?.id ?? (creating ? "create" : "closed")}
        open={creating || editing !== null}
        definition={editing}
        onClose={() => {
          setCreating(false)
          setEditing(null)
        }}
      />
      <DeleteDefinitionDialog
        definition={deleting}
        onClose={() => setDeleting(null)}
      />
    </div>
  )
}

function DefinitionRow({
  definition: d,
  onEdit,
  onDelete,
}: {
  definition: ProviderDefinition
  onEdit: () => void
  onDelete: () => void
}) {
  const [expanded, setExpanded] = useState(false)
  const connected = d.connected_instance_count ?? 0
  const vram =
    d.vram_required_bytes > 0
      ? `${(d.vram_required_bytes / 1024 ** 3).toFixed(1)} GiB`
      : "0"

  return (
    <>
      <TableRow>
        <TableCell className="font-medium">
          <button
            type="button"
            className="mr-2 inline-flex rounded p-0.5 align-middle hover:bg-accent"
            onClick={() => setExpanded(!expanded)}
            aria-label={
              expanded
                ? "Collapse definition detail"
                : "Expand definition detail"
            }
          >
            {expanded ? (
              <ChevronDown className="size-4 text-muted-foreground" />
            ) : (
              <ChevronRight className="size-4 text-muted-foreground" />
            )}
          </button>
          {d.alias}
        </TableCell>
        <TableCell>
          {d.provider_type === null ? (
            <Badge
              variant="outline"
              className="border-violet-500/30 bg-violet-500/15 font-mono text-violet-600 dark:text-violet-400"
            >
              awaiting_config
            </Badge>
          ) : (
            <StatusBadge status={d.provider_type} />
          )}
        </TableCell>
        <TableCell>
          <Badge
            variant={d.enabled ? "default" : "secondary"}
            className="capitalize"
          >
            {d.enabled ? "enabled" : "disabled"}
          </Badge>
        </TableCell>
        <TableCell className="font-mono">{d.capacity}</TableCell>
        <TableCell className="font-mono text-xs">{vram}</TableCell>
        <TableCell className="font-mono text-xs">
          {d.idle_timeout_seconds}
        </TableCell>
        <TableCell>
          {connected}
          <span className="ml-1 text-xs text-muted-foreground">
            connected / {(d.instances ?? []).length}
          </span>
        </TableCell>
        <TableCell className="text-right">
          <div className="inline-flex gap-1">
            <Button
              variant="ghost"
              size="icon-sm"
              onClick={onEdit}
              aria-label={`Edit ${d.alias}`}
            >
              <Pencil />
            </Button>
            <Button
              variant="ghost"
              size="icon-sm"
              className="text-destructive hover:bg-destructive/10"
              onClick={onDelete}
              aria-label={`Delete ${d.alias}`}
            >
              <Trash2 />
            </Button>
          </div>
        </TableCell>
      </TableRow>
      {expanded && (
        <TableRow className="bg-muted/40 hover:bg-muted/40">
          <TableCell colSpan={8} className="py-4">
            <DefinitionDetail definition={d} />
          </TableCell>
        </TableRow>
      )}
    </>
  )
}

function CopyButton({ value, label }: { value: string; label?: string }) {
  const [copied, setCopied] = useState(false)
  const [revealed, setRevealed] = useState(false)
  const display = revealed ? value : "•".repeat(Math.min(value.length, 28))
  return (
    <div className="flex items-center gap-2">
      <code className="rounded bg-muted px-2 py-1 font-mono text-xs break-all">
        {display}
      </code>
      <Button
        variant="ghost"
        size="icon-sm"
        onClick={() => setRevealed(!revealed)}
        aria-label={revealed ? "Hide" : "Reveal"}
      >
        {revealed ? <EyeOff /> : <Eye />}
      </Button>
      <Button
        variant="ghost"
        size="icon-sm"
        onClick={async () => {
          try {
            await navigator.clipboard.writeText(value)
            setCopied(true)
            setTimeout(() => setCopied(false), 2000)
          } catch {
            /* clipboard unavailable */
          }
        }}
        aria-label={label ?? "Copy to clipboard"}
      >
        {copied ? <Check className="text-emerald-500" /> : <Copy />}
      </Button>
    </div>
  )
}

function DefinitionDetail({
  definition: d,
}: {
  definition: ProviderDefinition
}) {
  const discovered = (d.model_metadata?.models ?? []) as unknown[]
  // H1: the admin API returns unredacted backend_config — strip every
  // x-secret leaf before rendering so stored secrets never hit the DOM.
  const { data: typeDetail } = useProviderType(d.provider_type)
  const displayConfig = useMemo(() => {
    if (d.backend_config === null) return null
    const schema = (typeDetail?.schema ?? null) as RJSFSchema | null
    if (!schema) return d.backend_config
    const stripped = stripSecrets(schema, d.backend_config)
    const secretPaths = collectSecretPaths(schema)
    if (secretPaths.length === 0) return stripped
    // Replace stripped leaves with a marker so the operator knows a
    // secret is stored there (the path exists, the value doesn't).
    let out: Record<string, unknown> = stripped
    for (const p of secretPaths) {
      if (getByDotPath(d.backend_config, p) !== undefined) {
        out = setByDotPath(out, p, "•••••• (secret — hidden)")
      }
    }
    return out
  }, [typeDetail, d.backend_config])
  return (
    <div className="grid gap-6 md:grid-cols-2">
      <div>
        <h4 className="mb-2 text-sm font-semibold">Registration token</h4>
        <p className="mb-1 text-xs text-muted-foreground">
          Set as PROVIDER_REGISTRATION_TOKEN in the provider container env.
        </p>
        <CopyButton value={d.registration_token} />
        <h4 className="mt-4 mb-2 text-sm font-semibold">Config fingerprint</h4>
        <code className="rounded bg-muted px-2 py-1 font-mono text-xs">
          {d.config_fingerprint === null
            ? "null — no config authored yet"
            : `${d.config_fingerprint.slice(0, 16)}…`}
        </code>
      </div>
      <div>
        <h4 className="mb-2 text-sm font-semibold">
          Discovered model metadata
        </h4>
        {discovered.length === 0 ? (
          <p className="text-xs text-muted-foreground">
            Nothing discovered yet — populated after the backend starts or a
            config update.
          </p>
        ) : (
          <pre className="max-h-48 overflow-auto rounded-md bg-muted p-3 font-mono text-xs">
            {JSON.stringify(d.model_metadata, null, 2)}
          </pre>
        )}
      </div>
      <div className="md:col-span-2">
        <h4 className="mb-2 text-sm font-semibold">backend_config</h4>
        {displayConfig === null ? (
          <p className="rounded-md border border-dashed border-violet-500/40 p-3 text-xs text-muted-foreground">
            No backend_config yet — this shell definition is waiting for its
            provider to register (which adopts the type). Then edit this
            definition to author the config against the committed schema; saving
            pushes it to the connected instance.
          </p>
        ) : (
          <>
            <p className="mb-1 text-xs text-muted-foreground">
              x-secret fields are hidden (write-only — edit via the schema form;
              leaving the input blank keeps the stored value).
            </p>
            <pre className="max-h-64 overflow-auto rounded-md bg-muted p-3 font-mono text-xs">
              {JSON.stringify(displayConfig, null, 2)}
            </pre>
          </>
        )}
        {(d.instances ?? []).length > 0 && (
          <>
            <h4 className="mt-4 mb-2 text-sm font-semibold">Instances</h4>
            <ul className="space-y-1 text-xs">
              {(d.instances ?? []).map((i) => (
                <li key={i.id} className="flex items-center gap-2">
                  <Link
                    to="/instances"
                    className="font-mono text-primary hover:underline"
                  >
                    {i.machine_uid ?? i.id.slice(0, 8)}
                  </Link>
                  <StatusBadge status={i.instance_status} />
                  <StatusBadge status={i.backend_status} />
                  <span className="font-mono text-muted-foreground">
                    :{i.port} · v{i.version}
                  </span>
                </li>
              ))}
            </ul>
          </>
        )}
      </div>
    </div>
  )
}

/**
 * The definition editor. The backend_config section is now a
 * schema-driven rjsf form (Phase 12) fetched from
 * AdminService.getProviderType for the selected provider type, with a
 * raw-JSON escape hatch. All other fields keep their previous shape.
 */
function DefinitionFormDialog({
  open,
  definition,
  onClose,
}: {
  open: boolean
  definition: ProviderDefinition | null
  onClose: () => void
}) {
  const queryClient = useQueryClient()
  const { showErrorToast, showSuccessToast } = useCustomToast()
  const isEdit = definition !== null
  const storedConfig = (definition?.backend_config ?? {}) as Record<
    string,
    unknown
  >

  const form = useForm<DefinitionFormValues>({
    resolver: zodResolver(definitionSchema),
    defaultValues: {
      alias: definition?.alias ?? "",
      provider_type: definition?.provider_type ?? "",
      capacity: String(definition?.capacity ?? 1),
      vram_required_bytes: String(definition?.vram_required_bytes ?? 0),
      idle_timeout_seconds: String(definition?.idle_timeout_seconds ?? 300),
      registration_token: definition?.registration_token ?? "",
      enabled: definition?.enabled ?? true,
    },
  })

  const providerType = form.watch("provider_type")

  const { data: providerTypes = [] } = useQuery({
    queryKey: providerTypeKeys.all,
    queryFn: async () =>
      ((await AdminService.listProviderTypes()).data ??
        []) as unknown as ProviderTypeSummary[],
    enabled: open,
    staleTime: 30_000,
  })

  const { data: typeDetail, isLoading: schemaLoading } = useQuery({
    queryKey: providerTypeKeys.detail(providerType ?? ""),
    queryFn: async () =>
      (
        await AdminService.getProviderType({
          path: { name: providerType as string },
        })
      ).data as Record<string, unknown>,
    enabled: open && providerType !== "" && providerType !== undefined,
    staleTime: 30_000,
  })

  const schema = (typeDetail?.schema ?? null) as RJSFSchema | null
  const schemaHasFields =
    schema !== null &&
    typeof schema === "object" &&
    Object.keys((schema as Record<string, unknown>).properties ?? {}).length > 0

  // Controlled backend_config editor state: `rawMode` toggles between
  // the schema form and the raw-JSON escape hatch.
  //
  // H1: every display surface (form data AND the raw textarea) is
  // seeded from the SECRET-STRIPPED stored config — the admin API
  // returns unredacted values, so the UI must never render them.
  // C1: the raw-mode submit path also runs restoreSecrets() so an
  // edit→toggle-raw→save cycle can't drop a stored secret.
  // H2: the form keeps `initialConfig` (the stripped stored config) as
  // the diff base so rjsf-materialized schema defaults are pruned on
  // submit (see pruneUntouchedDefaults) — an untouched save is
  // byte-identical to what was stored.
  const strippedInitial = useMemo<Record<string, unknown>>(
    () => (schema ? stripSecrets(schema, storedConfig) : storedConfig),
    [schema, storedConfig],
  )
  const [rawMode, setRawMode] = useState(false)
  const [configValue, setConfigValue] =
    useState<Record<string, unknown>>(strippedInitial)
  const [rawText, setRawText] = useState(() =>
    JSON.stringify(strippedInitial, null, 2),
  )
  const [rawError, setRawError] = useState<string | null>(null)
  const [extraErrors, setExtraErrors] =
    useState<ReturnType<typeof serverErrorsToErrorSchema>>(undefined)
  // Snapshot of the diff base at editor (re)init time; submit prunes
  // against THIS, not the live query result, so mid-edit refetches
  // can't shift the baseline under the operator.
  const [initialConfig, setInitialConfig] =
    useState<Record<string, unknown>>(strippedInitial)
  // Track the type the current form state was initialized for so a
  // provider_type switch resets the editor instead of leaking config.
  const [initType, setInitType] = useState<string>(
    definition?.provider_type ?? "",
  )

  // Re-initialize the editor when the type (or its schema) changes.
  if (providerType !== initType) {
    setInitType(providerType)
    const next = schema ? stripSecrets(schema, storedConfig) : storedConfig
    setConfigValue(next)
    setInitialConfig(next)
    setRawText(JSON.stringify(next, null, 2))
    setExtraErrors(undefined)
    setRawError(null)
  }
  // A schema that just arrived (async) re-strips the secrets.
  const [initSchemaFp, setInitSchemaFp] = useState<string | null>(
    (typeDetail?.schema_fingerprint as string) ?? null,
  )
  const nowFp = (typeDetail?.schema_fingerprint as string) ?? null
  if (nowFp !== initSchemaFp) {
    setInitSchemaFp(nowFp)
    if (schema) {
      const next = stripSecrets(schema, storedConfig)
      setConfigValue(next)
      setInitialConfig(next)
      setRawText(JSON.stringify(next, null, 2))
      setExtraErrors(undefined)
    }
  }

  // Permissive / empty schemas (bootstrap path) can't drive a form —
  // force the raw-JSON hatch.
  const effectiveRawMode = rawMode || !schemaHasFields

  const toggleRawMode = () => {
    if (effectiveRawMode) {
      // raw → form: parse first; invalid JSON keeps the operator in raw.
      try {
        const parsed = JSON.parse(rawText)
        if (typeof parsed !== "object" || parsed === null) throw new Error()
        setConfigValue(
          schema
            ? stripSecrets(schema, parsed as Record<string, unknown>)
            : parsed,
        )
        setRawError(null)
        setRawMode(false)
      } catch {
        setRawError("Invalid JSON — fix it before switching to the form.")
      }
    } else {
      // form → raw: snapshot current form data (never contains stored
      // secrets; a freshly typed secret is the operator's own input).
      setRawText(JSON.stringify(configValue, null, 2))
      setRawMode(true)
    }
  }

  const mutation = useMutation({
    mutationFn: async (values: DefinitionFormValues) => {
      // Phase 14: empty provider type = shell creation. No config is sent
      // (the API refuses backend_config on a shell).
      const shellCreate = !isEdit && values.provider_type === ""
      let backendConfig: Record<string, unknown>
      if (shellCreate) {
        backendConfig = {}
      } else if (effectiveRawMode) {
        try {
          const parsed = JSON.parse(rawText)
          if (
            typeof parsed !== "object" ||
            parsed === null ||
            Array.isArray(parsed)
          ) {
            throw new Error()
          }
          // C1: raw mode shows the secret-STRIPPED config, so merge the
          // stored secrets back for anything the operator didn't
          // explicitly set. An explicit non-empty value in the raw text
          // wins (rotate); a missing/empty one keeps the stored value.
          backendConfig = schema
            ? restoreSecrets(
                schema,
                parsed as Record<string, unknown>,
                storedConfig,
              )
            : (parsed as Record<string, unknown>)
        } catch {
          setRawError("backend_config must be valid JSON (an object).")
          throw new Error("backend_config must be valid JSON (an object)")
        }
      } else {
        setRawError(null)
        // H2: prune everything rjsf materialized (defaults/consts)
        // that the operator didn't actually change, then restore
        // write-only secrets they left empty.
        const pruned = schema
          ? pruneUntouchedDefaults(
              initialConfig,
              materializedDefaults(schema, initialConfig),
              configValue,
              schema,
            )
          : configValue
        backendConfig = schema
          ? restoreSecrets(schema, pruned, storedConfig)
          : pruned
      }
      const body = {
        alias: values.alias,
        provider_type:
          values.provider_type === "" ? null : values.provider_type,
        backend_config: backendConfig,
        capacity: Number(values.capacity),
        vram_required_bytes: Number(values.vram_required_bytes),
        idle_timeout_seconds: Number(values.idle_timeout_seconds),
        enabled: values.enabled,
      }
      setExtraErrors(undefined)
      if (isEdit && definition) {
        const patch: DefinitionPatch = { ...body }
        // Never null-out provider_type/backend_config in a PATCH (the API
        // refuses unset; a shell edit carries the current values).
        if (
          (patch as { provider_type?: string | null }).provider_type === null
        ) {
          delete (patch as { provider_type?: string | null }).provider_type
        }
        // Phase 14 (review F1): editing a SHELL must not silently author
        // a config. A shell's editor seeds from {} (no stored config);
        // unless the operator actually typed one, drop backend_config
        // from the PATCH entirely — sending {} would otherwise commit an
        // empty config (push + schedulable on defaults) on an adopted
        // shell, and 422 on a not-yet-adopted one.
        if (
          definition.backend_config === null &&
          JSON.stringify(backendConfig) === JSON.stringify({})
        ) {
          delete (patch as { backend_config?: unknown }).backend_config
        }
        if ((patch as { backend_config?: unknown }).backend_config === null) {
          delete (patch as { backend_config?: unknown }).backend_config
        }
        if (
          values.registration_token &&
          values.registration_token !== definition.registration_token
        ) {
          patch.registration_token = values.registration_token
        }
        return await AdminService.patchDefinition({
          path: { definition_id: definition.id },
          body: patch,
        })
      }
      const createBody: DefinitionCreate = { ...body }
      // Phase 14: a shell create omits backend_config entirely.
      if (shellCreate) {
        delete (createBody as { backend_config?: unknown }).backend_config
      }
      if (values.registration_token) {
        createBody.registration_token = values.registration_token
      }
      return await AdminService.createDefinition({ body: createBody })
    },
    onSuccess: (resp) => {
      queryClient.invalidateQueries({ queryKey: definitionKeys.all })
      queryClient.invalidateQueries({ queryKey: instanceKeys.all })
      const results = (resp?.data as unknown as ProviderDefinition)
        ?.config_update_results
      const s = summarizeResults(results)
      if (s.failed.length > 0) {
        showErrorToast(
          `Saved, but ${s.failed.length} instance(s) refused the config update: ${s.failed
            .map((f) => `${f.error}${f.step ? `@${f.step}` : ""}`)
            .join("; ")}`,
        )
      } else if (s.ok > 0) {
        showSuccessToast(
          `Definition saved — pushed new config to ${s.ok} connected instance(s).`,
        )
      } else if (s.noop > 0) {
        showSuccessToast(
          `Definition saved — ${s.noop} instance(s) already up to date.`,
        )
      } else {
        showSuccessToast(isEdit ? "Definition updated" : "Definition created")
      }
      onClose()
    },
    onError: (err: Error) => {
      // Surface the admin's 422 schema-validation per-field errors into
      // the form when we can map them.
      const detail = (
        err as {
          response?: { data?: { detail?: unknown } }
        }
      )?.response?.data?.detail
      const es = serverErrorsToErrorSchema(detail, schema)
      if (es) {
        setExtraErrors(es)
        if (
          detail &&
          typeof detail === "object" &&
          (detail as Record<string, unknown>).error ===
            "backend_config_schema_validation_failed"
        ) {
          showErrorToast(
            "backend_config failed schema validation — see field errors.",
          )
          return
        }
      }
      showErrorToast(extractError(err))
    },
  })

  return (
    <Dialog
      open={open}
      onOpenChange={(o) => {
        if (!o) onClose()
      }}
    >
      <DialogContent className="sm:max-w-3xl max-h-[90vh] overflow-y-auto">
        <DialogHeader>
          <DialogTitle>
            {isEdit ? `Edit ${definition?.alias}` : "Add definition"}
          </DialogTitle>
          <DialogDescription>
            {isEdit
              ? "Changing backend_config or capacity pushes provider.config.update to every connected instance (results shown after save)."
              : "A provider container registers against this definition with its registration token."}
          </DialogDescription>
        </DialogHeader>
        <Form {...form}>
          <form
            onSubmit={form.handleSubmit((v) => mutation.mutate(v))}
            className="space-y-4"
          >
            <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
              <FormField
                control={form.control}
                name="alias"
                render={({ field }) => (
                  <FormItem>
                    <FormLabel>Alias</FormLabel>
                    <FormControl>
                      <Input {...field} placeholder="mock-model" />
                    </FormControl>
                    <FormDescription>
                      Public model name used by clients in /v1.
                    </FormDescription>
                    <FormMessage />
                  </FormItem>
                )}
              />
              <FormField
                control={form.control}
                name="provider_type"
                render={({ field }) => (
                  <FormItem>
                    <FormLabel>Provider type</FormLabel>
                    <Select
                      onValueChange={(v) =>
                        field.onChange(v === "SHELL" ? "" : v)
                      }
                      value={field.value}
                      disabled={
                        isEdit &&
                        (definition?.provider_type ?? "") !== "" &&
                        (definition?.instances?.length ?? 0) > 0
                      }
                    >
                      <FormControl>
                        <SelectTrigger>
                          <SelectValue placeholder="Select type (or shell)" />
                        </SelectTrigger>
                      </FormControl>
                      <SelectContent>
                        {!isEdit && (
                          <SelectItem value="SHELL">
                            (shell — type adopted at registration)
                          </SelectItem>
                        )}
                        {providerTypes.map((t) => (
                          <SelectItem key={t.name} value={t.name}>
                            {t.name}
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                    <FormDescription>
                      {isEdit && (definition?.instances?.length ?? 0) > 0
                        ? "Locked: instances are attached (changing the type would break their binding)."
                        : isEdit && (definition?.provider_type ?? "") === ""
                          ? "Shell definition — the type is set by the provider's first registration (repair path)."
                          : "Pick a registered type, or create a shell and let the container adopt its type."}
                    </FormDescription>
                    <FormMessage />
                  </FormItem>
                )}
              />
            </div>

            <div className="grid grid-cols-1 gap-4 sm:grid-cols-3">
              <FormField
                control={form.control}
                name="capacity"
                render={({ field }) => (
                  <FormItem>
                    <FormLabel>Capacity</FormLabel>
                    <FormControl>
                      <Input type="number" min={1} step={1} {...field} />
                    </FormControl>
                    <FormMessage />
                  </FormItem>
                )}
              />
              <FormField
                control={form.control}
                name="vram_required_bytes"
                render={({ field }) => (
                  <FormItem>
                    <FormLabel>VRAM required (bytes)</FormLabel>
                    <FormControl>
                      <Input
                        type="number"
                        min={0}
                        step={1024 ** 3}
                        {...field}
                      />
                    </FormControl>
                    <FormMessage />
                  </FormItem>
                )}
              />
              <FormField
                control={form.control}
                name="idle_timeout_seconds"
                render={({ field }) => (
                  <FormItem>
                    <FormLabel>Idle timeout (s)</FormLabel>
                    <FormControl>
                      <Input type="number" min={0} step={30} {...field} />
                    </FormControl>
                    <FormMessage />
                  </FormItem>
                )}
              />
            </div>

            <div className="space-y-2">
              <div className="flex items-center justify-between gap-2">
                <FormLabel className="text-sm font-semibold">
                  backend_config
                </FormLabel>
                {schemaHasFields && (
                  <Button
                    type="button"
                    variant="outline"
                    size="sm"
                    onClick={toggleRawMode}
                  >
                    {effectiveRawMode ? (
                      <>
                        <FormInput /> Schema form
                      </>
                    ) : (
                      <>
                        <Code2 /> Raw JSON
                      </>
                    )}
                  </Button>
                )}
              </div>
              {!providerType ? (
                <p className="rounded-md border border-dashed p-3 text-xs text-muted-foreground">
                  {isEdit
                    ? "Shell definition — backend_config is authored here after the provider registers and adopts its type."
                    : "Select a provider type to load its schema. Pick “shell” to configure after registration."}
                </p>
              ) : schemaLoading ? (
                <p className="rounded-md border border-dashed p-3 text-xs text-muted-foreground">
                  Loading {providerType} schema…
                </p>
              ) : !schema ? (
                <p className="rounded-md border border-destructive/40 p-3 text-xs text-destructive">
                  Provider type "{providerType}" has no committed schema.
                </p>
              ) : effectiveRawMode ? (
                <>
                  {!schemaHasFields && (
                    <p className="flex items-center gap-1 text-xs text-amber-600 dark:text-amber-500">
                      <AlertTriangle className="size-3.5" />
                      This type's committed schema has no fields (permissive
                      bootstrap) — raw JSON only.
                    </p>
                  )}
                  {schemaHasFields && (
                    <p className="text-xs text-muted-foreground">
                      Secret fields are omitted from this view — edit them via
                      the schema form (leave blank to keep the current value).
                      Saving here keeps stored secrets intact.
                    </p>
                  )}
                  <Textarea
                    value={rawText}
                    rows={12}
                    spellCheck={false}
                    className="font-mono text-xs"
                    onChange={(e) => {
                      setRawText(e.target.value)
                      setRawError(null)
                    }}
                  />
                  {rawError && (
                    <p className="text-xs text-destructive">{rawError}</p>
                  )}
                </>
              ) : (
                <SchemaForm
                  schema={schema}
                  value={configValue}
                  onChange={setConfigValue}
                  extraErrors={extraErrors}
                />
              )}
            </div>

            <FormField
              control={form.control}
              name="registration_token"
              render={({ field }) => (
                <FormItem>
                  <FormLabel>Registration token</FormLabel>
                  <FormControl>
                    <Input
                      {...field}
                      placeholder={
                        isEdit
                          ? "(unchanged — clear and type to rotate)"
                          : "leave empty to auto-generate"
                      }
                    />
                  </FormControl>
                  <FormDescription>
                    {isEdit
                      ? "Keep as-is to retain the current token; the current value is shown on the row detail with copy."
                      : "Auto-generated when empty. Copy it into the provider container env after creating."}
                  </FormDescription>
                  <FormMessage />
                </FormItem>
              )}
            />

            <FormField
              control={form.control}
              name="enabled"
              render={({ field }) => (
                <FormItem className="flex flex-row items-center justify-between rounded-lg border p-3">
                  <div className="space-y-0.5">
                    <FormLabel>Enabled</FormLabel>
                    <FormDescription>
                      Disabled definitions are excluded from scheduling and
                      hidden from /v1/models.
                    </FormDescription>
                  </div>
                  <FormControl>
                    <Switch
                      checked={field.value}
                      onCheckedChange={field.onChange}
                    />
                  </FormControl>
                </FormItem>
              )}
            />

            <DialogFooter>
              <Button type="button" variant="outline" onClick={onClose}>
                Cancel
              </Button>
              <Button type="submit" disabled={mutation.isPending}>
                {mutation.isPending
                  ? "Saving…"
                  : isEdit
                    ? "Save & push"
                    : "Create"}
              </Button>
            </DialogFooter>
          </form>
        </Form>
      </DialogContent>
    </Dialog>
  )
}

function DeleteDefinitionDialog({
  definition,
  onClose,
}: {
  definition: ProviderDefinition | null
  onClose: () => void
}) {
  const queryClient = useQueryClient()
  const { showErrorToast, showSuccessToast } = useCustomToast()

  const mutation = useMutation({
    mutationFn: async () => {
      if (!definition) return
      return await AdminService.deleteDefinition({
        path: { definition_id: definition.id },
      })
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: definitionKeys.all })
      queryClient.invalidateQueries({ queryKey: instanceKeys.all })
      showSuccessToast("Definition deleted")
      onClose()
    },
    onError: (err: Error) => {
      const msg = extractError(err)
      showErrorToast(
        msg.includes("connected")
          ? `${msg} — tip: disable the definition instead of deleting it.`
          : msg,
      )
    },
  })

  const connectedCount = definition?.connected_instance_count ?? 0

  return (
    <Dialog
      open={definition !== null}
      onOpenChange={(o) => {
        if (!o) onClose()
      }}
    >
      <DialogContent className="sm:max-w-md">
        <DialogHeader>
          <DialogTitle>Delete definition</DialogTitle>
          <DialogDescription>
            Delete <span className="font-semibold">{definition?.alias}</span> (
            {definition?.provider_type})? Any offline instance rows are deleted
            with it.
            {connectedCount > 0 && (
              <span className="mt-2 block text-destructive">
                {connectedCount} instance(s) are currently connected — the
                server will refuse this delete. Disable the definition instead,
                or stop the provider containers first.
              </span>
            )}
          </DialogDescription>
        </DialogHeader>
        <DialogFooter>
          <Button type="button" variant="outline" onClick={onClose}>
            Cancel
          </Button>
          <Button
            type="button"
            variant="destructive"
            disabled={mutation.isPending || connectedCount > 0}
            onClick={() => mutation.mutate()}
          >
            {mutation.isPending ? "Deleting…" : "Delete"}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}
