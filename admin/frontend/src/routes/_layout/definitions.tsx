import { zodResolver } from "@hookform/resolvers/zod"
import type { RJSFSchema } from "@rjsf/utils"
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query"
import { createFileRoute, Link } from "@tanstack/react-router"
import {
  AlertTriangle,
  ChevronDown,
  ChevronRight,
  Code2,
  FormInput,
  Pencil,
  Plus,
  Trash2,
} from "lucide-react"
import { useEffect, useMemo, useState } from "react"
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
import { Checkbox } from "@/components/ui/checkbox"
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
import {
  Sheet,
  SheetContent,
  SheetDescription,
  SheetFooter,
  SheetHeader,
  SheetTitle,
} from "@/components/ui/sheet"
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
  useAgents,
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
  // Phase 16: provider_type is required (the Phase 14 shell is retired).
  provider_type: z.string().min(1, "provider_type is required").max(64),
  // Phase 18: endpoint kind this definition serves. Gated by the chosen
  // provider type's serves_modalities in the dialog.
  modality: z.enum(["llm", "embedding"]),
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
  enabled: z.boolean(),
  // Phase 16: placement. 'specific' requires ≥1 agent — enforced in the submit
  // handler (and server-side) rather than on the object schema, so the
  // zodResolver generic stays a plain ZodObject.
  agent_placement: z.enum(["any_of_type", "specific"]),
  agents: z.array(z.string()),
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
                <TableHead>Modality</TableHead>
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

      <DefinitionFormSheet
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
          <StatusBadge status={d.provider_type} />
        </TableCell>
        <TableCell>
          <Badge
            variant={d.modality === "embedding" ? "secondary" : "outline"}
            className="capitalize"
          >
            {d.modality ?? "llm"}
          </Badge>
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
          <TableCell colSpan={9} className="py-4">
            <DefinitionDetail definition={d} />
          </TableCell>
        </TableRow>
      )}
    </>
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
  const placementAgents = d.agents ?? []
  return (
    <div className="grid gap-6 md:grid-cols-2">
      <div>
        <h4 className="mb-2 text-sm font-semibold">Config fingerprint</h4>
        <code className="rounded bg-muted px-2 py-1 font-mono text-xs">
          {d.config_fingerprint ? `${d.config_fingerprint.slice(0, 16)}…` : "—"}
        </code>
      </div>
      <div>
        <h4 className="mb-2 text-sm font-semibold">Modality</h4>
        <p className="text-xs text-muted-foreground">
          <span className="font-mono">{d.modality ?? "llm"}</span> —{" "}
          {d.modality === "embedding"
            ? "served by POST /v1/embeddings."
            : "served by /v1/responses and /v1/chat/completions."}
        </p>
      </div>
      <div>
        <h4 className="mb-2 text-sm font-semibold">Placement</h4>
        <p className="text-xs text-muted-foreground">
          {d.agent_placement === "specific" ? (
            <>
              <span className="font-mono">specific</span> — hosted on{" "}
              {placementAgents.length} cherry-picked agent
              {placementAgents.length === 1 ? "" : "s"}
            </>
          ) : (
            <>
              <span className="font-mono">any_of_type</span> — hosted on every{" "}
              {d.provider_type} agent
            </>
          )}
        </p>
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
        <p className="mb-1 text-xs text-muted-foreground">
          x-secret fields are hidden (write-only — edit via the schema form;
          leaving the input blank keeps the stored value).
        </p>
        <pre className="max-h-64 overflow-auto rounded-md bg-muted p-3 font-mono text-xs">
          {JSON.stringify(displayConfig, null, 2)}
        </pre>
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
                  <StatusBadge status={i.agent_status} />
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
function DefinitionFormSheet({
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
      modality: definition?.modality === "embedding" ? "embedding" : "llm",
      capacity: String(definition?.capacity ?? 1),
      vram_required_bytes: String(definition?.vram_required_bytes ?? 0),
      idle_timeout_seconds: String(definition?.idle_timeout_seconds ?? 300),
      enabled: definition?.enabled ?? true,
      // Phase 16: placement.
      agent_placement:
        definition?.agent_placement === "specific" ? "specific" : "any_of_type",
      agents: definition?.agents ?? [],
    },
  })

  const providerType = form.watch("provider_type")
  const modality = form.watch("modality")
  const placement = form.watch("agent_placement")

  // Phase 16: agents available to cherry-pick for 'specific' placement,
  // filtered to the definition's provider_type (the backend enforces the
  // type match too — 422 on a mismatch). Poll only while the dialog is open.
  const { data: allAgents = [] } = useAgents(4000, open)
  const eligibleAgents = allAgents.filter(
    (a) => a.provider_type === providerType,
  )

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

  // Phase 18: which modalities the selected provider type can host. Gated by
  // its serves_modalities (intersected with the known enum); while the detail
  // is loading or declares none, fall back to just ["llm"].
  const servedModalities = useMemo<string[]>(() => {
    const known = ["llm", "embedding"]
    const raw = (typeDetail?.serves_modalities as string[] | undefined) ?? []
    const list = raw.filter((m) => known.includes(m))
    return list.length > 0 ? list : ["llm"]
  }, [typeDetail])

  // When the chosen type doesn't serve the current modality (e.g. after a
  // provider_type switch), reset to a served value — mirrors the
  // backend_config reset on a type change.
  useEffect(() => {
    if (!servedModalities.includes(modality)) {
      form.setValue("modality", servedModalities[0] as "llm" | "embedding")
    }
  }, [servedModalities, modality, form])

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
      let backendConfig: Record<string, unknown>
      if (effectiveRawMode) {
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
      // Phase 16: placement. 'specific' sends the cherry-picked agent ids;
      // 'any_of_type' sends an empty list (the backend clears the links).
      const agents = values.agent_placement === "specific" ? values.agents : []
      const body = {
        alias: values.alias,
        provider_type: values.provider_type,
        modality: values.modality,
        backend_config: backendConfig,
        capacity: Number(values.capacity),
        vram_required_bytes: Number(values.vram_required_bytes),
        idle_timeout_seconds: Number(values.idle_timeout_seconds),
        enabled: values.enabled,
        agent_placement: values.agent_placement,
        agents,
      }
      setExtraErrors(undefined)
      if (isEdit && definition) {
        const patch: DefinitionPatch = { ...body }
        return await AdminService.patchDefinition({
          path: { definition_id: definition.id },
          body: patch,
        })
      }
      const createBody: DefinitionCreate = { ...body }
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
    <Sheet
      open={open}
      onOpenChange={(o) => {
        if (!o) onClose()
      }}
    >
      <SheetContent className="flex w-full flex-col gap-0 p-0 sm:max-w-3xl">
        <SheetHeader className="border-b">
          <SheetTitle>
            {isEdit ? `Edit ${definition?.alias}` : "Add definition"}
          </SheetTitle>
          <SheetDescription>
            {isEdit
              ? "Changing backend_config or capacity pushes provider.config.update to every connected instance (results shown after save)."
              : "Agents of the chosen provider type host this definition according to its placement."}
          </SheetDescription>
        </SheetHeader>
        <Form {...form}>
          <form
            onSubmit={form.handleSubmit((v) => {
              // 'specific' placement needs at least one agent (server also
              // 422s an empty list).
              if (v.agent_placement === "specific" && v.agents.length === 0) {
                form.setError("agents", {
                  message: "select at least one agent for 'specific' placement",
                })
                return
              }
              form.clearErrors("agents")
              mutation.mutate(v)
            })}
            className="flex min-h-0 flex-1 flex-col"
          >
            <div className="min-h-0 flex-1 space-y-4 overflow-y-auto p-4">
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
                        onValueChange={(v) => {
                          field.onChange(v)
                          // Agents are type-specific: a retype invalidates any
                          // cherry-picked selection.
                          form.setValue("agents", [])
                        }}
                        value={field.value}
                        disabled={
                          isEdit && (definition?.instances?.length ?? 0) > 0
                        }
                      >
                        <FormControl>
                          <SelectTrigger>
                            <SelectValue placeholder="Select type" />
                          </SelectTrigger>
                        </FormControl>
                        <SelectContent>
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
                          : "The registered type whose schema renders backend_config and whose agents can host this model."}
                      </FormDescription>
                      <FormMessage />
                    </FormItem>
                  )}
                />
                <FormField
                  control={form.control}
                  name="modality"
                  render={({ field }) => (
                    <FormItem>
                      <FormLabel>Modality</FormLabel>
                      <Select
                        onValueChange={(v) => field.onChange(v)}
                        value={field.value}
                        disabled={
                          isEdit && (definition?.instances?.length ?? 0) > 0
                        }
                      >
                        <FormControl>
                          <SelectTrigger>
                            <SelectValue placeholder="Select modality" />
                          </SelectTrigger>
                        </FormControl>
                        <SelectContent>
                          {servedModalities.map((m) => (
                            <SelectItem
                              key={m}
                              value={m}
                              className="capitalize"
                            >
                              {m}
                            </SelectItem>
                          ))}
                        </SelectContent>
                      </Select>
                      <FormDescription>
                        {isEdit && (definition?.instances?.length ?? 0) > 0
                          ? "Locked: instances are attached (changing the modality would break their binding)."
                          : providerType
                            ? `Endpoint kind this alias serves. "${providerType}" hosts: ${servedModalities.join(", ")}.`
                            : "Pick a provider type first — available modalities depend on it."}
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

              <div className="space-y-3 rounded-lg border p-3">
                <FormLabel className="text-sm font-semibold">
                  Placement
                </FormLabel>
                <FormField
                  control={form.control}
                  name="agent_placement"
                  render={({ field }) => (
                    <FormItem>
                      <FormLabel>Agent placement</FormLabel>
                      <Select
                        onValueChange={field.onChange}
                        value={field.value}
                      >
                        <FormControl>
                          <SelectTrigger>
                            <SelectValue />
                          </SelectTrigger>
                        </FormControl>
                        <SelectContent>
                          <SelectItem value="any_of_type">
                            Any agent of this type
                          </SelectItem>
                          <SelectItem value="specific">
                            Specific agents…
                          </SelectItem>
                        </SelectContent>
                      </Select>
                      <FormDescription>
                        {providerType
                          ? `any_of_type hosts on every "${providerType}" agent; specific cherry-picks individual agents.`
                          : "Pick a provider type first — placement lists that type's agents."}
                      </FormDescription>
                      <FormMessage />
                    </FormItem>
                  )}
                />
                {placement === "specific" && (
                  <FormField
                    control={form.control}
                    name="agents"
                    render={({ field }) => (
                      <FormItem>
                        <FormLabel>
                          Host on{" "}
                          <span className="font-mono">
                            {providerType || "—"}
                          </span>{" "}
                          agents
                        </FormLabel>
                        {eligibleAgents.length === 0 ? (
                          <p className="rounded-md border border-dashed p-3 text-xs text-muted-foreground">
                            No {providerType || "—"} agents registered yet.
                            Start a provider container of this type, then pick
                            it here.
                          </p>
                        ) : (
                          <div className="max-h-48 space-y-1 overflow-auto rounded-md border p-2">
                            {eligibleAgents.map((a) => {
                              const checked = field.value.includes(a.id)
                              const cbId = `def-agent-${a.id}`
                              return (
                                <div
                                  key={a.id}
                                  className="flex items-center gap-2 rounded px-1 py-1 text-sm hover:bg-accent"
                                >
                                  <Checkbox
                                    id={cbId}
                                    checked={checked}
                                    onCheckedChange={(c) =>
                                      field.onChange(
                                        c
                                          ? [...field.value, a.id]
                                          : field.value.filter(
                                              (x) => x !== a.id,
                                            ),
                                      )
                                    }
                                  />
                                  <label
                                    htmlFor={cbId}
                                    className="flex flex-1 cursor-pointer items-center gap-2"
                                  >
                                    <span className="font-mono text-xs">
                                      {a.agent_id}
                                    </span>
                                    <span className="text-xs text-muted-foreground">
                                      @ {a.machine_uid ?? "—"} · :{a.base_port}
                                    </span>
                                  </label>
                                  <StatusBadge status={a.agent_status} />
                                  {!a.websocket_connected && (
                                    <span className="text-xs text-muted-foreground">
                                      (ws down)
                                    </span>
                                  )}
                                </div>
                              )
                            })}
                          </div>
                        )}
                        <FormMessage />
                      </FormItem>
                    )}
                  />
                )}
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
                    Select a provider type to load its schema and render
                    backend_config.
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
            </div>
            <SheetFooter className="flex-row justify-end border-t">
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
            </SheetFooter>
          </form>
        </Form>
      </SheetContent>
    </Sheet>
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
