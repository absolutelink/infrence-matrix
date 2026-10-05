import { zodResolver } from "@hookform/resolvers/zod"
import { useMutation, useQueryClient } from "@tanstack/react-query"
import { createFileRoute, Link } from "@tanstack/react-router"
import {
  Check,
  ChevronDown,
  ChevronRight,
  Copy,
  Eye,
  EyeOff,
  Pencil,
  Plus,
  RefreshCw,
  Trash2,
} from "lucide-react"
import { useState } from "react"
import { useForm } from "react-hook-form"
import { z } from "zod"

import type { DefinitionCreate, DefinitionPatch } from "@/client"
import { AdminService } from "@/client"
import { EmptyNudge } from "@/components/Common/EmptyNudge"
import { StatusBadge } from "@/components/Common/StatusBadge"
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
  useDefinitions,
} from "@/hooks/useAdminData"
import useCustomToast from "@/hooks/useCustomToast"
import { extractError } from "@/lib/errors"
import { BACKEND_CONFIG_EXAMPLES } from "@/lib/providerConfigExamples"
import type { ConfigUpdateResult, ProviderDefinition } from "@/types/admin"
import { PROVIDER_TYPES } from "@/types/admin"

export const Route = createFileRoute("/_layout/definitions")({
  component: DefinitionsPage,
  head: () => ({ meta: [{ title: "Definitions - Inference Matrix" }] }),
})

const definitionSchema = z.object({
  alias: z.string().min(1, "alias is required").max(255),
  provider_type: z.enum(PROVIDER_TYPES),
  backend_config_text: z.string().refine((t) => {
    if (!t.trim()) return true
    try {
      const parsed = JSON.parse(t)
      return (
        typeof parsed === "object" && parsed !== null && !Array.isArray(parsed)
      )
    } catch {
      return false
    }
  }, "backend_config must be valid JSON (an object)"),
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
            schedule it.
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
        // Remount per target: useForm defaultValues are read only at first
        // mount, so editing B after A would otherwise show A's values.
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
          {d.config_fingerprint.slice(0, 16)}…
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
        <pre className="max-h-64 overflow-auto rounded-md bg-muted p-3 font-mono text-xs">
          {JSON.stringify(d.backend_config, null, 2)}
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

  const form = useForm<DefinitionFormValues>({
    resolver: zodResolver(definitionSchema),
    defaultValues: {
      alias: definition?.alias ?? "",
      provider_type: (definition?.provider_type as never) ?? "mock",
      backend_config_text: JSON.stringify(
        definition?.backend_config ?? {},
        null,
        2,
      ),
      capacity: String(definition?.capacity ?? 1),
      vram_required_bytes: String(definition?.vram_required_bytes ?? 0),
      idle_timeout_seconds: String(definition?.idle_timeout_seconds ?? 300),
      registration_token: definition?.registration_token ?? "",
      enabled: definition?.enabled ?? true,
    },
  })

  const providerType = form.watch("provider_type")
  const schemaHint =
    BACKEND_CONFIG_EXAMPLES[
      (providerType ?? "mock") as keyof typeof BACKEND_CONFIG_EXAMPLES
    ]

  const mutation = useMutation({
    mutationFn: async (values: DefinitionFormValues) => {
      const backendConfig = values.backend_config_text.trim()
        ? JSON.parse(values.backend_config_text)
        : {}
      const body = {
        alias: values.alias,
        provider_type: values.provider_type,
        backend_config: backendConfig,
        capacity: Number(values.capacity),
        vram_required_bytes: Number(values.vram_required_bytes),
        idle_timeout_seconds: Number(values.idle_timeout_seconds),
        enabled: values.enabled,
      }
      if (isEdit && definition) {
        const patch: DefinitionPatch = { ...body }
        // Only send the token when it changed (empty = keep current).
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
    onError: (err: Error) => showErrorToast(extractError(err)),
  })

  return (
    <Dialog
      open={open}
      onOpenChange={(o) => {
        if (!o) onClose()
      }}
    >
      <DialogContent className="sm:max-w-2xl max-h-[90vh] overflow-y-auto">
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
                      onValueChange={field.onChange}
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
                        {PROVIDER_TYPES.map((t) => (
                          <SelectItem key={t} value={t}>
                            {t}
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                    <FormDescription>
                      {isEdit && (definition?.instances?.length ?? 0) > 0
                        ? "Locked: instances are attached (changing the type would break their binding)."
                        : "Must match the provider container's type."}
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

            <FormField
              control={form.control}
              name="backend_config_text"
              render={({ field }) => (
                <FormItem>
                  <div className="flex items-center justify-between">
                    <FormLabel>backend_config (JSON)</FormLabel>
                    <Button
                      type="button"
                      variant="outline"
                      size="sm"
                      onClick={() => {
                        const ex =
                          BACKEND_CONFIG_EXAMPLES[
                            providerType as keyof typeof BACKEND_CONFIG_EXAMPLES
                          ]
                        if (ex) {
                          field.onChange(JSON.stringify(ex.example, null, 2))
                        }
                      }}
                    >
                      <RefreshCw /> Load {providerType} example
                    </Button>
                  </div>
                  <FormControl>
                    <Textarea
                      {...field}
                      rows={12}
                      spellCheck={false}
                      className="font-mono text-xs"
                      placeholder='{"model": {...}, "args": {...}}'
                    />
                  </FormControl>
                  <FormDescription>{schemaHint?.hint}</FormDescription>
                  <FormMessage />
                </FormItem>
              )}
            />

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
