import { useMutation, useQueryClient } from "@tanstack/react-query"
import { createFileRoute, Link } from "@tanstack/react-router"
import {
  Eraser,
  HardDriveDownload,
  MoreVertical,
  Play,
  RefreshCw,
  RotateCw,
  ScrollText,
  Square,
} from "lucide-react"
import { useState } from "react"

import { AdminService } from "@/client"
import { LogsSheet } from "@/components/Common/LogsSheet"
import { ConnectionBadge, StatusBadge } from "@/components/Common/StatusBadge"
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu"
import { Label } from "@/components/ui/label"
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
import {
  definitionKeys,
  instanceKeys,
  useInstances,
  useProviderTypes,
} from "@/hooks/useAdminData"
import useCustomToast from "@/hooks/useCustomToast"
import { extractError } from "@/lib/errors"
import type {
  BackendActionResult,
  ProviderInstance,
  StorageActionResult,
} from "@/types/admin"
import { formatBytes } from "@/utils"

export const Route = createFileRoute("/_layout/instances")({
  component: InstancesPage,
  head: () => ({ meta: [{ title: "Instances - Inference Matrix" }] }),
})

// sha256 of canonical JSON {"type":"object"} — the admin's permissive
// bootstrap schema (Phase 12 transition). An instance reporting this fp
// never proved a real schema; it is not "waiting" on anything.
const PERMISSIVE_SCHEMA_FP =
  "a2c799262a3ce3c19ef5cdd983bf3d12b43ab3c426227091b909dcb7054738c0"

// Backend transitions the operator can drive by hand (provider_lib.ops).
// `stop` has no options; the other three may either accept (the boot runs
// in the background — a cold halogen-flash instance downloads its
// checkpoint and companions first, which is minutes-to-hours) or wait for
// the engine to report running.
type BackendActionKind = "start" | "stop" | "restart" | "initialize"

interface BackendActionTarget {
  kind: BackendActionKind
  instance: ProviderInstance
}

const BACKEND_ACTIONS: Record<
  BackendActionKind,
  { label: string; title: string; waits: boolean; waitLabel: string }
> = {
  start: {
    label: "Start backend",
    title: "Boot the backend now (no inference request needed)",
    waits: true,
    waitLabel: "Wait until the engine reports running",
  },
  stop: {
    label: "Stop backend",
    title: "Unload the backend (refused while requests are live)",
    waits: false,
    waitLabel: "",
  },
  restart: {
    label: "Restart backend",
    title: "Stop and start again — reloads the model/config",
    waits: true,
    waitLabel: "Wait until the engine reports running",
  },
  initialize: {
    label: "Reinitialize",
    title:
      "Re-register with the admin, adopt the definition config, reboot and re-read model metadata",
    waits: true,
    waitLabel: "Wait for the whole re-provision (may download for a long time)",
  },
}

function InstancesPage() {
  const { data: instances = [], isLoading } = useInstances()
  const { data: providerTypes = [] } = useProviderTypes()
  const committedByType = new Map(
    providerTypes.map((t) => [t.name, t.schema_fingerprint]),
  )
  const [cacheTarget, setCacheTarget] = useState<ProviderInstance | null>(null)
  const [pruneTarget, setPruneTarget] = useState<ProviderInstance | null>(null)
  const [logsTarget, setLogsTarget] = useState<ProviderInstance | null>(null)
  const [backendTarget, setBackendTarget] =
    useState<BackendActionTarget | null>(null)

  return (
    <div className="flex flex-col gap-6">
      <div>
        <h1 className="text-2xl font-bold tracking-tight">
          Provider Instances
        </h1>
        <p className="text-muted-foreground">
          One backend per instance. Status mirrors the provider WS events;
          storage actions run over the live socket.
        </p>
      </div>

      {isLoading ? (
        <p className="text-muted-foreground">Loading instances…</p>
      ) : instances.length === 0 ? (
        <div className="flex flex-col items-center justify-center py-12 text-center">
          <div className="rounded-full bg-muted p-4 mb-4">
            <HardDriveDownload className="h-8 w-8 text-muted-foreground" />
          </div>
          <h3 className="text-lg font-semibold">No instances yet</h3>
          <p className="text-muted-foreground">
            Create a machine + definition, then start a provider container
            pointing at the admin — it registers and shows up here.
          </p>
        </div>
      ) : (
        <div className="rounded-md border">
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Machine</TableHead>
                <TableHead>Alias</TableHead>
                <TableHead>Instance</TableHead>
                <TableHead>Backend</TableHead>
                <TableHead>WS</TableHead>
                <TableHead>Version</TableHead>
                <TableHead>Port</TableHead>
                <TableHead>Epoch</TableHead>
                <TableHead>Last seen</TableHead>
                <TableHead>Last request</TableHead>
                <TableHead>Config fp</TableHead>
                <TableHead className="text-right">Actions</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {instances.map((i) => (
                <TableRow key={i.id}>
                  <TableCell className="font-mono text-xs">
                    {i.machine_uid ?? "—"}
                  </TableCell>
                  <TableCell className="font-medium">{i.alias}</TableCell>
                  <TableCell>
                    <div className="flex flex-wrap items-center gap-1">
                      <StatusBadge status={i.agent_status} />
                      {(() => {
                        const committed = i.provider_type
                          ? committedByType.get(i.provider_type)
                          : undefined
                        if (
                          committed &&
                          committed !== PERMISSIVE_SCHEMA_FP &&
                          i.reported_schema_fingerprint !== committed
                        ) {
                          return (
                            <Link
                              to="/provider-types"
                              search={{ type: i.provider_type ?? "" }}
                              title={`Reported schema ${
                                i.reported_schema_fingerprint
                                  ? `${i.reported_schema_fingerprint.slice(0, 8)}…`
                                  : "none"
                              } ≠ committed ${committed.slice(0, 8)}… — registration refused until consensus`}
                            >
                              <StatusBadge
                                status="waiting_schema"
                                className="border-amber-500/40 bg-amber-500/10 text-amber-600 dark:text-amber-400"
                              />
                            </Link>
                          )
                        }
                        return null
                      })()}
                    </div>
                  </TableCell>
                  <TableCell>
                    {/* error_message carries the reason of the last status
                        transition (incl. an initializing heartbeat's engine
                        line and a failed boot's message). */}
                    <span title={i.error_message ?? undefined}>
                      <StatusBadge status={i.backend_status} />
                    </span>
                  </TableCell>
                  <TableCell>
                    <ConnectionBadge connected={i.websocket_connected} />
                  </TableCell>
                  <TableCell className="font-mono text-xs">
                    {i.version}
                  </TableCell>
                  <TableCell className="font-mono text-xs">{i.port}</TableCell>
                  <TableCell className="font-mono text-xs">{i.epoch}</TableCell>
                  <TableCell className="text-xs text-muted-foreground">
                    {relTime(i.last_seen)}
                  </TableCell>
                  <TableCell className="text-xs text-muted-foreground">
                    {relTime(i.last_request_at)}
                  </TableCell>
                  <TableCell>
                    <code className="font-mono text-xs text-muted-foreground">
                      {i.config_fingerprint
                        ? `${i.config_fingerprint.slice(0, 8)}…`
                        : "—"}
                    </code>
                  </TableCell>
                  <TableCell className="text-right">
                    <div className="inline-flex gap-1">
                      <Button
                        variant="outline"
                        size="sm"
                        onClick={() => setLogsTarget(i)}
                        title="View backend + provider logs"
                      >
                        <ScrollText /> Logs
                      </Button>
                      <BackendActionMenu
                        instance={i}
                        onPick={(kind) =>
                          setBackendTarget({ kind, instance: i })
                        }
                      />
                      <Button
                        variant="outline"
                        size="sm"
                        disabled={!i.websocket_connected}
                        onClick={() => setCacheTarget(i)}
                        title={
                          i.websocket_connected
                            ? "Clear prompt cache"
                            : "Instance websocket is down"
                        }
                      >
                        <Eraser /> Cache
                      </Button>
                      <Button
                        variant="outline"
                        size="sm"
                        disabled={!i.websocket_connected}
                        onClick={() => setPruneTarget(i)}
                        title={
                          i.websocket_connected
                            ? "Prune unused model files"
                            : "Instance websocket is down"
                        }
                      >
                        <HardDriveDownload /> Prune
                      </Button>
                    </div>
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </div>
      )}

      <StorageActionSheet
        kind="cache"
        instance={cacheTarget}
        onClose={() => setCacheTarget(null)}
      />
      <StorageActionSheet
        kind="prune"
        instance={pruneTarget}
        onClose={() => setPruneTarget(null)}
      />
      <BackendActionSheet
        target={backendTarget}
        onClose={() => setBackendTarget(null)}
      />
      <LogsSheet
        instance={logsTarget}
        onOpenChange={(o) => {
          if (!o) setLogsTarget(null)
        }}
      />
    </div>
  )
}

function relTime(iso: string | null): string {
  if (!iso) return "never"
  const diff = (Date.now() - new Date(iso).getTime()) / 1000
  if (diff < 5) return "just now"
  if (diff < 60) return `${Math.floor(diff)}s ago`
  if (diff < 3600) return `${Math.floor(diff / 60)}m ago`
  if (diff < 86400) return `${Math.floor(diff / 3600)}h ago`
  return new Date(iso).toLocaleString()
}

function StorageActionSheet({
  kind,
  instance,
  onClose,
}: {
  kind: "cache" | "prune"
  instance: ProviderInstance | null
  onClose: () => void
}) {
  const queryClient = useQueryClient()
  const { showErrorToast } = useCustomToast()
  const [dryRun, setDryRun] = useState(true)
  const [force, setForce] = useState(false)
  const [result, setResult] = useState<StorageActionResult | null>(null)

  const mutation = useMutation({
    mutationFn: async () => {
      if (!instance) return
      setResult(null)
      if (kind === "cache") {
        return await AdminService.clearCache({
          path: { instance_id: instance.id },
          body: { dry_run: dryRun, force },
        })
      }
      return await AdminService.pruneStorage({
        path: { instance_id: instance.id },
        body: { dry_run: dryRun },
      })
    },
    onSuccess: (resp) => {
      const data = (resp?.data ?? null) as unknown as StorageActionResult
      setResult(data)
      if (!dryRun) {
        queryClient.invalidateQueries({ queryKey: instanceKeys.all })
        queryClient.invalidateQueries({ queryKey: definitionKeys.all })
      }
    },
    onError: (err: Error) => showErrorToast(extractError(err)),
  })

  const title = kind === "cache" ? "Clear prompt cache" : "Prune unused storage"
  const desc =
    kind === "cache"
      ? "Deletes prompt-cache dirs only — never model files. Refused while the backend is in_use unless forced."
      : "Deletes MODELS_DIR files not referenced by the driver's resolved artifact set. Refuses to run blind."

  return (
    <Sheet
      open={instance !== null}
      onOpenChange={(o) => {
        if (!o) {
          setResult(null)
          setDryRun(true)
          setForce(false)
          onClose()
        }
      }}
    >
      <SheetContent className="flex w-full flex-col gap-0 p-0 sm:max-w-xl">
        <SheetHeader className="border-b">
          <SheetTitle>{title}</SheetTitle>
          <SheetDescription>
            {desc} Target:{" "}
            <span className="font-mono">
              {instance?.machine_uid} / {instance?.alias}
            </span>
          </SheetDescription>
        </SheetHeader>

        <div className="min-h-0 flex-1 space-y-3 overflow-y-auto p-4">
          <div className="flex items-center justify-between rounded-lg border p-3">
            <div>
              <Label className="text-sm font-medium">Dry run</Label>
              <p className="text-xs text-muted-foreground">
                Preview what would be deleted without touching files.
              </p>
            </div>
            <Switch checked={dryRun} onCheckedChange={setDryRun} />
          </div>
          {kind === "cache" && !dryRun && (
            <div className="flex items-center justify-between rounded-lg border p-3">
              <div>
                <Label className="text-sm font-medium">Force</Label>
                <p className="text-xs text-muted-foreground">
                  Override the backend_in_use refusal (may cause I/O errors on
                  live streams).
                </p>
              </div>
              <Switch checked={force} onCheckedChange={setForce} />
            </div>
          )}

          {result && (
            <Alert>
              <AlertTitle>
                {dryRun ? "Preview" : "Done"} —{" "}
                {formatBytes(result.bytes_freed ?? 0)}{" "}
                {dryRun ? "would be freed" : "freed"}
              </AlertTitle>
              <AlertDescription>
                {(result.deleted ?? []).length === 0 ? (
                  <p className="text-xs">Nothing to delete.</p>
                ) : (
                  <ul className="mt-1 max-h-48 overflow-auto space-y-0.5">
                    {(result.deleted ?? []).map((p) => (
                      <li key={p} className="font-mono text-xs break-all">
                        {p}
                      </li>
                    ))}
                  </ul>
                )}
                {kind === "prune" && (result.kept ?? []).length > 0 && (
                  <p className="mt-2 text-xs text-muted-foreground">
                    Kept (referenced artifacts):{" "}
                    <Badge variant="outline" className="ml-1">
                      {(result.kept ?? []).length} entries
                    </Badge>
                  </p>
                )}
              </AlertDescription>
            </Alert>
          )}
        </div>

        <SheetFooter className="flex-row justify-end border-t">
          <Button
            type="button"
            variant="outline"
            onClick={() => {
              setResult(null)
              onClose()
            }}
          >
            Close
          </Button>
          <Button
            type="button"
            variant={dryRun ? "secondary" : "destructive"}
            disabled={mutation.isPending}
            onClick={() => mutation.mutate()}
          >
            {mutation.isPending
              ? "Running…"
              : dryRun
                ? "Preview (dry run)"
                : "Execute"}
          </Button>
        </SheetFooter>
      </SheetContent>
    </Sheet>
  )
}

function BackendActionMenu({
  instance,
  onPick,
}: {
  instance: ProviderInstance
  onPick: (kind: BackendActionKind) => void
}) {
  const offline = !instance.websocket_connected
  return (
    <DropdownMenu modal={false}>
      <DropdownMenuTrigger asChild>
        <Button
          variant="outline"
          size="sm"
          disabled={offline}
          title={offline ? "Instance websocket is down" : "Backend controls"}
        >
          <MoreVertical /> Backend
        </Button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="end">
        <DropdownMenuItem
          title={BACKEND_ACTIONS.start.title}
          onSelect={() => onPick("start")}
        >
          <Play /> Start
        </DropdownMenuItem>
        <DropdownMenuItem onSelect={() => onPick("stop")}>
          <Square /> Stop
        </DropdownMenuItem>
        <DropdownMenuItem
          title={BACKEND_ACTIONS.restart.title}
          onSelect={() => onPick("restart")}
        >
          <RotateCw /> Restart
        </DropdownMenuItem>
        <DropdownMenuSeparator />
        <DropdownMenuItem onSelect={() => onPick("initialize")}>
          <RefreshCw /> Reinitialize
        </DropdownMenuItem>
      </DropdownMenuContent>
    </DropdownMenu>
  )
}

function BackendActionSheet({
  target,
  onClose,
}: {
  target: BackendActionTarget | null
  onClose: () => void
}) {
  const queryClient = useQueryClient()
  const { showErrorToast, showSuccessToast } = useCustomToast()
  const [wait, setWait] = useState(false)
  const [result, setResult] = useState<BackendActionResult | null>(null)
  const kind = target?.kind
  const spec = kind ? BACKEND_ACTIONS[kind] : null

  const mutation = useMutation({
    mutationFn: async () => {
      if (!target) return null
      setResult(null)
      const options = {
        path: { instance_id: target.instance.id },
        body: { wait_for_running: wait },
      }
      if (target.kind === "start") {
        return await AdminService.startBackend(options)
      }
      if (target.kind === "stop") {
        return await AdminService.stopBackend({
          path: { instance_id: target.instance.id },
        })
      }
      if (target.kind === "restart") {
        return await AdminService.restartBackend(options)
      }
      return await AdminService.initializeInstance(options)
    },
    onSuccess: (resp) => {
      const data = (resp?.data ?? null) as unknown as BackendActionResult
      setResult(data)
      queryClient.invalidateQueries({ queryKey: instanceKeys.all })
      queryClient.invalidateQueries({ queryKey: definitionKeys.all })
      if (data) {
        showSuccessToast(
          `${spec?.label}: backend ${data.backend_status ?? "unknown"}`,
        )
      }
    },
    onError: (err: Error) => showErrorToast(extractError(err)),
  })

  const reset = () => {
    setResult(null)
    setWait(false)
    onClose()
  }

  const desc: Record<BackendActionKind, string> = {
    start:
      "Boots the instance's one backend without waiting for an inference request. By default the request returns as soon as the provider accepts: a cold boot downloads its checkpoint and companions first, which can take a long time — watch the Backend column, or open Logs for the engine's own progress.",
    stop: "Unloads the backend and frees its memory. Refused while live requests hold slots — in-flight streams are never cancelled.",
    restart:
      "Stop then start. Refused while live requests hold slots. The driver re-applies the definition's config on the way back up.",
    initialize:
      "Re-runs the whole init lifecycle: a fresh registration with the admin (re-checks the version and schema gates, re-adopts capacity and backend_config, mints a new instance secret and rewrites the provider's cached config), then a reboot and a re-read of the model metadata. Use it after editing a definition by hand, or when an instance looks wedged on stale config.",
  }

  return (
    <Sheet
      open={target !== null}
      onOpenChange={(o) => {
        if (!o) reset()
      }}
    >
      <SheetContent className="flex w-full flex-col gap-0 p-0 sm:max-w-xl">
        <SheetHeader className="border-b">
          <SheetTitle>{spec?.label}</SheetTitle>
          <SheetDescription>
            {kind ? desc[kind] : ""} Target:{" "}
            <span className="font-mono">
              {target?.instance.machine_uid} / {target?.instance.alias}
            </span>
          </SheetDescription>
        </SheetHeader>

        <div className="min-h-0 flex-1 space-y-3 overflow-y-auto p-4">
          {spec?.waits && (
            <div className="flex items-center justify-between rounded-lg border p-3">
              <div>
                <Label className="text-sm font-medium">{spec.waitLabel}</Label>
                <p className="text-xs text-muted-foreground">
                  Off = 202 accepted and the transition runs in the background.
                  On = this request stays open until the engine is up (it can be
                  very long).
                </p>
              </div>
              <Switch checked={wait} onCheckedChange={setWait} />
            </div>
          )}

          {result && (
            <Alert>
              <AlertTitle>
                {result.accepted
                  ? "Accepted — running in the background"
                  : `Done — ${result.backend_status ?? "unknown"}`}
              </AlertTitle>
              <AlertDescription>
                <p className="text-xs">
                  Backend status:{" "}
                  <Badge variant="outline">
                    {result.backend_status ?? "—"}
                  </Badge>
                  {typeof result.capacity === "number" && (
                    <>
                      {" · capacity "}
                      {result.capacity}
                    </>
                  )}
                  {(result.api_port || result.backend_port) && (
                    <>
                      {" · ports "}
                      {result.api_port ?? result.backend_port}
                      {result.engine_port ? `/${result.engine_port}` : ""}
                    </>
                  )}
                </p>
                {result.accepted && (
                  <p className="mt-2 text-xs text-muted-foreground">
                    The Backend column refreshes on its own; a cold boot shows
                    initializing while the engine downloads or loads, then
                    running (or error).
                  </p>
                )}
                {result.no_config && (
                  <p className="mt-2 text-xs text-amber-600 dark:text-amber-400">
                    The definition still has no backend_config, so nothing was
                    booted — the registration was refreshed only. Author the
                    config on the Definitions page.
                  </p>
                )}
              </AlertDescription>
            </Alert>
          )}
        </div>

        <SheetFooter className="flex-row justify-end border-t">
          <Button type="button" variant="outline" onClick={reset}>
            Close
          </Button>
          <Button
            type="button"
            variant={kind === "stop" ? "destructive" : "secondary"}
            disabled={mutation.isPending}
            onClick={() => mutation.mutate()}
          >
            {mutation.isPending ? "Sending…" : spec?.label}
          </Button>
        </SheetFooter>
      </SheetContent>
    </Sheet>
  )
}
