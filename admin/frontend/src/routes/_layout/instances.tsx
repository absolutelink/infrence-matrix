import { useMutation, useQueryClient } from "@tanstack/react-query"
import { createFileRoute } from "@tanstack/react-router"
import { Eraser, HardDriveDownload } from "lucide-react"
import { useState } from "react"

import { AdminService } from "@/client"
import { ConnectionBadge, StatusBadge } from "@/components/Common/StatusBadge"
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert"
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
import { Label } from "@/components/ui/label"
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
} from "@/hooks/useAdminData"
import useCustomToast from "@/hooks/useCustomToast"
import { extractError } from "@/lib/errors"
import type { ProviderInstance, StorageActionResult } from "@/types/admin"
import { formatBytes } from "@/utils"

export const Route = createFileRoute("/_layout/instances")({
  component: InstancesPage,
  head: () => ({ meta: [{ title: "Instances - Inference Matrix" }] }),
})

// NOTE: live backend.logs / provider.logs are WS events only — there is
// no REST read endpoint for them, so the UI does not show log streams.
// (provider.logs is still Reserved in docs/ws-protocol.md §4.)

function InstancesPage() {
  const { data: instances = [], isLoading } = useInstances()
  const [cacheTarget, setCacheTarget] = useState<ProviderInstance | null>(null)
  const [pruneTarget, setPruneTarget] = useState<ProviderInstance | null>(null)

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
                    <StatusBadge status={i.instance_status} />
                  </TableCell>
                  <TableCell>
                    <StatusBadge status={i.backend_status} />
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

      <StorageActionDialog
        kind="cache"
        instance={cacheTarget}
        onClose={() => setCacheTarget(null)}
      />
      <StorageActionDialog
        kind="prune"
        instance={pruneTarget}
        onClose={() => setPruneTarget(null)}
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

function StorageActionDialog({
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
    <Dialog
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
      <DialogContent className="sm:max-w-lg">
        <DialogHeader>
          <DialogTitle>{title}</DialogTitle>
          <DialogDescription>
            {desc} Target:{" "}
            <span className="font-mono">
              {instance?.machine_uid} / {instance?.alias}
            </span>
          </DialogDescription>
        </DialogHeader>

        <div className="space-y-3">
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

        <DialogFooter>
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
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}
