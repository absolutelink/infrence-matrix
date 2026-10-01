import { useMutation, useQueryClient } from "@tanstack/react-query"
import type { ColumnDef } from "@tanstack/react-table"
import {
  ListOrdered,
  MoreHorizontal,
  Pencil,
  Play,
  Power,
  RefreshCw,
  Rocket,
  Terminal,
  Trash2,
  Zap,
} from "lucide-react"
import { useMemo, useRef, useState } from "react"
import { toast } from "sonner"

import { AgentsService, ServerInstancesService } from "@/client"
import { EditServerDialog } from "@/components/ServerInstances/EditServerDialog"
import type { HalogenOptions } from "@/components/ServerInstances/HalogenSettingsFields"
import { useLogPanel } from "@/components/ServerInstances/LogPanelContext"
import { MetadataDialog } from "@/components/ServerInstances/MetadataDialog"
import type { ServerOptions } from "@/components/ServerInstances/ServerSettingsFields"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu"
import { useQueueStatus } from "@/hook/useQueueStatus"
import { useTokenStats } from "@/hook/useTokenStats"

type ServerInstance = {
  id: string
  model_id: string
  model_name: string | null
  alias: string
  engine?: "llamacpp" | "halogen" | "halogen-flash"
  engine_options?: HalogenOptions
  status: string
  health_status: string
  error_message: string | null
  agent_id: string
  agent_name: string | null
  agent_host: string | null
  agent_port: number | null
  proxy_url: string | null
  started_at: string | null
  last_health_check: string | null
  cpu_usage_percent: number | null
  ram_usage_bytes: number | null
  vram_usage_bytes: number | null
  gpu_layers: number
  context_size: number
  flash_attn: boolean
  inactivity_timeout_seconds: number
  server_options?: ServerOptions
  model_metadata?: Record<string, unknown>
}

const formatRate = (rate: number | null | undefined) =>
  rate === null || rate === undefined ? "—" : rate.toFixed(1)

export function useColumns(): ColumnDef<ServerInstance>[] {
  const { status } = useQueueStatus()
  const { data: tokenData } = useTokenStats()

  const throughputById = useMemo(() => {
    const map = new Map<
      string,
      { decode: number | null | undefined; prefill: number | null | undefined }
    >()
    for (const server of tokenData?.servers ?? []) {
      map.set(server.id, {
        decode: server.decode_tokens_per_second,
        prefill: server.prefill_tokens_per_second,
      })
    }
    return map
  }, [tokenData])

  const queueById = useMemo(() => {
    const map = new Map<
      string,
      { available: number; capacity: number; queued: number }
    >()
    for (const server of status?.servers ?? []) {
      map.set(server.id, {
        available: server.available,
        capacity: server.capacity,
        queued: server.queued,
      })
    }
    return map
  }, [status])

  // Keep the latest lookup maps in refs so the column definitions (and their
  // cell render functions) stay referentially stable across data refreshes.
  // Otherwise TanStack Table rebuilds the column model every refresh, which
  // remounts cells (e.g. the actions dropdown) and drops their open state.
  const throughputRef = useRef(throughputById)
  throughputRef.current = throughputById
  const queueRef = useRef(queueById)
  queueRef.current = queueById

  return useMemo<ColumnDef<ServerInstance>[]>(
    () => [
      {
        accessorKey: "alias",
        header: "Alias",
        cell: ({ row }) => {
          const instance = row.original
          return (
            <div>
              <div className="font-medium">{instance.alias || "—"}</div>
              <div className="text-xs text-muted-foreground">
                {instance.engine === "halogen" ||
                instance.engine === "halogen-flash"
                  ? instance.engine === "halogen-flash"
                    ? "Halogen Flash"
                    : "Halogen"
                  : instance.model_name || "Unknown"}
              </div>
            </div>
          )
        },
      },
      {
        accessorKey: "agent_name",
        header: "Agent",
        cell: ({ row }) => {
          const instance = row.original
          return (
            <div>
              <div className="font-medium">
                {instance.agent_name || "Unknown"}
              </div>
              <div className="text-xs text-muted-foreground">
                {instance.agent_host}:{instance.agent_port}
              </div>
            </div>
          )
        },
      },
      {
        accessorKey: "status",
        header: "Status",
        cell: ({ row }) => {
          const instance = row.original
          const statusColors: Record<
            string,
            "default" | "secondary" | "destructive" | "outline"
          > = {
            running: "default",
            starting: "secondary",
            stopping: "secondary",
            stopped: "secondary",
            healthy: "default",
            unhealthy: "destructive",
            initialization_failed: "destructive",
            metadata_gathering: "secondary",
            preparing: "secondary",
            uninitialized: "outline",
            unknown: "outline",
          }

          const statusVariant = statusColors[instance.status] || "outline"
          const healthVariant =
            statusColors[instance.health_status] || "outline"
          const statusLabel =
            instance.status === "starting"
              ? "booting"
              : instance.status === "metadata_gathering"
                ? "gathering metadata"
                : instance.status

          return (
            <div className="flex flex-col items-start gap-1">
              <div className="flex gap-2">
                <Badge variant={statusVariant}>{statusLabel}</Badge>
                <Badge variant={healthVariant} className="text-xs">
                  {instance.health_status}
                </Badge>
              </div>
              {instance.last_health_check && (
                <span
                  className="text-[11px] text-muted-foreground"
                  title="Last updated"
                >
                  Updated{" "}
                  {new Date(instance.last_health_check).toLocaleTimeString()}
                </span>
              )}
              {instance.error_message && (
                <span
                  className="max-w-64 truncate text-xs text-destructive"
                  title={instance.error_message}
                >
                  {instance.error_message}
                </span>
              )}
            </div>
          )
        },
      },
      {
        id: "throughput",
        header: "Throughput",
        cell: ({ row }) => {
          const stats = throughputRef.current.get(row.original.id)
          return (
            <div className="flex flex-col gap-1 text-xs">
              <div className="flex items-center gap-1">
                <Zap className="h-3 w-3 text-emerald-500" />
                <span>{formatRate(stats?.decode)} tok/s</span>
              </div>
              <div className="flex items-center gap-1">
                <Rocket className="h-3 w-3 text-sky-500" />
                <span>{formatRate(stats?.prefill)} tok/s</span>
              </div>
            </div>
          )
        },
      },
      {
        id: "slots_queue",
        header: "Slots / Queue",
        cell: ({ row }) => {
          const slot = queueRef.current.get(row.original.id)
          const available = slot?.available ?? 0
          const capacity = slot?.capacity ?? 0
          const queued = slot?.queued ?? 0
          return (
            <div className="flex flex-col gap-1 text-xs">
              <div className="flex items-center gap-1">
                <span className="font-medium">
                  {available}/{capacity}
                </span>
                <span className="text-muted-foreground">slots</span>
              </div>
              <div className="flex items-center gap-1">
                <ListOrdered className="h-3 w-3" />
                <span>{queued} queued</span>
              </div>
            </div>
          )
        },
      },
      {
        id: "actions",
        cell: ({ row }) => {
          const instance = row.original

          return <InstanceActions instance={instance} />
        },
      },
    ],
    [],
  )
}

function InstanceActions({ instance }: { instance: ServerInstance }) {
  const queryClient = useQueryClient()
  const { openLogs } = useLogPanel()
  const [editOpen, setEditOpen] = useState(false)
  const [metadataOpen, setMetadataOpen] = useState(false)

  const stopMutation = useMutation({
    mutationFn: async () => {
      // Stop the llama-server on the agent via the command proxy.
      if (instance.agent_id && instance.agent_host && instance.agent_port) {
        await AgentsService.sendCommand({
          path: { agent_id: instance.agent_id },
          body: {
            method: "POST",
            path: "/servers/stop",
            body: { server_id: instance.id },
          },
        })
      }
      return ServerInstancesService.instancesStopServer({
        path: { server_id: instance.id },
      })
    },
    onSuccess: () => {
      toast.success("Server stopped")
      queryClient.invalidateQueries({ queryKey: ["server-instances"] })
    },
    onError: () => {
      toast.error("Failed to stop server")
    },
  })

  const startMutation = useMutation({
    mutationFn: async () =>
      ServerInstancesService.instancesRestartServer({
        path: { server_id: instance.id },
      }),
    onSuccess: () => {
      toast.success("Server start request sent")
      queryClient.invalidateQueries({ queryKey: ["server-instances"] })
    },
    onError: () => {
      toast.error("Failed to start server")
    },
  })

  const initializeMutation = useMutation({
    mutationFn: async () =>
      ServerInstancesService.instancesInitializeExistingServer({
        path: { server_id: instance.id },
      }),
    onSuccess: () => {
      toast.success("Server initialization started")
      queryClient.invalidateQueries({ queryKey: ["server-instances"] })
    },
    onError: () => {
      toast.error("Failed to initialize server")
    },
  })

  const deleteMutation = useMutation({
    mutationFn: async () =>
      ServerInstancesService.instancesDeleteServer({
        path: { server_id: instance.id },
      }),
    onSuccess: () => {
      toast.success("Server instance deleted")
      queryClient.invalidateQueries({ queryKey: ["server-instances"] })
    },
    onError: (error: Error) => {
      toast.error(`Failed to delete server: ${error.message}`)
    },
  })

  const canStart = ["stopped", "error"].includes(instance.status)
  const canEdit = [
    "stopped",
    "running",
    "error",
    "initialization_failed",
  ].includes(instance.status)
  const canInitialize = [
    "uninitialized",
    "stopped",
    "initialization_failed",
  ].includes(instance.status)

  return (
    <>
      <DropdownMenu>
        <DropdownMenuTrigger asChild>
          <Button variant="ghost" className="h-8 w-8 p-0">
            <span className="sr-only">Open menu</span>
            <MoreHorizontal className="h-4 w-4" />
          </Button>
        </DropdownMenuTrigger>
        <DropdownMenuContent align="end">
          <DropdownMenuLabel>Actions</DropdownMenuLabel>
          <DropdownMenuSeparator />
          <DropdownMenuItem onClick={() => openLogs(instance)}>
            <Terminal className="mr-2 h-4 w-4" />
            View Logs
          </DropdownMenuItem>
          <DropdownMenuItem
            onClick={() => setEditOpen(true)}
            disabled={!canEdit}
          >
            <Pencil className="mr-2 h-4 w-4" />
            Edit Settings
          </DropdownMenuItem>
          <DropdownMenuItem
            onClick={() => setMetadataOpen(true)}
            disabled={
              instance.status === "preparing" ||
              instance.status === "metadata_gathering"
            }
          >
            <Pencil className="mr-2 h-4 w-4" />
            Edit Metadata
          </DropdownMenuItem>
          <DropdownMenuSeparator />
          {canInitialize ? (
            <DropdownMenuItem
              onClick={() => initializeMutation.mutate()}
              disabled={initializeMutation.isPending}
            >
              <RefreshCw className="mr-2 h-4 w-4" />
              {instance.status === "initialization_failed"
                ? "Retry Initialization"
                : "Initialize / Refresh Metadata"}
            </DropdownMenuItem>
          ) : null}
          {instance.status === "running" ||
          instance.status === "starting" ||
          instance.status === "stopping" ? (
            <DropdownMenuItem
              className="text-destructive"
              onClick={() => stopMutation.mutate()}
              disabled={stopMutation.isPending}
            >
              <Power className="mr-2 h-4 w-4" />
              {instance.status === "running" ? "Stop Server" : "Force Stop"}
            </DropdownMenuItem>
          ) : null}
          {canStart ? (
            <DropdownMenuItem
              onClick={() => startMutation.mutate()}
              disabled={startMutation.isPending}
            >
              <Play className="mr-2 h-4 w-4" />
              {instance.status === "error" ? "Restart Server" : "Start Server"}
            </DropdownMenuItem>
          ) : null}
          <DropdownMenuSeparator />
          <DropdownMenuItem
            className="text-destructive"
            onClick={() => {
              if (
                window.confirm(
                  `Delete server instance "${instance.alias || instance.id}"?` +
                    (instance.status === "running"
                      ? " It will be stopped first."
                      : ""),
                )
              ) {
                deleteMutation.mutate()
              }
            }}
            disabled={deleteMutation.isPending}
          >
            <Trash2 className="mr-2 h-4 w-4" />
            Delete
          </DropdownMenuItem>
        </DropdownMenuContent>
      </DropdownMenu>

      <EditServerDialog
        isOpen={editOpen}
        onClose={() => setEditOpen(false)}
        instance={{
          ...instance,
          alias: instance.alias ?? "",
        }}
      />
      <MetadataDialog
        isOpen={metadataOpen}
        onClose={() => setMetadataOpen(false)}
        instance={instance}
      />
    </>
  )
}
