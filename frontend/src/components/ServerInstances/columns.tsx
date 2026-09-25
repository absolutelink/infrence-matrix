import { useMutation, useQueryClient } from "@tanstack/react-query"
import type { ColumnDef } from "@tanstack/react-table"
import {
  Activity,
  Cpu,
  MemoryStick,
  MoreHorizontal,
  Pencil,
  Play,
  Power,
  Terminal,
  Trash2,
} from "lucide-react"
import { useState } from "react"
import { toast } from "sonner"

import { AgentsService, ServerInstancesService } from "@/client"
import { EditServerDialog } from "@/components/ServerInstances/EditServerDialog"
import type { HalogenOptions } from "@/components/ServerInstances/HalogenSettingsFields"
import { useLogPanel } from "@/components/ServerInstances/LogPanelContext"
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

type ServerInstance = {
  id: string
  model_id: string
  model_name: string | null
  alias: string
  engine?: "llamacpp" | "halogen"
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
}

export const columns: ColumnDef<ServerInstance>[] = [
  {
    accessorKey: "alias",
    header: "Alias",
    cell: ({ row }) => {
      const instance = row.original
      return (
        <div>
          <div className="font-medium">{instance.alias || "—"}</div>
          <div className="text-xs text-muted-foreground">
            {instance.engine === "halogen"
              ? "Halogen"
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
          <div className="font-medium">{instance.agent_name || "Unknown"}</div>
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
        unknown: "outline",
      }

      const statusVariant = statusColors[instance.status] || "outline"
      const healthVariant = statusColors[instance.health_status] || "outline"
      const statusLabel =
        instance.status === "starting" ? "booting" : instance.status

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
    accessorKey: "resources",
    header: "Resources",
    cell: ({ row }) => {
      const instance = row.original

      const ramGB = instance.ram_usage_bytes
        ? (instance.ram_usage_bytes / 1073741824).toFixed(2)
        : null
      const vramGB = instance.vram_usage_bytes
        ? (instance.vram_usage_bytes / 1073741824).toFixed(2)
        : null

      return (
        <div className="flex flex-col gap-1 text-xs">
          {instance.cpu_usage_percent !== null && (
            <div className="flex items-center gap-1">
              <Cpu className="h-3 w-3" />
              <span>{instance.cpu_usage_percent.toFixed(1)}%</span>
            </div>
          )}
          {ramGB && (
            <div className="flex items-center gap-1">
              <MemoryStick className="h-3 w-3" />
              <span>{ramGB} GB</span>
            </div>
          )}
          {vramGB && (
            <div className="flex items-center gap-1">
              <Activity className="h-3 w-3" />
              <span>{vramGB} GB VRAM</span>
            </div>
          )}
          {!instance.cpu_usage_percent && !ramGB && !vramGB && (
            <span className="text-muted-foreground">N/A</span>
          )}
        </div>
      )
    },
  },
  {
    accessorKey: "started_at",
    header: "Uptime",
    cell: () => null,
  },
  {
    id: "actions",
    cell: ({ row }) => {
      const instance = row.original

      return <InstanceActions instance={instance} />
    },
  },
]

function InstanceActions({ instance }: { instance: ServerInstance }) {
  const queryClient = useQueryClient()
  const { openLogs } = useLogPanel()
  const [editOpen, setEditOpen] = useState(false)

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

  const canStart =
    instance.status !== "running" && instance.status !== "starting"

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
          <DropdownMenuItem onClick={() => setEditOpen(true)}>
            <Pencil className="mr-2 h-4 w-4" />
            Edit Settings
          </DropdownMenuItem>
          <DropdownMenuSeparator />
          {instance.status === "running" ? (
            <DropdownMenuItem
              className="text-destructive"
              onClick={() => stopMutation.mutate()}
              disabled={stopMutation.isPending}
            >
              <Power className="mr-2 h-4 w-4" />
              Stop Server
            </DropdownMenuItem>
          ) : null}
          {canStart ? (
            <DropdownMenuItem
              onClick={() => startMutation.mutate()}
              disabled={startMutation.isPending}
            >
              <Play className="mr-2 h-4 w-4" />
              Start Server
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
    </>
  )
}
