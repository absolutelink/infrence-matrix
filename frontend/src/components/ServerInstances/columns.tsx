import { useMutation, useQueryClient } from "@tanstack/react-query"
import type { ColumnDef } from "@tanstack/react-table"
import {
  Activity,
  Clock,
  Cpu,
  MemoryStick,
  MoreHorizontal,
  Power,
  Terminal,
} from "lucide-react"
import { useState } from "react"
import { toast } from "sonner"

import { AgentsService, ServerInstancesService } from "@/client"
import { ServerLogsSheet } from "@/components/ServerInstances/ServerLogsSheet"
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
  port: number
  status: string
  health_status: string
  agent_id: string
  agent_name: string | null
  agent_host: string | null
  agent_port: number | null
  proxy_url: string | null
  started_at: string | null
  total_requests: number
  cpu_usage_percent: number | null
  ram_usage_bytes: number | null
  vram_usage_bytes: number | null
}

export const columns: ColumnDef<ServerInstance>[] = [
  {
    accessorKey: "model_name",
    header: "Model",
    cell: ({ row }) => {
      const instance = row.original
      return (
        <div className="font-medium">{instance.model_name || "Unknown"}</div>
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

      return (
        <div className="flex gap-2">
          <Badge variant={statusVariant}>{instance.status}</Badge>
          <Badge variant={healthVariant} className="text-xs">
            {instance.health_status}
          </Badge>
        </div>
      )
    },
  },
  {
    accessorKey: "port",
    header: "Port",
    cell: ({ row }) => {
      const instance = row.original
      return (
        <div className="text-muted-foreground font-mono text-sm">
          {instance.port}
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
    cell: ({ row }) => {
      const instance = row.original
      if (!instance.started_at) {
        return <span className="text-muted-foreground">-</span>
      }

      const started = new Date(instance.started_at)
      const now = new Date()
      const diff = now.getTime() - started.getTime()
      const hours = Math.floor(diff / 3600000)
      const minutes = Math.floor((diff % 3600000) / 60000)

      if (hours > 0) {
        return (
          <div className="flex items-center gap-1">
            <Clock className="h-3 w-3" />
            <span>
              {hours}h {minutes}m
            </span>
          </div>
        )
      }
      if (minutes > 0) {
        return (
          <div className="flex items-center gap-1">
            <Clock className="h-3 w-3" />
            <span>{minutes}m</span>
          </div>
        )
      }
      return (
        <div className="flex items-center gap-1 text-green-600">
          <Clock className="h-3 w-3" />
          <span>Just started</span>
        </div>
      )
    },
  },
  {
    accessorKey: "total_requests",
    header: "Requests",
    cell: ({ row }) => {
      const instance = row.original
      return (
        <div className="text-muted-foreground">
          {instance.total_requests.toLocaleString()}
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
]

function InstanceActions({ instance }: { instance: ServerInstance }) {
  const queryClient = useQueryClient()
  const [logsOpen, setLogsOpen] = useState(false)

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
          <DropdownMenuItem onClick={() => setLogsOpen(true)}>
            <Terminal className="mr-2 h-4 w-4" />
            View Logs
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
        </DropdownMenuContent>
      </DropdownMenu>

      <ServerLogsSheet
        isOpen={logsOpen}
        onClose={() => setLogsOpen(false)}
        instance={instance}
      />
    </>
  )
}
