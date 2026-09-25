import { useQuery } from "@tanstack/react-query"
import { Activity, ChevronDown, Cpu, ListOrdered, Server } from "lucide-react"

import { AgentsService } from "@/client"
import { useQueueStatus } from "@/hook/useQueueStatus"

export function QueueStatusBar() {
  const { status, connected } = useQueueStatus()
  const { data: agents = [] } = useQuery({
    queryKey: ["queue-status-agents"],
    queryFn: async () => (await AgentsService.listAgents()).data.agents || [],
    refetchInterval: 10000,
  })

  const gpuAgents = agents.filter((agent: any) => agent.gpu_info?.vram_total)
  const usedVram = gpuAgents.reduce(
    (sum: number, agent: any) => sum + (agent.gpu_info?.vram_used || 0),
    0,
  )
  const totalVram = gpuAgents.reduce(
    (sum: number, agent: any) => sum + (agent.gpu_info?.vram_total || 0),
    0,
  )
  const gpuUtilization = gpuAgents.length
    ? gpuAgents.reduce(
        (sum: number, agent: any) => sum + (agent.gpu_info?.utilization || 0),
        0,
      ) / gpuAgents.length
    : null
  const formatBytes = (bytes: number) => `${(bytes / 1073741824).toFixed(1)} GB`

  if (!status) {
    return (
      <div className="ml-auto flex items-center gap-2 text-xs text-muted-foreground">
        <Activity className="h-3.5 w-3.5 animate-pulse" />
        <span>Connecting to scheduler</span>
      </div>
    )
  }

  const busy = status.queued > 0 || status.available === 0

  return (
    <details className="group relative ml-auto">
      <summary className="flex cursor-pointer list-none items-center gap-3 rounded-md px-2 py-1.5 text-xs text-muted-foreground transition-colors hover:bg-muted hover:text-foreground">
        <span className="flex items-center gap-1.5">
          <span
            className={`h-1.5 w-1.5 rounded-full ${busy ? "bg-amber-500" : "bg-emerald-500"}`}
          />
          {status.active} running
        </span>
        <span className="flex items-center gap-1.5">
          <ListOrdered className="h-3.5 w-3.5" />
          {status.queued} queued
        </span>
        {gpuUtilization !== null && (
          <span className="flex items-center gap-1.5">
            <Cpu className="h-3.5 w-3.5" />
            {gpuUtilization.toFixed(0)}% GPU
          </span>
        )}
        <ChevronDown className="h-3.5 w-3.5 transition-transform group-open:rotate-180" />
      </summary>
      <div className="absolute right-0 top-full z-20 mt-2 w-80 rounded-lg border bg-popover p-3 text-popover-foreground shadow-lg">
        <div className="mb-3 flex items-center justify-between">
          <div>
            <p className="text-sm font-medium">Inference capacity</p>
            <p className="text-xs text-muted-foreground">
              {status.available} of {status.capacity} slots available
            </p>
          </div>
          {gpuAgents.length > 0 && (
            <div className="mb-3 rounded-md bg-muted/60 px-2.5 py-2 text-xs">
              <div className="flex items-center justify-between">
                <span className="text-muted-foreground">GPU utilization</span>
                <span className="font-medium">
                  {gpuUtilization?.toFixed(0)}%
                </span>
              </div>
              <div className="mt-1 flex items-center justify-between">
                <span className="text-muted-foreground">VRAM usage</span>
                <span className="font-medium">
                  {formatBytes(usedVram)} / {formatBytes(totalVram)}
                </span>
              </div>
            </div>
          )}
          <Server className="h-4 w-4 text-muted-foreground" />
        </div>
        <div className="space-y-2">
          {status.servers.length === 0 ? (
            <p className="text-xs text-muted-foreground">No running servers</p>
          ) : (
            status.servers.map((server) => (
              <div
                key={server.id}
                className="rounded-md bg-muted/60 px-2.5 py-2"
              >
                <div className="flex items-center justify-between text-xs">
                  <span className="truncate font-medium">
                    {server.alias || server.model_id.slice(0, 8)}
                  </span>
                  <span className="text-muted-foreground">
                    {server.state === "booting"
                      ? "Booting"
                      : `${server.active}/${server.capacity} active`}
                  </span>
                </div>
                <div className="mt-1 text-[11px] text-muted-foreground">
                  {server.state === "booting"
                    ? "Waiting for llama.cpp health"
                    : server.telemetry_known
                      ? `${server.available} slots available`
                      : "Using health status; telemetry unavailable"}
                </div>
              </div>
            ))
          )}
        </div>
        {!connected && (
          <p className="mt-3 text-[11px] text-amber-600">
            Scheduler connection interrupted
          </p>
        )}
      </div>
    </details>
  )
}
