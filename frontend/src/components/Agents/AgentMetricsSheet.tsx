import { useQuery } from "@tanstack/react-query"
import { RefreshCw } from "lucide-react"

import { AgentsService } from "@/client"
import { Button } from "@/components/ui/button"
import {
  Sheet,
  SheetContent,
  SheetDescription,
  SheetHeader,
  SheetTitle,
} from "@/components/ui/sheet"

interface AgentMetricsSheetProps {
  isOpen: boolean
  onClose: () => void
  agentId: string
  agentName: string
}

function formatBytes(bytes: number): string {
  if (!bytes) {
    return "N/A"
  }
  const gb = bytes / 1073741824
  if (gb >= 1) {
    return `${gb.toFixed(1)} GB`
  }
  return `${(bytes / 1048576).toFixed(0)} MB`
}

export function AgentMetricsSheet({
  isOpen,
  onClose,
  agentId,
  agentName,
}: AgentMetricsSheetProps) {
  const metricsQuery = useQuery({
    queryKey: ["agent-metrics", agentId],
    queryFn: async () => {
      const [gpuInfo, gpuUsage, servers] = await Promise.all([
        AgentsService.sendCommand({
          path: { agent_id: agentId },
          body: { method: "GET", path: "/gpu" },
        }),
        AgentsService.sendCommand({
          path: { agent_id: agentId },
          body: { method: "GET", path: "/gpu/usage" },
        }),
        AgentsService.sendCommand({
          path: { agent_id: agentId },
          body: { method: "GET", path: "/servers/list" },
        }),
      ])

      return {
        gpuInfo: gpuInfo.data as Record<string, unknown>,
        gpuUsage: gpuUsage.data as Record<string, unknown>,
        servers: servers.data as { servers?: Array<Record<string, unknown>> },
      }
    },
    enabled: isOpen,
    refetchInterval: 5000,
  })

  const gpuInfo = metricsQuery.data?.gpuInfo ?? {}
  const gpuUsage = metricsQuery.data?.gpuUsage ?? {}
  const servers = metricsQuery.data?.servers?.servers ?? []

  return (
    <Sheet open={isOpen} onOpenChange={(open) => !open && onClose()}>
      <SheetContent side="right" className="flex w-full flex-col sm:max-w-md">
        <SheetHeader>
          <SheetTitle className="flex items-center gap-2">
            Metrics - {agentName}
            <Button
              variant="ghost"
              size="icon"
              className="h-6 w-6"
              onClick={() => metricsQuery.refetch()}
              disabled={metricsQuery.isFetching}
            >
              <RefreshCw className="h-4 w-4" />
            </Button>
          </SheetTitle>
          <SheetDescription>
            GPU and server statistics, refreshed every 5 seconds
          </SheetDescription>
        </SheetHeader>

        <div className="flex-1 space-y-6 overflow-y-auto p-4">
          {metricsQuery.isLoading && (
            <div className="text-muted-foreground">Loading metrics...</div>
          )}

          <div>
            <h4 className="mb-2 text-sm font-semibold">GPU</h4>
            <dl className="space-y-1 text-sm">
              <div className="flex justify-between">
                <dt className="text-muted-foreground">Name</dt>
                <dd>{String(gpuInfo.name ?? "Unknown")}</dd>
              </div>
              <div className="flex justify-between">
                <dt className="text-muted-foreground">Backend</dt>
                <dd>{String(gpuInfo.backend ?? "auto")}</dd>
              </div>
              <div className="flex justify-between">
                <dt className="text-muted-foreground">VRAM Total</dt>
                <dd>{formatBytes(Number(gpuInfo.vram_total ?? 0))}</dd>
              </div>
              <div className="flex justify-between">
                <dt className="text-muted-foreground">VRAM Used</dt>
                <dd>{formatBytes(Number(gpuInfo.vram_used ?? 0))}</dd>
              </div>
              {gpuUsage.gpu_percent !== undefined && (
                <div className="flex justify-between">
                  <dt className="text-muted-foreground">Utilization</dt>
                  <dd>{Number(gpuUsage.gpu_percent).toFixed(1)}%</dd>
                </div>
              )}
            </dl>
          </div>

          <div>
            <h4 className="mb-2 text-sm font-semibold">
              Servers ({servers.length})
            </h4>
            {servers.length === 0 ? (
              <div className="text-sm text-muted-foreground">
                No running llama.cpp servers
              </div>
            ) : (
              <dl className="space-y-2 text-sm">
                {servers.map((server) => (
                  <div
                    key={String(server.server_id)}
                    className="rounded-md border p-2"
                  >
                    <div className="font-medium">
                      {String(server.server_id)}
                    </div>
                    <div className="text-xs text-muted-foreground">
                      Model: {String(server.model_path ?? "N/A")}
                    </div>
                    <div className="text-xs text-muted-foreground">
                      Uptime:{" "}
                      {Math.floor(Number(server.uptime_seconds ?? 0) / 60)}m
                    </div>
                  </div>
                ))}
              </dl>
            )}
          </div>
        </div>
      </SheetContent>
    </Sheet>
  )
}
