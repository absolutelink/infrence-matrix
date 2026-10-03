import { useMutation, useQuery } from "@tanstack/react-query"
import {
  ChevronDown,
  Gauge,
  ListOrdered,
  MemoryStick,
  Rocket,
  Server,
  Trash2,
  Zap,
} from "lucide-react"
import { toast } from "sonner"

import { AgentsService, QueueService } from "@/client"
import { Button } from "@/components/ui/button"
import { useQueueStatus } from "@/hook/useQueueStatus"
import { useTokenStats } from "@/hook/useTokenStats"
import { uniqueGpuSnapshots } from "@/lib/gpuMetrics"

const formatTokens = (n: number) => {
  if (n >= 1_000_000_000) return `${(n / 1_000_000_000).toFixed(1)}B`
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`
  if (n >= 1_000) return `${(n / 1_000).toFixed(1)}k`
  return `${n}`
}

const formatRate = (rate: number | null | undefined) =>
  rate === null || rate === undefined ? "—" : rate.toFixed(1)

export function QueueStatusBar() {
  const { status, connected } = useQueueStatus()
  const { data: tokenData } = useTokenStats()

  const clearQueueMutation = useMutation({
    mutationFn: () => QueueService.clearInferenceQueue(),
    onSuccess: (result) => {
      toast.success(`Cleared ${result.data?.cancelled ?? 0} queued request(s)`)
    },
    onError: () => toast.error("Failed to clear the inference queue"),
  })
  const { data: agents = [] } = useQuery({
    queryKey: ["queue-status-agents"],
    queryFn: async () => (await AgentsService.listAgents()).data.agents || [],
    refetchInterval: 10000,
  })

  const gpuSnapshots = uniqueGpuSnapshots(agents)
  const usedVram = gpuSnapshots.reduce(
    (sum: number, gpu: any) => sum + (gpu.vram_used || 0),
    0,
  )
  const totalVram = gpuSnapshots.reduce(
    (sum: number, gpu: any) => sum + (gpu.vram_total || 0),
    0,
  )
  const gpuUtilization = gpuSnapshots.length
    ? gpuSnapshots.reduce(
        (sum: number, gpu: any) => sum + (gpu.utilization || 0),
        0,
      ) / gpuSnapshots.length
    : null
  const formatBytes = (bytes: number) => `${(bytes / 1073741824).toFixed(1)} GB`

  const live = tokenData?.global.live
  const day = tokenData?.global.last_24h

  if (!status) {
    return (
      <div className="ml-auto flex items-center gap-2 text-xs text-muted-foreground">
        <span>Connecting to scheduler</span>
      </div>
    )
  }

  const busy = status.queued > 0 || status.available === 0

  return (
    <details className="group relative ml-auto">
      <summary className="flex cursor-pointer list-none items-center gap-3 rounded-md px-2 py-1.5 text-xs text-muted-foreground transition-colors hover:bg-muted hover:text-foreground">
        <span
          className="flex items-center gap-1.5"
          title="Latest llama-server decode throughput metrics scrape"
        >
          <Zap className="h-3.5 w-3.5 text-emerald-500" />
          {formatRate(live?.decode_tokens_per_second)} tok/s
        </span>
        <span
          className="flex items-center gap-1.5"
          title="Latest llama-server prefill throughput metrics scrape"
        >
          <Rocket className="h-3.5 w-3.5 text-sky-500" />
          {formatRate(live?.prefill_tokens_per_second)} tok/s
        </span>
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
        <ChevronDown className="h-3.5 w-3.5 transition-transform group-open:rotate-180" />
      </summary>
      <div className="absolute right-0 top-full z-20 mt-2 w-96 rounded-lg border bg-popover p-4 text-popover-foreground shadow-lg">
        <div className="mb-3 flex items-center justify-between">
          <div>
            <p className="text-sm font-medium">Inference capacity</p>
            <p className="text-xs text-muted-foreground">
              {status.available} of {status.capacity} slots available
            </p>
          </div>
          <Server className="h-4 w-4 text-muted-foreground" />
        </div>

        {day && (
          <div className="mb-3 grid grid-cols-3 gap-2 text-xs">
            <WindowCard label="Last 24h" totals={tokenData!.global.last_24h} />
            <WindowCard
              label="Last 7 days"
              totals={tokenData!.global.last_7d}
            />
            <WindowCard
              label="Last 30 days"
              totals={tokenData!.global.last_30d}
            />
          </div>
        )}

        {gpuSnapshots.length > 0 && (
          <div className="mb-3 rounded-md bg-muted/60 px-2.5 py-2 text-xs">
            <div className="flex items-center justify-between">
              <span className="flex items-center gap-1.5 text-muted-foreground">
                <Gauge className="h-3.5 w-3.5" />
                GPU utilization
              </span>
              <span className="font-medium">{gpuUtilization?.toFixed(0)}%</span>
            </div>
            <div className="mt-1 flex items-center justify-between">
              <span className="flex items-center gap-1.5 text-muted-foreground">
                <MemoryStick className="h-3.5 w-3.5" />
                VRAM usage
              </span>
              <span className="font-medium">
                {formatBytes(usedVram)} / {formatBytes(totalVram)}
              </span>
            </div>
          </div>
        )}

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
                      ? `${server.available} slots available · ${server.queued} queued`
                      : "Using health status; telemetry unavailable"}
                </div>
              </div>
            ))
          )}
        </div>

        <div className="mt-4 border-t pt-3">
          <Button
            type="button"
            variant="outline"
            size="sm"
            className="w-full justify-center text-destructive hover:text-destructive"
            onClick={() => clearQueueMutation.mutate()}
            disabled={status.queued === 0 || clearQueueMutation.isPending}
          >
            <Trash2 className="mr-2 h-3.5 w-3.5" />
            {clearQueueMutation.isPending ? "Clearing queue..." : "Clear queue"}
          </Button>
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

function WindowCard({
  label,
  totals,
}: {
  label: string
  totals: {
    prompt_tokens: number
    completion_tokens: number
    total_tokens: number
  }
}) {
  return (
    <div className="rounded-md bg-muted/60 px-2.5 py-2">
      <p className="font-medium text-muted-foreground">{label}</p>
      <p className="mt-0.5 text-sm font-semibold">
        {formatTokens(totals.total_tokens)}
      </p>
      <p className="mt-0.5 text-[11px] text-muted-foreground">
        {formatTokens(totals.prompt_tokens)} in /{" "}
        {formatTokens(totals.completion_tokens)} out
      </p>
    </div>
  )
}
