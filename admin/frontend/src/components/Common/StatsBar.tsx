import { useMutation, useQueryClient } from "@tanstack/react-query"

import { AdminService } from "@/client"
import { Button } from "@/components/ui/button"
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover"
import { Separator } from "@/components/ui/separator"
import { Skeleton } from "@/components/ui/skeleton"
import {
  schedulerStatsKeys,
  useFleetMetrics,
  useSchedulerStats,
  useUsageStats,
} from "@/hooks/useAdminData"
import useCustomToast from "@/hooks/useCustomToast"
import { useIsMobile } from "@/hooks/useMobile"
import { extractError } from "@/lib/errors"
import type {
  ClearQueueResult,
  SchedulerStatsDefinition,
  UsageSample,
} from "@/types/admin"

// Phase 19: global live stats bar in the admin header. Three segments —
// rolling token rates, scheduler queue depth (with operator clear), and fleet
// VRAM/GPU. Each polls its own endpoint; values stay fresh without a reload.

function gib(bytes?: number): string {
  return ((bytes ?? 0) / 1024 ** 3).toFixed(1)
}

interface AliasRate {
  alias: string
  gen: number
  prompt: number
  requests: number
  completionTokens: number
}

// Group usage samples by definition alias and roll up per-model rates,
// matching the responses.tsx avg math (average over samples that have a
// non-zero generation rate).
function groupRatesByAlias(samples: UsageSample[]): AliasRate[] {
  const byAlias = new Map<string, UsageSample[]>()
  for (const s of samples) {
    const key = s.alias ?? "(unknown)"
    const bucket = byAlias.get(key)
    if (bucket) bucket.push(s)
    else byAlias.set(key, [s])
  }
  const rows: AliasRate[] = []
  for (const [alias, group] of byAlias) {
    const withRates = group.filter((s) => s.predicted_per_second > 0)
    const gen =
      withRates.length > 0
        ? withRates.reduce((a, s) => a + s.predicted_per_second, 0) /
          withRates.length
        : 0
    const prompt =
      withRates.length > 0
        ? withRates.reduce((a, s) => a + s.prompt_per_second, 0) /
          withRates.length
        : 0
    rows.push({
      alias,
      gen,
      prompt,
      requests: group.length,
      completionTokens: group.reduce((a, s) => a + s.completion_tokens, 0),
    })
  }
  return rows.sort((a, b) => b.requests - a.requests)
}

function RatesSegment() {
  const isMobile = useIsMobile()
  const { data, isLoading } = useUsageStats(50)
  const samples = data?.samples ?? []
  const withRates = samples.filter((s) => s.predicted_per_second > 0)
  const avgGen =
    withRates.length > 0
      ? withRates.reduce((a, s) => a + s.predicted_per_second, 0) /
        withRates.length
      : 0
  const avgPrompt =
    withRates.length > 0
      ? withRates.reduce((a, s) => a + s.prompt_per_second, 0) /
        withRates.length
      : 0
  const rows = groupRatesByAlias(samples)

  return (
    <Popover>
      <PopoverTrigger asChild>
        <Button
          variant="ghost"
          size="sm"
          className="h-8 gap-1.5 font-mono text-xs hover:bg-accent"
        >
          {isLoading && !data ? (
            <Skeleton className="h-3.5 w-24" />
          ) : (
            <>
              <span>{avgGen.toFixed(1)} tok/s</span>
              {!isMobile && (
                <span className="text-muted-foreground">
                  prompt {avgPrompt.toFixed(0)} t/s
                </span>
              )}
            </>
          )}
        </Button>
      </PopoverTrigger>
      <PopoverContent
        align="start"
        className="w-96 p-0"
        aria-label="Per-model token rates"
      >
        <div className="px-3 py-2 text-xs font-medium text-muted-foreground">
          Per-model rates · last {samples.length} requests
        </div>
        <Separator />
        {rows.length === 0 ? (
          <p className="px-3 py-6 text-center text-sm text-muted-foreground">
            No usage samples recorded yet.
          </p>
        ) : (
          <div className="max-h-80 overflow-y-auto">
            <div className="grid grid-cols-[1fr_auto_auto_auto] gap-x-3 gap-y-1 px-3 py-2 text-xs">
              <span className="font-medium text-muted-foreground">Model</span>
              <span className="text-right font-medium text-muted-foreground">
                gen t/s
              </span>
              <span className="text-right font-medium text-muted-foreground">
                prompt t/s
              </span>
              <span className="text-right font-medium text-muted-foreground">
                reqs · tok
              </span>
              {rows.map((r) => (
                <div
                  key={r.alias}
                  className="col-span-4 grid grid-cols-subgrid gap-x-3 border-t pt-1"
                >
                  <span className="truncate font-mono" title={r.alias}>
                    {r.alias}
                  </span>
                  <span className="text-right font-mono">
                    {r.gen.toFixed(1)}
                  </span>
                  <span className="text-right font-mono">
                    {r.prompt.toFixed(0)}
                  </span>
                  <span className="text-right font-mono text-muted-foreground">
                    {r.requests} · {r.completionTokens}
                  </span>
                </div>
              ))}
            </div>
          </div>
        )}
      </PopoverContent>
    </Popover>
  )
}

function QueueRow({ def }: { def: SchedulerStatsDefinition }) {
  const queryClient = useQueryClient()
  const { showSuccessToast, showErrorToast } = useCustomToast()
  const clear = useMutation({
    mutationFn: () =>
      AdminService.clearSchedulerQueue({ path: { alias: def.alias } }),
    onSuccess: (resp) => {
      const cleared =
        (resp?.data as unknown as ClearQueueResult | undefined)?.cleared ?? 0
      queryClient.invalidateQueries({ queryKey: schedulerStatsKeys.all })
      showSuccessToast(`Cleared ${cleared} waiter(s) from ${def.alias}.`)
    },
    onError: (err: Error) => showErrorToast(extractError(err)),
  })

  return (
    <div className="flex items-center gap-2 px-3 py-1.5 text-sm">
      <span
        className="min-w-0 flex-1 truncate font-mono text-xs"
        title={def.alias}
      >
        {def.alias}
      </span>
      <span className="flex shrink-0 items-center gap-1.5 font-mono text-xs text-muted-foreground">
        <span>q {def.queued}</span>
        <span>·</span>
        <span>a {def.active}</span>
      </span>
      <Button
        size="sm"
        variant="outline"
        className="h-7 shrink-0 px-2 text-xs"
        disabled={def.queued === 0 || clear.isPending}
        onClick={() => clear.mutate()}
      >
        {clear.isPending ? "Clearing…" : "Clear"}
      </Button>
    </div>
  )
}

function QueueSegment() {
  const queryClient = useQueryClient()
  const { showSuccessToast, showErrorToast } = useCustomToast()
  const { data, isLoading } = useSchedulerStats()
  const totals = data?.totals
  const defs = data?.definitions ?? []

  const clearAll = useMutation({
    mutationFn: () => AdminService.clearAllSchedulerQueues(),
    onSuccess: (resp) => {
      const cleared =
        (resp?.data as unknown as ClearQueueResult | undefined)?.cleared ?? 0
      queryClient.invalidateQueries({ queryKey: schedulerStatsKeys.all })
      showSuccessToast(`Cleared ${cleared} waiter(s) across all queues.`)
    },
    onError: (err: Error) => showErrorToast(extractError(err)),
  })

  return (
    <Popover>
      <PopoverTrigger asChild>
        <Button
          variant="ghost"
          size="sm"
          className="h-8 gap-1.5 font-mono text-xs hover:bg-accent"
        >
          {isLoading && !data ? (
            <Skeleton className="h-3.5 w-24" />
          ) : (
            <>
              <span>queue {totals?.queued ?? 0}</span>
              <span className="text-muted-foreground">
                · active {totals?.active ?? 0}
              </span>
            </>
          )}
        </Button>
      </PopoverTrigger>
      <PopoverContent
        align="start"
        className="w-96 p-0"
        aria-label="Scheduler queues"
      >
        <div className="px-3 py-2 text-xs font-medium text-muted-foreground">
          Scheduler queues
        </div>
        <Separator />
        {defs.length === 0 ? (
          <p className="px-3 py-6 text-center text-sm text-muted-foreground">
            No definitions.
          </p>
        ) : (
          <div className="max-h-80 overflow-y-auto py-1">
            {defs.map((d) => (
              <QueueRow key={d.alias} def={d} />
            ))}
          </div>
        )}
        <Separator />
        <div className="flex justify-end p-2">
          <Button
            size="sm"
            variant="secondary"
            className="h-7 px-2.5 text-xs"
            disabled={(totals?.queued ?? 0) === 0 || clearAll.isPending}
            onClick={() => clearAll.mutate()}
          >
            {clearAll.isPending ? "Clearing…" : "Clear all"}
          </Button>
        </div>
      </PopoverContent>
    </Popover>
  )
}

function VramSegment() {
  const { data, isLoading } = useFleetMetrics()
  if (isLoading && !data) {
    return <Skeleton className="h-3.5 w-32" />
  }
  const used = data?.vram.used_bytes ?? 0
  const total = data?.vram.total_bytes ?? 0
  const gpu = data?.gpu_utilization_percent ?? 0
  return (
    <span className="flex items-center gap-2 font-mono text-xs text-muted-foreground">
      <span title="Fleet VRAM used / total">
        VRAM {gib(used)}/{gib(total)} GiB
      </span>
      <span title="Fleet GPU utilization (mean of per-machine means)">
        GPU {gpu.toFixed(0)}%
      </span>
    </span>
  )
}

export function StatsBar() {
  const isMobile = useIsMobile()
  return (
    <div className="flex min-w-0 items-center gap-2">
      <RatesSegment />
      <Separator orientation="vertical" className="h-5" />
      <QueueSegment />
      {!isMobile && (
        <>
          <Separator orientation="vertical" className="h-5" />
          <VramSegment />
        </>
      )}
    </div>
  )
}

export default StatsBar
