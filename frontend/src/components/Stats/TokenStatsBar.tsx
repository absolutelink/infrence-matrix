import {
  ChevronDown,
  Coins,
  Gauge,
  MemoryStick,
  Rocket,
  Zap,
} from "lucide-react"

import { useTokenStats } from "@/hook/useTokenStats"

const formatTokens = (n: number) => {
  if (n >= 1_000_000_000) return `${(n / 1_000_000_000).toFixed(1)}B`
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`
  if (n >= 1_000) return `${(n / 1_000).toFixed(1)}k`
  return `${n}`
}

const formatRate = (rate: number | null | undefined) =>
  rate === null || rate === undefined ? "—" : rate.toFixed(1)

export function TokenStatsBar() {
  const { data } = useTokenStats()

  if (!data) {
    return null
  }

  const { global, servers } = data
  const live = global.live
  const day = global.last_24h

  return (
    <details className="group relative">
      <summary className="flex cursor-pointer list-none items-center gap-3 rounded-md px-2 py-1.5 text-xs text-muted-foreground transition-colors hover:bg-muted hover:text-foreground">
        <span
          className="flex items-center gap-1.5"
          title="Decode tokens/sec (last 60s)"
        >
          <Zap className="h-3.5 w-3.5 text-emerald-500" />
          {formatRate(live.decode_tokens_per_second)} tok/s
        </span>
        <span
          className="flex items-center gap-1.5"
          title="Prefill tokens/sec (last 60s)"
        >
          <Rocket className="h-3.5 w-3.5 text-sky-500" />
          {formatRate(live.prefill_tokens_per_second)} tok/s
        </span>
        <span
          className="flex items-center gap-1.5"
          title="New (uncached) input + generated output tokens in the last 24 hours"
        >
          <Coins className="h-3.5 w-3.5 text-amber-500" />
          {formatTokens(day.prompt_tokens)} in /{" "}
          {formatTokens(day.completion_tokens)} out
        </span>
        <ChevronDown className="h-3.5 w-3.5 transition-transform group-open:rotate-180" />
      </summary>
      <div className="absolute left-1/2 top-full z-20 mt-2 w-[26rem] -translate-x-1/2 rounded-lg border bg-popover p-4 text-popover-foreground shadow-lg">
        <div className="mb-3 flex items-center justify-between">
          <div>
            <p className="text-sm font-medium">Token throughput</p>
            <p className="text-xs text-muted-foreground">
              Live rates over the last {live.window_seconds}s of completed
              requests
            </p>
          </div>
          <Gauge className="h-4 w-4 text-muted-foreground" />
        </div>

        <div className="mb-3 grid grid-cols-3 gap-2 text-xs">
          <WindowCard label="Last 24h" totals={global.last_24h} />
          <WindowCard label="Last 7 days" totals={global.last_7d} />
          <WindowCard label="Last 30 days" totals={global.last_30d} />
        </div>

        <div className="space-y-2">
          <p className="text-xs font-medium text-muted-foreground">
            Servers by decode speed
          </p>
          {servers.length === 0 ? (
            <p className="text-xs text-muted-foreground">
              No token usage recorded in the last 30 days
            </p>
          ) : (
            servers.map((server) => (
              <div
                key={server.id}
                className="rounded-md bg-muted/60 px-2.5 py-2"
              >
                <div className="flex items-center justify-between text-xs">
                  <span className="truncate font-medium">{server.alias}</span>
                  <span className="flex items-center gap-1.5 text-muted-foreground">
                    <Zap className="h-3 w-3 text-emerald-500" />
                    {formatRate(server.decode_tokens_per_second)} tok/s
                    <Rocket className="ml-1 h-3 w-3 text-sky-500" />
                    {formatRate(server.prefill_tokens_per_second)} tok/s
                  </span>
                </div>
                <div className="mt-1 flex items-center justify-between text-[11px] text-muted-foreground">
                  <span className="flex items-center gap-1.5">
                    <MemoryStick className="h-3 w-3" />
                    7d: {formatTokens(server.last_7d.total_tokens)}
                  </span>
                  <span>30d: {formatTokens(server.last_30d.total_tokens)}</span>
                </div>
              </div>
            ))
          )}
        </div>
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
