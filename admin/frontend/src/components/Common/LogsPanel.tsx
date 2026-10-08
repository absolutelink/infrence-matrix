import { ArrowDownToLine, Download, Pause, Play, Search } from "lucide-react"
import { type ReactNode, useEffect, useMemo, useRef, useState } from "react"

import { StatusBadge } from "@/components/Common/StatusBadge"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { useInstanceLogs, useInstances } from "@/hooks/useAdminData"
import { cn } from "@/lib/utils"
import type { LogEntry, LogKind, ProviderInstance } from "@/types/admin"

// Rendered-line cap: matches the admin's Redis LOGS_CAP so the panel
// never holds more than the backend could return anyway. Simple tail
// cap instead of a virtualization dependency.
const MAX_LINES = 2000
const INITIAL_LIMIT = 500
// Within this many px of the bottom counts as "at bottom".
const AT_BOTTOM_PX = 24

function streamClass(entry: LogEntry): string {
  if (entry.stream === "stderr") {
    return "text-red-600 dark:text-red-400"
  }
  const t = entry.text
  if (/\b(ERROR|FATAL|CRITICAL)\b/.test(t)) {
    return "text-red-600 dark:text-red-300"
  }
  if (/\bWARN(ING)?\b/.test(t)) {
    return "text-amber-600 dark:text-amber-400"
  }
  if (/\b(DEBUG|TRACE)\b/.test(t)) {
    return "text-muted-foreground"
  }
  return ""
}

function highlight(text: string, query: string): ReactNode {
  const q = query.trim()
  if (!q) return text
  const lower = text.toLowerCase()
  const needle = q.toLowerCase()
  const parts: ReactNode[] = []
  let i = 0
  let key = 0
  for (;;) {
    const at = lower.indexOf(needle, i)
    if (at === -1) {
      parts.push(text.slice(i))
      break
    }
    if (at > i) parts.push(text.slice(i, at))
    parts.push(
      <mark
        key={key++}
        className="rounded-sm bg-yellow-500/40 text-inherit dark:bg-yellow-400/30"
      >
        {text.slice(at, at + needle.length)}
      </mark>,
    )
    i = at + needle.length
  }
  return parts
}

export function LogsPanel({
  instance,
  active,
}: {
  instance: ProviderInstance
  active: boolean
}) {
  // Live status badges come from the shared instances query (keyed by id)
  // so a tab opened earlier still reflects current agent/backend state;
  // the snapshot passed in is the fallback while that query is cold.
  const { data: allInstances = [] } = useInstances()
  const live = allInstances.find((i) => i.id === instance.id) ?? instance

  const [kind, setKind] = useState<LogKind>("backend")
  // Chronological accumulation (oldest index 0 → newest at the end).
  // The API returns newest-first; we reverse every batch before append.
  const [entries, setEntries] = useState<LogEntry[]>([])
  const [liveTail, setLiveTail] = useState(true)
  const [atBottom, setAtBottom] = useState(true)
  const [pendingNew, setPendingNew] = useState(0)
  const [showStdout, setShowStdout] = useState(true)
  const [showStderr, setShowStderr] = useState(true)
  const [query, setQuery] = useState("")
  const [gap, setGap] = useState(false)
  const [dropped, setDropped] = useState(0)
  const [skipped, setSkipped] = useState(0)

  const cursorRef = useRef(0)
  const atBottomRef = useRef(true)
  const scrollRef = useRef<HTMLDivElement>(null)
  const [since, setSince] = useState(0)

  // Reset-on-view-change (render-phase state adjustment with a STATE
  // prev-key, the replay-safe React docs pattern): switching kind resets
  // the cursor to 0 so the fresh initial fetch (since=0) reloads the
  // tail, and clears accumulated lines. A discarded render simply
  // re-derives from state on the retry.
  const viewKey = `${instance.id}|${kind}`
  const [prevViewKey, setPrevViewKey] = useState(viewKey)
  if (prevViewKey !== viewKey) {
    setPrevViewKey(viewKey)
    cursorRef.current = 0
    setSince(0)
    setEntries([])
    setPendingNew(0)
    setGap(false)
    setDropped(0)
    setSkipped(0)
    setAtBottom(true)
    atBottomRef.current = true
  }

  // Polling gating: `enabled: active` means only the active tab's panel
  // ever fetches — inactive tabs (kept mounted so their scrollback and
  // cursor survive a tab switch) issue no requests at all. `liveTail &&
  // active` keeps the ~2s refetch interval off for paused or background
  // tabs.
  const { data, isFetching, error } = useInstanceLogs(instance.id, {
    kind,
    since,
    limit: INITIAL_LIMIT,
    live: liveTail && active,
    enabled: active,
  })

  // Accumulate polled batches. entries arrive newest-first; filter to
  // seq > cursor, then merge into the chronological list via a
  // seq-keyed Map + sort so batches can never land out of order (a
  // kind switch can surface a cached window whose seqs interleave with
  // the freshly fetched ones). cursorRef only ever moves forward, never
  // regresses.
  useEffect(() => {
    if (!data) return
    const fresh = data.entries.filter((e) => e.seq > cursorRef.current)
    setGap(data.gap)
    setDropped(data.dropped)
    // H1: the backend trims to `limit` newest of everything unseen;
    // the difference is what we skipped between polls. Track the
    // cumulative total so the operator is told, not silently starved.
    const batchSkipped = Math.max(
      0,
      (data.unseen_total ?? 0) - data.entries.length,
    )
    if (batchSkipped > 0) setSkipped((s) => s + batchSkipped)
    if (fresh.length === 0) return
    const lastSeq = Math.max(...fresh.map((e) => e.seq))
    if (lastSeq > cursorRef.current) {
      cursorRef.current = lastSeq
      setSince(lastSeq)
    }
    // NOTE: counts all fresh lines, not the stream/search-filtered
    // subset — the badge is an ingest indicator, not a visible-count.
    if (!atBottomRef.current) setPendingNew((p) => p + fresh.length)
    setEntries((prev) => {
      const bySeq = new Map<number, LogEntry>()
      for (const e of prev) bySeq.set(e.seq, e)
      for (const e of fresh) bySeq.set(e.seq, e)
      const merged = [...bySeq.values()].sort((a, b) => a.seq - b.seq)
      return merged.length > MAX_LINES
        ? merged.slice(merged.length - MAX_LINES)
        : merged
    })
  }, [data])

  // Auto-scroll to bottom on new entries while pinned. (entries is
  // referenced so the effect re-runs after the list commits.)
  useEffect(() => {
    if (entries.length === 0) return
    const el = scrollRef.current
    if (el && atBottomRef.current) el.scrollTop = el.scrollHeight
  }, [entries])

  const onScroll = () => {
    const el = scrollRef.current
    if (!el) return
    const pinned =
      el.scrollHeight - el.scrollTop - el.clientHeight <= AT_BOTTOM_PX
    atBottomRef.current = pinned
    setAtBottom(pinned)
    if (pinned) setPendingNew(0)
  }

  const jumpToLatest = () => {
    const el = scrollRef.current
    if (el) el.scrollTop = el.scrollHeight
    atBottomRef.current = true
    setAtBottom(true)
    setPendingNew(0)
  }

  const visible = useMemo(() => {
    const q = query.trim().toLowerCase()
    return entries.filter((e) => {
      if (e.stream === "stderr" ? !showStderr : !showStdout) return false
      if (q && !e.text.toLowerCase().includes(q)) return false
      return true
    })
  }, [entries, query, showStdout, showStderr])

  const downloadTail = () => {
    const lines = visible
      .map((e) => `${e.ts} [${e.stream}] ${e.text}`)
      .join("\n")
    const blob = new Blob([`${lines}\n`], {
      type: "text/plain;charset=utf-8",
    })
    const url = URL.createObjectURL(blob)
    const a = document.createElement("a")
    a.href = url
    a.download = `${instance.alias ?? instance.id ?? "instance"}-${kind}.log`
    document.body.appendChild(a)
    a.click()
    a.remove()
    // Immediate revoke can cancel the download in Safari/Firefox.
    setTimeout(() => URL.revokeObjectURL(url), 1000)
  }

  return (
    <div className="flex min-h-0 flex-1 flex-col">
      <div className="flex flex-wrap items-center gap-2 border-b px-4 py-2">
        <Badge variant="outline" className="font-mono text-xs">
          {live.alias ?? live.id.slice(0, 8)}
        </Badge>
        {live.provider_type && (
          <Badge variant="secondary" className="font-mono text-xs">
            {live.provider_type}
          </Badge>
        )}
        <StatusBadge status={live.agent_status} />
        <StatusBadge status={live.backend_status} />
        <span className="font-mono text-xs text-muted-foreground">
          {live.machine_uid ?? "—"} · {live.id}
        </span>
      </div>

      <div className="flex flex-wrap items-center gap-2 border-b px-4 py-2">
        <Tabs value={kind} onValueChange={(v) => setKind(v as LogKind)}>
          <TabsList>
            <TabsTrigger value="backend">Backend</TabsTrigger>
            <TabsTrigger value="provider">Provider</TabsTrigger>
            <TabsTrigger value="all">All</TabsTrigger>
          </TabsList>
        </Tabs>

        <div className="flex items-center gap-1">
          <Button
            variant={showStdout ? "secondary" : "outline"}
            size="sm"
            onClick={() => setShowStdout((s) => !s)}
            title="Toggle stdout lines"
          >
            stdout
          </Button>
          <Button
            variant={showStderr ? "secondary" : "outline"}
            size="sm"
            className={cn(showStderr && "text-red-600 dark:text-red-400")}
            onClick={() => setShowStderr((s) => !s)}
            title="Toggle stderr lines"
          >
            stderr
          </Button>
        </div>

        <div className="relative min-w-40 flex-1">
          <Search className="pointer-events-none absolute top-1/2 left-2 size-4 -translate-y-1/2 text-muted-foreground" />
          <Input
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Search loaded lines…"
            className="h-8 pl-8 font-mono text-xs"
          />
        </div>

        <div className="flex items-center gap-1">
          <Button
            variant={liveTail ? "default" : "outline"}
            size="sm"
            onClick={() => setLiveTail((l) => !l)}
            title={liveTail ? "Pause live tailing" : "Resume live tailing"}
          >
            {liveTail ? <Pause /> : <Play />}
            {liveTail ? "Live" : "Paused"}
          </Button>
          <Button
            variant="outline"
            size="sm"
            onClick={downloadTail}
            disabled={visible.length === 0}
            title="Download visible entries as .log"
          >
            <Download />
            Download
          </Button>
        </div>
      </div>

      {gap && (
        <div className="border-b bg-amber-500/10 px-4 py-1.5 text-xs text-amber-700 dark:text-amber-400">
          Older log entries were dropped (ring/Redis cap).
          {dropped > 0 && (
            <span className="ml-1 font-mono">{dropped} lines lost</span>
          )}
        </div>
      )}
      {!gap && dropped > 0 && (
        <div className="border-b bg-muted/50 px-4 py-1.5 text-xs text-muted-foreground">
          Provider reported <span className="font-mono">{dropped}</span> dropped
          lines (buffer overflow upstream).
        </div>
      )}
      {skipped > 0 && (
        <div className="border-b bg-orange-500/10 px-4 py-1.5 text-xs text-orange-700 dark:text-orange-400">
          <span className="font-mono">{skipped}</span> newer entries skipped
          between refreshes — widen the tail or pause less.
        </div>
      )}
      {error && (
        <div className="border-b bg-destructive/10 px-4 py-1.5 text-xs text-destructive">
          {liveTail
            ? "Log fetch failed — retrying while live."
            : "Log fetch failed — Live paused, resume to retry."}
        </div>
      )}

      <div
        ref={scrollRef}
        onScroll={onScroll}
        className="relative min-h-0 flex-1 overflow-y-auto bg-zinc-950/95 p-3 font-mono text-xs leading-5 dark:bg-black"
      >
        {visible.length === 0 ? (
          <p className="p-2 text-muted-foreground">
            {isFetching
              ? "Loading logs…"
              : entries.length === 0
                ? "No log entries yet for this kind."
                : "No lines match the current filters."}
          </p>
        ) : (
          visible.map((e) => (
            <div
              key={`${kind}-${e.seq}`}
              className={cn("whitespace-pre-wrap break-all", streamClass(e))}
            >
              <span className="mr-2 text-muted-foreground select-none">
                {formatTs(e.ts)}
              </span>
              {e.stream === "stderr" && (
                <span className="mr-1 text-red-500/80 select-none">2&gt;</span>
              )}
              {highlight(e.text, query)}
            </div>
          ))
        )}
        {!atBottom && (
          <div
            className="sticky bottom-2 flex justify-center"
            aria-live="polite"
          >
            <Button
              size="sm"
              variant="secondary"
              className="shadow-lg"
              onClick={jumpToLatest}
            >
              <ArrowDownToLine />
              {pendingNew > 0
                ? `Jump to latest (${pendingNew} new)`
                : "Resume live / Jump to latest"}
            </Button>
          </div>
        )}
      </div>

      <div className="flex items-center justify-between border-t px-4 py-1.5 text-xs text-muted-foreground">
        <span>
          {visible.length} shown / {entries.length} loaded · cursor{" "}
          <span className="font-mono">{cursorRef.current}</span>
        </span>
      </div>
    </div>
  )
}

function formatTs(ts: string): string {
  if (!ts) return "--:--:--"
  const d = new Date(ts)
  if (Number.isNaN(d.getTime())) return ts.slice(0, 19)
  return d.toLocaleTimeString()
}
