/**
 * Cursor-merged log feed for a single server instance or benchmark run.
 *
 * Live agent events (log.lines / benchmark.log) carry agent-side sequence
 * numbers; polled history (`?after=<cursor>`) uses the same cursors. Lines
 * are appended exactly once — never content-matched — and the visible tail
 * is preserved across WebSocket disconnects and reconnects.
 */
import { useCallback, useEffect, useRef, useState } from "react"

import { AgentsService } from "@/client"
import {
  type AgentEvent,
  acquireStream,
  type StreamStatus,
} from "@/hook/agentEventStream"
import {
  type FeedLine,
  mergeEntries,
  parseHistoryResponse,
  type RawEntry,
  toLines,
  unionLines,
} from "@/hook/logMerge"

const POLL_INTERVAL_MS = 5000

async function fetchHistory(
  agentId: string,
  kind: "server" | "benchmark",
  feedId: string,
  after: number | null,
) {
  const cursorParam = after === null ? "" : `&after=${after}`
  const path =
    kind === "server"
      ? `/servers/logs/${feedId}?lines=300${cursorParam}`
      : `/benchmarks/logs/${feedId}?lines=500${cursorParam}`
  const response = await AgentsService.sendCommand({
    path: { agent_id: agentId },
    body: { method: "GET", path },
  })
  return parseHistoryResponse(
    response.data as Parameters<typeof parseHistoryResponse>[0],
  )
}

export function useLogFeed(options: {
  agentId: string
  feedId: string
  kind: "server" | "benchmark"
  enabled: boolean
}) {
  const { agentId, feedId, kind, enabled } = options
  const [lines, setLines] = useState<FeedLine[]>([])
  const [connected, setConnected] = useState(false)

  const tailRef = useRef<FeedLine[]>([])
  const cursorRef = useRef(0)
  const bootstrappedRef = useRef(false)
  const refetchTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null)

  const publish = useCallback((next: FeedLine[], cursor: number) => {
    tailRef.current = next
    cursorRef.current = cursor
    setLines(next)
  }, [])

  /**
   * Pull history. Without a bootstrap, adopt the retained tail wholesale;
   * afterwards fetch only lines past the cursor and merge them in.
   */
  const refetch = useCallback(async () => {
    if (!enabled || !agentId || !feedId) return
    try {
      if (!bootstrappedRef.current) {
        const history = await fetchHistory(agentId, kind, feedId, null)
        // Live events may have arrived while the fetch was in flight; union
        // by seq so nothing is lost or duplicated.
        const combined = unionLines(history.lines, tailRef.current)
        const lastSeq = combined.length
          ? combined[combined.length - 1].seq + 1
          : 0
        publish(combined, Math.max(history.nextCursor, lastSeq))
        bootstrappedRef.current = true
        return
      }
      const history = await fetchHistory(
        agentId,
        kind,
        feedId,
        cursorRef.current,
      )
      if (history.nextCursor < cursorRef.current) {
        // Feed reset (agent restart): adopt the new feed from scratch.
        publish(history.lines, history.nextCursor)
        return
      }
      // A reported gap only means older lines were evicted; the returned
      // entries still start at the oldest retained line, so the cursor
      // merge recovers everything we have not seen yet.
      const merged = mergeEntries(
        tailRef.current,
        cursorRef.current,
        history.lines,
      )
      if (merged.tail !== tailRef.current) {
        publish(merged.tail, merged.cursor)
      }
    } catch {
      // Transient command failures retry on the next scheduled tick.
    }
  }, [agentId, feedId, kind, enabled, publish])

  const scheduleRefetch = useCallback(
    (delay = POLL_INTERVAL_MS) => {
      if (refetchTimerRef.current) clearTimeout(refetchTimerRef.current)
      refetchTimerRef.current = setTimeout(() => {
        refetchTimerRef.current = null
        void refetch()
      }, delay)
    },
    [refetch],
  )

  useEffect(() => {
    if (!enabled || !agentId || !feedId) return

    // Fresh feed identity: start from an empty tail.
    tailRef.current = []
    cursorRef.current = 0
    bootstrappedRef.current = false
    setLines([])

    const { stream, release } = acquireStream(agentId)

    const onEvents = (events: AgentEvent[]) => {
      for (const event of events) {
        const isServer = kind === "server" && event.event === "log.lines"
        const isBenchmark =
          kind === "benchmark" && event.event === "benchmark.log"
        if (!isServer && !isBenchmark) continue
        const id = isServer ? event.data.server_id : event.data.run_id
        if (id !== feedId) continue
        const seqStart = Number(event.data.seq_start ?? NaN)
        if (Number.isNaN(seqStart)) continue
        if (bootstrappedRef.current && seqStart > cursorRef.current) {
          // Missed lines (backend/agent blip): reconcile from history.
          scheduleRefetch(0)
          continue
        }
        const entries = isServer
          ? toLines(event.data.lines as RawEntry[] | undefined, seqStart)
          : [
              {
                seq: seqStart,
                stream: String(event.data.stream ?? "stdout"),
                line: String(event.data.line ?? ""),
              },
            ]
        const merged = mergeEntries(tailRef.current, cursorRef.current, entries)
        if (merged.tail !== tailRef.current) {
          publish(merged.tail, merged.cursor)
        }
      }
    }

    const onStatus = (status: StreamStatus) => {
      const live = status === "connected"
      setConnected(live)
      if (live) {
        // Close the gap opened while disconnected as soon as we're back.
        scheduleRefetch(0)
      }
    }

    stream.addListener(onEvents)
    stream.addStatusListener(onStatus)
    setConnected(stream.getStatus() === "connected")

    // Poll while disconnected, and always bootstrap.
    const ensure = () => {
      if (stream.getStatus() !== "connected" || !bootstrappedRef.current) {
        void refetch()
      }
    }
    ensure()
    const interval = setInterval(ensure, POLL_INTERVAL_MS)

    return () => {
      stream.removeListener(onEvents)
      stream.removeStatusListener(onStatus)
      clearInterval(interval)
      if (refetchTimerRef.current) {
        clearTimeout(refetchTimerRef.current)
        refetchTimerRef.current = null
      }
      release()
    }
  }, [agentId, feedId, kind, enabled, publish, scheduleRefetch, refetch])

  return { lines, connected, refetch }
}
