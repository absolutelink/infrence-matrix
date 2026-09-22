import { useQuery } from "@tanstack/react-query"
import { RefreshCw } from "lucide-react"
import { useEffect, useRef, useState } from "react"

import { AgentsService } from "@/client"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import {
  Sheet,
  SheetContent,
  SheetDescription,
  SheetHeader,
  SheetTitle,
} from "@/components/ui/sheet"
import { useAgentEvents } from "@/hook/useAgentEvents"

interface ServerLogsSheetProps {
  isOpen: boolean
  onClose: () => void
  instance: {
    id: string
    model_name?: string | null
    agent_id: string
    agent_name?: string | null
    agent_host?: string | null
    agent_port?: number | null
  }
}

type LogLine = { stream: "stdout" | "stderr"; line: string }

const MAX_TAIL_LINES = 500

export function ServerLogsSheet({
  isOpen,
  onClose,
  instance,
}: ServerLogsSheetProps) {
  const logRef = useRef<HTMLDivElement>(null)
  const [autoScroll, setAutoScroll] = useState(true)

  // Live tail via the agent event stream
  const { events, connected } = useAgentEvents(instance.agent_id, isOpen)
  const [tail, setTail] = useState<LogLine[]>([])
  // Marks which historical lines are already seeded from the poll so live
  // lines are only appended once.
  const [seeded, setSeeded] = useState(false)

  useEffect(() => {
    setTail([])
    setSeeded(false)
  }, [])

  useEffect(() => {
    const newLines: LogLine[] = []
    for (const event of events) {
      if (
        event.event === "log.lines" &&
        (event.data as { server_id?: string }).server_id === instance.id
      ) {
        const lines = ((event.data as { lines?: LogLine[] }).lines ??
          []) as LogLine[]
        newLines.push(...lines)
      }
    }
    if (newLines.length > 0) {
      setTail((prev) => {
        const merged = [...prev, ...newLines]
        return merged.length > MAX_TAIL_LINES
          ? merged.slice(-MAX_TAIL_LINES)
          : merged
      })
    }
  }, [events, instance.id])

  // History: fetch the last 100 lines from the agent's buffer and seed the
  // tail before (or alongside) live lines.
  const historyQuery = useQuery({
    queryKey: ["server-logs-history", instance.id],
    queryFn: async () => {
      const response = await AgentsService.sendCommand({
        path: { agent_id: instance.agent_id },
        body: {
          method: "GET",
          path: `/servers/logs/${instance.id}?lines=100`,
        },
      })
      return response.data as {
        status?: string
        stdout?: string[]
        stderr?: string[]
      }
    },
    enabled: isOpen && Boolean(instance.agent_id),
    refetchOnWindowFocus: false,
  })

  useEffect(() => {
    if (!historyQuery.data || !connected) {
      return
    }
    if (seeded) {
      return
    }
    const history: LogLine[] = [
      ...(historyQuery.data.stdout ?? []).map((line) => ({
        stream: "stdout" as const,
        line,
      })),
      ...(historyQuery.data.stderr ?? []).map((line) => ({
        stream: "stderr" as const,
        line,
      })),
    ]
    if (history.length > 0) {
      setTail((prev) => {
        const merged = [...history, ...prev]
        return merged.length > MAX_TAIL_LINES
          ? merged.slice(-MAX_TAIL_LINES)
          : merged
      })
    }
    setSeeded(true)
  }, [historyQuery.data, connected, seeded])

  useEffect(() => {
    if (logRef.current && autoScroll) {
      logRef.current.scrollTop = logRef.current.scrollHeight
    }
  }, [autoScroll])

  const fallbackLines: LogLine[] = connected
    ? []
    : [
        ...(historyQuery.data?.stdout ?? []).map((line) => ({
          stream: "stdout" as const,
          line,
        })),
        ...(historyQuery.data?.stderr ?? []).map((line) => ({
          stream: "stderr" as const,
          line,
        })),
      ]

  const displayLines = connected ? tail : fallbackLines

  return (
    <Sheet open={isOpen} onOpenChange={(open) => !open && onClose()}>
      <SheetContent side="right" className="flex w-full flex-col sm:max-w-2xl">
        <SheetHeader>
          <SheetTitle className="flex items-center gap-2">
            Logs - {instance.model_name || instance.id}
            <Badge variant={connected ? "default" : "secondary"}>
              {connected ? "Live" : "Polling"}
            </Badge>
            <Button
              variant="ghost"
              size="icon"
              className="h-6 w-6"
              onClick={() => historyQuery.refetch()}
              disabled={historyQuery.isFetching}
            >
              <RefreshCw className="h-4 w-4" />
            </Button>
          </SheetTitle>
          <SheetDescription>
            llama.cpp output from {instance.agent_name || "agent"}
            {connected ? " (live tail)" : " (refreshed every 5 seconds)"}
          </SheetDescription>
        </SheetHeader>

        <label className="flex items-center gap-2 px-4 text-sm">
          <input
            type="checkbox"
            checked={autoScroll}
            onChange={(e) => setAutoScroll(e.target.checked)}
          />
          Auto-scroll
        </label>

        <div
          ref={logRef}
          className="flex-1 overflow-y-auto rounded-md bg-black p-4 font-mono text-xs leading-relaxed text-green-400"
        >
          {displayLines.length === 0 && (
            <div className="text-muted-foreground">
              {connected
                ? "Connected - waiting for log output..."
                : "No logs available yet."}
            </div>
          )}

          {connected && seeded && tail.length > 0 && (
            <div className="text-muted-foreground mb-2 text-[10px] uppercase tracking-wider border-b border-muted-foreground/20 pb-1">
              Last {Math.min(tail.length, 100)} lines before attaching live
            </div>
          )}

          {displayLines.map((entry, i) => (
            <div
              key={`${i}-${entry.line.slice(0, 12)}`}
              className={
                entry.stream === "stderr"
                  ? "whitespace-pre-wrap text-red-400"
                  : "whitespace-pre-wrap"
              }
            >
              {entry.line}
            </div>
          ))}
        </div>
      </SheetContent>
    </Sheet>
  )
}
