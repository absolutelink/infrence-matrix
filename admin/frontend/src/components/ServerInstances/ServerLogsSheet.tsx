import { RefreshCw } from "lucide-react"
import { useEffect, useRef, useState } from "react"

import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { useLogFeed } from "@/hook/useLogFeed"

interface ServerLogsSheetProps {
  isOpen: boolean
  instance: {
    id: string
    model_name?: string | null
    agent_id: string
    agent_name?: string | null
    agent_host?: string | null
    agent_port?: number | null
  }
}

export function ServerLogsSheet({ isOpen, instance }: ServerLogsSheetProps) {
  const logRef = useRef<HTMLDivElement>(null)
  const [autoScroll, setAutoScroll] = useState(true)

  const { lines, connected, refetch } = useLogFeed({
    agentId: instance.agent_id,
    feedId: instance.id,
    kind: "server",
    enabled: isOpen,
  })

  useEffect(() => {
    if (logRef.current && autoScroll && lines.length > 0) {
      logRef.current.scrollTop = logRef.current.scrollHeight
    }
  }, [autoScroll, lines.length])

  return (
    <div className="flex h-full min-h-0 flex-col p-3">
      <div className="flex items-center justify-between gap-2 pb-2">
        <h2 className="flex items-center gap-2 font-semibold">
          Logs - {instance.model_name || instance.id}
          <Badge variant={connected ? "default" : "secondary"}>
            {connected ? "Live" : "Polling"}
          </Badge>
          <Button
            variant="ghost"
            size="icon"
            className="h-6 w-6"
            onClick={() => refetch()}
          >
            <RefreshCw className="h-4 w-4" />
          </Button>
        </h2>
        <div className="text-sm text-muted-foreground">
          llama.cpp output from {instance.agent_name || "agent"}
          {connected ? " (live tail)" : " (refreshed every 5 seconds)"}
        </div>
      </div>

      <label className="flex items-center gap-2 pb-2 text-sm">
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
        {lines.length === 0 && (
          <div className="text-muted-foreground">
            {connected
              ? "Connected - waiting for log output..."
              : "Loading log history..."}
          </div>
        )}

        {lines.map((entry) => (
          <div
            key={entry.seq}
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
    </div>
  )
}
