import { RefreshCw } from "lucide-react"
import { useEffect, useRef } from "react"

import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { useLogFeed } from "@/hook/useLogFeed"

interface Props {
  isOpen: boolean
  instance: { id: string; run_id?: string | null; agent_id: string }
}

export function BenchmarkLogs({ isOpen, instance }: Props) {
  const logRef = useRef<HTMLDivElement>(null)
  const runId = instance.run_id ?? instance.id

  const { lines, connected, refetch } = useLogFeed({
    agentId: instance.agent_id,
    feedId: runId,
    kind: "benchmark",
    enabled: isOpen,
  })

  useEffect(() => {
    if (logRef.current) logRef.current.scrollTop = logRef.current.scrollHeight
  }, [])

  return (
    <div className="flex h-full min-h-0 flex-col p-3">
      <div className="flex items-center gap-2 pb-2 font-semibold">
        Benchmark logs{" "}
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
      </div>
      <div
        ref={logRef}
        className="flex-1 overflow-y-auto rounded-md bg-black p-4 font-mono text-xs leading-relaxed text-green-400"
      >
        {lines.length === 0 && (
          <div className="text-muted-foreground">
            {connected
              ? "Connected - waiting for benchmark output..."
              : "Loading benchmark logs..."}
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
