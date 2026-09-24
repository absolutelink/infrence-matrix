import { useEffect, useRef, useState } from "react"
import { Badge } from "@/components/ui/badge"
import { useAgentEvents } from "@/hook/useAgentEvents"

interface Props {
  isOpen: boolean
  instance: { id: string; run_id?: string | null; agent_id: string }
}

export function BenchmarkLogs({ isOpen, instance }: Props) {
  const logRef = useRef<HTMLDivElement>(null)
  const [lines, setLines] = useState<Array<{ stream: string; line: string }>>(
    [],
  )
  const { events, connected } = useAgentEvents(instance.agent_id, isOpen)

  useEffect(() => {
    const next: Array<{ stream: string; line: string }> = []
    for (const event of events) {
      if (
        event.event !== "benchmark.log" ||
        event.data.run_id !== (instance.run_id ?? instance.id)
      )
        continue
      next.push({
        stream: String(event.data.stream ?? "stdout"),
        line: String(event.data.line ?? ""),
      })
    }
    setLines(next.slice(-500))
  }, [events, instance.id, instance.run_id])

  useEffect(() => {
    if (logRef.current) logRef.current.scrollTop = logRef.current.scrollHeight
  }, [])

  return (
    <div className="flex h-full min-h-0 flex-col p-3">
      <div className="flex items-center gap-2 pb-2 font-semibold">
        Benchmark logs{" "}
        <Badge variant={connected ? "default" : "secondary"}>
          {connected ? "Live" : "Disconnected"}
        </Badge>
      </div>
      <div
        ref={logRef}
        className="flex-1 overflow-y-auto rounded-md bg-black p-4 font-mono text-xs leading-relaxed text-green-400"
      >
        {lines.length === 0 && (
          <div className="text-muted-foreground">
            {connected
              ? "Connected - waiting for benchmark output..."
              : "No live log connection."}
          </div>
        )}
        {lines.map((entry, index) => (
          <div
            key={`${index}-${entry.line.slice(0, 12)}`}
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
