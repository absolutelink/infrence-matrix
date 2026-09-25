import { Badge } from "@/components/ui/badge"
import {
  Sheet,
  SheetContent,
  SheetDescription,
  SheetHeader,
  SheetTitle,
} from "@/components/ui/sheet"
import { useAgentEvents } from "@/hook/useAgentEvents"

interface AgentEventsSheetProps {
  isOpen: boolean
  onClose: () => void
  agentId: string
  agentName: string
}

const eventColors: Record<
  string,
  "default" | "secondary" | "destructive" | "outline"
> = {
  "server.started": "default",
  "server.stopped": "secondary",
  "server.error": "destructive",
  "server.health": "outline",
  "download.completed": "default",
  "download.failed": "destructive",
  "download.started": "default",
  "download.progress": "outline",
  "gpu.usage": "outline",
}

function formatEvent(e: {
  event: string
  data: Record<string, unknown>
}): string {
  const d = e.data
  switch (e.event) {
    case "server.started":
      return `Server ${d.server_id} started on port ${d.port}`
    case "server.stopped":
      return `Server ${d.server_id} stopped`
    case "server.error":
      return `Server ${d.server_id} error: ${d.error}`
    case "server.health":
      return `Server ${d.server_id} is ${d.status}${d.error ? `: ${d.error}` : ""}`
    case "download.started":
      return `Download started: ${d.filename} (${d.repo_id})`
    case "download.progress":
      return `Download ${d.filename}: ${d.progress_percent}% (${d.bytes_downloaded}/${d.total_bytes ?? "?"} bytes)`
    case "download.completed":
      return `Download complete: ${d.filename}`
    case "download.failed":
      return `Download failed: ${d.filename} - ${d.error}`
    case "gpu.usage":
      return `GPU: ${(Number(d.vram_used) / 1073741824).toFixed(1)} GB VRAM, ${Number(d.utilization).toFixed(0)}% util`
    default:
      return JSON.stringify(d)
  }
}

export function AgentEventsSheet({
  isOpen,
  onClose,
  agentId,
  agentName,
}: AgentEventsSheetProps) {
  const { events, connected } = useAgentEvents(agentId, isOpen)

  return (
    <Sheet open={isOpen} onOpenChange={(open) => !open && onClose()}>
      <SheetContent side="right" className="flex w-full flex-col sm:max-w-2xl">
        <SheetHeader>
          <SheetTitle className="flex items-center gap-2">
            Events - {agentName}
            <Badge variant={connected ? "default" : "secondary"}>
              {connected ? "Live" : "Connecting..."}
            </Badge>
          </SheetTitle>
          <SheetDescription>
            Real-time events from this agent: servers, downloads, GPU usage
          </SheetDescription>
        </SheetHeader>

        <div className="flex-1 overflow-y-auto p-4">
          {!connected && events.length === 0 && (
            <div className="text-muted-foreground">
              Waiting for events... (make sure the agent is online)
            </div>
          )}

          {connected && events.length === 0 && (
            <div className="text-muted-foreground">
              Connected. Waiting for events... Try starting a server or
              downloading a model on this agent.
            </div>
          )}

          <div className="space-y-2">
            {[...events].reverse().map((e, i) => (
              <div
                key={`${e.timestamp}-${i}`}
                className="rounded-md border p-2 text-sm"
              >
                <div className="flex items-center gap-2">
                  <Badge variant={eventColors[e.event] || "outline"}>
                    {e.event}
                  </Badge>
                  <span className="text-xs text-muted-foreground">
                    {new Date(e.timestamp).toLocaleTimeString()}
                  </span>
                </div>
                <div className="mt-1 text-muted-foreground">
                  {formatEvent(e)}
                </div>
              </div>
            ))}
          </div>
        </div>
      </SheetContent>
    </Sheet>
  )
}
