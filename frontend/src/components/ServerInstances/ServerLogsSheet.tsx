import { useQuery } from "@tanstack/react-query"
import { RefreshCw } from "lucide-react"
import { useEffect, useRef } from "react"

import { AgentsService } from "@/client"
import { Button } from "@/components/ui/button"
import {
  Sheet,
  SheetContent,
  SheetDescription,
  SheetHeader,
  SheetTitle,
} from "@/components/ui/sheet"

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

export function ServerLogsSheet({
  isOpen,
  onClose,
  instance,
}: ServerLogsSheetProps) {
  const logRef = useRef<HTMLDivElement>(null)

  const logsQuery = useQuery({
    queryKey: ["server-logs", instance.id],
    queryFn: async () => {
      const response = await AgentsService.sendCommand({
        path: { agent_id: instance.agent_id },
        body: {
          method: "GET",
          path: `/servers/logs/${instance.id}`,
        },
      })
      return response.data as {
        status?: string
        stdout?: string[]
        stderr?: string[]
      }
    },
    enabled: isOpen && Boolean(instance.agent_id),
    refetchInterval: 5000,
  })

  useEffect(() => {
    if (logRef.current) {
      logRef.current.scrollTop = logRef.current.scrollHeight
    }
  }, [])

  return (
    <Sheet open={isOpen} onOpenChange={(open) => !open && onClose()}>
      <SheetContent side="right" className="flex w-full flex-col sm:max-w-2xl">
        <SheetHeader>
          <SheetTitle className="flex items-center gap-2">
            Logs - {instance.model_name || instance.id}
            <Button
              variant="ghost"
              size="icon"
              className="h-6 w-6"
              onClick={() => logsQuery.refetch()}
              disabled={logsQuery.isFetching}
            >
              <RefreshCw className="h-4 w-4" />
            </Button>
          </SheetTitle>
          <SheetDescription>
            llama.cpp output from {instance.agent_name || "agent"}, refreshed
            every 5 seconds
          </SheetDescription>
        </SheetHeader>

        <div
          ref={logRef}
          className="flex-1 overflow-y-auto rounded-md bg-black p-4 font-mono text-xs leading-relaxed text-green-400"
        >
          {logsQuery.isLoading && (
            <div className="text-muted-foreground">Loading logs...</div>
          )}

          {!logsQuery.isLoading && !logsQuery.data && (
            <div className="text-muted-foreground">
              No logs available for this server.
            </div>
          )}

          {(logsQuery.data?.stdout ?? []).map((line, i) => (
            <div key={`o-${i}`} className="whitespace-pre-wrap">
              {line}
            </div>
          ))}
          {(logsQuery.data?.stderr ?? []).map((line, i) => (
            <div key={`e-${i}`} className="whitespace-pre-wrap text-red-400">
              {line}
            </div>
          ))}
        </div>
      </SheetContent>
    </Sheet>
  )
}
