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

interface AgentLogsSheetProps {
  isOpen: boolean
  onClose: () => void
  agentId: string
  agentName: string
}

export function AgentLogsSheet({
  isOpen,
  onClose,
  agentId,
  agentName,
}: AgentLogsSheetProps) {
  const logRef = useRef<HTMLDivElement>(null)

  const logsQuery = useQuery({
    queryKey: ["agent-logs", agentId],
    queryFn: async () => {
      const servers = await AgentsService.sendCommand({
        path: { agent_id: agentId },
        body: { method: "GET", path: "/servers/list" },
      })

      const serverList =
        (servers.data as { servers?: Array<{ server_id: string }> })?.servers ??
        []

      const logs = await Promise.all(
        serverList.map(async (server) => {
          const logResponse = await AgentsService.sendCommand({
            path: { agent_id: agentId },
            body: { method: "GET", path: `/servers/logs/${server.server_id}` },
          })
          return {
            server_id: server.server_id,
            ...(logResponse.data as {
              status?: string
              stdout?: string[]
              stderr?: string[]
            }),
          }
        }),
      )

      return logs
    },
    enabled: isOpen,
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
            Logs - {agentName}
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
            llama.cpp server output, refreshed every 5 seconds
          </SheetDescription>
        </SheetHeader>

        <div
          ref={logRef}
          className="flex-1 overflow-y-auto rounded-md bg-black p-4 font-mono text-xs leading-relaxed text-green-400"
        >
          {logsQuery.isLoading && (
            <div className="text-muted-foreground">Loading logs...</div>
          )}

          {!logsQuery.isLoading && (logsQuery.data ?? []).length === 0 && (
            <div className="text-muted-foreground">
              No running llama.cpp servers on this agent. Start a server to see
              logs.
            </div>
          )}

          {(logsQuery.data ?? []).map((server) => (
            <div key={server.server_id} className="mb-6">
              <div className="mb-2 font-semibold text-white">
                Server: {server.server_id}
                {server.status && (
                  <span className="ml-2 text-xs text-muted-foreground">
                    ({server.status})
                  </span>
                )}
              </div>
              {(server.stdout ?? []).map((line, i) => (
                <div key={`o-${i}`} className="whitespace-pre-wrap">
                  {line}
                </div>
              ))}
              {(server.stderr ?? []).map((line, i) => (
                <div
                  key={`e-${i}`}
                  className="whitespace-pre-wrap text-red-400"
                >
                  {line}
                </div>
              ))}
            </div>
          ))}
        </div>
      </SheetContent>
    </Sheet>
  )
}
