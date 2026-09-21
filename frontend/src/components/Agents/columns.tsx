import type { ColumnDef } from "@tanstack/react-table"
import { Activity, MoreHorizontal, Terminal, Trash2 } from "lucide-react"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu"

type Agent = {
  id: string
  name: string
  host: string
  port: number
  status: "online" | "offline" | "unreachable"
  websocket_connected: boolean
  gpu_info?: {
    name?: string
    vram_total?: number
    backend?: string
  } | null
  last_seen?: string | null
}

export const columns: ColumnDef<Agent>[] = [
  {
    accessorKey: "name",
    header: "Name",
    cell: ({ row }) => {
      const agent = row.original
      return (
        <div className="font-medium">
          {agent.name}
          {agent.websocket_connected && (
            <Badge variant="default" className="ml-2 h-5 text-xs">
              Connected
            </Badge>
          )}
        </div>
      )
    },
  },
  {
    accessorKey: "host",
    header: "Host",
    cell: ({ row }) => {
      const agent = row.original
      return (
        <div className="text-muted-foreground">
          {agent.host}:{agent.port}
        </div>
      )
    },
  },
  {
    accessorKey: "status",
    header: "Status",
    cell: ({ row }) => {
      const agent = row.original
      const statusColors: Record<
        string,
        "default" | "secondary" | "destructive" | "outline"
      > = {
        online: "default",
        offline: "secondary",
        unreachable: "destructive",
      }
      return (
        <Badge variant={statusColors[agent.status] || "outline"}>
          {agent.status}
        </Badge>
      )
    },
  },
  {
    accessorKey: "gpu_info",
    header: "GPU",
    cell: ({ row }) => {
      const agent = row.original
      const gpuName = agent.gpu_info?.name || "Unknown"
      const vram = agent.gpu_info?.vram_total
      const vramFormatted = vram
        ? `${(vram / 1073741824).toFixed(1)} GB`
        : "N/A"

      return (
        <div className="text-sm text-muted-foreground">
          <div className="font-medium">{gpuName}</div>
          <div className="text-xs">{vramFormatted} VRAM</div>
        </div>
      )
    },
  },
  {
    accessorKey: "last_seen",
    header: "Last Seen",
    cell: ({ row }) => {
      const agent = row.original
      if (!agent.last_seen) {
        return <span className="text-muted-foreground">Never</span>
      }

      const lastSeen = new Date(agent.last_seen)
      const now = new Date()
      const diff = now.getTime() - lastSeen.getTime()
      const minutes = Math.floor(diff / 60000)
      const hours = Math.floor(minutes / 60)

      if (minutes < 1) {
        return <span className="text-green-600 font-medium">Just now</span>
      }
      if (minutes < 60) {
        return <span>{minutes}m ago</span>
      }
      if (hours < 24) {
        return <span>{hours}h ago</span>
      }
      return <span>{lastSeen.toLocaleDateString()}</span>
    },
  },
  {
    id: "actions",
    cell: ({ row }) => {
      const agent = row.original

      return (
        <DropdownMenu>
          <DropdownMenuTrigger asChild>
            <Button variant="ghost" className="h-8 w-8 p-0">
              <span className="sr-only">Open menu</span>
              <MoreHorizontal className="h-4 w-4" />
            </Button>
          </DropdownMenuTrigger>
          <DropdownMenuContent align="end">
            <DropdownMenuLabel>Actions</DropdownMenuLabel>
            <DropdownMenuSeparator />
            <DropdownMenuItem
              onClick={() => navigator.clipboard.writeText(agent.id)}
            >
              Copy Agent ID
            </DropdownMenuItem>
            <DropdownMenuItem>
              <Terminal className="mr-2 h-4 w-4" />
              View Logs
            </DropdownMenuItem>
            <DropdownMenuItem>
              <Activity className="mr-2 h-4 w-4" />
              View Metrics
            </DropdownMenuItem>
            <DropdownMenuSeparator />
            <DropdownMenuItem className="text-destructive">
              <Trash2 className="mr-2 h-4 w-4" />
              Delete Agent
            </DropdownMenuItem>
          </DropdownMenuContent>
        </DropdownMenu>
      )
    },
  },
]
