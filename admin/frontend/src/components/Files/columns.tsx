import type { ColumnDef } from "@tanstack/react-table"
import type { FileData } from "@/client"
import { Badge } from "@/components/ui/badge"
import { formatBytes } from "@/utils"
import FileActionsMenu from "./FileActionsMenu"

export const columns: ColumnDef<FileData>[] = [
  {
    accessorKey: "filename",
    header: "Name",
    cell: ({ row }) => (
      <div className="font-medium truncate max-w-[280px]">
        {row.getValue("filename")}
      </div>
    ),
  },
  {
    accessorKey: "purpose",
    header: "Purpose",
    cell: ({ row }) => (
      <Badge variant="secondary">{row.getValue("purpose")}</Badge>
    ),
  },
  {
    accessorKey: "bytes",
    header: "Size",
    cell: ({ row }) => {
      const size = row.getValue("bytes") as number
      return <div>{formatBytes(size)}</div>
    },
  },
  {
    accessorKey: "status",
    header: "Status",
    cell: ({ row }) => {
      const status = row.getValue("status") as string
      const variant =
        status === "uploaded"
          ? "default"
          : status === "processing"
            ? "outline"
            : "destructive"
      return (
        <Badge variant={variant as "default" | "outline" | "destructive"}>
          {status}
        </Badge>
      )
    },
  },
  {
    accessorKey: "created_at",
    header: "Created",
    cell: ({ row }) => {
      const ts = row.getValue("created_at") as number
      return (
        <div className="text-sm text-muted-foreground">
          {new Date(ts * 1000).toLocaleString()}
        </div>
      )
    },
  },
  {
    id: "actions",
    cell: ({ row }) => <FileActionsMenu file={row.original} />,
  },
]
