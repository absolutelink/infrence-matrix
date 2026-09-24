import type { ColumnDef } from "@tanstack/react-table"
import type { Model } from "@/client"
import { formatBytes } from "@/utils"
import ModelActionsMenu from "./ModelActionsMenu"

export const columns: ColumnDef<Model>[] = [
  {
    accessorKey: "name",
    header: "Name",
    cell: ({ row }) => (
      <div className="font-medium">{row.getValue("name")}</div>
    ),
  },
  {
    accessorKey: "architecture",
    header: "Architecture",
    cell: ({ row }) => (
      <div>
        <div>{row.getValue("architecture")}</div>
        <div className="text-xs text-muted-foreground">
          {row.original.model_type || "llm"}
        </div>
      </div>
    ),
  },
  {
    accessorKey: "quantization",
    header: "Quantization",
    cell: ({ row }) => (
      <div className="font-mono text-xs">{row.getValue("quantization")}</div>
    ),
  },
  {
    accessorKey: "size_bytes",
    header: "Size",
    cell: ({ row }) => {
      const size = row.getValue("size_bytes") as number
      return <div>{formatBytes(size)}</div>
    },
  },
  {
    accessorKey: "parameter_count",
    header: "Parameters",
    cell: ({ row }) => {
      const params = row.getValue("parameter_count") as number | null
      if (!params) return "-"
      if (params >= 1_000_000_000) {
        return <div>{(params / 1_000_000_000).toFixed(1)}B</div>
      }
      if (params >= 1_000_000) {
        return <div>{(params / 1_000_000).toFixed(1)}M</div>
      }
      return <div>{params}</div>
    },
  },
  {
    accessorKey: "source",
    header: "Source",
    cell: ({ row }) => (
      <div className="capitalize">{row.getValue("source")}</div>
    ),
  },
  {
    id: "actions",
    cell: ({ row }) => <ModelActionsMenu model={row.original} />,
  },
]
