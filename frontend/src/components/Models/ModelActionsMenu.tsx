import { MoreHorizontal, Pencil, Trash2 } from "lucide-react"
import { type Model } from "@/client"

import { Button } from "@/components/ui/button"
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu"
import DeleteModel from "./DeleteModel"
import { useState } from "react"
import EditModel from "./EditModel"

interface ModelActionsMenuProps {
  model: Model
}

export default function ModelActionsMenu({ model }: ModelActionsMenuProps) {
  const [showEditDialog, setShowEditDialog] = useState(false)
  const [showDeleteDialog, setShowDeleteDialog] = useState(false)

  return (
    <>
      <DropdownMenu>
        <DropdownMenuTrigger asChild>
          <Button variant="ghost" className="h-8 w-8 p-0">
            <span className="sr-only">Open menu</span>
            <MoreHorizontal className="h-4 w-4" />
          </Button>
        </DropdownMenuTrigger>
        <DropdownMenuContent align="end">
          <DropdownMenuItem onClick={() => setShowEditDialog(true)}>
            <Pencil className="mr-2 h-4 w-4" />
            Edit
          </DropdownMenuItem>
          <DropdownMenuSeparator />
          <DropdownMenuItem
            onClick={() => setShowDeleteDialog(true)}
            className="text-destructive"
          >
            <Trash2 className="mr-2 h-4 w-4" />
            Delete
          </DropdownMenuItem>
        </DropdownMenuContent>
      </DropdownMenu>

      <EditModel
        isOpen={showEditDialog}
        onClose={() => setShowEditDialog(false)}
        model={model}
      />
      <DeleteModel
        isOpen={showDeleteDialog}
        onClose={() => setShowDeleteDialog(false)}
        modelId={model.id}
      />
    </>
  )
}
