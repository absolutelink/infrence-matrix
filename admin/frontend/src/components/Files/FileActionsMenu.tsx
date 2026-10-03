import { Download, MoreHorizontal, Trash2 } from "lucide-react"
import { useState } from "react"
import type { FileData } from "@/client"
import { Button } from "@/components/ui/button"
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu"
import { getApiUrl } from "@/utils"
import DeleteFile from "./DeleteFile"

interface FileActionsMenuProps {
  file: FileData
}

export default function FileActionsMenu({ file }: FileActionsMenuProps) {
  const [showDeleteDialog, setShowDeleteDialog] = useState(false)

  const handleDownload = () => {
    const url = `${getApiUrl()}/v1/files/${file.id}/content`
    window.open(url, "_blank", "noopener,noreferrer")
  }

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
          <DropdownMenuItem onClick={handleDownload}>
            <Download className="mr-2 h-4 w-4" />
            Download
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

      <DeleteFile
        isOpen={showDeleteDialog}
        onClose={() => setShowDeleteDialog(false)}
        fileId={file.id}
        filename={file.filename}
      />
    </>
  )
}
