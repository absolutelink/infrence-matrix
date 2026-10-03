import { useQueryClient } from "@tanstack/react-query"
import { useState } from "react"
import { V1FilesService } from "@/client"
import { Button } from "@/components/ui/button"
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog"
import useCustomToast from "@/hooks/useCustomToast"

interface DeleteFileProps {
  isOpen: boolean
  onClose: () => void
  fileId: string
  filename: string
}

export default function DeleteFile({
  isOpen,
  onClose,
  fileId,
  filename,
}: DeleteFileProps) {
  const queryClient = useQueryClient()
  const { showSuccessToast, showErrorToast } = useCustomToast()
  const [isDeleting, setIsDeleting] = useState(false)

  const deleteFile = async () => {
    setIsDeleting(true)
    try {
      await V1FilesService.v1.deleteFile({ path: { file_id: fileId } })
      showSuccessToast("File deleted successfully")
      onClose()
      queryClient.invalidateQueries({ queryKey: ["files"] })
    } catch (error: any) {
      showErrorToast(error.message || "Failed to delete file")
    } finally {
      setIsDeleting(false)
    }
  }

  return (
    <Dialog open={isOpen} onOpenChange={onClose}>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>Delete File</DialogTitle>
          <DialogDescription>
            Are you sure you want to delete{" "}
            <span className="font-medium text-foreground">{filename}</span>?
            This action cannot be undone.
          </DialogDescription>
        </DialogHeader>
        <DialogFooter>
          <Button variant="outline" onClick={onClose} disabled={isDeleting}>
            Cancel
          </Button>
          <Button
            onClick={deleteFile}
            disabled={isDeleting}
            variant="destructive"
          >
            {isDeleting ? "Deleting..." : "Delete"}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}
