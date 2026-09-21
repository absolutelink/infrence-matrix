import { useQueryClient } from "@tanstack/react-query"
import { useState } from "react"
import { ModelsService } from "@/client"
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

interface DeleteModelProps {
  isOpen: boolean
  onClose: () => void
  modelId: string
}

export default function DeleteModel({
  isOpen,
  onClose,
  modelId,
}: DeleteModelProps) {
  const queryClient = useQueryClient()
  const { showSuccessToast, showErrorToast } = useCustomToast()
  const [isDeleting, setIsDeleting] = useState(false)

  const deleteModel = async () => {
    setIsDeleting(true)
    try {
      await ModelsService.deleteModel({ path: { id: modelId } })
      showSuccessToast("Model deleted successfully")
      onClose()
      queryClient.invalidateQueries({ queryKey: ["models"] })
    } catch (error: any) {
      showErrorToast(error.message || "Failed to delete model")
    } finally {
      setIsDeleting(false)
    }
  }

  return (
    <Dialog open={isOpen} onOpenChange={onClose}>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>Delete Model</DialogTitle>
          <DialogDescription>
            Are you sure you want to delete this model? This action cannot be
            undone.
          </DialogDescription>
        </DialogHeader>
        <DialogFooter>
          <Button variant="outline" onClick={onClose} disabled={isDeleting}>
            Cancel
          </Button>
          <Button
            onClick={deleteModel}
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
