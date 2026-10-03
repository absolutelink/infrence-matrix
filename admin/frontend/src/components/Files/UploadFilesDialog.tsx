import { useQueryClient } from "@tanstack/react-query"
import { Upload, X } from "lucide-react"
import { useRef, useState } from "react"
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
import { Label } from "@/components/ui/label"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"
import useCustomToast from "@/hooks/useCustomToast"
import { formatBytes } from "@/utils"
import { DEFAULT_PURPOSE, FILE_PURPOSES } from "./purposes"

interface UploadFilesDialogProps {
  isOpen: boolean
  onClose: () => void
}

export default function UploadFilesDialog({
  isOpen,
  onClose,
}: UploadFilesDialogProps) {
  const queryClient = useQueryClient()
  const { showSuccessToast, showErrorToast } = useCustomToast()
  const fileInputRef = useRef<HTMLInputElement>(null)
  const [files, setFiles] = useState<File[]>([])
  const [purpose, setPurpose] = useState<string>(DEFAULT_PURPOSE)
  const [isUploading, setIsUploading] = useState(false)
  const [isDragging, setIsDragging] = useState(false)

  const addFiles = (incoming: FileList | null) => {
    if (!incoming) return
    const list = Array.from(incoming)
    setFiles((prev) => {
      const seen = new Set(prev.map((f) => `${f.name}:${f.size}`))
      const merged = [...prev]
      for (const file of list) {
        const key = `${file.name}:${file.size}`
        if (!seen.has(key)) {
          merged.push(file)
          seen.add(key)
        }
      }
      return merged
    })
  }

  const removeFile = (index: number) => {
    setFiles((prev) => prev.filter((_, i) => i !== index))
  }

  const reset = () => {
    setFiles([])
    setPurpose(DEFAULT_PURPOSE)
    setIsDragging(false)
    if (fileInputRef.current) fileInputRef.current.value = ""
  }

  const handleUpload = async () => {
    if (files.length === 0) {
      showErrorToast("Select at least one file to upload")
      return
    }
    setIsUploading(true)
    let succeeded = 0
    const failures: string[] = []
    for (const file of files) {
      try {
        await V1FilesService.v1.uploadFile({
          body: { file, purpose },
        })
        succeeded += 1
      } catch (error: any) {
        failures.push(`${file.name}: ${error.message || "upload failed"}`)
      }
    }
    setIsUploading(false)

    if (succeeded > 0) {
      showSuccessToast(
        `${succeeded} file${succeeded === 1 ? "" : "s"} uploaded`,
      )
      queryClient.invalidateQueries({ queryKey: ["files"] })
    }
    if (failures.length > 0) {
      showErrorToast(failures.join("; "))
    }
    if (succeeded > 0) {
      reset()
      onClose()
    }
  }

  return (
    <Dialog
      open={isOpen}
      onOpenChange={(open) => {
        if (!open) {
          reset()
          onClose()
        }
      }}
    >
      <DialogContent>
        <DialogHeader>
          <DialogTitle>Upload Files</DialogTitle>
          <DialogDescription>
            Drag and drop one or more files, or click to browse.
          </DialogDescription>
        </DialogHeader>

        <div className="space-y-4">
          <div className="space-y-1.5">
            <Label className="text-sm">Purpose</Label>
            <Select value={purpose} onValueChange={setPurpose}>
              <SelectTrigger>
                <SelectValue placeholder="Select a purpose" />
              </SelectTrigger>
              <SelectContent>
                {FILE_PURPOSES.map((p) => (
                  <SelectItem key={p} value={p}>
                    {p}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>

          <button
            type="button"
            aria-label="Drop files here or click to browse"
            onClick={() => fileInputRef.current?.click()}
            onDragOver={(e) => {
              e.preventDefault()
              setIsDragging(true)
            }}
            onDragLeave={() => setIsDragging(false)}
            onDrop={(e) => {
              e.preventDefault()
              setIsDragging(false)
              addFiles(e.dataTransfer.files)
            }}
            className={`flex min-h-[140px] w-full cursor-pointer flex-col items-center justify-center gap-2 rounded-lg border-2 border-dashed p-4 text-center transition-colors ${
              isDragging
                ? "border-primary bg-primary/5 text-foreground"
                : "border-border text-muted-foreground hover:border-primary/50 hover:text-foreground"
            }`}
          >
            <Upload className="h-8 w-8" />
            <span className="font-medium">
              Drop files here, or click to browse
            </span>
            <span className="text-xs">Multiple files supported</span>
          </button>
          <input
            ref={fileInputRef}
            type="file"
            multiple
            className="hidden"
            onChange={(e) => addFiles(e.target.files)}
          />

          {files.length > 0 && (
            <ul className="space-y-1">
              {files.map((file, index) => (
                <li
                  key={`${file.name}-${file.size}-${index}`}
                  className="flex items-center justify-between rounded-md bg-muted px-3 py-2 text-sm"
                >
                  <span className="truncate">{file.name}</span>
                  <span className="ml-2 flex items-center gap-2 text-muted-foreground">
                    {formatBytes(file.size)}
                    <button
                      type="button"
                      aria-label={`Remove ${file.name}`}
                      onClick={() => removeFile(index)}
                      className="text-muted-foreground hover:text-foreground"
                    >
                      <X className="h-4 w-4" />
                    </button>
                  </span>
                </li>
              ))}
            </ul>
          )}
        </div>

        <DialogFooter>
          <Button
            variant="outline"
            onClick={() => {
              reset()
              onClose()
            }}
            disabled={isUploading}
          >
            Cancel
          </Button>
          <Button
            onClick={handleUpload}
            disabled={isUploading || files.length === 0}
          >
            {isUploading
              ? "Uploading..."
              : `Upload ${files.length > 0 ? files.length : ""}`.trim()}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}
