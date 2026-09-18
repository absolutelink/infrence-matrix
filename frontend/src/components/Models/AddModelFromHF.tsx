import { useState, useEffect } from "react"
import { useMutation, useQueryClient } from "@tanstack/react-query"
import { X } from "lucide-react"

import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog"
import { Button } from "@/components/ui/button"
import { Label } from "@/components/ui/label"
import { Input } from "@/components/ui/input"
import { LoadingButton } from "@/components/ui/loading-button"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"
import { ModelsService } from "@/client"
import useCustomToast from "@/hooks/useCustomToast"

interface HuggingFaceModel {
  id: string
  modelId: string
  author: string
  downloads: number
  likes: number
}

interface AddModelFromHFProps {
  model: HuggingFaceModel
  onClose: () => void
}

interface GGUFFile {
  path: string
  size: number
}

export function AddModelFromHF({ model, onClose }: AddModelFromHFProps) {
  const queryClient = useQueryClient()
  const { showSuccessToast, showErrorToast } = useCustomToast()
  const [files, setFiles] = useState<GGUFFile[]>([])
  const [selectedFile, setSelectedFile] = useState<string>("")
  const [isLoadingFiles, setIsLoadingFiles] = useState(false)

  // Fetch GGUF files for this model
  useEffect(() => {
    const fetchFiles = async () => {
      setIsLoadingFiles(true)
      try {
        const token = localStorage.getItem("access_token")
        const baseUrl = import.meta.env.VITE_API_URL || "http://localhost:8000"
        const response = await fetch(
          `${baseUrl}/api/v1/huggingface/models/files?repo_id=${encodeURIComponent(model.modelId)}`,
          {
            headers: {
              "Authorization": `Bearer ${token}`,
            },
          }
        )
        if (response.ok) {
          const data = await response.json()
          setFiles(data)
          if (data.length > 0) {
            setSelectedFile(data[0].path)
          }
        }
      } catch (error) {
        console.error("Failed to fetch files:", error)
      } finally {
        setIsLoadingFiles(false)
      }
    }

    fetchFiles()
  }, [model.modelId])

  const createModelMutation = useMutation({
    mutationFn: async (modelData: any) => {
      return await ModelsService.createModel({ body: modelData })
    },
    onSuccess: () => {
      showSuccessToast("Model added successfully from HuggingFace")
      queryClient.invalidateQueries({ queryKey: ["models"] })
      onClose()
    },
    onError: (error: any) => {
      showErrorToast(error.message || "Failed to add model")
    },
  })

  const handleAddModel = () => {
    if (!selectedFile) {
      showErrorToast("Please select a GGUF file")
      return
    }

    const file = files.find(f => f.path === selectedFile)
    const modelData: any = {
      name: `${model.modelId}/${selectedFile.split('/').pop()}`,
      path: `/models/${model.modelId}/${selectedFile.split('/').pop()}`,
      size_bytes: file?.size || 0,
      architecture: "llama",
      quantization: selectedFile.includes("Q4_K_M") ? "Q4_K_M" : 
                   selectedFile.includes("Q5_K_M") ? "Q5_K_M" : 
                   selectedFile.includes("Q8_0") ? "Q8_0" : "unknown",
      supports_embeddings: false,
      supports_vision: false,
      context_length: 4096,
      tags: ["huggingface", model.modelId.split('/')[0]],
      source: "huggingface",
      source_repo_id: model.modelId,
      source_url: `https://huggingface.co/${model.modelId}`,
      source_file: selectedFile,
    }
    
    // Only include optional fields if they have values
    if (file?.size) modelData.size_bytes = file.size
    if (model.modelId.includes("7B") || model.modelId.includes("7b")) modelData.parameter_count = 7000000000
    else if (model.modelId.includes("13B") || model.modelId.includes("13b")) modelData.parameter_count = 13000000000
    else if (model.modelId.includes("70B") || model.modelId.includes("70b")) modelData.parameter_count = 70000000000

    createModelMutation.mutate(modelData)
  }

  return (
    <Dialog open={true} onOpenChange={onClose}>
      <DialogContent className="max-w-2xl">
        <DialogHeader>
          <DialogTitle>Add Model from HuggingFace</DialogTitle>
          <DialogDescription>
            Select a GGUF file to download and add to your models
          </DialogDescription>
        </DialogHeader>

        <div className="space-y-4 py-4">
          <div>
            <Label>Model Repository</Label>
            <div className="text-sm text-muted-foreground mt-1">
              {model.modelId}
            </div>
          </div>

          <div>
            <Label>GGUF File</Label>
            {isLoadingFiles ? (
              <div className="text-sm text-muted-foreground">
                Loading available files...
              </div>
            ) : files.length > 0 ? (
              <Select value={selectedFile} onValueChange={setSelectedFile}>
                <SelectTrigger className="mt-1">
                  <SelectValue placeholder="Select a file" />
                </SelectTrigger>
                <SelectContent>
                  {files.map((file) => (
                    <SelectItem key={file.path} value={file.path}>
                      {file.path.split('/').pop()} ({(file.size / 1024 / 1024 / 1024).toFixed(2)} GB)
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            ) : (
              <div className="text-sm text-destructive">
                No GGUF files found in this repository
              </div>
            )}
          </div>

          <div className="flex justify-end gap-2 pt-4">
            <Button variant="outline" onClick={onClose}>
              Cancel
            </Button>
            <LoadingButton
              onClick={handleAddModel}
              loading={createModelMutation.isPending || !selectedFile}
              disabled={!selectedFile}
            >
              Add Model
            </LoadingButton>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  )
}
