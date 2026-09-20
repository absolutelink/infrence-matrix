import { useState, useEffect } from "react"
import { useMutation, useQueryClient } from "@tanstack/react-query"

import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog"
import { Button } from "@/components/ui/button"
import { Label } from "@/components/ui/label"
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

interface GGUFFileGroup {
  id: string
  name: string
  files: GGUFFile[]
  totalSize: number
  fileCount: number
  isSplit: boolean
}

// Detect if files are part of a split model (e.g., model-00001-of-00003.gguf)
const groupSplitFiles = (files: GGUFFile[]): GGUFFileGroup[] => {
  const splitPattern = /^(.+?)-(\d+)-of-(\d+)\.gguf$/
  const groups = new Map<string, GGUFFileGroup>()
  const singleFiles: GGUFFileGroup[] = []

  files.forEach(file => {
    const filename = file.path.split('/').pop() || file.path
    const match = filename.match(splitPattern)
    
    if (match) {
      const [, baseName, , totalParts] = match
      const groupId = `${baseName}-${totalParts}`
      
      if (!groups.has(groupId)) {
        groups.set(groupId, {
          id: groupId,
          name: `${baseName} (${totalParts} parts)`,
          files: [],
          totalSize: 0,
          fileCount: parseInt(totalParts),
          isSplit: true,
        })
      }
      
      const group = groups.get(groupId)!
      group.files.push(file)
      group.totalSize += file.size
    } else {
      singleFiles.push({
        id: file.path,
        name: filename,
        files: [file],
        totalSize: file.size,
        fileCount: 1,
        isSplit: false,
      })
    }
  })

  return [...Array.from(groups.values()), ...singleFiles]
}

export function AddModelFromHF({ model, onClose }: AddModelFromHFProps) {
  const queryClient = useQueryClient()
  const { showSuccessToast, showErrorToast } = useCustomToast()
  const [fileGroups, setFileGroups] = useState<GGUFFileGroup[]>([])
  const [selectedGroup, setSelectedGroup] = useState<string>("")
  const [isLoadingFiles, setIsLoadingFiles] = useState(false)
  const [parameterCount, setParameterCount] = useState<number | undefined>(undefined)

  // Fetch GGUF files for this model and group split files
  useEffect(() => {
    const fetchFiles = async () => {
      setIsLoadingFiles(true)
      try {
        const token = localStorage.getItem("access_token")
        const baseUrl = (window as any).APP_CONFIG?.API_URL || 
                        import.meta.env.VITE_API_URL || 
                        window.location.origin
        const response = await fetch(
          `${baseUrl}/api/v1/huggingface/models/files?repo_id=${encodeURIComponent(model.modelId)}`,
          {
            headers: {
              "Authorization": `Bearer ${token}`,
            },
          }
        )
        if (response.ok) {
          const data: GGUFFile[] = await response.json()
          const groups = groupSplitFiles(data)
          setFileGroups(groups)
          if (groups.length > 0) {
            setSelectedGroup(groups[0].id)
          }
        }
      } catch (error) {
        console.error("Failed to fetch files:", error)
      } finally {
        setIsLoadingFiles(false)
      }
    }

    // Fetch parameter count from HuggingFace config
    const fetchParams = async () => {
      try {
        const token = localStorage.getItem("access_token")
        const baseUrl = (window as any).APP_CONFIG?.API_URL || 
                        import.meta.env.VITE_API_URL || 
                        window.location.origin
        const response = await fetch(
          `${baseUrl}/api/v1/huggingface/models/params?repo_id=${encodeURIComponent(model.modelId)}`,
          {
            headers: {
              "Authorization": `Bearer ${token}`,
            },
          }
        )
        if (response.ok) {
          const data = await response.json()
          if (data.parameter_count) {
            setParameterCount(data.parameter_count)
          }
        }
      } catch (error) {
        console.error("Failed to fetch parameter count:", error)
      }
    }

    fetchFiles()
    fetchParams()
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
    if (!selectedGroup) {
      showErrorToast("Please select a GGUF file")
      return
    }

    const group = fileGroups.find(g => g.id === selectedGroup)
    if (!group) return

    // For split models, use the first file's name pattern
    const primaryFile = group.files[0]
    const baseName = group.isSplit ? group.name.split(' (')[0] : primaryFile.path.split('/').pop()
    
    const modelData: any = {
      name: `${model.modelId}/${baseName}${group.isSplit ? ' (split)' : ''}`,
      path: `/models/${model.modelId}/${baseName}`,
      size_bytes: group.totalSize,
      architecture: "llama",
      quantization: baseName?.includes("Q4_K_M") ? "Q4_K_M" : 
                   baseName?.includes("Q5_K_M") ? "Q5_K_M" : 
                   baseName?.includes("Q8_0") ? "Q8_0" : "unknown",
      supports_embeddings: false,
      supports_vision: false,
      context_length: 4096,
      tags: ["huggingface", model.modelId.split('/')[0]],
      source: "huggingface",
      source_repo_id: model.modelId,
      source_url: `https://huggingface.co/${model.modelId}`,
      source_file: group.isSplit ? JSON.stringify(group.files.map(f => f.path)) : primaryFile.path,
    }
    
    if (group.totalSize) modelData.size_bytes = group.totalSize
    // Use actual parameter count from HuggingFace if available
    if (parameterCount) {
      modelData.parameter_count = parameterCount
    } else if (model.modelId.includes("7B") || model.modelId.includes("7b")) {
      modelData.parameter_count = 7000000000
    } else if (model.modelId.includes("13B") || model.modelId.includes("13b")) {
      modelData.parameter_count = 13000000000
    } else if (model.modelId.includes("70B") || model.modelId.includes("70b")) {
      modelData.parameter_count = 70000000000
    }

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
            <Label>GGUF Files</Label>
            {isLoadingFiles ? (
              <div className="text-sm text-muted-foreground">
                Loading available files...
              </div>
            ) : fileGroups.length > 0 ? (
              <Select value={selectedGroup} onValueChange={setSelectedGroup}>
                <SelectTrigger className="mt-1">
                  <SelectValue placeholder="Select files to download" />
                </SelectTrigger>
                <SelectContent>
                  {fileGroups.map((group) => (
                    <SelectItem key={group.id} value={group.id}>
                      {group.name} ({(group.totalSize / 1024 / 1024 / 1024).toFixed(2)} GB{group.isSplit ? `, ${group.fileCount} files` : ''})
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
              loading={createModelMutation.isPending || !selectedGroup}
              disabled={!selectedGroup}
            >
              Add Model{selectedGroup && fileGroups.find(g => g.id === selectedGroup)?.isSplit ? " (Multiple Files)" : ""}
            </LoadingButton>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  )
}
