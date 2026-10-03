import { useMutation, useQueryClient } from "@tanstack/react-query"
import { useEffect, useMemo, useState } from "react"
import { ModelsService } from "@/client"
import { Button } from "@/components/ui/button"
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog"
import { Label } from "@/components/ui/label"
import { LoadingButton } from "@/components/ui/loading-button"
import {
  Select,
  SelectContent,
  SelectGroup,
  SelectItem,
  SelectLabel,
  SelectSeparator,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"
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
  model_type?: string
  quantization?: string | null
  is_aux?: boolean
}

interface GGUFFileGroup {
  id: string
  name: string
  files: GGUFFile[]
  totalSize: number
  fileCount: number
  isSplit: boolean
}

interface GGUFSelection {
  id: string
  label: string
  files: GGUFFile[]
  totalSize: number
  fileCount: number
  isSplit: boolean
  quantization: string | null
  isAux: boolean
}

const MODEL_TYPE_OPTIONS = ["llm", "mtp", "mmproj", "dflash"] as const

const AUX_TYPES = ["mtp", "mmproj", "dflash"]

const formatGB = (bytes: number): string =>
  (bytes / 1024 / 1024 / 1024).toFixed(2)

// Detect if files are part of a split model (e.g., model-00001-of-00003.gguf)
const groupSplitFiles = (files: GGUFFile[]): GGUFFileGroup[] => {
  const splitPattern = /^(.+?)-(\d+)-of-(\d+)\.gguf$/
  const groups = new Map<string, GGUFFileGroup>()
  const singleFiles: GGUFFileGroup[] = []

  files.forEach((file) => {
    const filename = file.path.split("/").pop() || file.path
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
          fileCount: parseInt(totalParts, 10),
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

// Group main (non-aux) files by their quantization tag.
const groupByQuantization = (files: GGUFFile[]): GGUFSelection[] => {
  const map = new Map<string, GGUFFile[]>()
  for (const file of files) {
    if (!file.quantization) continue
    const bucket = map.get(file.quantization) ?? []
    bucket.push(file)
    map.set(file.quantization, bucket)
  }
  return Array.from(map.entries()).map(([tag, bucket]) => ({
    id: `quant:${tag}`,
    label: tag,
    files: bucket,
    totalSize: bucket.reduce((sum, f) => sum + f.size, 0),
    fileCount: bucket.length,
    isSplit: bucket.length > 1,
    quantization: tag,
    isAux: false,
  }))
}

const toOtherSelections = (groups: GGUFFileGroup[]): GGUFSelection[] =>
  groups.map((group) => ({
    id: `file:${group.id}`,
    label: group.name,
    files: group.files,
    totalSize: group.totalSize,
    fileCount: group.fileCount,
    isSplit: group.isSplit,
    quantization: null,
    isAux: false,
  }))

const toAuxSelections = (files: GGUFFile[]): GGUFSelection[] =>
  files.map((file) => ({
    id: `aux:${file.path}`,
    label: file.path.split("/").pop() || file.path,
    files: [file],
    totalSize: file.size,
    fileCount: 1,
    isSplit: false,
    quantization: file.quantization ?? null,
    isAux: true,
  }))

export function AddModelFromHF({ model, onClose }: AddModelFromHFProps) {
  const queryClient = useQueryClient()
  const { showSuccessToast, showErrorToast } = useCustomToast()
  const [files, setFiles] = useState<GGUFFile[]>([])
  const [selectedMainId, setSelectedMainId] = useState<string>("")
  const [selectedAuxId, setSelectedAuxId] = useState<string>("")
  const [isLoadingFiles, setIsLoadingFiles] = useState(false)
  const [parameterCount, setParameterCount] = useState<number | undefined>(
    undefined,
  )
  const [modelType, setModelType] = useState<string>("llm")

  // Fetch GGUF files for this model
  useEffect(() => {
    const fetchFiles = async () => {
      setIsLoadingFiles(true)
      try {
        const baseUrl =
          (window as any).APP_CONFIG?.API_URL ||
          import.meta.env.VITE_API_URL ||
          window.location.origin
        const response = await fetch(
          `${baseUrl}/api/v1/huggingface/models/files?repo_id=${encodeURIComponent(model.modelId)}`,
        )
        if (response.ok) {
          const data: GGUFFile[] = await response.json()
          setFiles(data)
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
        const baseUrl =
          (window as any).APP_CONFIG?.API_URL ||
          import.meta.env.VITE_API_URL ||
          window.location.origin
        const response = await fetch(
          `${baseUrl}/api/v1/huggingface/models/params?repo_id=${encodeURIComponent(model.modelId)}`,
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

  // Split fetched files into main (llm) and auxiliary groups.
  const { quantSelections, otherSelections, auxSelections } = useMemo(() => {
    const mainFiles = files.filter((f) => !f.is_aux)
    const auxFiles = files.filter((f) => f.is_aux)
    const quant = groupByQuantization(mainFiles)
    const untagged = mainFiles.filter((f) => !f.quantization)
    const other = toOtherSelections(groupSplitFiles(untagged))
    const aux = toAuxSelections(auxFiles)
    return {
      quantSelections: quant,
      otherSelections: other,
      auxSelections: aux,
    }
  }, [files])

  const mainSelections = useMemo(
    () => [...quantSelections, ...otherSelections],
    [quantSelections, otherSelections],
  )

  // Default selections when the file list changes.
  useEffect(() => {
    if (mainSelections.length > 0 && !selectedMainId) {
      setSelectedMainId(mainSelections[0].id)
    }
    if (auxSelections.length > 0 && !selectedAuxId) {
      setSelectedAuxId(auxSelections[0].id)
    }
  }, [mainSelections, auxSelections, selectedMainId, selectedAuxId])

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

  const buildBaseModelData = () => {
    const modelData: any = {
      architecture: "llama",
      supports_embeddings: false,
      supports_vision: false,
      tags: ["huggingface", model.modelId.split("/")[0]],
      source: "huggingface",
      source_repo_id: model.modelId,
      source_url: `https://huggingface.co/${model.modelId}`,
    }
    if (parameterCount) {
      modelData.parameter_count = parameterCount
    } else if (model.modelId.includes("7B") || model.modelId.includes("7b")) {
      modelData.parameter_count = 7000000000
    } else if (model.modelId.includes("13B") || model.modelId.includes("13b")) {
      modelData.parameter_count = 13000000000
    } else if (model.modelId.includes("70B") || model.modelId.includes("70b")) {
      modelData.parameter_count = 70000000000
    }
    return modelData
  }

  const handleAddModel = () => {
    if (AUX_TYPES.includes(modelType)) {
      if (!selectedAuxId) {
        showErrorToast("Please select an auxiliary file")
        return
      }
      const sel = auxSelections.find((s) => s.id === selectedAuxId)
      if (!sel) return
      const auxPath = sel.files[0].path
      const auxName = auxPath.split("/").pop() || auxPath

      const modelData = buildBaseModelData()
      modelData.name = `${model.modelId}/${auxName}`
      modelData.path = `/models/${model.modelId}/${auxName}`
      modelData.size_bytes = sel.totalSize
      modelData.model_type = modelType
      modelData.quantization = sel.quantization ?? "unknown"
      modelData.source_files = [auxPath]
      modelData.source_file = auxPath

      createModelMutation.mutate(modelData)
      return
    }

    // Main (llm) selection from the Quantizations / Other files sections.
    if (!selectedMainId) {
      showErrorToast("Please select a GGUF file")
      return
    }
    const sel = mainSelections.find((s) => s.id === selectedMainId)
    if (!sel) return

    const allPaths = sel.files.map((f) => f.path)
    const splitPrimary =
      allPaths.find((p) => /-00001-of-\d+\.gguf$/.test(p)) || allPaths[0]

    const baseName = sel.quantization
      ? sel.label
      : sel.isSplit
        ? sel.label.split(" (")[0]
        : (allPaths[0]?.split("/").pop() ?? sel.label)

    const modelData = buildBaseModelData()
    modelData.name = `${model.modelId}/${baseName}${sel.isSplit ? " (split)" : ""}`
    modelData.path = `/models/${model.modelId}/${baseName}`
    modelData.size_bytes = sel.totalSize
    modelData.model_type = "llm"
    modelData.quantization = sel.quantization ?? "unknown"
    modelData.source_files = allPaths
    modelData.source_file = sel.isSplit ? splitPrimary : allPaths[0]

    createModelMutation.mutate(modelData)
  }

  const isAuxSelected = AUX_TYPES.includes(modelType)
  const hasMainOptions = mainSelections.length > 0
  const hasAuxOptions = auxSelections.length > 0

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
            <Label>Model Type</Label>
            <Select value={modelType} onValueChange={setModelType}>
              <SelectTrigger className="mt-1">
                <SelectValue placeholder="Select model type" />
              </SelectTrigger>
              <SelectContent>
                {MODEL_TYPE_OPTIONS.map((type) => (
                  <SelectItem key={type} value={type}>
                    {type}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>

          {isLoadingFiles ? (
            <div className="text-sm text-muted-foreground">
              Loading available files...
            </div>
          ) : isAuxSelected ? (
            <div>
              <Label>Auxiliary File</Label>
              {hasAuxOptions ? (
                <Select value={selectedAuxId} onValueChange={setSelectedAuxId}>
                  <SelectTrigger className="mt-1">
                    <SelectValue placeholder="Select an auxiliary file" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectGroup>
                      <SelectLabel>Auxiliary files</SelectLabel>
                      {auxSelections.map((sel) => (
                        <SelectItem key={sel.id} value={sel.id}>
                          {sel.label} ({formatGB(sel.totalSize)} GB)
                        </SelectItem>
                      ))}
                    </SelectGroup>
                  </SelectContent>
                </Select>
              ) : (
                <div className="text-sm text-destructive mt-1">
                  No auxiliary files found in this repository
                </div>
              )}
            </div>
          ) : hasMainOptions ? (
            <div>
              <Label>Quantization / Files</Label>
              <Select value={selectedMainId} onValueChange={setSelectedMainId}>
                <SelectTrigger className="mt-1">
                  <SelectValue placeholder="Select a quantization or file" />
                </SelectTrigger>
                <SelectContent>
                  {quantSelections.length > 0 && (
                    <SelectGroup>
                      <SelectLabel>Quantizations</SelectLabel>
                      {quantSelections.map((sel) => (
                        <SelectItem key={sel.id} value={sel.id}>
                          {sel.label} ({formatGB(sel.totalSize)} GB
                          {sel.isSplit ? `, ${sel.fileCount} files` : ""})
                        </SelectItem>
                      ))}
                    </SelectGroup>
                  )}
                  {otherSelections.length > 0 && (
                    <>
                      {quantSelections.length > 0 && <SelectSeparator />}
                      <SelectGroup>
                        <SelectLabel>Other files</SelectLabel>
                        {otherSelections.map((sel) => (
                          <SelectItem key={sel.id} value={sel.id}>
                            {sel.label} ({formatGB(sel.totalSize)} GB
                            {sel.isSplit ? `, ${sel.fileCount} files` : ""})
                          </SelectItem>
                        ))}
                      </SelectGroup>
                    </>
                  )}
                </SelectContent>
              </Select>
            </div>
          ) : (
            <div className="text-sm text-destructive">
              No GGUF files found in this repository
            </div>
          )}

          <div className="flex justify-end gap-2 pt-4">
            <Button variant="outline" onClick={onClose}>
              Cancel
            </Button>
            <LoadingButton
              onClick={handleAddModel}
              loading={
                createModelMutation.isPending ||
                (isAuxSelected ? !selectedAuxId : !selectedMainId)
              }
              disabled={isAuxSelected ? !selectedAuxId : !selectedMainId}
            >
              Add Model
            </LoadingButton>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  )
}
