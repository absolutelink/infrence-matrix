import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query"
import { useEffect, useState } from "react"
import { toast } from "sonner"
import type { ServerInstanceResponse } from "@/client"
import { ServerInstancesService } from "@/client"
import { Button } from "@/components/ui/button"
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { LoadingButton } from "@/components/ui/loading-button"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"
import type {
  BenchmarkDefinition,
  BenchmarkDefinitionInput,
} from "@/lib/benchmarkApi"
import { benchmarkApi } from "@/lib/benchmarkApi"

interface Props {
  open: boolean
  definition?: BenchmarkDefinition
  onClose: () => void
}

export function DefinitionDialog({ open, definition, onClose }: Props) {
  const queryClient = useQueryClient()
  const [name, setName] = useState("")
  const [description, setDescription] = useState("")
  const [serverId, setServerId] = useState("")
  const [promptSizes, setPromptSizes] = useState("512")
  const [generationSizes, setGenerationSizes] = useState("128")
  const [repetitions, setRepetitions] = useState("3")
  const [batchSize, setBatchSize] = useState("512")
  const [ubatchSize, setUbatchSize] = useState("")
  const [contextSize, setContextSize] = useState("")
  const [gpuLayers, setGpuLayers] = useState("")
  const [flashAttn, setFlashAttn] = useState(true)

  const serversQuery = useQuery({
    queryKey: ["server-instances"],
    queryFn: async () => {
      const response =
        await ServerInstancesService.instancesListServerInstances()
      return (response.data?.server_instances ?? []) as ServerInstanceResponse[]
    },
    enabled: open,
  })

  useEffect(() => {
    if (!open) return
    setName(definition?.name ?? "")
    setDescription(definition?.description ?? "")
    const config = definition?.config ?? {}
    setServerId(definition?.source_server_instance_id ?? "")
    setPromptSizes(
      ((config.prompt_sizes as number[] | undefined) ?? [512]).join(", "),
    )
    setGenerationSizes(
      ((config.generation_sizes as number[] | undefined) ?? [128]).join(", "),
    )
    setRepetitions(String(config.repetitions ?? 3))
    setBatchSize(String(config.batch_size ?? 512))
    setUbatchSize(String(config.ubatch_size ?? ""))
    setContextSize(String(config.context_size ?? ""))
    setGpuLayers(String(config.gpu_layers ?? ""))
    setFlashAttn(config.flash_attn !== false)
  }, [definition, open])

  const mutation = useMutation({
    mutationFn: async () => {
      const numbers = (value: string) =>
        value
          .split(",")
          .map((item) => Number(item.trim()))
          .filter((item) => Number.isFinite(item))
      const body: BenchmarkDefinitionInput = {
        name: name.trim(),
        description: description.trim(),
        source_server_instance_id: serverId,
        config: {
          prompt_sizes: numbers(promptSizes),
          generation_sizes: numbers(generationSizes),
          repetitions: Number(repetitions),
          batch_size: batchSize ? Number(batchSize) : null,
          ubatch_size: ubatchSize ? Number(ubatchSize) : null,
          context_size: contextSize ? Number(contextSize) : null,
          gpu_layers: gpuLayers ? Number(gpuLayers) : null,
          flash_attn: flashAttn,
        },
      }
      return definition
        ? benchmarkApi.updateDefinition(definition.id, body)
        : benchmarkApi.createDefinition(body)
    },
    onSuccess: () => {
      toast.success(definition ? "Definition updated" : "Definition created")
      queryClient.invalidateQueries({ queryKey: ["benchmark-definitions"] })
      onClose()
    },
    onError: (error) =>
      toast.error(
        error instanceof Error ? error.message : "Could not save definition",
      ),
  })

  return (
    <Dialog open={open} onOpenChange={(value) => !value && onClose()}>
      <DialogContent className="max-w-2xl">
        <DialogHeader>
          <DialogTitle>
            {definition
              ? "Edit benchmark definition"
              : "New benchmark definition"}
          </DialogTitle>
          <DialogDescription>
            Select a saved server definition and configure the llama-bench
            workload.
          </DialogDescription>
        </DialogHeader>
        <div className="space-y-4 py-2">
          <div>
            <Label>Server definition</Label>
            <Select value={serverId} onValueChange={setServerId}>
              <SelectTrigger className="mt-1">
                <SelectValue placeholder="Select a server definition" />
              </SelectTrigger>
              <SelectContent>
                {serversQuery.data?.map((server) => (
                  <SelectItem key={server.id} value={server.id}>
                    {server.alias} - {server.model_name ?? server.model_id} (
                    {server.status})
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
          <div className="grid grid-cols-2 gap-4">
            <div>
              <Label htmlFor="benchmark-prompts">Prompt sizes</Label>
              <Input
                id="benchmark-prompts"
                className="mt-1"
                value={promptSizes}
                onChange={(event) => setPromptSizes(event.target.value)}
                placeholder="512, 2048"
              />
            </div>
            <div>
              <Label htmlFor="benchmark-generation">Generation sizes</Label>
              <Input
                id="benchmark-generation"
                className="mt-1"
                value={generationSizes}
                onChange={(event) => setGenerationSizes(event.target.value)}
                placeholder="128, 512"
              />
            </div>
          </div>
          <div className="grid grid-cols-2 gap-4">
            <div>
              <Label htmlFor="benchmark-repetitions">Repetitions</Label>
              <Input
                id="benchmark-repetitions"
                type="number"
                min={1}
                className="mt-1"
                value={repetitions}
                onChange={(event) => setRepetitions(event.target.value)}
              />
            </div>
            <div>
              <Label htmlFor="benchmark-batch">Batch size</Label>
              <Input
                id="benchmark-batch"
                type="number"
                min={1}
                className="mt-1"
                value={batchSize}
                onChange={(event) => setBatchSize(event.target.value)}
              />
            </div>
          </div>
          <div className="grid grid-cols-2 gap-4">
            <div>
              <Label htmlFor="benchmark-ubatch">Ubatch size</Label>
              <Input
                id="benchmark-ubatch"
                type="number"
                min={1}
                className="mt-1"
                value={ubatchSize}
                onChange={(event) => setUbatchSize(event.target.value)}
              />
            </div>
            <div>
              <Label htmlFor="benchmark-context">Context size override</Label>
              <Input
                id="benchmark-context"
                type="number"
                min={256}
                className="mt-1"
                value={contextSize}
                onChange={(event) => setContextSize(event.target.value)}
              />
            </div>
          </div>
          <div className="grid grid-cols-2 gap-4">
            <div>
              <Label htmlFor="benchmark-gpu">GPU layers override</Label>
              <Input
                id="benchmark-gpu"
                type="number"
                min={0}
                className="mt-1"
                value={gpuLayers}
                onChange={(event) => setGpuLayers(event.target.value)}
              />
            </div>
            <label className="flex items-center gap-2 pt-7 text-sm">
              <input
                type="checkbox"
                checked={flashAttn}
                onChange={(event) => setFlashAttn(event.target.checked)}
              />
              Flash attention
            </label>
          </div>
          <div>
            <Label htmlFor="benchmark-name">Name</Label>
            <Input
              id="benchmark-name"
              className="mt-1"
              value={name}
              onChange={(event) => setName(event.target.value)}
              placeholder="Throughput smoke test"
            />
          </div>
          <div>
            <Label htmlFor="benchmark-description">Description</Label>
            <Input
              id="benchmark-description"
              className="mt-1"
              value={description}
              onChange={(event) => setDescription(event.target.value)}
              placeholder="Short description"
            />
          </div>
        </div>
        <DialogFooter>
          <Button variant="outline" onClick={onClose}>
            Cancel
          </Button>
          <LoadingButton
            loading={mutation.isPending}
            disabled={!name.trim() || !serverId}
            onClick={() => mutation.mutate()}
          >
            Save definition
          </LoadingButton>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}
