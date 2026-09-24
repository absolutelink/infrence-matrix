import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query"
import { useEffect, useState } from "react"
import { toast } from "sonner"

import type { Model, ServerInstanceResponse } from "@/client"
import { AgentsService, ModelsService, ServerInstancesService } from "@/client"
import { Button } from "@/components/ui/button"
import { Checkbox } from "@/components/ui/checkbox"
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
  copyFrom?: ServerInstanceResponse
  onClose: () => void
}

const batchValues = [128, 256, 512, 1024, 2048, 4096]
const ubatchValues = [128, 256, 512, 1024, 2048]

export function DefinitionDialog({
  open,
  definition,
  copyFrom,
  onClose,
}: Props) {
  const queryClient = useQueryClient()
  const [name, setName] = useState("")
  const [description, setDescription] = useState("")
  const [modelId, setModelId] = useState("")
  const [agentId, setAgentId] = useState("")
  const [mmprojModelId, setMmprojModelId] = useState("none")
  const [gpuLayers, setGpuLayers] = useState("35")
  const [contextSize, setContextSize] = useState("4096")
  const [mtpDraftMax, setMtpDraftMax] = useState("0")
  const [flashAttn, setFlashAttn] = useState(true)
  const [promptSizes, setPromptSizes] = useState("512")
  const [generationSizes, setGenerationSizes] = useState("128")
  const [repetitions, setRepetitions] = useState("3")
  const [batchSize, setBatchSize] = useState("512")
  const [ubatchSize, setUbatchSize] = useState("none")

  const modelsQuery = useQuery({
    queryKey: ["models"],
    queryFn: async () => (await ModelsService.readModels()).data ?? [],
    enabled: open,
  })
  const agentsQuery = useQuery({
    queryKey: ["agents"],
    queryFn: async () => (await AgentsService.listAgents()).data?.agents ?? [],
    enabled: open,
  })

  useEffect(() => {
    if (!open) return
    const config = definition?.config ?? {}
    const source = copyFrom
    setName(definition?.name ?? (source ? `${source.alias} benchmark` : ""))
    setDescription(definition?.description ?? "")
    setModelId(String(config.model_id ?? source?.model_id ?? ""))
    setAgentId(String(config.agent_id ?? source?.agent_id ?? ""))
    setMmprojModelId(
      String(config.mmproj_model_id ?? source?.mmproj_model_id ?? "none"),
    )
    setGpuLayers(String(config.gpu_layers ?? source?.gpu_layers ?? 35))
    setContextSize(String(config.context_size ?? source?.context_size ?? 4096))
    setMtpDraftMax(String(config.mtp_draft_max ?? source?.mtp_draft_max ?? 0))
    setFlashAttn(
      config.flash_attn === false ? false : source?.flash_attn !== false,
    )
    setPromptSizes(
      ((config.prompt_sizes as number[] | undefined) ?? [512]).join(", "),
    )
    setGenerationSizes(
      ((config.generation_sizes as number[] | undefined) ?? [128]).join(", "),
    )
    setRepetitions(String(config.repetitions ?? 3))
    setBatchSize(String(config.batch_size ?? 512))
    setUbatchSize(config.ubatch_size ? String(config.ubatch_size) : "none")
  }, [copyFrom, definition, open])

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
        source_server_instance_id:
          copyFrom?.id ?? definition?.source_server_instance_id,
        config: {
          model_id: modelId,
          agent_id: agentId,
          mmproj_model_id: mmprojModelId === "none" ? null : mmprojModelId,
          gpu_layers: Number(gpuLayers),
          context_size: Number(contextSize),
          mtp_draft_max: mtpDraftMax === "0" ? null : Number(mtpDraftMax),
          flash_attn: flashAttn,
          prompt_sizes: numbers(promptSizes),
          generation_sizes: numbers(generationSizes),
          repetitions: Number(repetitions),
          batch_size: Number(batchSize),
          ubatch_size: ubatchSize === "none" ? null : Number(ubatchSize),
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

  const models = ((modelsQuery.data ?? []) as Model[]).filter(
    (model) => model.model_type === "llm",
  )
  const mmprojModels = ((modelsQuery.data ?? []) as Model[]).filter(
    (model) => model.model_type === "mmproj",
  )
  const agents = (agentsQuery.data ?? []) as Array<{
    id: string
    name: string
    status: string
  }>

  return (
    <Dialog open={open} onOpenChange={(value) => !value && onClose()}>
      <DialogContent className="max-h-[90vh] overflow-y-auto sm:max-w-2xl">
        <DialogHeader>
          <DialogTitle>
            {definition
              ? "Edit benchmark definition"
              : "New benchmark definition"}
          </DialogTitle>
          <DialogDescription>
            Configure the model, server settings, and llama-bench workload.
          </DialogDescription>
        </DialogHeader>
        <div className="grid grid-cols-2 gap-4 py-2">
          <div className="col-span-2">
            <Label htmlFor="benchmark-name">Name</Label>
            <Input
              id="benchmark-name"
              className="mt-1"
              value={name}
              onChange={(event) => setName(event.target.value)}
            />
          </div>
          <div className="col-span-2">
            <Label htmlFor="benchmark-description">Description</Label>
            <Input
              id="benchmark-description"
              className="mt-1"
              value={description}
              onChange={(event) => setDescription(event.target.value)}
            />
          </div>
          <div className="col-span-2">
            <Label>Model</Label>
            <Select value={modelId} onValueChange={setModelId}>
              <SelectTrigger className="mt-1">
                <SelectValue placeholder="Select a model" />
              </SelectTrigger>
              <SelectContent>
                {models.map((model) => (
                  <SelectItem key={model.id} value={model.id as string}>
                    {model.name}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
          <div className="col-span-2">
            <Label>Agent</Label>
            <Select value={agentId} onValueChange={setAgentId}>
              <SelectTrigger className="mt-1">
                <SelectValue placeholder="Select an agent" />
              </SelectTrigger>
              <SelectContent>
                {agents.map((agent) => (
                  <SelectItem key={agent.id} value={agent.id}>
                    {agent.name} ({agent.status})
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
          <div className="col-span-2">
            <Label>mmproj (vision projector)</Label>
            <Select value={mmprojModelId} onValueChange={setMmprojModelId}>
              <SelectTrigger className="mt-1">
                <SelectValue placeholder="None" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="none">None</SelectItem>
                {mmprojModels.map((model) => (
                  <SelectItem key={model.id} value={model.id as string}>
                    {model.name}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
          <div>
            <Label htmlFor="benchmark-gpu">GPU layers</Label>
            <Input
              id="benchmark-gpu"
              type="number"
              min={0}
              max={1000}
              className="mt-1"
              value={gpuLayers}
              onChange={(event) => setGpuLayers(event.target.value)}
            />
          </div>
          <div>
            <Label htmlFor="benchmark-context">Context size</Label>
            <Input
              id="benchmark-context"
              type="number"
              min={256}
              step={256}
              className="mt-1"
              value={contextSize}
              onChange={(event) => setContextSize(event.target.value)}
            />
          </div>
          <div>
            <Label htmlFor="benchmark-mtp">MTP Draft N-Max</Label>
            <Input
              id="benchmark-mtp"
              type="number"
              min={0}
              className="mt-1"
              value={mtpDraftMax}
              onChange={(event) => setMtpDraftMax(event.target.value)}
            />
          </div>
          <Label
            htmlFor="benchmark-flash-attn"
            className="flex items-end gap-2 pb-2 text-sm"
          >
            <Checkbox
              id="benchmark-flash-attn"
              checked={flashAttn}
              onCheckedChange={(checked) => setFlashAttn(checked === true)}
            />
            Flash attention
          </Label>
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
            <Label>Batch size</Label>
            <Select value={batchSize} onValueChange={setBatchSize}>
              <SelectTrigger className="mt-1">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {batchValues.map((value) => (
                  <SelectItem key={value} value={String(value)}>
                    {value}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
          <div>
            <Label>Ubatch size</Label>
            <Select value={ubatchSize} onValueChange={setUbatchSize}>
              <SelectTrigger className="mt-1">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="none">Default</SelectItem>
                {ubatchValues.map((value) => (
                  <SelectItem key={value} value={String(value)}>
                    {value}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
        </div>
        <DialogFooter>
          <Button variant="outline" onClick={onClose}>
            Cancel
          </Button>
          <LoadingButton
            loading={mutation.isPending}
            disabled={!name.trim() || !modelId || !agentId}
            onClick={() => mutation.mutate()}
          >
            Save definition
          </LoadingButton>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}

export function CopyServerDialog({
  open,
  onClose,
  onSelect,
}: {
  open: boolean
  onClose: () => void
  onSelect: (server: ServerInstanceResponse) => void
}) {
  const [serverId, setServerId] = useState("")
  const serversQuery = useQuery({
    queryKey: ["server-instances"],
    queryFn: async () =>
      (await ServerInstancesService.instancesListServerInstances()).data
        ?.server_instances ?? [],
    enabled: open,
  })
  return (
    <Dialog open={open} onOpenChange={(value) => !value && onClose()}>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>Copy from server instances</DialogTitle>
          <DialogDescription>
            Choose a server definition to populate a new benchmark definition.
          </DialogDescription>
        </DialogHeader>
        <Select value={serverId} onValueChange={setServerId}>
          <SelectTrigger>
            <SelectValue placeholder="Select a server instance" />
          </SelectTrigger>
          <SelectContent>
            {serversQuery.data?.map((server) => (
              <SelectItem key={server.id} value={server.id}>
                {server.alias} - {server.model_name ?? server.model_id}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
        <DialogFooter>
          <Button variant="outline" onClick={onClose}>
            Cancel
          </Button>
          <Button
            disabled={!serverId}
            onClick={() => {
              const server = serversQuery.data?.find(
                (item) => item.id === serverId,
              )
              if (server) onSelect(server)
            }}
          >
            Copy settings
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}
