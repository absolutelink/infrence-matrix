import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query"
import { useState } from "react"
import { toast } from "sonner"
import type { Model, StartServerRequest } from "@/client"
import { AgentsService, ModelsService, ServerInstancesService } from "@/client"
import { ModelSelect } from "@/components/Models/ModelSelect"
import {
  type GufoOptions,
  GufoSettingsFields,
} from "@/components/ServerInstances/GufoSettingsFields"
import {
  type HalogenFlashOptions,
  HalogenFlashSettingsFields,
} from "@/components/ServerInstances/HalogenFlashSettingsFields"
import {
  type HalogenOptions,
  HalogenSettingsFields,
} from "@/components/ServerInstances/HalogenSettingsFields"
import {
  type ServerOptions,
  ServerSettingsFields,
  validateServerOptions,
} from "@/components/ServerInstances/ServerSettingsFields"
import { Button } from "@/components/ui/button"
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
import {
  Sheet,
  SheetContent,
  SheetDescription,
  SheetFooter,
  SheetHeader,
  SheetTitle,
} from "@/components/ui/sheet"

interface StartServerDialogProps {
  isOpen: boolean
  onClose: () => void
}

export function StartServerDialog({ isOpen, onClose }: StartServerDialogProps) {
  const queryClient = useQueryClient()
  const [modelId, setModelId] = useState<string>("")
  const [agentId, setAgentId] = useState<string>("")
  const [alias, setAlias] = useState("")
  const [gpuLayers, setGpuLayers] = useState("35")
  const [contextSize, setContextSize] = useState("4096")
  const [vramRequired, setVramRequired] = useState("")
  const [mmprojModelId, setMmprojModelId] = useState<string>("none")
  const [dflashModelId, setDflashModelId] = useState<string>("none")
  const [mtpDraftMax, setMtpDraftMax] = useState("0")
  const [serverOptions, setServerOptions] = useState<ServerOptions>({})
  const [engine, setEngine] = useState<
    "llamacpp" | "halogen" | "halogen-flash" | "gufo"
  >("llamacpp")
  const [engineOptions, setEngineOptions] = useState<
    HalogenOptions | HalogenFlashOptions | GufoOptions
  >({})

  const modelsQuery = useQuery({
    queryKey: ["models"],
    queryFn: async () => {
      const response = await ModelsService.readModels()
      return response.data
    },
    enabled: isOpen,
  })

  const agentsQuery = useQuery({
    queryKey: ["agents"],
    queryFn: async () => {
      const response = await AgentsService.listAgents()
      return response.data.agents || []
    },
    enabled: isOpen,
  })

  const startMutation = useMutation({
    mutationFn: async () => {
      const body: StartServerRequest = {
        alias: alias.trim(),
        gpu_layers: Number(gpuLayers) || 35,
        context_size: Number(contextSize) || 4096,
        mtp_draft_max: mtpDraftMax === "0" ? null : Number(mtpDraftMax),
        server_options: engine === "llamacpp" ? serverOptions : {},
        engine,
        engine_options: engine !== "llamacpp" ? engineOptions : {},
      }
      if (vramRequired.trim()) {
        body.vram_required_bytes = Math.round(
          Number(vramRequired) * 1024 * 1024 * 1024,
        )
      }
      if (engine === "llamacpp" || engine === "gufo") {
        body.model_id = modelId
      }
      if (agentId && agentId !== "auto") {
        body.agent_id = agentId
      }
      if (engine === "llamacpp" && mmprojModelId && mmprojModelId !== "none") {
        body.mmproj_model_id = mmprojModelId
      }
      if (engine === "llamacpp" && dflashModelId !== "none") {
        body.dflash_model_id = dflashModelId
      }
      return ServerInstancesService.instancesStartServer({ body })
    },
    onSuccess: () => {
      toast.success("Server created; model preparation requested")
      queryClient.invalidateQueries({ queryKey: ["server-instances"] })
      onClose()
    },
    onError: (error: unknown) => {
      const detail = (error as { body?: { detail?: string } }).body?.detail
      toast.error(detail || "Failed to create server")
    },
  })

  const allModels = (modelsQuery.data ?? []) as Model[]
  const agents = (agentsQuery.data ?? []) as Array<{
    id: string
    name: string
    status: string
    platform?: string
    type?: string
    gpu_info?: {
      npu?: {
        available?: boolean
        reasons?: string[]
      }
    }
  }>
  const compatibleAgents = agents.filter((agent) => {
    if (engine === "gufo") {
      return agent.platform === "gufo" && agent.type === "gufo"
    }
    if (engine === "halogen" || engine === "halogen-flash") {
      return agent.platform === engine && agent.type === "rocm"
    }
    return agent.platform !== "halogen" && agent.platform !== "gufo"
  })

  const selectedAgent =
    agentId && agentId !== "auto"
      ? compatibleAgents.find((agent) => agent.id === agentId)
      : compatibleAgents[0]
  const npuInfo = selectedAgent?.gpu_info?.npu
  const npuAvailable = npuInfo?.available === true
  const npuReasons = npuInfo?.reasons ?? []

  return (
    <Sheet
      open={isOpen}
      onOpenChange={(open) => {
        if (!open) {
          onClose()
        }
      }}
    >
      <SheetContent className="w-full gap-0 overflow-hidden p-0 sm:max-w-xl">
        <SheetHeader className="border-b px-6 py-5">
          <SheetTitle>Create Server</SheetTitle>
          <SheetDescription>
            Create a server configuration. The assigned agent will prepare the
            files, gather OpenAI model metadata, and leave the server stopped
            when initialization completes.
          </SheetDescription>
        </SheetHeader>

        <div className="min-h-0 flex-1 overflow-y-auto px-6 py-5">
          <div className="space-y-4">
            <div>
              <Label>Engine</Label>
              <Select
                value={engine}
                onValueChange={(value) => {
                  setEngine(
                    value as "llamacpp" | "halogen" | "halogen-flash" | "gufo",
                  )
                  setAgentId("")
                  setServerOptions({})
                  setEngineOptions({})
                }}
              >
                <SelectTrigger className="mt-1">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="llamacpp">llama.cpp</SelectItem>
                  <SelectItem value="halogen">Halogen</SelectItem>
                  <SelectItem value="halogen-flash">Halogen Flash</SelectItem>
                  <SelectItem value="gufo">Gufo</SelectItem>
                </SelectContent>
              </Select>
            </div>
            <div>
              <Label htmlFor="start-alias">Alias</Label>
              <Input
                id="start-alias"
                value={alias}
                onChange={(e) => setAlias(e.target.value)}
                placeholder="Public name clients will use (e.g. qwen-4b)"
                className="mt-1"
              />
            </div>
            {engine === "halogen" ? (
              <HalogenSettingsFields
                options={engineOptions as HalogenOptions}
                onChange={setEngineOptions}
              />
            ) : engine === "halogen-flash" ? (
              <HalogenFlashSettingsFields
                options={engineOptions as HalogenFlashOptions}
                onChange={setEngineOptions}
                npuAvailable={npuAvailable}
                npuReasons={npuReasons}
              />
            ) : engine === "gufo" ? (
              <GufoSettingsFields
                options={engineOptions as GufoOptions}
                onChange={setEngineOptions}
                models={(modelsQuery.data ?? []) as Model[]}
              />
            ) : (
              <ServerSettingsFields
                options={serverOptions}
                onChange={setServerOptions}
                mtpDraftMax={mtpDraftMax}
                onMtpDraftMaxChange={setMtpDraftMax}
              />
            )}
            {engine === "llamacpp" && (
              <div>
                <Label>dflash draft model</Label>
                <ModelSelect
                  className="mt-1"
                  models={allModels}
                  modelType="dflash"
                  value={dflashModelId}
                  onValueChange={setDflashModelId}
                  placeholder="None (standard MTP)"
                  extraItems={[{ value: "none", label: "None" }]}
                />
              </div>
            )}
            {(engine === "llamacpp" || engine === "gufo") && (
              <div>
                <Label>Model</Label>
                <ModelSelect
                  className="mt-1"
                  models={allModels}
                  modelType="llm"
                  value={modelId}
                  onValueChange={setModelId}
                  placeholder="Select a model"
                />
              </div>
            )}

            <div>
              <Label>Agent</Label>
              <Select value={agentId} onValueChange={setAgentId}>
                <SelectTrigger className="mt-1">
                  <SelectValue placeholder="Auto (first online agent)" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="auto">
                    Auto (first online agent)
                  </SelectItem>
                  {compatibleAgents.map((agent) => (
                    <SelectItem key={agent.id} value={agent.id}>
                      {agent.name} ({agent.status})
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            <div>
              <Label htmlFor="start-vram-required">Required VRAM (GB)</Label>
              <Input
                id="start-vram-required"
                type="number"
                min={0}
                step={0.1}
                value={vramRequired}
                onChange={(e) => setVramRequired(e.target.value)}
                placeholder="Auto"
                className="mt-1"
              />
              <p className="mt-1 text-xs text-muted-foreground">
                Leave blank to estimate from the model file size.
              </p>
            </div>

            {engine === "llamacpp" && (
              <div>
                <Label>mmproj (vision projector)</Label>
                <ModelSelect
                  className="mt-1"
                  models={allModels}
                  modelType="mmproj"
                  value={mmprojModelId}
                  onValueChange={setMmprojModelId}
                  placeholder="None (no --mmproj flag)"
                  extraItems={[{ value: "none", label: "None" }]}
                />
              </div>
            )}

            {engine === "llamacpp" && (
              <div className="grid grid-cols-1 gap-4 sm:grid-cols-3">
                <div>
                  <Label htmlFor="start-gpu-layers">GPU layers</Label>
                  <Input
                    id="start-gpu-layers"
                    type="number"
                    min={0}
                    max={1000}
                    value={gpuLayers}
                    onChange={(e) => setGpuLayers(e.target.value)}
                    className="mt-1"
                  />
                </div>
                <div>
                  <Label htmlFor="start-context-size">Context size</Label>
                  <Input
                    id="start-context-size"
                    type="number"
                    min={256}
                    max={1048576}
                    step={256}
                    value={contextSize}
                    onChange={(e) => setContextSize(e.target.value)}
                    className="mt-1"
                  />
                </div>
              </div>
            )}
          </div>
        </div>

        <SheetFooter className="border-t px-6 py-4 sm:flex-row sm:justify-end">
          <Button variant="outline" onClick={onClose}>
            Cancel
          </Button>
          <LoadingButton
            onClick={() => startMutation.mutate()}
            loading={startMutation.isPending}
            disabled={
              ((engine === "llamacpp" || engine === "gufo") && !modelId) ||
              !alias.trim() ||
              (engine === "llamacpp" &&
                validateServerOptions(serverOptions) !== null)
            }
          >
            Create Server
          </LoadingButton>
        </SheetFooter>
      </SheetContent>
    </Sheet>
  )
}
