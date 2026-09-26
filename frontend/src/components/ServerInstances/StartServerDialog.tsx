import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query"
import { useState } from "react"
import { toast } from "sonner"
import type { Model, StartServerRequest } from "@/client"
import { AgentsService, ModelsService, ServerInstancesService } from "@/client"
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
    "llamacpp" | "halogen" | "halogen-flash"
  >("llamacpp")
  const [engineOptions, setEngineOptions] = useState<
    HalogenOptions | HalogenFlashOptions
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
      if (engine === "llamacpp") {
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

  const models = ((modelsQuery.data ?? []) as Model[]).filter(
    (model) => model.model_type === "llm",
  )
  const mmprojModels = ((modelsQuery.data ?? []) as Model[]).filter(
    (model) => model.model_type === "mmproj",
  )
  const dflashModels = ((modelsQuery.data ?? []) as Model[]).filter(
    (model) => model.model_type === "dflash",
  )
  const agents = (agentsQuery.data ?? []) as Array<{
    id: string
    name: string
    status: string
    platform?: string
    type?: string
  }>
  const compatibleAgents = agents.filter((agent) =>
    engine === "halogen" || engine === "halogen-flash"
      ? agent.platform === engine && agent.type === "rocm"
      : agent.platform !== "halogen",
  )

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
                  setEngine(value as "llamacpp" | "halogen" | "halogen-flash")
                  setAgentId("")
                  setServerOptions({})
                }}
              >
                <SelectTrigger className="mt-1">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="llamacpp">llama.cpp</SelectItem>
                  <SelectItem value="halogen">Halogen</SelectItem>
                  <SelectItem value="halogen-flash">Halogen Flash</SelectItem>
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
                options={engineOptions}
                onChange={setEngineOptions}
              />
            ) : engine === "halogen-flash" ? (
              <HalogenFlashSettingsFields
                options={engineOptions as HalogenFlashOptions}
                onChange={setEngineOptions}
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
                <Select value={dflashModelId} onValueChange={setDflashModelId}>
                  <SelectTrigger className="mt-1">
                    <SelectValue placeholder="None (standard MTP)" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="none">None</SelectItem>
                    {dflashModels.map((model) => (
                      <SelectItem key={model.id} value={model.id as string}>
                        {model.name}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
            )}
            {engine === "llamacpp" && (
              <div>
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
                <Select value={mmprojModelId} onValueChange={setMmprojModelId}>
                  <SelectTrigger className="mt-1">
                    <SelectValue placeholder="None (no --mmproj flag)" />
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
              (engine === "llamacpp" && !modelId) ||
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
