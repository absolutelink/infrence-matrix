import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query"
import { useEffect, useState } from "react"
import { toast } from "sonner"
import type { Model } from "@/client"
import { ModelsService, ServerInstancesService } from "@/client"
import {
  type ServerOptions,
  ServerSettingsFields,
  validateServerOptions,
} from "@/components/ServerInstances/ServerSettingsFields"
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

interface EditServerDialogProps {
  isOpen: boolean
  onClose: () => void
  instance: {
    id: string
    model_id: string
    model_name: string | null
    mmproj_model_id?: string | null
    status: string
    alias: string
    gpu_layers: number
    context_size: number
    flash_attn: boolean
    mtp_draft_max?: number | null
    inactivity_timeout_seconds: number
    server_options?: ServerOptions
  }
}

export function EditServerDialog({
  isOpen,
  onClose,
  instance,
}: EditServerDialogProps) {
  const queryClient = useQueryClient()
  const [modelId, setModelId] = useState("")
  const [alias, setAlias] = useState("")
  const [gpuLayers, setGpuLayers] = useState("35")
  const [contextSize, setContextSize] = useState("4096")
  const [flashAttn, setFlashAttn] = useState(true)
  const [inactivityTimeout, setInactivityTimeout] = useState("300")
  const [mmprojModelId, setMmprojModelId] = useState<string>("none")
  const [mtpDraftMax, setMtpDraftMax] = useState("0")
  const [serverOptions, setServerOptions] = useState<ServerOptions>({})

  const modelsQuery = useQuery({
    queryKey: ["models"],
    queryFn: async () => {
      const response = await ModelsService.readModels()
      return response.data
    },
    enabled: isOpen,
  })

  useEffect(() => {
    if (isOpen) {
      setModelId(instance.model_id)
      setAlias(instance.alias)
      setGpuLayers(String(instance.gpu_layers))
      setContextSize(String(instance.context_size))
      setFlashAttn(instance.flash_attn)
      setInactivityTimeout(String(instance.inactivity_timeout_seconds))
      setMmprojModelId(instance.mmproj_model_id || "none")
      setMtpDraftMax(
        instance.mtp_draft_max ? String(instance.mtp_draft_max) : "0",
      )
      setServerOptions(instance.server_options || {})
    }
  }, [isOpen, instance])

  const wasRunning = instance.status === "running"

  const updateMutation = useMutation({
    mutationFn: async () => {
      return ServerInstancesService.instancesUpdateServer({
        path: { server_id: instance.id },
        body: {
          alias: alias.trim() || instance.alias,
          model_id: modelId,
          gpu_layers: Number(gpuLayers),
          context_size: Number(contextSize),
          flash_attn: flashAttn,
          inactivity_timeout_seconds: Number(inactivityTimeout),
          mmproj_model_id: mmprojModelId === "none" ? "" : mmprojModelId,
          mtp_draft_max: mtpDraftMax === "0" ? null : Number(mtpDraftMax),
          server_options: serverOptions,
        },
      })
    },
    onSuccess: (data) => {
      const status = (data as unknown as { status?: string }).status
      if (status === "restarting") {
        toast.success("Settings saved — restarting server")
      } else {
        toast.success("Settings saved")
      }
      queryClient.invalidateQueries({ queryKey: ["server-instances"] })
      onClose()
    },
    onError: (error: Error) => {
      toast.error(`Failed to update server: ${error.message}`)
    },
  })

  return (
    <Dialog
      open={isOpen}
      onOpenChange={(open) => {
        if (!open) {
          onClose()
        }
      }}
    >
      <DialogContent>
        <DialogHeader>
          <DialogTitle>Edit Server Settings</DialogTitle>
          <DialogDescription>
            {instance.model_name || "Server"}
            {wasRunning
              ? " — saving will stop the server and restart it with the new settings/model"
              : ""}
          </DialogDescription>
        </DialogHeader>

        <div className="grid grid-cols-2 gap-4 py-2">
          <div className="col-span-2">
            <Label htmlFor="alias">Alias</Label>
            <Input
              id="alias"
              value={alias}
              onChange={(e) => setAlias(e.target.value)}
              placeholder="Public name clients use in requests"
              className="mt-1"
            />
          </div>
          <div className="col-span-2">
            <Label>Model</Label>
            <Select value={modelId} onValueChange={setModelId}>
              <SelectTrigger className="mt-1">
                <SelectValue placeholder="Select a model" />
              </SelectTrigger>
              <SelectContent>
                {((modelsQuery.data ?? []) as Model[])
                  .filter((model) => model.model_type === "llm")
                  .map((model) => (
                    <SelectItem
                      key={model.id ?? model.name}
                      value={model.id ?? model.name}
                    >
                      {model.name}
                    </SelectItem>
                  ))}
              </SelectContent>
            </Select>
          </div>
          <div className="col-span-2">
            <Label>mmproj (vision projector)</Label>
            <Select value={mmprojModelId} onValueChange={setMmprojModelId}>
              <SelectTrigger className="mt-1">
                <SelectValue placeholder="None (no --mmproj flag)" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="none">None</SelectItem>
                {((modelsQuery.data ?? []) as Model[])
                  .filter((model) => model.model_type === "mmproj")
                  .map((model) => (
                    <SelectItem
                      key={model.id ?? model.name}
                      value={model.id ?? model.name}
                    >
                      {model.name}
                    </SelectItem>
                  ))}
              </SelectContent>
            </Select>
          </div>
          <div>
            <Label htmlFor="gpu-layers">GPU layers</Label>
            <Input
              id="gpu-layers"
              type="number"
              min={0}
              max={1000}
              value={gpuLayers}
              onChange={(e) => setGpuLayers(e.target.value)}
              className="mt-1"
            />
          </div>
          <div className="col-span-2">
            <ServerSettingsFields
              options={serverOptions}
              onChange={setServerOptions}
            />
          </div>
          <div>
            <Label htmlFor="context-size">Context size</Label>
            <Input
              id="context-size"
              type="number"
              min={256}
              max={1048576}
              step={256}
              value={contextSize}
              onChange={(e) => setContextSize(e.target.value)}
              className="mt-1"
            />
          </div>
          <div>
            <Label htmlFor="inactivity-timeout">
              Inactivity timeout (seconds)
            </Label>
            <Input
              id="inactivity-timeout"
              type="number"
              min={0}
              max={86400}
              value={inactivityTimeout}
              onChange={(e) => setInactivityTimeout(e.target.value)}
              className="mt-1"
            />
          </div>
          <div className="flex items-end pb-2">
            <Label
              htmlFor="flash-attn"
              className="flex items-center gap-2 text-sm font-medium"
            >
              <Checkbox
                id="flash-attn"
                checked={flashAttn}
                onCheckedChange={(checked) => setFlashAttn(checked === true)}
              />
              Flash attention
            </Label>
          </div>

          <div>
            <Label htmlFor="mtp-draft-max">MTP Draft N-Max</Label>
            <Input
              id="mtp-draft-max"
              type="number"
              min={0}
              value={mtpDraftMax}
              onChange={(e) => setMtpDraftMax(e.target.value)}
              className="mt-1"
              placeholder="0 (no flags)"
            />
          </div>
        </div>

        <DialogFooter>
          <Button variant="outline" onClick={onClose}>
            Cancel
          </Button>
          <LoadingButton
            onClick={() => updateMutation.mutate()}
            loading={updateMutation.isPending}
            disabled={validateServerOptions(serverOptions) !== null}
          >
            {wasRunning ? "Save & Restart" : "Save"}
          </LoadingButton>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}
