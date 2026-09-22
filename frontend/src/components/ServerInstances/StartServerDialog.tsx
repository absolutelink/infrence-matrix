import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query"
import { useState } from "react"
import { toast } from "sonner"
import type { Model, StartServerRequest } from "@/client"
import { AgentsService, ModelsService, ServerInstancesService } from "@/client"
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
import { LoadingButton } from "@/components/ui/loading-button"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"

interface StartServerDialogProps {
  isOpen: boolean
  onClose: () => void
}

export function StartServerDialog({ isOpen, onClose }: StartServerDialogProps) {
  const queryClient = useQueryClient()
  const [modelId, setModelId] = useState<string>("")
  const [agentId, setAgentId] = useState<string>("")

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
        model_id: modelId,
      }
      if (agentId && agentId !== "auto") {
        body.agent_id = agentId
      }
      return ServerInstancesService.instancesStartServer({ body })
    },
    onSuccess: () => {
      toast.success("Server start request sent")
      queryClient.invalidateQueries({ queryKey: ["server-instances"] })
      onClose()
    },
    onError: () => {
      toast.error("Failed to start server")
    },
  })

  const models = (modelsQuery.data ?? []) as Model[]
  const agents = (agentsQuery.data ?? []) as Array<{
    id: string
    name: string
    status: string
  }>

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
          <DialogTitle>Start Server</DialogTitle>
          <DialogDescription>
            Start a llama.cpp server for a model. If the model file is not on
            the agent yet, it will be downloaded from HuggingFace first.
          </DialogDescription>
        </DialogHeader>

        <div className="space-y-4 py-2">
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

          <div>
            <Label>Agent</Label>
            <Select value={agentId} onValueChange={setAgentId}>
              <SelectTrigger className="mt-1">
                <SelectValue placeholder="Auto (first online agent)" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="auto">Auto (first online agent)</SelectItem>
                {agents.map((agent) => (
                  <SelectItem key={agent.id} value={agent.id}>
                    {agent.name} ({agent.status})
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
            onClick={() => startMutation.mutate()}
            loading={startMutation.isPending}
            disabled={!modelId}
          >
            Start Server
          </LoadingButton>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}
