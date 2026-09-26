import { useMutation, useQueryClient } from "@tanstack/react-query"
import { useEffect, useState } from "react"
import { toast } from "sonner"

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

type MetadataInstance = {
  id: string
  alias: string
  model_metadata?: Record<string, unknown>
}

const capabilityNames = [
  "chat",
  "completions",
  "embeddings",
  "vision",
  "tools",
  "reasoning",
]

export function MetadataDialog({
  isOpen,
  onClose,
  instance,
}: {
  isOpen: boolean
  onClose: () => void
  instance: MetadataInstance
}) {
  const queryClient = useQueryClient()
  const metadata = instance.model_metadata ?? {}
  const [description, setDescription] = useState("")
  const [ownedBy, setOwnedBy] = useState("")
  const [maxContextLength, setMaxContextLength] = useState("")
  const [capabilities, setCapabilities] = useState<Record<string, boolean>>({})

  useEffect(() => {
    if (!isOpen) return
    const discovered = (metadata.capabilities ?? {}) as Record<string, unknown>
    setDescription(
      typeof metadata.description === "string" ? metadata.description : "",
    )
    setOwnedBy(typeof metadata.owned_by === "string" ? metadata.owned_by : "")
    const discoveredContext =
      metadata.max_context_length ??
      metadata.context_length ??
      metadata.max_model_len
    setMaxContextLength(
      typeof discoveredContext === "number" ? String(discoveredContext) : "",
    )
    setCapabilities(
      Object.fromEntries(
        capabilityNames.map((name) => [name, discovered[name] === true]),
      ),
    )
  }, [isOpen, metadata])

  const mutation = useMutation({
    mutationFn: () =>
      ServerInstancesService.instancesUpdateServerMetadata({
        path: { server_id: instance.id },
        body: {
          description: description || null,
          owned_by: ownedBy || null,
          max_context_length: maxContextLength
            ? Number(maxContextLength)
            : null,
          capabilities,
        },
      }),
    onSuccess: () => {
      toast.success("Model metadata saved")
      queryClient.invalidateQueries({ queryKey: ["server-instances"] })
      onClose()
    },
    onError: () => toast.error("Failed to save model metadata"),
  })

  return (
    <Dialog open={isOpen} onOpenChange={(open) => !open && onClose()}>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>Edit Model Metadata</DialogTitle>
          <DialogDescription>
            Override discovered metadata for {instance.alias}. Runtime settings
            are unchanged.
          </DialogDescription>
        </DialogHeader>
        <div className="space-y-4 py-2">
          <div>
            <Label htmlFor="metadata-description">Description</Label>
            <Input
              id="metadata-description"
              value={description}
              onChange={(event) => setDescription(event.target.value)}
              className="mt-1"
            />
          </div>
          <div className="grid grid-cols-2 gap-4">
            <div>
              <Label htmlFor="metadata-owned-by">Owned by</Label>
              <Input
                id="metadata-owned-by"
                value={ownedBy}
                onChange={(event) => setOwnedBy(event.target.value)}
                className="mt-1"
              />
            </div>
            <div>
              <Label htmlFor="metadata-context">Max context length</Label>
              <Input
                id="metadata-context"
                type="number"
                min={1}
                value={maxContextLength}
                onChange={(event) => setMaxContextLength(event.target.value)}
                className="mt-1"
              />
            </div>
          </div>
          <div>
            <Label>Capabilities</Label>
            <div className="mt-2 grid grid-cols-2 gap-2 text-sm">
              {capabilityNames.map((name) => (
                <label key={name} className="flex items-center gap-2">
                  <input
                    type="checkbox"
                    checked={capabilities[name] === true}
                    onChange={(event) =>
                      setCapabilities((current) => ({
                        ...current,
                        [name]: event.target.checked,
                      }))
                    }
                  />
                  {name}
                </label>
              ))}
            </div>
          </div>
        </div>
        <DialogFooter>
          <Button variant="outline" onClick={onClose}>
            Cancel
          </Button>
          <LoadingButton
            loading={mutation.isPending}
            onClick={() => mutation.mutate()}
          >
            Save Metadata
          </LoadingButton>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}
