import { useMutation, useQueryClient } from "@tanstack/react-query"
import { useEffect, useRef, useState } from "react"
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

const reasoningEfforts = ["none", "low", "medium", "high", "xhigh"] as const
type ReasoningEffort = (typeof reasoningEfforts)[number]

type ReasoningConfig = {
  supported: boolean
  efforts: ReasoningEffort[]
  default: ReasoningEffort | ""
}

function readReasoningConfig(
  metadata: Record<string, unknown>,
  reasoningEnabled: boolean,
): ReasoningConfig {
  const block = metadata.reasoning
  if (block && typeof block === "object") {
    const value = block as {
      supported?: boolean
      efforts?: unknown
      default?: unknown
    }
    const efforts = Array.isArray(value.efforts)
      ? value.efforts.filter((e): e is ReasoningEffort =>
          reasoningEfforts.includes(e as ReasoningEffort),
        )
      : []
    return {
      supported: value.supported !== false,
      efforts,
      default:
        typeof value.default === "string" &&
        reasoningEfforts.includes(value.default as ReasoningEffort)
          ? (value.default as ReasoningEffort)
          : "",
    }
  }
  // Legacy rows: a bare capability boolean with no level detail.
  return {
    supported: reasoningEnabled,
    efforts: reasoningEnabled
      ? reasoningEfforts.filter((e) => e !== "none")
      : [],
    default: "",
  }
}

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
  const [reasoning, setReasoning] = useState<ReasoningConfig>({
    supported: false,
    efforts: [],
    default: "",
  })

  const prevOpenRef = useRef(false)
  useEffect(() => {
    if (isOpen && !prevOpenRef.current) {
      const discovered = (metadata.capabilities ?? {}) as Record<
        string,
        unknown
      >
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
      const caps = Object.fromEntries(
        capabilityNames.map((name) => [name, discovered[name] === true]),
      )
      setCapabilities(caps)
      setReasoning(readReasoningConfig(metadata, caps.reasoning === true))
    }
    prevOpenRef.current = isOpen
  }, [isOpen, metadata])

  const toggleReasoning = (enabled: boolean) => {
    setCapabilities((current) => ({ ...current, reasoning: enabled }))
    setReasoning((current) => ({
      ...current,
      supported: enabled,
      efforts:
        enabled && current.efforts.length === 0
          ? reasoningEfforts.filter((e) => e !== "none")
          : current.efforts,
      default: enabled ? current.default : "",
    }))
  }

  const toggleEffort = (effort: ReasoningEffort) => {
    setReasoning((current) => {
      const has = current.efforts.includes(effort)
      const efforts = has
        ? current.efforts.filter((e) => e !== effort)
        : reasoningEfforts.filter(
            (e) => e === effort || current.efforts.includes(e),
          )
      return {
        ...current,
        efforts,
        default:
          current.default !== "" && efforts.includes(current.default)
            ? current.default
            : "",
      }
    })
  }

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
          reasoning: {
            supported: reasoning.supported,
            efforts: reasoning.supported ? reasoning.efforts : [],
            default:
              reasoning.supported && reasoning.default
                ? reasoning.default
                : null,
          },
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
                      name === "reasoning"
                        ? toggleReasoning(event.target.checked)
                        : setCapabilities((current) => ({
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
          {capabilities.reasoning === true && (
            <div className="rounded-md border p-3">
              <Label>Reasoning levels</Label>
              <p className="text-xs text-muted-foreground mt-1 mb-2">
                Only the selected levels can be requested for this model.
              </p>
              <div className="grid grid-cols-3 gap-2 text-sm">
                {reasoningEfforts.map((effort) => (
                  <label key={effort} className="flex items-center gap-2">
                    <input
                      type="checkbox"
                      checked={reasoning.efforts.includes(effort)}
                      onChange={() => toggleEffort(effort)}
                    />
                    {effort}
                  </label>
                ))}
              </div>
              <div className="mt-3">
                <Label htmlFor="reasoning-default">Default level</Label>
                <select
                  id="reasoning-default"
                  className="mt-1 w-full rounded-md border bg-background px-3 py-2 text-sm"
                  value={reasoning.default}
                  onChange={(event) =>
                    setReasoning((current) => ({
                      ...current,
                      default: event.target.value as ReasoningEffort | "",
                    }))
                  }
                >
                  <option value="">No default (model default)</option>
                  {reasoning.efforts.map((effort) => (
                    <option key={effort} value={effort}>
                      {effort}
                    </option>
                  ))}
                </select>
              </div>
            </div>
          )}
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
