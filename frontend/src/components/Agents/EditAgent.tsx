import { useQueryClient } from "@tanstack/react-query"
import { useState } from "react"
import { toast } from "sonner"
import { AgentsService } from "@/client"
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

interface EditAgentProps {
  agent: {
    id: string
    name: string
    host: string
    port: number
  }
  isOpen: boolean
  onClose: () => void
}

export function EditAgent({ agent, isOpen, onClose }: EditAgentProps) {
  const queryClient = useQueryClient()
  const [isLoading, setIsLoading] = useState(false)
  const [formData, setFormData] = useState({
    name: agent?.name || "",
    host: agent?.host || "",
    port: agent?.port?.toString() || "8080",
  })

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setIsLoading(true)

    try {
      await AgentsService.updateAgent({
        agentId: agent.id,
        body: {
          name: formData.name,
          host: formData.host,
          port: parseInt(formData.port, 10),
        },
      })

      toast.success("Agent updated successfully")
      queryClient.invalidateQueries({ queryKey: ["agents"] })
      onClose()
      setFormData({
        name: agent?.name || "",
        host: agent?.host || "",
        port: agent?.port?.toString() || "8080",
      })
    } catch (error) {
      toast.error("Failed to update agent")
      console.error(error)
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <Dialog open={isOpen} onOpenChange={onClose}>
      <DialogContent className="sm:max-w-[500px]">
        <DialogHeader>
          <DialogTitle>Edit Agent</DialogTitle>
          <DialogDescription>
            Update the configuration for this agent.
          </DialogDescription>
        </DialogHeader>
        <form onSubmit={handleSubmit}>
          <div className="grid gap-4 py-4">
            <div className="grid gap-2">
              <Label htmlFor="name">Agent Name</Label>
              <Input
                id="name"
                value={formData.name}
                onChange={(e) =>
                  setFormData((prev) => ({ ...prev, name: e.target.value }))
                }
                placeholder="GPU-Agent-1"
                required
              />
            </div>
            <div className="grid gap-2">
              <Label htmlFor="host">Host</Label>
              <Input
                id="host"
                value={formData.host}
                onChange={(e) =>
                  setFormData((prev) => ({ ...prev, host: e.target.value }))
                }
                placeholder="10.100.2.100 or agent.example.com"
                required
              />
              <p className="text-xs text-muted-foreground">
                Hostname or IP address of the agent machine
              </p>
            </div>
            <div className="grid gap-2">
              <Label htmlFor="port">Port</Label>
              <Input
                id="port"
                type="number"
                value={formData.port}
                onChange={(e) =>
                  setFormData((prev) => ({ ...prev, port: e.target.value }))
                }
                placeholder="8080"
                required
              />
              <p className="text-xs text-muted-foreground">
                Agent API port (default: 8080)
              </p>
            </div>
          </div>
          <DialogFooter>
            <Button type="button" variant="outline" onClick={onClose}>
              Cancel
            </Button>
            <Button type="submit" disabled={isLoading}>
              {isLoading ? "Updating..." : "Update Agent"}
            </Button>
          </DialogFooter>
        </form>
      </DialogContent>
    </Dialog>
  )
}