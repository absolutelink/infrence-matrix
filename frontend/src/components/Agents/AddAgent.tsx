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

interface AddAgentProps {
  isOpen: boolean
  onClose: () => void
}

export function AddAgent({ isOpen, onClose }: AddAgentProps) {
  const queryClient = useQueryClient()
  const [isLoading, setIsLoading] = useState(false)
  const [formData, setFormData] = useState({
    agent_id: "",
    name: "",
    host: "",
    port: "8080",
  })

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setIsLoading(true)

    try {
      await AgentsService.registerAgent({
        body: {
          agent_id: formData.agent_id || crypto.randomUUID(),
          name: formData.name,
          host: formData.host,
          port: parseInt(formData.port, 10),
          gpu_info: null,
        },
      })

      toast.success("Agent registered successfully")
      queryClient.invalidateQueries({ queryKey: ["agents"] })
      onClose()
      setFormData({
        agent_id: "",
        name: "",
        host: "",
        port: "8080",
      })
    } catch (error) {
      toast.error("Failed to register agent")
      console.error(error)
    } finally {
      setIsLoading(false)
    }
  }

  const generateAgentId = () => {
    const id = crypto.randomUUID()
    setFormData((prev) => ({ ...prev, agent_id: id }))
  }

  return (
    <Dialog open={isOpen} onOpenChange={onClose}>
      <DialogContent className="sm:max-w-[500px]">
        <DialogHeader>
          <DialogTitle>Add Agent</DialogTitle>
          <DialogDescription>
            Register a new agent to manage llama.cpp servers and handle
            inference requests.
          </DialogDescription>
        </DialogHeader>
        <form onSubmit={handleSubmit}>
          <div className="grid gap-4 py-4">
            <div className="grid gap-2">
              <Label htmlFor="agent_id">Agent ID</Label>
              <div className="flex gap-2">
                <Input
                  id="agent_id"
                  value={formData.agent_id}
                  onChange={(e) =>
                    setFormData((prev) => ({
                      ...prev,
                      agent_id: e.target.value,
                    }))
                  }
                  placeholder="agent-1"
                  required
                />
                <Button
                  type="button"
                  variant="outline"
                  onClick={generateAgentId}
                >
                  Generate
                </Button>
              </div>
              <p className="text-xs text-muted-foreground">
                Unique identifier for this agent
              </p>
            </div>
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
              {isLoading ? "Registering..." : "Register Agent"}
            </Button>
          </DialogFooter>
        </form>
      </DialogContent>
    </Dialog>
  )
}
