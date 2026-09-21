import { useSuspenseQuery } from "@tanstack/react-query"
import { createFileRoute } from "@tanstack/react-router"
import { Plus, Server } from "lucide-react"
import { Suspense, useState } from "react"

import { AgentsService } from "@/client"
import { AddAgent } from "@/components/Agents/AddAgent"
import { columns } from "@/components/Agents/columns"
import { DataTable } from "@/components/Common/DataTable"
import { Button } from "@/components/ui/button"

function getAgentsQueryOptions() {
  return {
    queryFn: async () => {
      const response = await AgentsService.listAgents()
      return response.data.agents || []
    },
    queryKey: ["agents"],
  }
}

export const Route = createFileRoute("/_layout/agents")({
  component: Agents,
  head: () => ({
    meta: [
      {
        title: "Agents - Inference Matrix",
      },
    ],
  }),
})

function AgentsTableContent() {
  const { data: agents } = useSuspenseQuery(getAgentsQueryOptions())

  if (!agents || agents.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center text-center py-12">
        <div className="rounded-full bg-muted p-4 mb-4">
          <Server className="h-8 w-8 text-muted-foreground" />
        </div>
        <h3 className="text-lg font-semibold">No agents registered</h3>
        <p className="text-muted-foreground">
          Add an agent to start managing llama.cpp servers
        </p>
      </div>
    )
  }

  return <DataTable columns={columns as any} data={agents as any} />
}

function AgentsTable() {
  return (
    <Suspense
      fallback={
        <div className="flex items-center justify-center py-12">
          Loading agents...
        </div>
      }
    >
      <AgentsTableContent />
    </Suspense>
  )
}

function Agents() {
  const [isAddAgentOpen, setIsAddAgentOpen] = useState(false)

  return (
    <div className="flex flex-col gap-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold tracking-tight">Agents</h1>
          <p className="text-muted-foreground">
            Manage agents that run llama.cpp servers
          </p>
        </div>
        <Button onClick={() => setIsAddAgentOpen(true)}>
          <Plus className="mr-2 h-4 w-4" />
          Add Agent
        </Button>
      </div>
      <AgentsTable />
      <AddAgent
        isOpen={isAddAgentOpen}
        onClose={() => setIsAddAgentOpen(false)}
      />
    </div>
  )
}
