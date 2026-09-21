import { useSuspenseQuery } from "@tanstack/react-query"
import { createFileRoute } from "@tanstack/react-router"
import { Plus, Server } from "lucide-react"
import { Suspense } from "react"

// TODO: Import from client when backend is running
const ServerInstancesService = {
  listServerInstances: async () => ({ data: { server_instances: [] } }),
}

import { DataTable } from "@/components/Common/DataTable"
import { columns } from "@/components/ServerInstances/columns"
import { Button } from "@/components/ui/button"

function getServerInstancesQueryOptions() {
  return {
    queryFn: async () => {
      const response = await ServerInstancesService.listServerInstances()
      return response.data.server_instances || []
    },
    queryKey: ["server-instances"],
  }
}

export const Route = createFileRoute("/_layout/server-instances")({
  component: ServerInstances,
  head: () => ({
    meta: [
      {
        title: "Server Instances - Inference Matrix",
      },
    ],
  }),
})

function ServerInstancesTableContent() {
  const { data: instances } = useSuspenseQuery(getServerInstancesQueryOptions())

  if (!instances || instances.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center text-center py-12">
        <div className="rounded-full bg-muted p-4 mb-4">
          <Server className="h-8 w-8 text-muted-foreground" />
        </div>
        <h3 className="text-lg font-semibold">No server instances running</h3>
        <p className="text-muted-foreground">
          Start a server instance to begin inference
        </p>
      </div>
    )
  }

  return <DataTable columns={columns} data={instances} />
}

function ServerInstancesTable() {
  return (
    <Suspense
      fallback={
        <div className="flex items-center justify-center py-12">
          Loading server instances...
        </div>
      }
    >
      <ServerInstancesTableContent />
    </Suspense>
  )
}

function ServerInstances() {
  return (
    <div className="flex flex-col gap-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold tracking-tight">
            Server Instances
          </h1>
          <p className="text-muted-foreground">
            Manage running llama.cpp server instances
          </p>
        </div>
        <Button>
          <Plus className="mr-2 h-4 w-4" />
          Start Server
        </Button>
      </div>
      <ServerInstancesTable />
    </div>
  )
}
