import { useSuspenseQuery } from "@tanstack/react-query"
import { createFileRoute } from "@tanstack/react-router"
import {
  Activity,
  Clock,
  Cpu,
  MemoryStick,
  MessageSquare,
  Server,
  TrendingUp,
} from "lucide-react"
import { Suspense } from "react"

import { AgentsService, ModelsService, ServerInstancesService } from "@/client"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"

export const Route = createFileRoute("/_layout/")({
  component: Dashboard,
  head: () => ({
    meta: [
      {
        title: "Dashboard - Inference Matrix",
      },
    ],
  }),
})

function getModelsQueryOptions() {
  return {
    queryFn: async () =>
      (await ModelsService.readModels({ query: { skip: 0, limit: 100 } })).data,
    queryKey: ["dashboard-models"],
  }
}

function getAgentsQueryOptions() {
  return {
    queryFn: async () => {
      const response = await AgentsService.listAgents()
      return response.data.agents || []
    },
    queryKey: ["dashboard-agents"],
  }
}

function getServerInstancesQueryOptions() {
  return {
    queryFn: async () =>
      (await ServerInstancesService.instancesListServerInstances()).data
        .server_instances || [],
    queryKey: ["dashboard-server-instances"],
    refetchInterval: 10000,
  }
}

function MetricCard({
  title,
  value,
  description,
  icon: Icon,
  trend,
}: {
  title: string
  value: string | number
  description?: string
  icon: React.ElementType
  trend?: {
    value: number
    label: string
  }
}) {
  return (
    <Card>
      <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
        <CardTitle className="text-sm font-medium">{title}</CardTitle>
        <Icon className="h-4 w-4 text-muted-foreground" />
      </CardHeader>
      <CardContent>
        <div className="text-2xl font-bold">{value}</div>
        {description && (
          <p className="text-xs text-muted-foreground mt-1">{description}</p>
        )}
        {trend && (
          <div className="flex items-center mt-2 text-xs">
            <TrendingUp className="h-3 w-3 mr-1 text-green-600" />
            <span className="text-green-600 font-medium">{trend.label}</span>
          </div>
        )}
      </CardContent>
    </Card>
  )
}

function DashboardContent() {
  const { data: models } = useSuspenseQuery(getModelsQueryOptions())
  const { data: agents } = useSuspenseQuery(getAgentsQueryOptions())
  const { data: serverInstances } = useSuspenseQuery(
    getServerInstancesQueryOptions(),
  )

  const onlineAgents = agents.filter((a: any) => a.status === "online").length
  const runningServers = serverInstances.filter(
    (s: any) => s.status === "running",
  ).length
  const healthyServers = serverInstances.filter(
    (s: any) => s.status === "running" && s.health_status === "healthy",
  ).length
  const totalRequests = serverInstances.reduce(
    (acc: number, s: any) => acc + (s.total_requests || 0),
    0,
  )
  const gpuAgents = agents.filter((agent: any) => agent.gpu_info?.vram_total)
  const totalVram = gpuAgents.reduce(
    (sum: number, agent: any) => sum + (agent.gpu_info?.vram_total || 0),
    0,
  )
  const usedVram = gpuAgents.reduce(
    (sum: number, agent: any) => sum + (agent.gpu_info?.vram_used || 0),
    0,
  )
  const gpuUtilization = gpuAgents.length
    ? gpuAgents.reduce(
        (sum: number, agent: any) => sum + (agent.gpu_info?.utilization || 0),
        0,
      ) / gpuAgents.length
    : null
  const formatBytes = (bytes: number) => `${(bytes / 1073741824).toFixed(1)} GB`

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold tracking-tight">Dashboard</h1>
        <p className="text-muted-foreground">
          Overview of your inference infrastructure
        </p>
      </div>

      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-6">
        <MetricCard
          title="Total Models"
          value={models?.length || 0}
          description="GGUF models available"
          icon={Server}
        />
        <MetricCard
          title="Agents"
          value={`${onlineAgents}/${agents?.length || 0}`}
          description="Online agents"
          icon={Cpu}
          trend={
            onlineAgents === agents?.length
              ? { value: 100, label: "All online" }
              : undefined
          }
        />
        <MetricCard
          title="Running Servers"
          value={runningServers}
          description="Active llama.cpp instances"
          icon={Activity}
        />
        <MetricCard
          title="Healthy Servers"
          value={`${healthyServers}/${runningServers}`}
          description="Ready for inference"
          icon={Activity}
        />
        <MetricCard
          title="Total Requests"
          value={totalRequests.toLocaleString()}
          description="All-time inference requests"
          icon={MessageSquare}
        />
        <MetricCard
          title="GPU Utilization"
          value={
            gpuUtilization === null ? "—" : `${gpuUtilization.toFixed(0)}%`
          }
          description={`${gpuAgents.length} agent${gpuAgents.length === 1 ? "" : "s"} reporting`}
          icon={Cpu}
        />
        <MetricCard
          title="VRAM Usage"
          value={
            totalVram
              ? `${formatBytes(usedVram)} / ${formatBytes(totalVram)}`
              : "—"
          }
          description="Across connected agents"
          icon={MemoryStick}
        />
      </div>

      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-7">
        <Card className="col-span-4">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Activity className="h-5 w-5" />
              Recent Activity
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              {(serverInstances as any[]).slice(0, 5).map((instance: any) => (
                <div
                  key={instance.id}
                  className="flex items-center justify-between border-b pb-2 last:border-0"
                >
                  <div className="space-y-1">
                    <p className="text-sm font-medium">
                      {instance.model_name || "Unknown Model"}
                    </p>
                    <p className="text-xs text-muted-foreground">
                      Agent: {instance.agent_name || "Unknown"}
                    </p>
                  </div>
                  <div className="text-right">
                    <p className="text-sm font-medium">
                      {instance.total_requests.toLocaleString()} requests
                    </p>
                    <p className="text-xs text-muted-foreground">
                      Port: {instance.port}
                    </p>
                  </div>
                </div>
              ))}
              {serverInstances.length === 0 && (
                <p className="text-center text-muted-foreground py-4">
                  No server activity yet
                </p>
              )}
            </div>
          </CardContent>
        </Card>

        <Card className="col-span-3">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Clock className="h-5 w-5" />
              System Status
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="flex items-center justify-between">
              <div className="space-y-1">
                <p className="text-sm font-medium">Models Available</p>
                <p className="text-xs text-muted-foreground">
                  GGUF models in library
                </p>
              </div>
              <div className="text-2xl font-bold">{models?.length || 0}</div>
            </div>
            <div className="flex items-center justify-between">
              <div className="space-y-1">
                <p className="text-sm font-medium">Online Agents</p>
                <p className="text-xs text-muted-foreground">
                  Connected agents
                </p>
              </div>
              <div className="text-2xl font-bold text-green-600">
                {onlineAgents}
              </div>
            </div>
            <div className="flex items-center justify-between">
              <div className="space-y-1">
                <p className="text-sm font-medium">Running Servers</p>
                <p className="text-xs text-muted-foreground">
                  Active inference servers
                </p>
              </div>
              <div className="text-2xl font-bold">{runningServers}</div>
            </div>
            <div className="flex items-center justify-between">
              <div className="space-y-1">
                <p className="text-sm font-medium">Total Requests</p>
                <p className="text-xs text-muted-foreground">
                  All-time inference count
                </p>
              </div>
              <div className="text-2xl font-bold">
                {totalRequests.toLocaleString()}
              </div>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  )
}

function Dashboard() {
  return (
    <Suspense
      fallback={
        <div className="flex items-center justify-center py-12">
          Loading dashboard...
        </div>
      }
    >
      <DashboardContent />
    </Suspense>
  )
}
