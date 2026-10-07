import { useQuery } from "@tanstack/react-query"
import { createFileRoute } from "@tanstack/react-router"

import { AdminService } from "@/client"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { useDefinitions, useInstances } from "@/hooks/useAdminData"

export const Route = createFileRoute("/_layout/settings")({
  component: SettingsPage,
  head: () => ({ meta: [{ title: "Settings - Inference Matrix" }] }),
})

function SettingsPage() {
  const { data: health } = useQuery({
    queryKey: ["admin-health"],
    queryFn: async () =>
      (await AdminService.adminHealth()).data as Record<string, string>,
    refetchInterval: 10000,
  })
  const { data: definitions = [] } = useDefinitions()
  const { data: instances = [] } = useInstances()

  const connected = instances.filter((i) => i.websocket_connected).length

  return (
    <div className="flex flex-col gap-6">
      <div>
        <h1 className="text-2xl font-bold tracking-tight">Settings</h1>
        <p className="text-muted-foreground">
          Admin deployment info and pointers.
        </p>
      </div>

      <div className="grid gap-4 sm:grid-cols-3">
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm text-muted-foreground">
              Admin version
            </CardTitle>
          </CardHeader>
          <CardContent className="text-2xl font-bold font-mono">
            {health?.version ?? "…"}
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm text-muted-foreground">
              Connected providers
            </CardTitle>
          </CardHeader>
          <CardContent className="text-2xl font-bold">
            {connected}
            <span className="ml-2 text-xs font-normal text-muted-foreground">
              of {instances.length} instances
            </span>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm text-muted-foreground">
              Enabled aliases
            </CardTitle>
          </CardHeader>
          <CardContent className="text-2xl font-bold">
            {definitions.filter((d) => d.enabled).length}
            <span className="ml-2 text-xs font-normal text-muted-foreground">
              of {definitions.length} definitions
            </span>
          </CardContent>
        </Card>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Deploying provider containers</CardTitle>
        </CardHeader>
        <CardContent className="space-y-3 text-sm text-muted-foreground">
          <p>
            Each provider agent is a hardware-local container bound to a
            machine + one backend type; it may host one or more backends of that
            type. It needs no database — everything comes from env:
          </p>
          <pre className="overflow-auto rounded-md bg-muted p-3 font-mono text-xs">
            {`MACHINE_UID=<machine uid created here>
MACHINE_SECRET=<the machine's registration_secret, shown on Machines>
AGENT_ID=<stable id for this agent, e.g. agent-1>
ADMIN_BASE_URL=http://<admin-host>:8000
PROVIDER_PORT=8081            # optional; base port for this agent's backends
CACHE_DIR=/cache
MODELS_DIR=/models
METRICS_CATEGORIES="gpu_usage vram os_ram cpu storage"   # optional, space-delimited; never 'inference'`}
          </pre>
          <p>
            Full walkthrough: <code>deployment.md</code> (root of the repo).
            Remember the version hard-fail: deploy the admin first, then every
            provider container must match its VERSION exactly.
          </p>
        </CardContent>
      </Card>
    </div>
  )
}
