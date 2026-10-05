import { createFileRoute, Link } from "@tanstack/react-router"
import {
  ArrowRight,
  Coins,
  Cpu,
  Gauge,
  MessagesSquare,
  Package,
  Server,
} from "lucide-react"
import { Suspense } from "react"

import { EmptyNudge } from "@/components/Common/EmptyNudge"
import { StatusBadge } from "@/components/Common/StatusBadge"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Skeleton } from "@/components/ui/skeleton"
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table"
import {
  useDefinitions,
  useInstances,
  useOverview,
  useResponses,
} from "@/hooks/useAdminData"

export const Route = createFileRoute("/_layout/")({
  component: Dashboard,
  head: () => ({ meta: [{ title: "Dashboard - Inference Matrix" }] }),
})

function MetricCard({
  title,
  value,
  sub,
  icon: Icon,
  to,
}: {
  title: string
  value: string | number
  sub?: string
  icon: React.ElementType
  to?: string
}) {
  const card = (
    <Card className="transition-colors hover:bg-accent/50">
      <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
        <CardTitle className="text-sm font-medium text-muted-foreground">
          {title}
        </CardTitle>
        <Icon className="size-4 text-muted-foreground" />
      </CardHeader>
      <CardContent>
        <div className="text-2xl font-bold">{value}</div>
        {sub && <p className="text-xs text-muted-foreground">{sub}</p>}
      </CardContent>
    </Card>
  )
  return to ? (
    <Link to={to} className="block">
      {card}
    </Link>
  ) : (
    card
  )
}

function DashboardContent() {
  const { data: overview } = useOverview()
  const { data: instances = [] } = useInstances()
  const { data: definitions = [] } = useDefinitions()
  const { data: recent } = useResponses(8, 0)

  const enabledAliases = definitions.filter((d) => d.enabled).length

  return (
    <div className="flex flex-col gap-6">
      <div>
        <h1 className="text-2xl font-bold tracking-tight">Dashboard</h1>
        <p className="text-muted-foreground">
          Inference Matrix fleet overview — machines, definitions, live
          instances, and recent turns.
        </p>
      </div>

      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <MetricCard
          title="Machines"
          value={overview?.machines ?? 0}
          icon={Server}
          to="/machines"
        />
        <MetricCard
          title="Definitions"
          value={overview?.definitions ?? 0}
          sub={`${enabledAliases} enabled`}
          icon={Package}
          to="/definitions"
        />
        <MetricCard
          title="Connected instances"
          value={overview?.instances_connected ?? 0}
          sub={`${overview?.instances ?? 0} total · ${overview?.backends_running ?? 0} backends running`}
          icon={Cpu}
          to="/instances"
        />
        <MetricCard
          title="Live inference"
          value={`${overview?.active_requests ?? 0} active`}
          sub={`${overview?.queued_requests ?? 0} queued`}
          icon={Gauge}
        />
      </div>

      <div className="grid gap-4 lg:grid-cols-3">
        <Card className="lg:col-span-2">
          <CardHeader className="flex flex-row items-center justify-between">
            <CardTitle>Instance status</CardTitle>
            <Link
              to="/instances"
              className="text-sm text-muted-foreground hover:text-foreground inline-flex items-center gap-1"
            >
              All instances <ArrowRight className="size-3" />
            </Link>
          </CardHeader>
          <CardContent>
            {instances.length === 0 ? (
              <EmptyNudge
                text="No provider instances yet."
                actionLabel="Create a definition"
                to="/definitions"
              />
            ) : (
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>Machine</TableHead>
                    <TableHead>Alias</TableHead>
                    <TableHead>Instance</TableHead>
                    <TableHead>Backend</TableHead>
                    <TableHead>WS</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {instances.map((i) => (
                    <TableRow key={i.id}>
                      <TableCell className="font-mono text-xs">
                        {i.machine_uid ?? "—"}
                      </TableCell>
                      <TableCell className="font-medium">{i.alias}</TableCell>
                      <TableCell>
                        <StatusBadge status={i.instance_status} />
                      </TableCell>
                      <TableCell>
                        <StatusBadge status={i.backend_status} />
                      </TableCell>
                      <TableCell>
                        <span
                          className={
                            i.websocket_connected
                              ? "text-emerald-500"
                              : "text-muted-foreground"
                          }
                        >
                          {i.websocket_connected ? "●" : "○"} ep{i.epoch}
                        </span>
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            )}
          </CardContent>
        </Card>

        <div className="flex flex-col gap-4">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2 text-sm">
                <Coins className="size-4" /> Token usage
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-1 text-sm">
              <div className="flex justify-between">
                <span className="text-muted-foreground">Input</span>
                <span className="font-mono">{overview?.input_tokens ?? 0}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted-foreground">Output</span>
                <span className="font-mono">
                  {overview?.output_tokens ?? 0}
                </span>
              </div>
              <div className="flex justify-between border-t pt-1 font-medium">
                <span>Total</span>
                <span className="font-mono">{overview?.total_tokens ?? 0}</span>
              </div>
              <Link
                to="/responses"
                className="mt-2 inline-flex items-center gap-1 text-xs text-primary hover:underline"
              >
                Usage details <ArrowRight className="size-3" />
              </Link>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2 text-sm">
                <MessagesSquare className="size-4" /> Recent responses
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-2">
              {(recent?.responses ?? []).length === 0 ? (
                <p className="text-sm text-muted-foreground">
                  Nothing yet — try the Playground.
                </p>
              ) : (
                (recent?.responses ?? []).map((r) => (
                  <div
                    key={r.id}
                    className="flex items-center justify-between gap-2 text-sm"
                  >
                    <div className="min-w-0 truncate">
                      <span className="font-medium">{r.model_alias}</span>{" "}
                      <span className="text-xs text-muted-foreground">
                        {r.api_format === "chat_completions" ? "chat" : "resp"}
                      </span>
                    </div>
                    <div className="flex items-center gap-2 shrink-0">
                      <span className="font-mono text-xs text-muted-foreground">
                        {r.total_tokens} tok
                      </span>
                      <StatusBadge status={r.status} />
                    </div>
                  </div>
                ))
              )}
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  )
}

function Dashboard() {
  return (
    <Suspense
      fallback={
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
          {Array.from({ length: 4 }).map((_, i) => (
            <Skeleton key={i} className="h-28 rounded-lg" />
          ))}
        </div>
      }
    >
      <DashboardContent />
    </Suspense>
  )
}
