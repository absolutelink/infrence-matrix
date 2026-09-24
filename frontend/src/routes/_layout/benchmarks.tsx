import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query"
import { createFileRoute } from "@tanstack/react-router"
import {
  Activity,
  Clock3,
  FileJson,
  Pencil,
  Play,
  Plus,
  Square,
  Trash2,
  XCircle,
} from "lucide-react"
import { useState } from "react"
import { toast } from "sonner"
import type { ServerInstanceResponse } from "@/client"
import {
  CopyServerDialog,
  DefinitionDialog,
} from "@/components/Benchmarks/DefinitionDialog"
import { useLogPanel } from "@/components/ServerInstances/LogPanelContext"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card"
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import type { BenchmarkDefinition, BenchmarkRun } from "@/lib/benchmarkApi"
import { benchmarkApi } from "@/lib/benchmarkApi"

export const Route = createFileRoute("/_layout/benchmarks")({
  component: Benchmarks,
  head: () => ({ meta: [{ title: "Benchmarks - Inference Matrix" }] }),
})

const activeStatuses = new Set([
  "queued",
  "pending",
  "waiting_for_idle",
  "stopping_servers",
  "idle_timeout",
  "running",
  "starting",
])
const date = (value?: string | null) =>
  value ? new Date(value).toLocaleString() : "-"
const statusVariant = (status: string) =>
  activeStatuses.has(status)
    ? "default"
    : status === "failed"
      ? "destructive"
      : "secondary"

function Benchmarks() {
  const [dialogOpen, setDialogOpen] = useState(false)
  const [copyDialogOpen, setCopyDialogOpen] = useState(false)
  const [editing, setEditing] = useState<BenchmarkDefinition>()
  const [copyFrom, setCopyFrom] = useState<ServerInstanceResponse>()
  const definitions = useQuery({
    queryKey: ["benchmark-definitions"],
    queryFn: benchmarkApi.listDefinitions,
  })
  const runs = useQuery({
    queryKey: ["benchmark-runs"],
    queryFn: benchmarkApi.listRuns,
    refetchInterval: 5000,
  })
  const queryClient = useQueryClient()
  const { openLogs } = useLogPanel()
  const action = useMutation({
    mutationFn: ({
      id,
      kind,
    }: {
      id: string
      kind: "cancel" | "abort" | "force-stop"
    }) =>
      kind === "cancel"
        ? benchmarkApi.cancelRun(id)
        : kind === "abort"
          ? benchmarkApi.abortRun(id)
          : benchmarkApi.forceStopRun(id),
    onSuccess: () => {
      toast.success("Run action sent")
      queryClient.invalidateQueries({ queryKey: ["benchmark-runs"] })
    },
    onError: () => toast.error("Could not update benchmark run"),
  })
  const deleteDefinition = useMutation({
    mutationFn: benchmarkApi.deleteDefinition,
    onSuccess: () => {
      toast.success("Definition deleted")
      queryClient.invalidateQueries({ queryKey: ["benchmark-definitions"] })
    },
    onError: () => toast.error("Could not delete definition"),
  })
  const runDefinition = useMutation({
    mutationFn: benchmarkApi.runDefinition,
    onSuccess: (run) => {
      toast.success("Benchmark queued")
      queryClient.invalidateQueries({ queryKey: ["benchmark-runs"] })
      if (run.agent_id) {
        openLogs({
          id: run.id,
          kind: "benchmark",
          run_id: run.id,
          agent_id: run.agent_id,
          agent_name: run.agent_name,
        })
      }
    },
    onError: () => toast.error("Could not queue benchmark"),
  })

  const confirmAction = (message: string, callback: () => void) => {
    if (window.confirm(message)) callback()
  }
  const list = runs.data ?? []
  const queue = list.filter((run) => activeStatuses.has(run.status))

  return (
    <div className="flex flex-col gap-6">
      <div className="flex flex-wrap items-center justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold tracking-tight">Benchmarks</h1>
          <p className="text-muted-foreground">
            Define repeatable workloads and inspect their performance results.
          </p>
        </div>
        <div className="flex flex-wrap gap-2">
          <Button variant="outline" onClick={() => setCopyDialogOpen(true)}>
            Copy from server instances
          </Button>
          <Button
            onClick={() => {
              setEditing(undefined)
              setCopyFrom(undefined)
              setDialogOpen(true)
            }}
          >
            <Plus className="mr-2 h-4 w-4" />
            New definition
          </Button>
        </div>
      </div>
      <Tabs defaultValue="definitions">
        <TabsList>
          <TabsTrigger value="definitions">Definitions</TabsTrigger>
          <TabsTrigger value="runs">
            Queue & history{" "}
            <Badge variant="secondary" className="ml-2">
              {queue.length}
            </Badge>
          </TabsTrigger>
        </TabsList>
        <TabsContent value="definitions" className="mt-4">
          <Card>
            <CardHeader>
              <CardTitle>Benchmark definitions</CardTitle>
              <CardDescription>
                Reusable configurations submitted to the benchmark runner.
              </CardDescription>
            </CardHeader>
            <CardContent>
              {definitions.isLoading ? (
                <p className="py-8 text-center text-muted-foreground">
                  Loading definitions...
                </p>
              ) : definitions.isError ? (
                <p className="py-8 text-center text-destructive">
                  Could not load definitions.
                </p>
              ) : (
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead>Name</TableHead>
                      <TableHead>Description</TableHead>
                      <TableHead>Updated</TableHead>
                      <TableHead className="text-right">Actions</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {(definitions.data ?? []).map((definition) => (
                      <TableRow key={definition.id}>
                        <TableCell className="font-medium">
                          {definition.name}
                        </TableCell>
                        <TableCell className="max-w-md truncate text-muted-foreground">
                          {definition.description || "-"}
                        </TableCell>
                        <TableCell>
                          {date(definition.updated_at ?? definition.created_at)}
                        </TableCell>
                        <TableCell>
                          <div className="flex justify-end gap-1">
                            <Button
                              variant="ghost"
                              size="icon"
                              title="Run"
                              onClick={() =>
                                runDefinition.mutate(definition.id)
                              }
                              disabled={runDefinition.isPending}
                            >
                              <Play className="h-4 w-4" />
                            </Button>
                            <Button
                              variant="ghost"
                              size="icon"
                              title="Edit"
                              onClick={() => {
                                setEditing(definition)
                                setDialogOpen(true)
                              }}
                            >
                              <Pencil className="h-4 w-4" />
                            </Button>
                            <Button
                              variant="ghost"
                              size="icon"
                              title="Delete"
                              onClick={() =>
                                confirmAction(
                                  `Delete ${definition.name}?`,
                                  () => deleteDefinition.mutate(definition.id),
                                )
                              }
                            >
                              <Trash2 className="h-4 w-4 text-destructive" />
                            </Button>
                          </div>
                        </TableCell>
                      </TableRow>
                    ))}
                    {definitions.data?.length === 0 && (
                      <TableRow>
                        <TableCell
                          colSpan={4}
                          className="py-10 text-center text-muted-foreground"
                        >
                          No definitions yet.
                        </TableCell>
                      </TableRow>
                    )}
                  </TableBody>
                </Table>
              )}
            </CardContent>
          </Card>
        </TabsContent>
        <TabsContent value="runs" className="mt-4">
          <div className="grid gap-4 md:grid-cols-3">
            <Card>
              <CardHeader className="pb-3">
                <CardDescription>Queued or running</CardDescription>
                <CardTitle className="text-3xl">{queue.length}</CardTitle>
              </CardHeader>
              <CardContent>
                <Activity className="h-5 w-5 text-primary" />
              </CardContent>
            </Card>
            <Card>
              <CardHeader className="pb-3">
                <CardDescription>Completed runs</CardDescription>
                <CardTitle className="text-3xl">
                  {list.filter((run) => !activeStatuses.has(run.status)).length}
                </CardTitle>
              </CardHeader>
              <CardContent>
                <Clock3 className="h-5 w-5 text-muted-foreground" />
              </CardContent>
            </Card>
            <Card>
              <CardHeader className="pb-3">
                <CardDescription>Last result</CardDescription>
                <CardTitle className="truncate text-base">
                  {list.find((run) => run.results)?.definition_name ??
                    "No results"}
                </CardTitle>
              </CardHeader>
              <CardContent>
                <FileJson className="h-5 w-5 text-muted-foreground" />
              </CardContent>
            </Card>
          </div>
          <Card className="mt-4">
            <CardHeader>
              <CardTitle>Run history</CardTitle>
              <CardDescription>
                Active runs refresh every five seconds.
              </CardDescription>
            </CardHeader>
            <CardContent>
              <RunsTable
                runs={list}
                onAction={(id, kind) =>
                  confirmAction(
                    `${kind === "force-stop" ? "Force stop" : kind[0].toUpperCase() + kind.slice(1)} this run?`,
                    () => action.mutate({ id, kind }),
                  )
                }
              />
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
      <DefinitionDialog
        open={dialogOpen}
        definition={editing}
        copyFrom={copyFrom}
        onClose={() => setDialogOpen(false)}
      />
      <CopyServerDialog
        open={copyDialogOpen}
        onClose={() => setCopyDialogOpen(false)}
        onSelect={(server) => {
          setCopyFrom(server)
          setEditing(undefined)
          setCopyDialogOpen(false)
          setDialogOpen(true)
        }}
      />
    </div>
  )
}

function RunsTable({
  runs,
  onAction,
}: {
  runs: BenchmarkRun[]
  onAction: (id: string, kind: "cancel" | "abort" | "force-stop") => void
}) {
  const [result, setResult] = useState<{ id: string; value: unknown }>()
  const results = useMutation({
    mutationFn: benchmarkApi.getResults,
    onSuccess: (value, id) => setResult({ id, value }),
    onError: () => toast.error("Could not load results"),
  })
  return (
    <Table>
      <TableHeader>
        <TableRow>
          <TableHead>Definition</TableHead>
          <TableHead>Status</TableHead>
          <TableHead>Created</TableHead>
          <TableHead>Finished</TableHead>
          <TableHead className="text-right">Actions</TableHead>
        </TableRow>
      </TableHeader>
      <TableBody>
        {runs.map((run) => (
          <TableRow key={run.id}>
            <TableCell>
              <div className="font-medium">
                {run.definition_name ?? run.definition_id ?? "Unknown"}
              </div>
              <div className="font-mono text-xs text-muted-foreground">
                {run.id}
              </div>
            </TableCell>
            <TableCell>
              <Badge variant={statusVariant(run.status)}>{run.status}</Badge>
              {run.error && (
                <div className="max-w-48 truncate text-xs text-destructive">
                  {run.error}
                </div>
              )}
            </TableCell>
            <TableCell>{date(run.created_at ?? run.started_at)}</TableCell>
            <TableCell>{date(run.finished_at)}</TableCell>
            <TableCell>
              <div className="flex justify-end gap-1">
                {run.results != null && (
                  <Button
                    variant="ghost"
                    size="icon"
                    title="View results"
                    onClick={() =>
                      setResult({ id: run.id, value: run.results })
                    }
                  >
                    <FileJson className="h-4 w-4" />
                  </Button>
                )}
                {activeStatuses.has(run.status) && (
                  <>
                    <Button
                      variant="ghost"
                      size="icon"
                      title="Cancel"
                      onClick={() => onAction(run.id, "cancel")}
                    >
                      <XCircle className="h-4 w-4" />
                    </Button>
                    <Button
                      variant="ghost"
                      size="icon"
                      title="Abort"
                      onClick={() => onAction(run.id, "abort")}
                    >
                      <Square className="h-4 w-4" />
                    </Button>
                    <Button
                      variant="ghost"
                      size="icon"
                      title="Force stop"
                      onClick={() => onAction(run.id, "force-stop")}
                    >
                      <Square className="h-4 w-4 text-destructive" />
                    </Button>
                  </>
                )}
                {!activeStatuses.has(run.status) && !run.results && (
                  <Button
                    variant="ghost"
                    size="icon"
                    title="Load results"
                    onClick={() => results.mutate(run.id)}
                  >
                    <FileJson className="h-4 w-4" />
                  </Button>
                )}
              </div>
            </TableCell>
          </TableRow>
        ))}
        {runs.length === 0 && (
          <TableRow>
            <TableCell
              colSpan={5}
              className="py-10 text-center text-muted-foreground"
            >
              No benchmark runs yet.
            </TableCell>
          </TableRow>
        )}
        {result && (
          <TableRow>
            <TableCell colSpan={5}>
              <div className="rounded-md bg-muted p-4">
                <div className="mb-2 flex items-center justify-between">
                  <span className="font-medium">Results for {result.id}</span>
                  <Button
                    variant="ghost"
                    size="icon"
                    onClick={() => setResult(undefined)}
                  >
                    <XCircle className="h-4 w-4" />
                  </Button>
                </div>
                <pre className="max-h-80 overflow-auto whitespace-pre-wrap text-xs">
                  {JSON.stringify(result.value, null, 2)}
                </pre>
              </div>
            </TableCell>
          </TableRow>
        )}
      </TableBody>
    </Table>
  )
}
