import { useMutation, useQueryClient } from "@tanstack/react-query"
import { createFileRoute, Link } from "@tanstack/react-router"
import { ChevronDown, ChevronRight, Trash2 } from "lucide-react"
import { useMemo, useState } from "react"
import { AdminService } from "@/client"
import { EmptyNudge } from "@/components/Common/EmptyNudge"
import { ConnectionBadge, StatusBadge } from "@/components/Common/StatusBadge"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog"
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table"
import {
  agentKeys,
  useAgents,
  useDefinitions,
  useProviderTypes,
} from "@/hooks/useAdminData"
import useCustomToast from "@/hooks/useCustomToast"
import { extractError } from "@/lib/errors"
import type { ProviderAgent, ProviderDefinition } from "@/types/admin"

export const Route = createFileRoute("/_layout/agents")({
  component: AgentsPage,
  head: () => ({ meta: [{ title: "Agents - Inference Matrix" }] }),
})

// Phase 16: a provider agent is one hardware-local container bound to a
// (machine, provider_type, agent_id) triple. It owns 1..N backends
// (ProviderInstances) of its single type and holds the one WebSocket the admin
// commands it over. Backends nest under each agent in the API payload.
function AgentsPage() {
  const { data: agents = [], isLoading } = useAgents()
  const { data: definitions = [] } = useDefinitions()
  const { data: providerTypes = [], isPending: typesPending } =
    useProviderTypes()

  // provider_type → per-agent running cap (0 = unlimited).
  const maxRunningByType = useMemo(
    () =>
      new Map(providerTypes.map((t) => [t.name, t.max_running_backends ?? 0])),
    [providerTypes],
  )
  // definition id → alias (backends reference their definition by id).
  const aliasById = useMemo(
    () => new Map(definitions.map((d) => [d.id, d.alias])),
    [definitions],
  )
  // definition id → definition (for the placement summary in the detail).
  const defById = useMemo(
    () => new Map(definitions.map((d) => [d.id, d])),
    [definitions],
  )

  return (
    <div className="flex flex-col gap-6">
      <div>
        <h1 className="text-2xl font-bold tracking-tight">Provider Agents</h1>
        <p className="text-muted-foreground">
          Hardware-local containers, one per (machine, provider type). Each
          agent hosts the backends placed on it and holds a single WebSocket.
          See{" "}
          <Link to="/definitions" className="text-primary hover:underline">
            definitions
          </Link>{" "}
          for placement.
        </p>
      </div>

      {isLoading ? (
        <p className="text-muted-foreground">Loading agents…</p>
      ) : agents.length === 0 ? (
        <EmptyNudge
          text="No provider agents registered"
          actionLabel="Create a machine + definition"
          to="/machines"
        />
      ) : (
        <div className="rounded-md border">
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Machine</TableHead>
                <TableHead>Type</TableHead>
                <TableHead>Agent</TableHead>
                <TableHead>Status</TableHead>
                <TableHead>WS</TableHead>
                <TableHead>Version</TableHead>
                <TableHead>Max running</TableHead>
                <TableHead>Backends</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {agents.map((a) => (
                <AgentRow
                  key={a.id}
                  agent={a}
                  maxRunning={maxRunningByType.get(a.provider_type) ?? 0}
                  typesPending={typesPending}
                  aliasById={aliasById}
                  defById={defById}
                />
              ))}
            </TableBody>
          </Table>
        </div>
      )}
    </div>
  )
}

function AgentRow({
  agent: a,
  maxRunning,
  typesPending,
  aliasById,
  defById,
}: {
  agent: ProviderAgent
  maxRunning: number
  typesPending: boolean
  aliasById: Map<string, string>
  defById: Map<string, ProviderDefinition>
}) {
  const [expanded, setExpanded] = useState(false)
  const backends = a.backends ?? []
  return (
    <>
      <TableRow>
        <TableCell className="font-mono text-xs">
          {a.machine_uid ?? "—"}
        </TableCell>
        <TableCell>
          <StatusBadge status={a.provider_type} />
        </TableCell>
        <TableCell className="font-medium">
          <button
            type="button"
            className="mr-2 inline-flex rounded p-0.5 align-middle hover:bg-accent"
            onClick={() => setExpanded(!expanded)}
            aria-label={
              expanded ? "Collapse agent detail" : "Expand agent detail"
            }
          >
            {expanded ? (
              <ChevronDown className="size-4 text-muted-foreground" />
            ) : (
              <ChevronRight className="size-4 text-muted-foreground" />
            )}
          </button>
          <span className="font-mono text-xs">{a.agent_id}</span>
        </TableCell>
        <TableCell>
          <StatusBadge status={a.agent_status} />
        </TableCell>
        <TableCell>
          <ConnectionBadge connected={a.websocket_connected} />
        </TableCell>
        <TableCell className="font-mono text-xs">{a.version}</TableCell>
        <TableCell className="font-mono text-xs">
          {typesPending ? "…" : maxRunning === 0 ? "∞" : maxRunning}
        </TableCell>
        <TableCell>
          {backends.length}{" "}
          <span className="text-xs text-muted-foreground">hosted</span>
        </TableCell>
      </TableRow>
      {expanded && (
        <TableRow className="bg-muted/40 hover:bg-muted/40">
          <TableCell colSpan={8} className="py-4">
            <AgentDetail
              agent={a}
              maxRunning={maxRunning}
              typesPending={typesPending}
              aliasById={aliasById}
              defById={defById}
            />
          </TableCell>
        </TableRow>
      )}
    </>
  )
}

function AgentDetail({
  agent: a,
  maxRunning,
  typesPending,
  aliasById,
  defById,
}: {
  agent: ProviderAgent
  maxRunning: number
  typesPending: boolean
  aliasById: Map<string, string>
  defById: Map<string, ProviderDefinition>
}) {
  const [deleting, setDeleting] = useState(false)
  // Which definitions are placed on this agent — mirrors the server's
  // _placed_definitions (providers.py): only ENABLED definitions, and in
  // BOTH branches (any_of_type of the type ∪ specific linking this agent).
  const placed = useMemo(() => {
    const out: { alias: string; placement: string }[] = []
    for (const [, d] of defById) {
      const isSpecific =
        d.enabled &&
        d.agent_placement === "specific" &&
        (d.agents ?? []).includes(a.id)
      const isAnyOfType =
        d.enabled &&
        d.agent_placement === "any_of_type" &&
        d.provider_type === a.provider_type
      if (isSpecific || isAnyOfType) {
        out.push({ alias: d.alias, placement: d.agent_placement })
      }
    }
    return out
  }, [defById, a.id, a.provider_type])

  const backends = a.backends ?? []
  return (
    <div className="grid gap-6 md:grid-cols-2">
      <div>
        <h4 className="mb-2 text-sm font-semibold">
          Hosted backends ({backends.length})
          {!typesPending && maxRunning > 0 && (
            <span className="ml-2 text-xs font-normal text-muted-foreground">
              cap {maxRunning} running / agent
            </span>
          )}
        </h4>
        {backends.length === 0 ? (
          <p className="text-xs text-muted-foreground">
            This agent hosts no backends yet — place a definition of type{" "}
            <span className="font-mono">{a.provider_type}</span> on it.
          </p>
        ) : (
          <ul className="space-y-1 text-xs">
            {backends.map((b) => (
              <li key={b.id} className="flex items-center gap-2">
                <Link
                  to="/instances"
                  className="font-mono text-primary hover:underline"
                >
                  {aliasById.get(b.provider_definition_id) ??
                    b.provider_definition_id.slice(0, 8)}
                </Link>
                <StatusBadge status={b.backend_status} />
                <span className="font-mono text-muted-foreground">
                  :{b.port}
                </span>
                <code className="text-muted-foreground">
                  {b.config_fingerprint
                    ? `${b.config_fingerprint.slice(0, 8)}…`
                    : "—"}
                </code>
              </li>
            ))}
          </ul>
        )}
      </div>
      <div>
        <h4 className="mb-2 text-sm font-semibold">
          Placed definitions ({placed.length})
        </h4>
        {placed.length === 0 ? (
          <p className="text-xs text-muted-foreground">
            No definitions currently resolve to this agent.
          </p>
        ) : (
          <ul className="space-y-1 text-xs">
            {placed.map((p) => (
              <li key={p.alias} className="flex items-center gap-2">
                <Link
                  to="/definitions"
                  className="font-medium text-primary hover:underline"
                >
                  {p.alias}
                </Link>
                <Badge variant="outline" className="font-mono text-[10px]">
                  {p.placement}
                </Badge>
              </li>
            ))}
          </ul>
        )}
      </div>
      <div className="md:col-span-2">
        <h4 className="mb-2 text-sm font-semibold">Agent detail</h4>
        <dl className="grid grid-cols-2 gap-x-6 gap-y-1 text-xs sm:grid-cols-4">
          <Field k="agent id" v={a.agent_id} mono />
          <Field k="base port" v={String(a.base_port)} mono />
          <Field k="epoch" v={String(a.epoch)} mono />
          <Field k="version" v={a.version} mono />
          <Field
            k="reported schema"
            v={
              a.reported_schema_fingerprint
                ? `${a.reported_schema_fingerprint.slice(0, 12)}…`
                : "—"
            }
            mono
          />
          <Field k="last seen" v={a.last_seen ?? "never"} />
          <Field k="assigned GPUs" v={String(a.assigned_gpus?.length ?? 0)} />
          <Field k="specific links" v={String(a.definition_count ?? 0)} />
        </dl>
        {a.error_message && (
          <p className="mt-2 rounded-md border border-destructive/40 bg-destructive/5 p-2 text-xs text-destructive">
            {a.error_message}
          </p>
        )}
        <div className="mt-4 flex items-center justify-between gap-2 border-t pt-3">
          <p className="text-xs text-muted-foreground">
            {a.websocket_connected
              ? "Connected agents cannot be deleted — stop/redeploy the container first."
              : "Remove this decommissioned/renamed agent and its ghost backends."}
          </p>
          <Button
            type="button"
            variant="destructive"
            size="sm"
            disabled={a.websocket_connected}
            onClick={() => setDeleting(true)}
            title={
              a.websocket_connected
                ? "connected — stop the agent first"
                : "Delete this agent"
            }
          >
            <Trash2 />
            Delete
          </Button>
        </div>
      </div>
      <DeleteAgentDialog
        agent={a}
        open={deleting}
        onClose={() => setDeleting(false)}
      />
    </div>
  )
}

function DeleteAgentDialog({
  agent,
  open,
  onClose,
}: {
  agent: ProviderAgent
  open: boolean
  onClose: () => void
}) {
  const queryClient = useQueryClient()
  const { showErrorToast, showSuccessToast } = useCustomToast()

  const mutation = useMutation({
    mutationFn: async () =>
      await AdminService.deleteAgent({ path: { agent_id: agent.id } }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: agentKeys.all })
      showSuccessToast("Agent deleted")
      onClose()
    },
    onError: (err: Error) => showErrorToast(extractError(err)),
  })

  return (
    <Dialog
      open={open}
      onOpenChange={(o) => {
        if (!o) onClose()
      }}
    >
      <DialogContent className="sm:max-w-md">
        <DialogHeader>
          <DialogTitle>Delete agent</DialogTitle>
          <DialogDescription>
            Delete agent{" "}
            <span className="font-mono font-semibold">{agent.agent_id}</span> (
            {agent.provider_type})? This removes its{" "}
            {agent.backends?.length ?? 0} backend row(s) and placement links.
            This cannot be undone.
          </DialogDescription>
        </DialogHeader>
        <DialogFooter>
          <Button type="button" variant="outline" onClick={onClose}>
            Cancel
          </Button>
          <Button
            type="button"
            variant="destructive"
            disabled={mutation.isPending}
            onClick={() => mutation.mutate()}
          >
            {mutation.isPending ? "Deleting…" : "Delete"}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}

function Field({ k, v, mono }: { k: string; v: string; mono?: boolean }) {
  return (
    <div className="flex flex-col">
      <dt className="text-muted-foreground">{k}</dt>
      <dd className={mono ? "font-mono" : ""}>{v}</dd>
    </div>
  )
}
