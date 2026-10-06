// Provider Types page (Phase 12 E5): the schema registry view.
//
// Lists every registered provider type (name, status, committed
// fingerprint, instance count, consensus summary) and a detail panel
// with the committed schema (section tree + raw JSON). When a type is
// `consensus_pending` / `conflict`, the pending banner shows the
// pending fingerprint, the voter roster, the waiting_on list, and
// Commit / Dismiss operator overrides (docs/ws-protocol.md §2).

import { useMutation, useQueryClient } from "@tanstack/react-query"
import { createFileRoute, Link } from "@tanstack/react-router"
import { Check, ChevronDown, ChevronRight, ShieldCheck, X } from "lucide-react"
import { useEffect, useMemo, useState } from "react"

import { AdminService } from "@/client"
import { StatusBadge } from "@/components/Common/StatusBadge"
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert"
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
  providerTypeKeys,
  useProviderType,
  useProviderTypes,
} from "@/hooks/useAdminData"
import useCustomToast from "@/hooks/useCustomToast"
import { extractError } from "@/lib/errors"
import type {
  ProviderTypeConsensus,
  ProviderTypeDetail,
  ProviderTypeSummary,
} from "@/types/admin"

export const Route = createFileRoute("/_layout/provider-types")({
  // `?type=<name>` deep-links the Instances waiting_schema badge to a
  // type's pending consensus panel.
  validateSearch: (search: Record<string, unknown>): { type?: string } => ({
    type: typeof search.type === "string" ? search.type : undefined,
  }),
  component: ProviderTypesPage,
  head: () => ({
    meta: [{ title: "Provider Types - Inference Matrix" }],
  }),
})

function fpShort(fp: string | null | undefined): string {
  return fp ? `${fp.slice(0, 12)}…` : "—"
}

function consensusSummaryLine(c: ProviderTypeConsensus): string {
  if (c.pending_fingerprint) {
    return `${c.voter_count}/${c.universe_count} voted`
  }
  return `${c.on_committed}/${c.universe_count} on committed`
}

function ProviderTypesPage() {
  const { data: types = [], isLoading } = useProviderTypes()
  const { type: typeParam } = Route.useSearch()
  const [selected, setSelected] = useState<string | null>(typeParam ?? null)
  // N3: follow the ?type= param on every search change (e.g. arriving
  // via the instances waiting_schema badge while already mounted), not
  // just the initial state.
  useEffect(() => {
    if (typeParam) setSelected(typeParam)
  }, [typeParam])

  return (
    <div className="flex flex-col gap-6">
      <div>
        <h1 className="text-2xl font-bold tracking-tight">Provider Types</h1>
        <p className="text-muted-foreground">
          Schema registry: each type's committed{" "}
          <code className="rounded bg-muted px-1 font-mono text-xs">
            schema.json
          </code>{" "}
          gates definition validation and registrations. A changed schema needs
          every instance of the type to vote, or an operator force-commit (
          <Link to="/definitions" className="text-primary hover:underline">
            docs
          </Link>
          ).
        </p>
      </div>

      {isLoading ? (
        <p className="text-muted-foreground">Loading provider types…</p>
      ) : types.length === 0 ? (
        <div className="rounded-md border p-8 text-center text-sm text-muted-foreground">
          No provider types registered yet — they appear when the first provider
          container of each type registers.
        </div>
      ) : (
        <div className="rounded-md border">
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Name</TableHead>
                <TableHead>Status</TableHead>
                <TableHead>Committed fp</TableHead>
                <TableHead>Pending fp</TableHead>
                <TableHead>Consensus</TableHead>
                <TableHead className="text-right">Detail</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {types.map((t) => (
                <TypeRow
                  key={t.name}
                  type={t}
                  expanded={selected === t.name}
                  onToggle={() =>
                    setSelected(selected === t.name ? null : t.name)
                  }
                />
              ))}
            </TableBody>
          </Table>
        </div>
      )}
    </div>
  )
}

function TypeRow({
  type: t,
  expanded,
  onToggle,
}: {
  type: ProviderTypeSummary
  expanded: boolean
  onToggle: () => void
}) {
  const pending = t.consensus.pending_fingerprint != null
  return (
    <>
      <TableRow>
        <TableCell className="font-medium">
          <button
            type="button"
            className="mr-2 inline-flex rounded p-0.5 align-middle hover:bg-accent"
            onClick={onToggle}
            aria-label={expanded ? "Collapse" : "Expand"}
          >
            {expanded ? (
              <ChevronDown className="size-4 text-muted-foreground" />
            ) : (
              <ChevronRight className="size-4 text-muted-foreground" />
            )}
          </button>
          {t.name}
        </TableCell>
        <TableCell>
          <StatusBadge status={t.status} />
        </TableCell>
        <TableCell>
          <code className="font-mono text-xs text-muted-foreground">
            {fpShort(t.schema_fingerprint)}
          </code>
        </TableCell>
        <TableCell>
          {pending ? (
            <code className="font-mono text-xs text-amber-600 dark:text-amber-500">
              {fpShort(t.consensus.pending_fingerprint)}
            </code>
          ) : (
            <span className="text-xs text-muted-foreground">—</span>
          )}
        </TableCell>
        <TableCell className="text-xs text-muted-foreground">
          {consensusSummaryLine(t.consensus)}
        </TableCell>
        <TableCell className="text-right">
          <Button variant="ghost" size="sm" onClick={onToggle}>
            {expanded ? "Hide" : "View schema"}
          </Button>
        </TableCell>
      </TableRow>
      {expanded && (
        <TableRow className="bg-muted/40 hover:bg-muted/40">
          <TableCell colSpan={6} className="py-4">
            <TypeDetailPanel name={t.name} />
          </TableCell>
        </TableRow>
      )}
    </>
  )
}

function TypeDetailPanel({ name }: { name: string }) {
  const { data: detail, isLoading } = useProviderType(name)
  if (isLoading || !detail) {
    return <p className="text-sm text-muted-foreground">Loading schema…</p>
  }
  return (
    <div className="flex flex-col gap-4">
      <ConsensusBanner detail={detail} />
      <SchemaTree schema={detail.schema as Record<string, unknown>} />
      <RawSchemaView detail={detail} />
    </div>
  )
}

function ConsensusBanner({ detail }: { detail: ProviderTypeDetail }) {
  const queryClient = useQueryClient()
  const { showErrorToast, showSuccessToast } = useCustomToast()
  const [confirming, setConfirming] = useState<"commit" | "dismiss" | null>(
    null,
  )
  const c = detail.consensus
  const pending = c.pending_fingerprint != null
  const conflicted = detail.status === "conflict"

  const override = useMutation({
    mutationFn: async (action: "commit" | "dismiss") => {
      if (action === "commit") {
        return await AdminService.commitPending({ path: { name: detail.name } })
      }
      return await AdminService.dismissPending({ path: { name: detail.name } })
    },
    onSuccess: (_resp, action) => {
      queryClient.invalidateQueries({ queryKey: providerTypeKeys.all })
      queryClient.invalidateQueries({
        queryKey: providerTypeKeys.detail(detail.name),
      })
      showSuccessToast(
        action === "commit"
          ? `Force-committed pending schema for ${detail.name}. Affects future registrations and new/edited definitions only.`
          : `Dismissed the pending schema for ${detail.name} — the type is active on the committed schema.`,
      )
      setConfirming(null)
    },
    onError: (err: Error) => {
      showErrorToast(extractError(err))
      setConfirming(null)
    },
  })

  if (!pending) {
    return (
      <div className="flex items-center gap-2 text-xs text-muted-foreground">
        <ShieldCheck className="size-4 text-emerald-500" />
        No schema pending — all registrations must present{" "}
        <code className="rounded bg-muted px-1 font-mono">
          {fpShort(detail.schema_fingerprint)}
        </code>
        .
      </div>
    )
  }

  return (
    <Alert variant={conflicted ? "destructive" : "default"}>
      <AlertTitle>
        {conflicted
          ? `Schema conflict — a third fingerprint was presented while ${c.pending_fingerprint?.slice(
              0,
              12,
            )}… awaits consensus`
          : `Schema consensus pending for ${detail.name}`}
      </AlertTitle>
      <AlertDescription>
        <div className="grid gap-2 text-xs">
          <p>
            Pending fingerprint:{" "}
            <code className="rounded bg-muted px-1 font-mono">
              {fpShort(c.pending_fingerprint)}
            </code>{" "}
            (committed:{" "}
            <code className="rounded bg-muted px-1 font-mono">
              {fpShort(c.committed_fingerprint)}
            </code>
            )
          </p>
          <p>
            Voted ({c.voter_count}/{c.universe_count}):{" "}
            {c.voters.length === 0 ? (
              <span className="text-muted-foreground">none yet</span>
            ) : (
              c.voters.map((v) => (
                <Badge key={v} variant="outline" className="ml-1 font-mono">
                  {v.slice(0, 8)}
                </Badge>
              ))
            )}
          </p>
          <p>
            Waiting on ({c.waiting_on.length}):{" "}
            {c.waiting_on.length === 0 ? (
              <span className="text-muted-foreground">nobody</span>
            ) : (
              c.waiting_on.map((v) => (
                <Badge key={v} variant="secondary" className="ml-1 font-mono">
                  {v.slice(0, 8)}
                </Badge>
              ))
            )}
          </p>
          <p className="text-muted-foreground">
            Until consensus, instances presenting the pending schema are refused
            with 409 <code>schema_pending</code> and stay in{" "}
            <code>waiting_schema</code>. Force-commit promotes the pending
            schema immediately (future registrations + new/edited definitions
            only — connected old-schema instances keep running).
          </p>
          <div className="mt-1 flex gap-2">
            <Button
              size="sm"
              onClick={() => setConfirming("commit")}
              disabled={override.isPending}
            >
              <Check /> Commit pending
            </Button>
            <Button
              size="sm"
              variant="outline"
              onClick={() => setConfirming("dismiss")}
              disabled={override.isPending}
            >
              <X /> Dismiss
            </Button>
          </div>
        </div>
      </AlertDescription>

      <Dialog
        open={confirming !== null}
        onOpenChange={(o) => {
          if (!o) setConfirming(null)
        }}
      >
        <DialogContent className="sm:max-w-md">
          <DialogHeader>
            <DialogTitle>
              {confirming === "commit"
                ? "Force-commit pending schema"
                : "Dismiss pending schema"}
            </DialogTitle>
            <DialogDescription>
              {confirming === "commit"
                ? `Promote ${fpShort(c.pending_fingerprint)} to committed for ${detail.name}? Connected instances on the old schema keep running until they upgrade and re-register.`
                : `Drop the pending schema ${fpShort(c.pending_fingerprint)} for ${detail.name}? Instances will keep being refused until they present the committed fingerprint again.`}
            </DialogDescription>
          </DialogHeader>
          <DialogFooter>
            <Button
              type="button"
              variant="outline"
              onClick={() => setConfirming(null)}
            >
              Cancel
            </Button>
            <Button
              type="button"
              variant={confirming === "dismiss" ? "destructive" : "default"}
              disabled={override.isPending}
              onClick={() => confirming && override.mutate(confirming)}
            >
              {override.isPending
                ? "Working…"
                : confirming === "commit"
                  ? "Commit"
                  : "Dismiss"}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </Alert>
  )
}

/** Collapsible section tree of the committed schema: top-level sections
 *  with their field names, x-flag, and short type hints. */
function SchemaTree({ schema }: { schema: Record<string, unknown> }) {
  const [openSections, setOpenSections] = useState<Record<string, boolean>>({
    artifacts: true,
  })
  const sections = useMemo(() => {
    const props = (schema.properties ?? {}) as Record<string, unknown>
    return Object.entries(props).map(([key, value]) => {
      const rec = (value ?? {}) as Record<string, unknown>
      const fields = (rec.properties ?? {}) as Record<string, unknown>
      return {
        key,
        title: (rec.title as string) ?? key,
        order:
          typeof rec["x-order"] === "number" ? (rec["x-order"] as number) : 999,
        fields: Object.entries(fields).map(([fname, fvalue]) => {
          const fr = (fvalue ?? {}) as Record<string, unknown>
          const ref = fr.$ref
          const typeLabel =
            typeof ref === "string"
              ? ref.split("/").pop()
              : Array.isArray(fr.type)
                ? fr.type.join("|")
                : ((fr.type as string) ?? (fr.oneOf ? "oneOf" : "?"))
          return {
            name: fname,
            title: (fr.title as string) ?? fname,
            flag: fr["x-flag"] as string | undefined,
            type: String(typeLabel ?? "?"),
            supported: fr["x-supported"] !== false,
            secret: fr["x-secret"] === true,
          }
        }),
      }
    })
  }, [schema])

  if (sections.length === 0) {
    return (
      <p className="text-xs text-muted-foreground">
        This type's committed schema declares no sections (permissive
        bootstrap).
      </p>
    )
  }

  return (
    <div className="flex flex-col gap-1">
      {sections
        .sort((a, b) => a.order - b.order)
        .map((s) => {
          const open = openSections[s.key] ?? false
          return (
            <div key={s.key} className="rounded-md border bg-card">
              <button
                type="button"
                className="flex w-full items-center gap-2 px-3 py-2 text-left hover:bg-accent/50"
                onClick={() =>
                  setOpenSections((prev) => ({ ...prev, [s.key]: !open }))
                }
                aria-expanded={open}
              >
                {open ? (
                  <ChevronDown className="size-3.5 text-muted-foreground" />
                ) : (
                  <ChevronRight className="size-3.5 text-muted-foreground" />
                )}
                <span className="text-sm font-semibold">{s.title}</span>
                <span className="text-xs text-muted-foreground">
                  {s.key} · {s.fields.length} fields
                </span>
              </button>
              {open && (
                <div className="border-t px-3 py-2">
                  <Table>
                    <TableHeader>
                      <TableRow>
                        <TableHead className="text-xs">Field</TableHead>
                        <TableHead className="text-xs">Type</TableHead>
                        <TableHead className="text-xs">x-flag</TableHead>
                        <TableHead className="text-xs">Notes</TableHead>
                      </TableRow>
                    </TableHeader>
                    <TableBody>
                      {s.fields.map((f) => (
                        <TableRow key={f.name}>
                          <TableCell className="py-1">
                            <span className="font-mono text-xs">{f.name}</span>
                            <span className="ml-2 text-xs text-muted-foreground">
                              {f.title}
                            </span>
                          </TableCell>
                          <TableCell className="py-1 font-mono text-xs">
                            {f.type}
                          </TableCell>
                          <TableCell className="py-1 font-mono text-xs text-muted-foreground">
                            {f.flag ?? "—"}
                          </TableCell>
                          <TableCell className="py-1 space-x-1">
                            {!f.supported && (
                              <Badge
                                variant="outline"
                                className="text-[9px] text-zinc-500"
                              >
                                not wired
                              </Badge>
                            )}
                            {f.secret && (
                              <Badge
                                variant="outline"
                                className="text-[9px] text-violet-600 dark:text-violet-400"
                              >
                                secret
                              </Badge>
                            )}
                          </TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </div>
              )}
            </div>
          )
        })}
    </div>
  )
}

function RawSchemaView({ detail }: { detail: ProviderTypeDetail }) {
  const [open, setOpen] = useState(false)
  return (
    <div className="rounded-md border bg-card">
      <button
        type="button"
        className="flex w-full items-center gap-2 px-3 py-2 text-left hover:bg-accent/50"
        onClick={() => setOpen(!open)}
        aria-expanded={open}
      >
        {open ? (
          <ChevronDown className="size-3.5 text-muted-foreground" />
        ) : (
          <ChevronRight className="size-3.5 text-muted-foreground" />
        )}
        <span className="text-sm font-semibold">Raw schema JSON</span>
      </button>
      {open && (
        <pre className="max-h-96 overflow-auto border-t p-3 font-mono text-xs">
          {JSON.stringify(detail, null, 2)}
        </pre>
      )}
    </div>
  )
}
