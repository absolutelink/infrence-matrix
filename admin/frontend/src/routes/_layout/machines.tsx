import { zodResolver } from "@hookform/resolvers/zod"
import { useMutation, useQueryClient } from "@tanstack/react-query"
import { createFileRoute } from "@tanstack/react-router"
import {
  Check,
  Copy,
  Eye,
  EyeOff,
  Pencil,
  Plus,
  RotateCw,
  Trash2,
} from "lucide-react"
import { useState } from "react"
import { useForm } from "react-hook-form"
import { z } from "zod"
import { AdminService } from "@/client"
import { EmptyNudge } from "@/components/Common/EmptyNudge"
import { StatusBadge } from "@/components/Common/StatusBadge"
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
  Form,
  FormControl,
  FormDescription,
  FormField,
  FormItem,
  FormLabel,
  FormMessage,
} from "@/components/ui/form"
import { Input } from "@/components/ui/input"
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table"
import { machineKeys, useMachines } from "@/hooks/useAdminData"
import { useCopyToClipboard } from "@/hooks/useCopyToClipboard"
import useCustomToast from "@/hooks/useCustomToast"
import { extractError } from "@/lib/errors"
import type { Machine } from "@/types/admin"

export const Route = createFileRoute("/_layout/machines")({
  component: MachinesPage,
  head: () => ({ meta: [{ title: "Machines - Inference Matrix" }] }),
})

const machineSchema = z.object({
  uid: z.string().min(1, "uid is required").max(255),
  name: z.string().min(1, "name is required").max(255),
  host: z.string().max(255).optional(),
  dns: z.string().max(255).optional(),
  ip: z.string().max(255).optional(),
  total_vram_bytes: z
    .string()
    .refine((t) => Number.isInteger(Number(t)) && Number(t) >= 0, {
      message: "must be an integer ≥ 0",
    }),
})

type MachineFormValues = z.infer<typeof machineSchema>

function MachinesPage() {
  const { data: machines = [], isLoading } = useMachines()
  const [editing, setEditing] = useState<Machine | null>(null)
  const [creating, setCreating] = useState(false)
  const [deleting, setDeleting] = useState<Machine | null>(null)

  return (
    <div className="flex flex-col gap-6">
      <div className="flex items-start justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold tracking-tight">Machines</h1>
          <p className="text-muted-foreground">
            Hosts that provider instances register against. Create the machine
            (uid) before starting a provider container with that MACHINE_UID.
          </p>
        </div>
        <Button onClick={() => setCreating(true)}>
          <Plus /> Add machine
        </Button>
      </div>

      {isLoading ? (
        <p className="text-muted-foreground">Loading machines…</p>
      ) : machines.length === 0 ? (
        <EmptyNudge
          text="No machines registered"
          actionLabel="Create your first machine"
          onAction={() => setCreating(true)}
        />
      ) : (
        <div className="rounded-md border">
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>UID</TableHead>
                <TableHead>Name</TableHead>
                <TableHead>Address</TableHead>
                <TableHead>Total VRAM</TableHead>
                <TableHead>Agents</TableHead>
                <TableHead className="text-right">Actions</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {machines.map((m) => (
                <MachineRow
                  key={m.id}
                  machine={m}
                  onEdit={() => setEditing(m)}
                  onDelete={() => setDeleting(m)}
                />
              ))}
            </TableBody>
          </Table>
        </div>
      )}

      <MachineFormDialog
        // Remount per target: useForm defaultValues are read only at first
        // mount, so editing B after A would otherwise show A's values.
        key={editing?.id ?? (creating ? "create" : "closed")}
        open={creating || editing !== null}
        machine={editing}
        onClose={() => {
          setCreating(false)
          setEditing(null)
        }}
      />
      <DeleteMachineDialog
        machine={deleting}
        onClose={() => setDeleting(null)}
      />
    </div>
  )
}

function MachineRow({
  machine,
  onEdit,
  onDelete,
}: {
  machine: Machine
  onEdit: () => void
  onDelete: () => void
}) {
  const [expanded, setExpanded] = useState(false)
  const address = machine.dns || machine.host || machine.ip || "—"
  const vramGb = machine.total_vram_bytes
    ? `${(machine.total_vram_bytes / 1024 ** 3).toFixed(1)} GiB`
    : "0"

  return (
    <>
      <TableRow
        className="cursor-pointer"
        onClick={() => setExpanded(!expanded)}
      >
        <TableCell className="font-mono text-xs">{machine.uid}</TableCell>
        <TableCell className="font-medium">{machine.name}</TableCell>
        <TableCell className="text-muted-foreground">{address}</TableCell>
        <TableCell className="font-mono text-xs">{vramGb}</TableCell>
        <TableCell>
          {machine.agent_count ?? 0}{" "}
          <span className="text-xs text-muted-foreground">attached</span>
        </TableCell>
        <TableCell className="text-right">
          <div className="inline-flex gap-1">
            <Button
              variant="ghost"
              size="icon-sm"
              onClick={onEdit}
              aria-label={`Edit ${machine.name}`}
            >
              <Pencil />
            </Button>
            <Button
              variant="ghost"
              size="icon-sm"
              className="text-destructive hover:bg-destructive/10"
              onClick={onDelete}
              aria-label={`Delete ${machine.name}`}
            >
              <Trash2 />
            </Button>
          </div>
        </TableCell>
      </TableRow>
      {expanded && (
        <TableRow className="bg-muted/40 hover:bg-muted/40">
          <TableCell colSpan={6} className="py-4">
            <div className="mb-4">
              <MachineSecretPanel machine={machine} />
            </div>
            <HardwareView machine={machine} />
          </TableCell>
        </TableRow>
      )}
    </>
  )
}

function HardwareView({ machine }: { machine: Machine }) {
  const gpus = machine.hardware?.gpus ?? []
  const cpu = machine.hardware?.cpu as Record<string, unknown> | undefined
  const ram = machine.hardware?.ram as Record<string, unknown> | undefined
  return (
    <div className="grid gap-4 md:grid-cols-3">
      <div>
        <h4 className="mb-2 text-sm font-semibold">GPUs ({gpus.length})</h4>
        {gpus.length === 0 ? (
          <p className="text-xs text-muted-foreground">
            No GPU report yet — hardware merges from provider registration.
          </p>
        ) : (
          <ul className="space-y-1">
            {gpus.map((g, idx) => (
              <li key={g.uuid ?? idx} className="text-xs">
                <span className="font-medium">{g.name ?? "GPU"}</span>{" "}
                <StatusBadge status={g.vendor ?? "unknown"} />{" "}
                {g.total_vram_bytes
                  ? `${(g.total_vram_bytes / 1024 ** 3).toFixed(1)} GiB`
                  : ""}
                {g.uuid && (
                  <span className="ml-1 font-mono text-muted-foreground">
                    {g.uuid.slice(0, 12)}
                  </span>
                )}
              </li>
            ))}
          </ul>
        )}
      </div>
      <div>
        <h4 className="mb-2 text-sm font-semibold">CPU</h4>
        <KvList obj={cpu} empty="No CPU report." />
      </div>
      <div>
        <h4 className="mb-2 text-sm font-semibold">RAM</h4>
        <KvList obj={ram} empty="No RAM report." />
      </div>
    </div>
  )
}

function KvList({
  obj,
  empty,
}: {
  obj: Record<string, unknown> | undefined
  empty: string
}) {
  if (!obj || Object.keys(obj).length === 0) {
    return <p className="text-xs text-muted-foreground">{empty}</p>
  }
  return (
    <ul className="space-y-0.5">
      {Object.entries(obj).map(([k, v]) => (
        <li key={k} className="flex justify-between text-xs">
          <span className="text-muted-foreground">{k}</span>
          <span className="font-mono">{String(v)}</span>
        </li>
      ))}
    </ul>
  )
}

function MachineSecretPanel({ machine }: { machine: Machine }) {
  const queryClient = useQueryClient()
  const { showErrorToast, showSuccessToast } = useCustomToast()
  const [revealed, setRevealed] = useState(false)
  const [secret, setSecret] = useState(machine.registration_secret ?? "")
  const [copied, copy] = useCopyToClipboard()

  const rotate = useMutation({
    mutationFn: async () =>
      await AdminService.rotateMachineSecret({
        path: { machine_id: machine.id },
      }),
    onSuccess: (resp) => {
      const next = (resp?.data as Record<string, unknown>)
        ?.registration_secret as string | undefined
      if (next) {
        setSecret(next)
        setRevealed(true)
      }
      queryClient.invalidateQueries({ queryKey: machineKeys.all })
      showSuccessToast(
        "Registration secret rotated — redeploy every agent on this machine with the new MACHINE_SECRET.",
      )
    },
    onError: (err: Error) => showErrorToast(extractError(err)),
  })

  const masked = "•".repeat(24)

  return (
    <div className="rounded-md border bg-background p-3">
      <div className="mb-1 flex items-center justify-between gap-2">
        <h4 className="text-sm font-semibold">Registration secret</h4>
        <span className="text-xs text-muted-foreground">
          paste as <code className="font-mono">MACHINE_SECRET</code> in provider
          containers on this machine
        </span>
      </div>
      <div className="flex flex-wrap items-center gap-2">
        <code className="min-w-0 flex-1 truncate rounded bg-muted px-2 py-1 font-mono text-xs">
          {secret ? (revealed ? secret : masked) : "—"}
        </code>
        <Button
          type="button"
          variant="outline"
          size="sm"
          onClick={() => setRevealed((r) => !r)}
          disabled={!secret}
          aria-label={revealed ? "Hide secret" : "Reveal secret"}
        >
          {revealed ? <EyeOff /> : <Eye />}
          {revealed ? "Hide" : "Reveal"}
        </Button>
        <Button
          type="button"
          variant="outline"
          size="sm"
          disabled={!secret}
          onClick={async () => {
            if (secret && (await copy(secret))) {
              showSuccessToast("Registration secret copied")
            } else {
              showErrorToast("Copy failed")
            }
          }}
          aria-label="Copy secret"
        >
          {copied ? <Check /> : <Copy />}
          {copied ? "Copied" : "Copy"}
        </Button>
        <Button
          type="button"
          variant="secondary"
          size="sm"
          disabled={rotate.isPending}
          onClick={() => rotate.mutate()}
          title="Mint a new secret (agents keep the old one until redeployed)"
        >
          <RotateCw />
          {rotate.isPending ? "Rotating…" : "Rotate"}
        </Button>
      </div>
    </div>
  )
}

function MachineFormDialog({
  open,
  machine,
  onClose,
}: {
  open: boolean
  machine: Machine | null
  onClose: () => void
}) {
  const queryClient = useQueryClient()
  const { showErrorToast, showSuccessToast } = useCustomToast()
  const isEdit = machine !== null

  const form = useForm<MachineFormValues>({
    resolver: zodResolver(machineSchema),
    defaultValues: {
      uid: machine?.uid ?? "",
      name: machine?.name ?? "",
      host: machine?.host ?? "",
      dns: machine?.dns ?? "",
      ip: machine?.ip ?? "",
      total_vram_bytes: String(machine?.total_vram_bytes ?? 0),
    },
  })

  const mutation = useMutation({
    mutationFn: async (values: MachineFormValues) => {
      const body = {
        uid: values.uid,
        name: values.name,
        host: values.host || null,
        dns: values.dns || null,
        ip: values.ip || null,
        total_vram_bytes: Number(values.total_vram_bytes),
      }
      if (isEdit && machine) {
        // uid is immutable server-side; don't send it.
        const { uid: _uid, ...patch } = body
        return await AdminService.patchMachine({
          path: { machine_id: machine.id },
          body: patch,
        })
      }
      return await AdminService.createMachine({ body })
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: machineKeys.all })
      showSuccessToast(isEdit ? "Machine updated" : "Machine created")
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
      <DialogContent className="sm:max-w-lg">
        <DialogHeader>
          <DialogTitle>{isEdit ? "Edit machine" : "Add machine"}</DialogTitle>
          <DialogDescription>
            {isEdit
              ? "Update machine details. The uid cannot be changed — it is referenced by provider containers and Redis leases."
              : "Register a host that provider instances will register against."}
          </DialogDescription>
        </DialogHeader>
        <Form {...form}>
          <form
            onSubmit={form.handleSubmit((v) => mutation.mutate(v))}
            className="space-y-4"
          >
            <FormField
              control={form.control}
              name="uid"
              render={({ field }) => (
                <FormItem>
                  <FormLabel>UID</FormLabel>
                  <FormControl>
                    <Input
                      {...field}
                      disabled={isEdit}
                      placeholder="core-2-111"
                      className={isEdit ? "opacity-60" : ""}
                    />
                  </FormControl>
                  <FormDescription>
                    Matches the provider container&apos;s MACHINE_UID env.
                    {isEdit && " Immutable."}
                  </FormDescription>
                  <FormMessage />
                </FormItem>
              )}
            />
            <FormField
              control={form.control}
              name="name"
              render={({ field }) => (
                <FormItem>
                  <FormLabel>Name</FormLabel>
                  <FormControl>
                    <Input {...field} placeholder="Provider Host 111" />
                  </FormControl>
                  <FormMessage />
                </FormItem>
              )}
            />
            <div className="grid grid-cols-1 gap-4 sm:grid-cols-3">
              <FormField
                control={form.control}
                name="host"
                render={({ field }) => (
                  <FormItem>
                    <FormLabel>Host</FormLabel>
                    <FormControl>
                      <Input {...field} placeholder="hostname" />
                    </FormControl>
                    <FormMessage />
                  </FormItem>
                )}
              />
              <FormField
                control={form.control}
                name="dns"
                render={({ field }) => (
                  <FormItem>
                    <FormLabel>DNS</FormLabel>
                    <FormControl>
                      <Input {...field} placeholder="box.local" />
                    </FormControl>
                    <FormMessage />
                  </FormItem>
                )}
              />
              <FormField
                control={form.control}
                name="ip"
                render={({ field }) => (
                  <FormItem>
                    <FormLabel>IP</FormLabel>
                    <FormControl>
                      <Input {...field} placeholder="10.0.0.5" />
                    </FormControl>
                    <FormMessage />
                  </FormItem>
                )}
              />
            </div>
            <FormField
              control={form.control}
              name="total_vram_bytes"
              render={({ field }) => (
                <FormItem>
                  <FormLabel>Total VRAM (bytes)</FormLabel>
                  <FormControl>
                    <Input type="number" min={0} step={1024 ** 3} {...field} />
                  </FormControl>
                  <FormDescription>
                    Scheduler admission budget. Refreshed from provider hardware
                    reports.
                  </FormDescription>
                  <FormMessage />
                </FormItem>
              )}
            />
            <p className="text-xs text-muted-foreground">
              Address preference: dns → host → ip (reachable_address()).
            </p>
            <DialogFooter>
              <Button type="button" variant="outline" onClick={onClose}>
                Cancel
              </Button>
              <Button type="submit" disabled={mutation.isPending}>
                {mutation.isPending ? "Saving…" : isEdit ? "Save" : "Create"}
              </Button>
            </DialogFooter>
          </form>
        </Form>
      </DialogContent>
    </Dialog>
  )
}

function DeleteMachineDialog({
  machine,
  onClose,
}: {
  machine: Machine | null
  onClose: () => void
}) {
  const queryClient = useQueryClient()
  const { showErrorToast, showSuccessToast } = useCustomToast()

  const mutation = useMutation({
    mutationFn: async () => {
      if (!machine) return
      return await AdminService.deleteMachine({
        path: { machine_id: machine.id },
      })
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: machineKeys.all })
      showSuccessToast("Machine deleted")
      onClose()
    },
    onError: (err: Error) => showErrorToast(extractError(err)),
  })

  return (
    <Dialog
      open={machine !== null}
      onOpenChange={(o) => {
        if (!o) onClose()
      }}
    >
      <DialogContent className="sm:max-w-md">
        <DialogHeader>
          <DialogTitle>Delete machine</DialogTitle>
          <DialogDescription>
            Delete machine{" "}
            <span className="font-mono font-semibold">{machine?.uid}</span> (
            {machine?.name})? This cannot be undone. Refused while provider
            instances are attached.
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
