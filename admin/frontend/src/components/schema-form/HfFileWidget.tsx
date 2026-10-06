// HfFileWidget (Phase 12 E3): the custom rjsf field behind
// `x-widget: "hf-file"` / `$ref: #/$defs/hfFile`.
//
// Renders the current artifact descriptor compactly
// ({source:hf,repo,file[,revision]} → "repo/file", {path} → path) and
// opens a picker dialog backed by the admin's HF proxy endpoints
// (AdminService.searchModels / AdminService.listRepoFiles). The local
// path mode emits {"path": ...}. The descriptor shape matches the
// oneOf in provider/README.md; the driver resolves it via
// provider_lib.downloader.ensure_artifact.
//
// Label/description/x-flag/secret/not-wired chrome is rendered by the
// surrounding HintFieldTemplate — this component is the control only.

import type { FieldProps } from "@rjsf/utils"
import { useQuery } from "@tanstack/react-query"
import { Search, X } from "lucide-react"
import { useEffect, useState } from "react"

import { AdminService } from "@/client"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table"
import { extractError } from "@/lib/errors"

import { asRecord, describeHfDescriptor } from "./keywords"

type HfDescriptor = {
  source: "hf"
  repo: string
  file: string
  revision?: string
}

type RepoFileEntry = {
  rfilename: string
  size: number | null
  kind: string
  quantization: string | null
  model_type: string | null
  is_aux: boolean
}

function formatSize(bytes: number | null): string {
  if (bytes == null) return "—"
  if (bytes === 0) return "0 B"
  const k = 1024
  const sizes = ["B", "KB", "MB", "GB", "TB"]
  const i = Math.floor(Math.log(bytes) / Math.log(k))
  return `${parseFloat((bytes / k ** i).toFixed(2))} ${sizes[i]}`
}

function HfFilePickerDialog({
  open,
  onClose,
  onSelect,
}: {
  open: boolean
  onClose: () => void
  onSelect: (descriptor: HfDescriptor) => void
}) {
  const [query, setQuery] = useState("")
  const [debounced, setDebounced] = useState("")
  const [repo, setRepo] = useState<string | null>(null)
  const [showAll, setShowAll] = useState(false)

  useEffect(() => {
    const t = setTimeout(() => setDebounced(query.trim()), 350)
    return () => clearTimeout(t)
  }, [query])

  const search = useQuery({
    queryKey: ["hf-search", debounced],
    queryFn: async () =>
      ((
        await AdminService.searchModels({
          query: { search: debounced, limit: 20 },
        })
      ).data ?? []) as Array<Record<string, unknown>>,
    enabled: open && debounced.length > 0 && repo === null,
  })

  const files = useQuery({
    queryKey: ["hf-files", repo, showAll],
    queryFn: async () =>
      ((
        await AdminService.listRepoFiles({
          query: { repo_id: repo as string, include_all: showAll },
        })
      ).data ?? {}) as { repo_id: string; files: RepoFileEntry[] },
    enabled: open && repo !== null,
  })

  const reset = () => {
    setQuery("")
    setDebounced("")
    setRepo(null)
    setShowAll(false)
  }

  return (
    <Dialog
      open={open}
      onOpenChange={(o) => {
        if (!o) {
          reset()
          onClose()
        }
      }}
    >
      <DialogContent className="sm:max-w-2xl max-h-[85vh] overflow-y-auto">
        <DialogHeader>
          <DialogTitle>
            {repo ? `Files in ${repo}` : "Search HuggingFace"}
          </DialogTitle>
          <DialogDescription>
            Pick a model repo, then choose the artifact file — stored as{" "}
            <code className="rounded bg-muted px-1 font-mono text-[11px]">
              {"{source:hf,repo,file}"}
            </code>{" "}
            and downloaded by the provider at start.
          </DialogDescription>
        </DialogHeader>

        {repo === null ? (
          <>
            <div className="relative">
              <Search className="absolute left-2.5 top-2.5 size-4 text-muted-foreground" />
              <Input
                autoFocus
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder="e.g. Qwen3 GGUF"
                className="pl-8"
              />
            </div>
            {debounced.length === 0 ? (
              <p className="py-6 text-center text-sm text-muted-foreground">
                Type to search HuggingFace models.
              </p>
            ) : search.isLoading ? (
              <p className="py-6 text-center text-sm text-muted-foreground">
                Searching…
              </p>
            ) : search.isError ? (
              <p className="py-6 text-center text-sm text-destructive">
                {extractError(search.error)}
              </p>
            ) : (search.data ?? []).length === 0 ? (
              <p className="py-6 text-center text-sm text-muted-foreground">
                No repos found.
              </p>
            ) : (
              <div className="rounded-md border">
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead>Repo</TableHead>
                      <TableHead>Author</TableHead>
                      <TableHead className="text-right">DL</TableHead>
                      <TableHead className="text-right">Likes</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {(search.data ?? []).map((m) => {
                      const id = String(m.id ?? "")
                      if (!id) return null
                      return (
                        <TableRow
                          key={id}
                          className="cursor-pointer hover:bg-accent/50"
                          onClick={() => setRepo(id)}
                        >
                          <TableCell className="font-mono text-xs">
                            {id}
                          </TableCell>
                          <TableCell className="text-xs text-muted-foreground">
                            {String(m.author ?? "")}
                          </TableCell>
                          <TableCell className="text-right font-mono text-xs">
                            {Number(m.downloads ?? 0).toLocaleString()}
                          </TableCell>
                          <TableCell className="text-right font-mono text-xs">
                            {Number(m.likes ?? 0).toLocaleString()}
                          </TableCell>
                        </TableRow>
                      )
                    })}
                  </TableBody>
                </Table>
              </div>
            )}
          </>
        ) : (
          <>
            <div className="flex items-center justify-between gap-2">
              <Button variant="outline" size="sm" onClick={() => setRepo(null)}>
                ← Back to search
              </Button>
              <label className="flex items-center gap-2 text-xs text-muted-foreground">
                <input
                  type="checkbox"
                  checked={showAll}
                  onChange={(e) => setShowAll(e.target.checked)}
                />
                show unclassified files
              </label>
            </div>
            {files.isLoading ? (
              <p className="py-6 text-center text-sm text-muted-foreground">
                Loading files…
              </p>
            ) : files.isError ? (
              <p className="py-6 text-center text-sm text-destructive">
                {extractError(files.error)}
              </p>
            ) : (files.data?.files ?? []).length === 0 ? (
              <p className="py-6 text-center text-sm text-muted-foreground">
                No recognized artifact files in this repo.
              </p>
            ) : (
              <div className="rounded-md border">
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead>File</TableHead>
                      <TableHead>Kind</TableHead>
                      <TableHead>Quant</TableHead>
                      <TableHead className="text-right">Size</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {(files.data?.files ?? []).map((f) => (
                      <TableRow
                        key={f.rfilename}
                        className="cursor-pointer hover:bg-accent/50"
                        onClick={() => {
                          onSelect({ source: "hf", repo, file: f.rfilename })
                          reset()
                        }}
                      >
                        <TableCell className="font-mono text-xs break-all">
                          {f.rfilename}
                          {f.is_aux && (
                            <span className="ml-1 text-[10px] text-muted-foreground">
                              (aux)
                            </span>
                          )}
                        </TableCell>
                        <TableCell>
                          <Badge
                            variant="secondary"
                            className="font-mono text-[10px]"
                          >
                            {f.kind}
                          </Badge>
                        </TableCell>
                        <TableCell className="font-mono text-xs">
                          {f.quantization ?? "—"}
                        </TableCell>
                        <TableCell className="text-right font-mono text-xs">
                          {formatSize(f.size)}
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </div>
            )}
          </>
        )}
      </DialogContent>
    </Dialog>
  )
}

/**
 * The rjsf custom field registered as `hfFile` (wired via `ui:field`
 * from buildUiSchema when a property is an hf-file artifact
 * descriptor). Control row: current descriptor summary + Select/Change
 * (picker dialog) + Local path (inline {"path": ...} editor) + clear.
 */
export function HfFileField(props: FieldProps) {
  const {
    name,
    fieldPathId,
    formData,
    required,
    disabled,
    readonly,
    onChange,
  } = props
  const [pickerOpen, setPickerOpen] = useState(false)
  const [localMode, setLocalMode] = useState(false)
  const [localPath, setLocalPath] = useState(
    typeof asRecord(formData)?.path === "string"
      ? (asRecord(formData)?.path as string)
      : "",
  )

  const rec = asRecord(formData)
  const summary = describeHfDescriptor(formData)
  const isLocal = typeof rec?.path === "string"

  return (
    <div className="flex flex-col gap-2">
      <div className="flex items-center gap-2">
        {summary ? (
          <code
            id={`${name}-hffile`}
            className="min-w-0 flex-1 truncate rounded-md border bg-muted/50 px-2 py-1.5 font-mono text-xs"
            title={summary}
          >
            {!isLocal && (
              <Badge
                variant="outline"
                className="mr-2 h-4 border-sky-500/40 px-1 text-[9px] text-sky-600 dark:text-sky-400"
              >
                hf
              </Badge>
            )}
            {summary}
          </code>
        ) : (
          <span className="flex-1 rounded-md border border-dashed px-2 py-1.5 text-xs text-muted-foreground">
            no artifact selected
          </span>
        )}
        <Button
          type="button"
          variant="outline"
          size="sm"
          disabled={disabled || readonly}
          onClick={() => {
            setLocalMode(false)
            setPickerOpen(true)
          }}
        >
          {summary && !isLocal ? "Change" : "Select"}
        </Button>
        <Button
          type="button"
          variant="ghost"
          size="sm"
          disabled={disabled || readonly}
          onClick={() => {
            setLocalMode(!localMode)
            setLocalPath(isLocal ? (rec?.path as string) : "")
          }}
          title="Use a local path descriptor"
        >
          Local path
        </Button>
        {!required && summary && (
          <Button
            type="button"
            variant="ghost"
            size="icon-sm"
            disabled={disabled || readonly}
            onClick={() => onChange(undefined, fieldPathId.path)}
            aria-label={`Clear ${name}`}
          >
            <X />
          </Button>
        )}
      </div>

      {localMode && (
        <div className="flex items-end gap-2 rounded-md border bg-muted/30 p-2">
          <div className="flex-1">
            <Label
              htmlFor={`${name}-localpath`}
              className="text-xs text-muted-foreground"
            >
              Local path on the provider machine
            </Label>
            <Input
              id={`${name}-localpath`}
              value={localPath}
              disabled={disabled || readonly}
              placeholder="/models/my-model.gguf"
              className="mt-1 font-mono text-xs"
              onChange={(e) => setLocalPath(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter") {
                  e.preventDefault()
                  if (localPath.trim() !== "") {
                    onChange({ path: localPath.trim() }, fieldPathId.path)
                    setLocalMode(false)
                  }
                }
              }}
            />
          </div>
          <Button
            type="button"
            size="sm"
            disabled={disabled || readonly || localPath.trim() === ""}
            onClick={() => {
              onChange({ path: localPath.trim() }, fieldPathId.path)
              setLocalMode(false)
            }}
          >
            Use path
          </Button>
        </div>
      )}

      <HfFilePickerDialog
        open={pickerOpen}
        onClose={() => setPickerOpen(false)}
        onSelect={(d) => {
          onChange(d, fieldPathId.path)
          setPickerOpen(false)
        }}
      />
    </div>
  )
}
