import { Info } from "lucide-react"
import { Checkbox } from "@/components/ui/checkbox"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip"

export type HalogenFlashOptions = {
  kv_slots?: number
  kv_pool_positions?: number
  kv_pool_fit?: number
  host_reserve_gib?: number
  ctx?: number
  max_tok?: number
  rope_yarn?: number
  admit_chunk?: number
  indexer_budget?: number
  max_tokens_cap?: number
  max_tokens_default?: number
  queue_timeout?: number
  keepalive_timeout?: number
  sse_keepalive_s?: number
  temperature?: number
  top_p?: number
  top_k?: number
  min_p?: number
  presence_penalty?: number
  frequency_penalty?: number
  reasoning_effort?: "minimal" | "low" | "medium" | "high" | "xhigh"
  enable_thinking?: number
  max_thinking_tokens?: number
  thinking_answer_room?: number
  drafter_default?: number
  mtp_depth?: number
  cache_dir_enabled?: boolean
  prompt_cache?: number
  prefill_chunk?: number
  cache_entries?: number
  cache_branches?: number
  cache_snap3?: number
  cache_full?: number
  cache_inplace?: number
  cache_disk_gib?: number
  cache_prune_old?: number
  composable_context?: number
  composable_context_floor?: number
  composable_context_bytes?: number
  pld?: string
  spec_adapt?: string
  grammar?: number
  vision_tower?: number
  vision_max_pixels?: number
  npu_models?: string[]
}

type Props = {
  options: HalogenFlashOptions
  onChange: (options: HalogenFlashOptions) => void
  npuAvailable?: boolean
  npuReasons?: string[]
}

type NumberField = readonly [keyof HalogenFlashOptions, string, number, string]

export const NPU_MODEL_CHOICES = [
  {
    id: "qwen3-embedding-0.6b",
    suffix: "embed",
    label: "Embeddings",
    description: "qwen3-embedding-0.6b — /v1/embeddings on the NPU.",
  },
  {
    id: "qwen3-reranker-0.6b",
    suffix: "rerank",
    label: "Rerank",
    description: "qwen3-reranker-0.6b — /v1/rerank on the NPU.",
  },
  {
    id: "decider-0.8b",
    suffix: "decide",
    label: "Decisions",
    description: "decider-0.8b — /v1/decisions single-pass classification.",
  },
  {
    id: "qwen3guard-gen-0.6b",
    suffix: "guard",
    label: "Moderation",
    description: "qwen3guard-gen-0.6b — /v1/moderations on the NPU.",
  },
  {
    id: "qwen3.5-2b",
    suffix: "nano",
    label: "Nano generation",
    description: "qwen3.5-2b — short chat/completions jobs on the NPU.",
  },
] as const

const descriptions: Record<string, string> = {
  kv_slots: "Maximum number of conversations generating at once.",
  kv_pool_positions:
    "Total attention positions reserved across all conversations.",
  kv_pool_fit:
    "Automatically lower the KV pool if it does not fit device memory.",
  host_reserve_gib: "System RAM to leave free when sizing the KV pool.",
  ctx: "Maximum context length for one request.",
  max_tok:
    "Largest single prefill call; keep this below the full context size.",
  rope_yarn:
    "RoPE scaling factor for contexts beyond the model's native context.",
  admit_chunk:
    "Prompt chunk size used while other conversations are generating.",
  indexer_budget:
    "Sparse-attention token budget. Higher values improve long-context recall at a cost.",
  max_tokens_cap: "Hard upper bound for a request's max_tokens.",
  max_tokens_default:
    "Token budget used when a request does not provide max_tokens.",
  queue_timeout: "Seconds a request may wait in the engine queue.",
  keepalive_timeout: "Seconds an idle HTTP keep-alive connection remains open.",
  sse_keepalive_s:
    "Seconds between SSE keepalive comments during quiet streaming.",
  temperature:
    "Default sampling temperature when a request does not provide one.",
  top_p: "Default nucleus sampling cutoff.",
  top_k:
    "Default number of highest-probability tokens considered; zero disables it.",
  min_p: "Default minimum probability relative to the most likely token.",
  presence_penalty:
    "Default penalty for tokens already present in the response.",
  frequency_penalty:
    "Default penalty based on how often tokens already appear.",
  max_thinking_tokens:
    "Default maximum tokens that may be spent in the thinking block.",
  thinking_answer_room:
    "Tokens reserved for the final answer when no thinking budget is supplied.",
  mtp_depth: "Number of tokens proposed by the speculative draft head.",
  prefill_chunk:
    "Tokens processed per prefill call and, in cache mode 1, the resume granularity.",
  cache_entries: "Number of prompt-cache entries retained using LRU eviction.",
  cache_branches: "Number of resume branches retained for each conversation.",
  cache_disk_gib: "Maximum disk usage for the managed prompt-cache directory.",
  cache_dir_enabled:
    "Persist prompt-cache entries under the agent's managed cache directory.",
  composable_context_floor:
    "Smallest message size eligible for composable context reuse.",
  composable_context_bytes:
    "Host-memory budget for retained composable messages.",
  pld: "Prompt lookup drafting parameters as N,K; zero disables prompt lookup.",
  spec_adapt: "Adaptive drafting parameters as window,floor,retry-tokens.",
  vision_max_pixels: "Maximum image pixels before downscaling.",
}

function OptionLabel({ name, label }: { name: string; label: string }) {
  return (
    <div className="flex items-center gap-1">
      <Label htmlFor={`halogen-flash-${name}`}>{label}</Label>
      <Tooltip>
        <TooltipTrigger asChild>
          <button
            type="button"
            aria-label={`About ${label}`}
            className="text-muted-foreground hover:text-foreground"
          >
            <Info className="size-3.5" />
          </button>
        </TooltipTrigger>
        <TooltipContent className="max-w-xs">
          {descriptions[name]}
        </TooltipContent>
      </Tooltip>
    </div>
  )
}

export function HalogenFlashSettingsFields({
  options,
  onChange,
  npuAvailable = false,
  npuReasons = [],
}: Props) {
  const enabledNpu = new Set(options.npu_models ?? [])

  const toggleNpuModel = (id: string, checked: boolean) => {
    const next = new Set(enabledNpu)
    if (checked) next.add(id)
    else next.delete(id)
    const list = NPU_MODEL_CHOICES.map((c) => c.id).filter((m) => next.has(m))
    const updated = { ...options }
    if (list.length) updated.npu_models = list
    else delete updated.npu_models
    onChange(updated)
  }

  const setNumber = (key: keyof HalogenFlashOptions, value: string) => {
    const next = { ...options }
    if (value === "") delete next[key]
    else next[key] = Number(value) as never
    onChange(next)
  }

  const setText = (key: keyof HalogenFlashOptions, value: string) => {
    const next = { ...options }
    if (value === "") delete next[key]
    else next[key] = value as never
    onChange(next)
  }

  const numberFields = (fields: readonly NumberField[]) => (
    <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
      {fields.map(([key, label, min, step]) => (
        <div key={key}>
          <OptionLabel name={key} label={label} />
          <Input
            id={`halogen-flash-${key}`}
            type="number"
            min={min}
            step={step}
            value={(options[key] as number | string | undefined) ?? ""}
            onChange={(event) => setNumber(key, event.target.value)}
            className="mt-1"
            placeholder="Agent default"
          />
        </div>
      ))}
    </div>
  )

  const toggle = (key: keyof HalogenFlashOptions, label: string) => (
    <div className="flex items-center gap-2 text-sm">
      <Checkbox
        checked={options[key] === 1}
        onCheckedChange={(checked) =>
          onChange({ ...options, [key]: checked === true ? 1 : undefined })
        }
      />
      <OptionLabel name={key} label={label} />
    </div>
  )

  return (
    <div className="space-y-4 rounded-md border p-4">
      <p className="text-sm text-muted-foreground">
        Leave fields empty to use the Halogen Flash defaults. Settings are
        passed as environment variables when the process starts.
      </p>

      <details open className="rounded-md border px-4">
        <summary className="cursor-pointer py-3 font-medium">
          Context and memory
        </summary>
        <div className="space-y-4 pb-4">
          {numberFields([
            ["kv_slots", "KV slots", 1, "1"],
            ["kv_pool_positions", "KV pool positions", 1, "1"],
            ["host_reserve_gib", "Host reserve (GiB)", 0, "1"],
            ["ctx", "Context size", 256, "256"],
            ["max_tok", "Maximum prefill tokens", 1, "1"],
            ["rope_yarn", "YaRN factor", 0, "0.1"],
            ["admit_chunk", "Admission chunk", -1, "1"],
            ["indexer_budget", "Attention budget", 2048, "16"],
          ])}
          <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
            <div>
              <OptionLabel name="kv_pool_fit" label="KV pool fit" />
              <Select
                value={options.kv_pool_fit?.toString() ?? "default"}
                onValueChange={(value) =>
                  setNumber("kv_pool_fit", value === "default" ? "" : value)
                }
              >
                <SelectTrigger className="mt-1">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="default">Agent default</SelectItem>
                  <SelectItem value="1">Fit automatically</SelectItem>
                  <SelectItem value="0">Allocate exactly</SelectItem>
                </SelectContent>
              </Select>
            </div>
          </div>
        </div>
      </details>

      <details className="rounded-md border px-4">
        <summary className="cursor-pointer py-3 font-medium">
          Request defaults
        </summary>
        <div className="space-y-4 pb-4">
          {numberFields([
            ["max_tokens_cap", "Maximum output tokens", 1, "1"],
            ["max_tokens_default", "Default output tokens", 1, "1"],
            ["queue_timeout", "Queue timeout (seconds)", 1, "1"],
            ["keepalive_timeout", "Keepalive timeout (seconds)", 0, "1"],
            ["sse_keepalive_s", "SSE keepalive (seconds)", 0, "1"],
            ["temperature", "Temperature", 0, "0.01"],
            ["top_p", "Top P", 0, "0.01"],
            ["top_k", "Top K", 0, "1"],
            ["min_p", "Min P", 0, "0.01"],
            ["presence_penalty", "Presence penalty", -2, "0.01"],
            ["frequency_penalty", "Frequency penalty", -2, "0.01"],
            ["max_thinking_tokens", "Maximum thinking tokens", 0, "1"],
            ["thinking_answer_room", "Thinking answer room", 0, "1"],
            ["mtp_depth", "MTP depth", 0, "1"],
          ])}
          <div>
            <OptionLabel name="reasoning_effort" label="Reasoning effort" />
            <Select
              value={options.reasoning_effort ?? "default"}
              onValueChange={(value) =>
                onChange({
                  ...options,
                  reasoning_effort:
                    value === "default"
                      ? undefined
                      : (value as HalogenFlashOptions["reasoning_effort"]),
                })
              }
            >
              <SelectTrigger className="mt-1">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="default">Agent default</SelectItem>
                {(["minimal", "low", "medium", "high", "xhigh"] as const).map(
                  (value) => (
                    <SelectItem key={value} value={value}>
                      {value}
                    </SelectItem>
                  ),
                )}
              </SelectContent>
            </Select>
          </div>
          <div className="grid gap-3 sm:grid-cols-2">
            {toggle("enable_thinking", "Enable thinking")}
            {toggle("drafter_default", "Enable draft head")}
          </div>
        </div>
      </details>

      <details className="rounded-md border px-4">
        <summary className="cursor-pointer py-3 font-medium">
          Prompt cache
        </summary>
        <div className="space-y-4 pb-4">
          <div>
            <OptionLabel name="prompt_cache" label="Prompt cache mode" />
            <Select
              value={options.prompt_cache?.toString() ?? "default"}
              onValueChange={(value) =>
                setNumber("prompt_cache", value === "default" ? "" : value)
              }
            >
              <SelectTrigger className="mt-1">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="default">Agent default</SelectItem>
                <SelectItem value="0">Off</SelectItem>
                <SelectItem value="1">Chunk boundaries only</SelectItem>
                <SelectItem value="2">Resume anywhere</SelectItem>
              </SelectContent>
            </Select>
          </div>
          {numberFields([
            ["prefill_chunk", "Prefill chunk", 1, "1"],
            ["cache_entries", "Cache entries", 1, "1"],
            ["cache_branches", "Cache branches", 1, "1"],
            ["cache_disk_gib", "Cache disk limit (GiB)", 0, "1"],
            ["composable_context_floor", "Composable context floor", 0, "1"],
            ["composable_context_bytes", "Composable context bytes", 0, "1"],
          ])}
          <div className="flex items-center gap-2 text-sm">
            <Checkbox
              id="halogen-flash-cache-dir-enabled"
              checked={options.cache_dir_enabled === true}
              onCheckedChange={(checked) =>
                onChange({ ...options, cache_dir_enabled: checked === true })
              }
            />
            <OptionLabel
              name="cache_dir_enabled"
              label="Persist cache to disk"
            />
          </div>
          <div className="grid gap-3 sm:grid-cols-2">
            {toggle("cache_snap3", "Cache user-message start")}
            {toggle("cache_full", "Cache end of request")}
            {toggle("cache_inplace", "In-place cache snapshots")}
            {toggle("cache_prune_old", "Prune old cache builds")}
            {toggle("composable_context", "Composable context preview")}
          </div>
        </div>
      </details>

      <details className="rounded-md border px-4">
        <summary className="cursor-pointer py-3 font-medium">
          Drafting and vision
        </summary>
        <div className="space-y-4 pb-4">
          {(["pld", "spec_adapt"] as const).map((key) => (
            <div key={key}>
              <OptionLabel
                name={key}
                label={
                  key === "pld" ? "Prompt lookup (N,K)" : "Adaptive drafting"
                }
              />
              <Input
                id={`halogen-flash-${key}`}
                value={options[key] ?? ""}
                onChange={(event) => setText(key, event.target.value)}
                className="mt-1"
                placeholder="Agent default"
              />
            </div>
          ))}
          <div className="grid gap-3 sm:grid-cols-2">
            {toggle("grammar", "Structured output grammar")}
            {toggle("vision_tower", "Enable vision")}
          </div>
          {numberFields([["vision_max_pixels", "Vision max pixels", 1, "1"]])}
        </div>
      </details>

      <details className="rounded-md border px-4">
        <summary className="cursor-pointer py-3 font-medium">
          NPU small models (Ryzen AI)
        </summary>
        <div className="space-y-4 pb-4">
          {!npuAvailable ? (
            <p className="text-sm text-muted-foreground">
              The selected agent&apos;s host does not report a usable NPU.
              Enabling these models will fail at start unless the host has the
              NPU driver, XRT with its NPU plugin, and the GPU fabric clock held
              at its top speed.
            </p>
          ) : null}
          {npuReasons.length > 0 ? (
            <ul className="list-disc space-y-1 pl-5 text-xs text-muted-foreground">
              {npuReasons.map((reason) => (
                <li key={reason}>{reason}</li>
              ))}
            </ul>
          ) : null}
          <div className="grid gap-3 sm:grid-cols-2">
            {NPU_MODEL_CHOICES.map((choice) => (
              <div key={choice.id} className="flex items-start gap-2 text-sm">
                <Checkbox
                  id={`halogen-flash-npu-${choice.suffix}`}
                  disabled={!npuAvailable && !enabledNpu.has(choice.id)}
                  checked={enabledNpu.has(choice.id)}
                  onCheckedChange={(checked) =>
                    toggleNpuModel(choice.id, checked === true)
                  }
                />
                <div>
                  <Label htmlFor={`halogen-flash-npu-${choice.suffix}`}>
                    {choice.label}
                    <span className="ml-1 font-mono text-xs text-muted-foreground">
                      -{choice.suffix}
                    </span>
                  </Label>
                  <p className="text-xs text-muted-foreground">
                    {choice.description}
                  </p>
                </div>
              </div>
            ))}
          </div>
          {enabledNpu.size > 0 ? (
            <p className="text-xs text-muted-foreground">
              Enabled models are served on the same port as the Flash model and
              are reachable as{" "}
              <span className="font-mono">
                {Array.from(enabledNpu)
                  .map(
                    (id) =>
                      `&lt;alias&gt;-${
                        NPU_MODEL_CHOICES.find((c) => c.id === id)?.suffix ??
                        "?"
                      }`,
                  )
                  .join(", ")}
              </span>
              . Saving changes restarts the server and downloads the model
              files.
            </p>
          ) : null}
        </div>
      </details>
    </div>
  )
}
