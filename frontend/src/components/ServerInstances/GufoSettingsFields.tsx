import { Info } from "lucide-react"
import type { Model } from "@/client"
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

export type GufoOptions = {
  context?: number
  mmproj?: string
  max_tokens?: number
  temperature?: number
  top_k?: number
  top_p?: number
  min_p?: number
  min_keep?: number
  seed?: number
  repeat_penalty?: number
  repeat_last_n?: number
  frequency_penalty?: number
  presence_penalty?: number
  think?: "on" | "off" | "auto"
  reasoning_effort?:
    | "auto"
    | "minimal"
    | "low"
    | "medium"
    | "high"
    | "xhigh"
    | "max"
  preserve_thinking?: "on" | "off" | "auto"
  speculative?: "dflash2" | "dspark" | "mtp" | "off"
  dflash_model?: string
  dspark_model?: string
  mtp_model?: string
  draft_policy?: "fixed" | "adaptive"
  draft_tokens?: number
  min_draft_tokens?: number
  prefill_chunk?: number
  max_pending?: number
  max_pending_per_client?: number
  request_timeout_ms?: number
  max_output_bytes?: number
  max_buffered_output_bytes?: number
  max_buffered_output_total?: number
  cache_disk?: string
  cache_disk_bytes?: number
  cache_disk_staging_bytes?: number
  sessions?: number
  max_connections?: number
  max_request_bytes?: number
  api_key?: string
  verbose?: boolean
  log_progress?: boolean
}

type Props = {
  options: GufoOptions
  onChange: (options: GufoOptions) => void
  models?: Model[]
}

type NumberField = readonly [keyof GufoOptions, string, number, string]

const descriptions: Record<string, string> = {
  context:
    "Maximum context capacity in tokens per session. 0 uses the model's native context. Memory use depends on the model and the number of resident sessions.",
  mmproj:
    "Qwen BF16 vision sidecar for image input. Auto-discovered beside the model when left unset.",
  max_tokens:
    "Default new-token limit per request. -1 generates until EOS or the context is full. Reasoning tokens count toward this limit.",
  temperature:
    "Randomness scale; 0 selects greedy decoding. Low values give predictable answers, high values more varied ones.",
  top_k: "Keep only the K highest-logit tokens. 0 disables the filter.",
  top_p:
    "Nucleus sampling: keep the smallest token set whose cumulative probability reaches P.",
  min_p:
    "Discard tokens whose probability falls below P times the best token's probability. Stays consistent across models.",
  min_keep:
    "Always keep at least N candidates after the sampling filters so the sampler never collapses to a single choice.",
  seed: "Fixed RNG seed for reproducible generation. -1 uses a random seed.",
  repeat_penalty:
    "Multiplicative penalty for recently used tokens; the anti-loop knob. 1.0 disables it.",
  repeat_last_n:
    "How far back (in tokens) the repetition penalties look. 0 disables the window.",
  frequency_penalty:
    "Penalty that grows with each token's occurrence count, reducing word-for-word repetition.",
  presence_penalty:
    "Flat penalty for any token already generated, pushing the model toward new words and topics.",
  think:
    'Default reasoning mode: whether the model shows its thinking before answering. "auto" uses the model default (enabled for Qwen3.8 and DeepSeek V4 Flash).',
  reasoning_effort:
    "Model instructions controlling reasoning depth, not a token limit. Auto selects Qwen xhigh or DeepSeek high; more effort can increase response time and generated tokens.",
  preserve_thinking:
    "Whether earlier turns' thinking is kept in the conversation history. Helps follow-ups build on prior reasoning at the cost of context.",
  speculative:
    "Draft backend for speculative decoding: dflash2 for Qwen3.8-27B, dspark for DeepSeek V4 Flash, mtp for models with a supported MTP head, or off.",
  dflash_model:
    "Path to the Qwen DFlash2 companion draft GGUF. Pick from the model library.",
  dspark_model:
    "Path to the DeepSeek V4 Flash DSpark support GGUF. Pick from the model library.",
  mtp_model:
    "Path to the MTP draft GGUF (for Qwen3.8-Flash-Next, the mtp-...-shared-*.gguf sidecar).",
  draft_policy:
    "DFlash2 block-length policy. adaptive varies the draft length with acceptance; fixed always drafts the full block.",
  draft_tokens:
    "Maximum speculative draft tokens evaluated per step. Wider drafts go faster when accepted and waste work when rejected.",
  min_draft_tokens:
    "Adaptive draft floor. DFlash2, DSpark, and Flash-Next MTP require the default of 1.",
  prefill_chunk:
    "Maximum prompt tokens processed between active decode rounds. Larger chunks prefill faster but leave less room for concurrent decoding.",
  max_pending:
    "Maximum queued generation requests. Over the limit, clients wait or are refused rather than exhausting memory.",
  max_pending_per_client:
    "Queue cap per client IP so one client cannot hog the whole queue.",
  request_timeout_ms:
    "Queue plus generation timeout in milliseconds; requests are killed after it. 0 disables the timeout.",
  max_output_bytes: "Maximum generated bytes per request.",
  max_buffered_output_bytes:
    "Maximum generated-but-not-yet-delivered stream bytes buffered per request; protects against slow clients.",
  max_buffered_output_total:
    "The same stream buffer budget across all requests.",
  cache_disk:
    "Directory for the opt-in restart-safe continuation cache. Repeated or shared prompts get much faster after restarts; costs disk space.",
  cache_disk_bytes: "Retained disk-cache byte budget.",
  cache_disk_staging_bytes:
    "RAM limit for queued snapshots and each disk read. 0 = auto (at most 1 GiB and 1/8 of available RAM). Must fit alongside the cache budget.",
  sessions:
    "Preallocated GPU request sessions (-j). More sessions let more requests compute at once, at the cost of memory per session.",
  max_connections:
    "Maximum simultaneous HTTP connections. Clients beyond the limit queue or are refused instead of piling up.",
  max_request_bytes:
    "Maximum HTTP request body size. Matters mostly for large payloads such as images.",
  api_key:
    "If set, every request requires an Authorization: Bearer <key> header, including health checks.",
  verbose: "Chattier server logs.",
  log_progress: "Log live prefill and decode progress.",
}

function OptionLabel({ name, label }: { name: string; label: string }) {
  return (
    <div className="flex items-center gap-1">
      <Label htmlFor={`gufo-${name}`}>{label}</Label>
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

export function GufoSettingsFields({ options, onChange, models = [] }: Props) {
  const setNumber = (key: keyof GufoOptions, value: string) => {
    const next = { ...options }
    if (value === "") delete next[key]
    else next[key] = Number(value) as never
    onChange(next)
  }

  const setText = (key: keyof GufoOptions, value: string) => {
    const next = { ...options }
    if (value === "") delete next[key]
    else next[key] = value as never
    onChange(next)
  }

  const setEnum = (key: keyof GufoOptions, value: string) => {
    const next = { ...options }
    if (value === "default") delete next[key]
    else next[key] = value as never
    onChange(next)
  }

  const setBool = (key: keyof GufoOptions, value: string) => {
    const next = { ...options }
    if (value === "default") delete next[key]
    else next[key] = (value === "on") as never
    onChange(next)
  }

  const numberFields = (fields: readonly NumberField[]) => (
    <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
      {fields.map(([key, label, min, step]) => (
        <div key={key}>
          <OptionLabel name={key} label={label} />
          <Input
            id={`gufo-${key}`}
            type="number"
            min={min}
            step={step}
            value={(options[key] as number | string | undefined) ?? ""}
            onChange={(event) => setNumber(key, event.target.value)}
            className="mt-1"
            placeholder="Server default"
          />
        </div>
      ))}
    </div>
  )

  const textFields = (
    fields: readonly (readonly [keyof GufoOptions, string])[],
  ) => (
    <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
      {fields.map(([key, label]) => (
        <div key={key}>
          <OptionLabel name={key} label={label} />
          <Input
            id={`gufo-${key}`}
            value={(options[key] as string | undefined) ?? ""}
            onChange={(event) => setText(key, event.target.value)}
            className="mt-1"
            placeholder="Server default"
          />
        </div>
      ))}
    </div>
  )

  const enumSelect = (
    key: keyof GufoOptions,
    label: string,
    values: readonly string[],
  ) => (
    <div>
      <OptionLabel name={key} label={label} />
      <Select
        value={(options[key] as string | undefined) ?? "default"}
        onValueChange={(value) => setEnum(key, value)}
      >
        <SelectTrigger className="mt-1">
          <SelectValue />
        </SelectTrigger>
        <SelectContent>
          <SelectItem value="default">Server default</SelectItem>
          {values.map((value) => (
            <SelectItem key={value} value={value}>
              {value}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
    </div>
  )

  const boolSelect = (key: keyof GufoOptions, label: string) => (
    <div>
      <OptionLabel name={key} label={label} />
      <Select
        value={
          options[key] === undefined
            ? "default"
            : options[key] === true
              ? "on"
              : "off"
        }
        onValueChange={(value) => setBool(key, value)}
      >
        <SelectTrigger className="mt-1">
          <SelectValue />
        </SelectTrigger>
        <SelectContent>
          <SelectItem value="default">Server default</SelectItem>
          <SelectItem value="on">on</SelectItem>
          <SelectItem value="off">off</SelectItem>
        </SelectContent>
      </Select>
    </div>
  )

  const modelPathSelect = (
    key: keyof GufoOptions,
    label: string,
    modelType: string,
  ) => {
    const current = options[key] as string | undefined
    const isLibraryValue = models.some((model) => model.path === current)
    return (
      <div>
        <OptionLabel name={key} label={label} />
        <Select
          value={current ?? "default"}
          onValueChange={(value) => setEnum(key, value)}
        >
          <SelectTrigger className="mt-1">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="default">Server default</SelectItem>
            {models
              .filter((model) => model.model_type === modelType)
              .map((model) => (
                <SelectItem key={model.id} value={model.path as string}>
                  {model.name}
                </SelectItem>
              ))}
            {current !== undefined && !isLibraryValue && (
              <SelectItem value={current}>{current}</SelectItem>
            )}
          </SelectContent>
        </Select>
      </div>
    )
  }

  return (
    <div className="space-y-4 rounded-md border p-4">
      <p className="text-sm text-muted-foreground">
        Leave fields empty or at &quot;Server default&quot; to use Gufo&apos;s
        own defaults. Settings are passed as command-line flags when the process
        starts.
      </p>

      <details open className="rounded-md border px-4">
        <summary className="cursor-pointer py-3 font-medium">
          Model &amp; context
        </summary>
        <div className="space-y-4 pb-4">
          {modelPathSelect("mmproj", "Vision sidecar (mmproj)", "mmproj")}
          {numberFields([["context", "Context tokens per session", 0, "1024"]])}
        </div>
      </details>

      <details className="rounded-md border px-4">
        <summary className="cursor-pointer py-3 font-medium">
          Sampling defaults
        </summary>
        <div className="space-y-4 pb-4">
          {numberFields([
            ["max_tokens", "Max tokens per response", -1, "1"],
            ["temperature", "Temperature", 0, "0.01"],
            ["top_k", "Top K", 0, "1"],
            ["top_p", "Top P", 0, "0.01"],
            ["min_p", "Min P", 0, "0.01"],
            ["min_keep", "Min keep", 0, "1"],
            ["seed", "Seed", -1, "1"],
            ["repeat_penalty", "Repeat penalty", 0, "0.01"],
            ["repeat_last_n", "Repeat last N", 0, "1"],
            ["frequency_penalty", "Frequency penalty", -2, "0.01"],
            ["presence_penalty", "Presence penalty", -2, "0.01"],
          ])}
        </div>
      </details>

      <details className="rounded-md border px-4">
        <summary className="cursor-pointer py-3 font-medium">
          Reasoning defaults
        </summary>
        <div className="grid grid-cols-1 gap-4 sm:grid-cols-3">
          {enumSelect("think", "Think mode", ["auto", "on", "off"])}
          {enumSelect("reasoning_effort", "Reasoning effort", [
            "auto",
            "minimal",
            "low",
            "medium",
            "high",
            "xhigh",
            "max",
          ])}
          {enumSelect("preserve_thinking", "Preserve thinking", [
            "auto",
            "on",
            "off",
          ])}
        </div>
      </details>

      <details className="rounded-md border px-4">
        <summary className="cursor-pointer py-3 font-medium">
          Speculative decoding
        </summary>
        <div className="space-y-4 pb-4">
          <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
            {enumSelect("speculative", "Draft backend", [
              "dflash2",
              "dspark",
              "mtp",
              "off",
            ])}
            {enumSelect("draft_policy", "Draft policy", ["fixed", "adaptive"])}
          </div>
          <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
            {modelPathSelect("dflash_model", "DFlash2 draft model", "dflash")}
            {modelPathSelect("dspark_model", "DSpark support model", "llm")}
            {modelPathSelect("mtp_model", "MTP draft model", "mtp")}
          </div>
          {numberFields([
            ["draft_tokens", "Draft tokens per step", 1, "1"],
            ["min_draft_tokens", "Min draft tokens", 1, "1"],
          ])}
        </div>
      </details>

      <details className="rounded-md border px-4">
        <summary className="cursor-pointer py-3 font-medium">
          Scheduling &amp; limits
        </summary>
        <div className="space-y-4 pb-4">
          {numberFields([
            ["prefill_chunk", "Prefill chunk", 1, "1"],
            ["max_pending", "Max pending requests", 0, "1"],
            ["max_pending_per_client", "Max pending per client", 0, "1"],
            ["request_timeout_ms", "Request timeout (ms)", 0, "1000"],
            ["max_output_bytes", "Max output bytes", 1, "1"],
            ["max_buffered_output_bytes", "Max buffered output bytes", 1, "1"],
            ["max_buffered_output_total", "Max buffered output total", 1, "1"],
          ])}
        </div>
      </details>

      <details className="rounded-md border px-4">
        <summary className="cursor-pointer py-3 font-medium">
          Disk cache
        </summary>
        <div className="space-y-4 pb-4">
          {textFields([["cache_disk", "Cache directory"]] as const)}
          {numberFields([
            ["cache_disk_bytes", "Cache size (bytes)", 0, "1"],
            ["cache_disk_staging_bytes", "Staging bytes", 0, "1"],
          ])}
        </div>
      </details>

      <details className="rounded-md border px-4">
        <summary className="cursor-pointer py-3 font-medium">Server</summary>
        <div className="space-y-4 pb-4">
          {numberFields([
            ["sessions", "GPU sessions", 1, "1"],
            ["max_connections", "Max connections", 1, "1"],
            ["max_request_bytes", "Max request bytes", 1, "1"],
          ])}
          {textFields([["api_key", "API key"]] as const)}
          <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
            {boolSelect("log_progress", "Log progress")}
            {boolSelect("verbose", "Verbose logging")}
          </div>
        </div>
      </details>
    </div>
  )
}
