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

type CacheType =
  | "f32"
  | "f16"
  | "bf16"
  | "q8_0"
  | "q4_0"
  | "q4_1"
  | "iq4_nl"
  | "q5_0"
  | "q5_1"
  | "turbo4"

export type ServerOptions = {
  threads?: number
  threads_batch?: number
  batch_size?: number
  ubatch_size?: number
  cache_type_k?: CacheType
  cache_type_v?: CacheType
  cache_prompt?: boolean
  cache_reuse?: number
  ctx_checkpoints?: number
  checkpoint_every?: number
  cache_ram?: number
  slot_save_path?: string
  kv_offload?: boolean
  kv_unified?: boolean
  no_mmap?: boolean
  no_cache_idle_slots?: boolean
  device?: string
  temperature?: number
  top_k?: number
  top_p?: number
  min_p?: number
  repeat_penalty?: number
  presence_penalty?: number
  frequency_penalty?: number
  seed?: number
  parallel?: number
  cont_batching?: boolean
  warmup?: boolean
  reasoning?: "on" | "off" | "auto"
  reasoning_budget?: number
  spec_draft_p_min?: number
  strict_mtp_qwen?: boolean
}

export function validateServerOptions(options: ServerOptions): string | null {
  if (options.cache_reuse !== undefined && options.cache_prompt === false) {
    return "Cache reuse requires prompt caching to be enabled."
  }
  if (options.top_p !== undefined && (options.top_p < 0 || options.top_p > 1)) {
    return "Top-p must be between 0 and 1."
  }
  if (options.min_p !== undefined && (options.min_p < 0 || options.min_p > 1)) {
    return "Min-p must be between 0 and 1."
  }
  if (
    options.spec_draft_p_min !== undefined &&
    (options.spec_draft_p_min < 0 || options.spec_draft_p_min > 1)
  ) {
    return "MTP draft probability minimum must be between 0 and 1."
  }
  return null
}

type Props = {
  options: ServerOptions
  onChange: (options: ServerOptions) => void
  mtpDraftMax: string
  onMtpDraftMaxChange: (value: string) => void
}

const cacheTypes: CacheType[] = [
  "f32",
  "f16",
  "bf16",
  "q8_0",
  "q4_0",
  "q4_1",
  "iq4_nl",
  "q5_0",
  "q5_1",
  "turbo4",
]

export function ServerSettingsFields({
  options,
  onChange,
  mtpDraftMax,
  onMtpDraftMaxChange,
}: Props) {
  const set = (key: keyof ServerOptions, value: string | boolean) => {
    const next = { ...options }
    if (typeof value === "boolean") {
      next[key] = value as never
    } else if (value === "") {
      delete next[key]
    } else {
      next[key] = Number(value) as never
    }
    onChange(next)
  }

  const setText = (key: keyof ServerOptions, value: string) => {
    const next = { ...options }
    if (value === "") delete next[key]
    else next[key] = value as never
    onChange(next)
  }

  const numericFields = (
    fields: ReadonlyArray<readonly [keyof ServerOptions, string]>,
  ) => (
    <div className="grid grid-cols-2 gap-4">
      {fields.map(([key, label]) => (
        <div key={key}>
          <Label htmlFor={`option-${key}`}>{label}</Label>
          <Input
            id={`option-${key}`}
            type="number"
            step={
              key === "top_p" ||
              key === "min_p" ||
              key === "temperature" ||
              key === "spec_draft_p_min"
                ? "0.01"
                : "1"
            }
            value={(options[key] as number | undefined) ?? ""}
            onChange={(event) => set(key, event.target.value)}
            className="mt-1"
          />
        </div>
      ))}
    </div>
  )

  const toggles = (
    fields: ReadonlyArray<readonly [keyof ServerOptions, string]>,
  ) => (
    <div className="grid gap-3 sm:grid-cols-2">
      {fields.map(([key, label]) => (
        <Label key={key} className="flex items-center gap-2">
          <Checkbox
            checked={(options[key] as boolean | undefined) === true}
            onCheckedChange={(checked) => set(key, checked === true)}
          />
          {label}
        </Label>
      ))}
    </div>
  )

  return (
    <div className="space-y-3 border-t pt-4">
      <details className="rounded-md border px-4">
        <summary className="cursor-pointer py-3 font-medium">
          Runtime and batching
        </summary>
        <div className="space-y-4 pb-4">
          <p className="text-sm text-muted-foreground">
            Leave fields empty to use llama.cpp defaults.
          </p>
          <div className="grid grid-cols-2 gap-4">
            <div>
              <Label>Device</Label>
              <Select
                value={options.device ?? "default"}
                onValueChange={(value) =>
                  setText("device", value === "default" ? "" : value)
                }
              >
                <SelectTrigger className="mt-1">
                  <SelectValue placeholder="Use default device" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="default">Use default device</SelectItem>
                  <SelectItem value="ROCm0">ROCm0</SelectItem>
                  <SelectItem value="Vulkan0">Vulkan0</SelectItem>
                  <SelectItem value="CPU">CPU</SelectItem>
                </SelectContent>
              </Select>
            </div>
            <div>
              <Label>Batch size</Label>
              <Select
                value={options.batch_size?.toString() ?? "none"}
                onValueChange={(value) =>
                  set("batch_size", value === "none" ? "" : value)
                }
              >
                <SelectTrigger className="mt-1">
                  <SelectValue placeholder="Use llama.cpp default" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="none">Use llama.cpp default</SelectItem>
                  {[256, 512, 1024, 2048, 4096, 8192].map((value) => (
                    <SelectItem key={value} value={value.toString()}>
                      {value}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
            <div>
              <Label>Ubatch size</Label>
              <Select
                value={options.ubatch_size?.toString() ?? "none"}
                onValueChange={(value) =>
                  set("ubatch_size", value === "none" ? "" : value)
                }
              >
                <SelectTrigger className="mt-1">
                  <SelectValue placeholder="Use llama.cpp default" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="none">Use llama.cpp default</SelectItem>
                  {[128, 256, 512, 1024, 2048].map((value) => (
                    <SelectItem key={value} value={value.toString()}>
                      {value}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
          </div>
          {numericFields([
            ["threads", "Generation threads"],
            ["threads_batch", "Batch threads"],
            ["parallel", "Parallel slots"],
          ])}
          {toggles([
            ["cont_batching", "Continuous batching"],
            ["kv_unified", "Unified KV"],
            ["warmup", "Warmup"],
            ["no_mmap", "Disable mmap"],
          ])}
        </div>
      </details>

      <details className="rounded-md border px-4">
        <summary className="cursor-pointer py-3 font-medium">
          KV cache and prompt caching
        </summary>
        <div className="space-y-4 pb-4">
          <div className="grid grid-cols-2 gap-4">
            {(["cache_type_k", "cache_type_v"] as const).map((key) => (
              <div key={key}>
                <Label>
                  {key === "cache_type_k" ? "KV type K" : "KV type V"}
                </Label>
                <Select
                  value={options[key] ?? "default"}
                  onValueChange={(value) =>
                    setText(key, value === "default" ? "" : value)
                  }
                >
                  <SelectTrigger className="mt-1">
                    <SelectValue placeholder="Use llama.cpp default" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="default">
                      Use llama.cpp default
                    </SelectItem>
                    {cacheTypes.map((value) => (
                      <SelectItem key={value} value={value}>
                        {value}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
            ))}
          </div>
          {numericFields([
            ["cache_reuse", "Cache reuse minimum"],
            ["cache_ram", "Cache RAM (MiB)"],
            ["ctx_checkpoints", "Context checkpoints"],
            ["checkpoint_every", "Checkpoint interval"],
          ])}
          <div>
            <Label htmlFor="option-slot_save_path">Slot save path</Label>
            <Input
              id="option-slot_save_path"
              value={options.slot_save_path ?? ""}
              onChange={(event) =>
                setText("slot_save_path", event.target.value)
              }
              className="mt-1"
              placeholder="Use llama.cpp default"
            />
          </div>
          {toggles([
            ["cache_prompt", "Prompt caching"],
            ["kv_offload", "KV cache offload"],
            ["no_cache_idle_slots", "Disable idle-slot cache"],
          ])}
        </div>
      </details>

      <details className="rounded-md border px-4">
        <summary className="cursor-pointer py-3 font-medium">Sampling</summary>
        <div className="space-y-4 pb-4">
          {numericFields([
            ["temperature", "Temperature"],
            ["top_k", "Top-k"],
            ["top_p", "Top-p"],
            ["min_p", "Min-p"],
            ["repeat_penalty", "Repeat penalty"],
            ["presence_penalty", "Presence penalty"],
            ["frequency_penalty", "Frequency penalty"],
            ["seed", "Seed"],
          ])}
        </div>
      </details>

      <details className="rounded-md border px-4">
        <summary className="cursor-pointer py-3 font-medium">
          MTP and reasoning
        </summary>
        <div className="space-y-4 pb-4">
          <div>
            <Label htmlFor="option-mtp-draft-max">MTP draft N-Max</Label>
            <Input
              id="option-mtp-draft-max"
              type="number"
              min={0}
              value={mtpDraftMax}
              onChange={(event) => onMtpDraftMaxChange(event.target.value)}
              className="mt-1"
              placeholder="0 (disabled)"
            />
          </div>
          {numericFields([
            ["spec_draft_p_min", "MTP draft probability minimum"],
          ])}
          <div>
            <Label>Reasoning mode</Label>
            <Select
              value={options.reasoning ?? "default"}
              onValueChange={(value) =>
                setText("reasoning", value === "default" ? "" : value)
              }
            >
              <SelectTrigger className="mt-1">
                <SelectValue placeholder="Use model default" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="default">Use model default</SelectItem>
                <SelectItem value="off">Off</SelectItem>
                <SelectItem value="on">On</SelectItem>
                <SelectItem value="auto">Auto</SelectItem>
              </SelectContent>
            </Select>
          </div>
          {numericFields([["reasoning_budget", "Reasoning budget"]])}
          {toggles([["strict_mtp_qwen", "Strict Qwen MTP"]])}
        </div>
      </details>

      {validateServerOptions(options) && (
        <p className="text-sm text-destructive">
          {validateServerOptions(options)}
        </p>
      )}
    </div>
  )
}
