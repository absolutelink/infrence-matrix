import { Checkbox } from "@/components/ui/checkbox"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"

export type HalogenFlashOptions = {
  kv_slots?: number
  kv_pool_positions?: number
  ctx?: number
  max_tokens_cap?: number
  max_tokens_default?: number
  queue_timeout?: number
  temperature?: number
  top_p?: number
  top_k?: number
  reasoning_effort?: "minimal" | "low" | "medium" | "high" | "xhigh"
  max_thinking_tokens?: number
  cache_dir_enabled?: boolean
  prompt_cache?: number
  prefill_chunk?: number
  cache_entries?: number
  cache_branches?: number
  cache_snap3?: number
  cache_full?: number
  cache_disk_gib?: number
  cache_prune_old?: number
  composable_context?: number
  composable_context_floor?: number
  composable_context_bytes?: number
  vision_tower?: number
}

type Props = {
  options: HalogenFlashOptions
  onChange: (options: HalogenFlashOptions) => void
}

export function HalogenFlashSettingsFields({ options, onChange }: Props) {
  const setNumber = (key: keyof HalogenFlashOptions, value: string) => {
    const next = { ...options }
    if (value === "") delete next[key]
    else next[key] = Number(value) as never
    onChange(next)
  }

  const fields = [
    ["kv_slots", "KV slots", 1],
    ["kv_pool_positions", "KV pool positions", 1],
    ["ctx", "Context size", 256],
    ["max_tokens_cap", "Maximum output tokens", 1],
    ["max_tokens_default", "Default output tokens", 1],
    ["queue_timeout", "Queue timeout (seconds)", 1],
    ["temperature", "Temperature", 0],
    ["top_p", "Top P", 0],
    ["top_k", "Top K", 0],
    ["max_thinking_tokens", "Maximum thinking tokens", 0],
  ] as const

  const cacheFields = [
    ["prompt_cache", "Prompt cache mode", 0],
    ["prefill_chunk", "Prefill chunk", 1],
    ["cache_entries", "Cache entries", 1],
    ["cache_branches", "Cache branches", 1],
    ["cache_snap3", "Cache snapshot 3", 0],
    ["cache_full", "Cache full", 0],
    ["cache_disk_gib", "Cache disk limit (GiB)", 0],
    ["cache_prune_old", "Prune old cache", 0],
    ["composable_context", "Composable context", 0],
    ["composable_context_floor", "Composable context floor", 0],
    ["composable_context_bytes", "Composable context bytes", 0],
  ] as const

  return (
    <div className="space-y-4 rounded-md border p-4">
      <p className="text-sm text-muted-foreground">
        Flash settings are applied through environment variables when the server
        starts. Model and tokenizer files are managed by the agent.
      </p>
      <section className="space-y-3">
        <h3 className="text-sm font-medium">Runtime</h3>
        <div className="grid grid-cols-2 gap-4">
          {fields.map(([key, label, min]) => (
            <div key={key}>
              <Label htmlFor={`halogen-flash-${key}`}>{label}</Label>
              <Input
                id={`halogen-flash-${key}`}
                type="number"
                min={min}
                step={key === "ctx" ? 256 : undefined}
                value={options[key] ?? ""}
                onChange={(event) => setNumber(key, event.target.value)}
                className="mt-1"
                placeholder="Agent default"
              />
            </div>
          ))}
        </div>
        <label
          htmlFor="halogen-flash-vision-tower"
          className="flex items-center gap-2 text-sm"
        >
          <Checkbox
            id="halogen-flash-vision-tower"
            checked={options.vision_tower === 1}
            onCheckedChange={(checked) =>
              onChange({
                ...options,
                vision_tower: checked === true ? 1 : undefined,
              })
            }
          />
          Enable Vision
        </label>
      </section>
      <section className="space-y-3 border-t pt-4">
        <h3 className="text-sm font-medium">Prompt Cache</h3>
        <label
          htmlFor="halogen-flash-cache-dir-enabled"
          className="flex items-center gap-2 text-sm"
        >
          <Checkbox
            id="halogen-flash-cache-dir-enabled"
            checked={options.cache_dir_enabled === true}
            onCheckedChange={(checked) =>
              onChange({ ...options, cache_dir_enabled: checked === true })
            }
          />
          Enable disk prompt cache
        </label>
        <p className="text-xs text-muted-foreground">
          Stores this server&apos;s cache in its managed UUID directory under
          the agent cache path.
        </p>
        <div className="grid grid-cols-2 gap-4">
          {cacheFields.map(([key, label, min]) => (
            <div key={key}>
              <Label htmlFor={`halogen-flash-${key}`}>{label}</Label>
              <Input
                id={`halogen-flash-${key}`}
                type="number"
                min={min}
                step={key === "prefill_chunk" ? 1 : undefined}
                value={options[key] ?? ""}
                onChange={(event) => setNumber(key, event.target.value)}
                className="mt-1"
                placeholder="Agent default"
              />
            </div>
          ))}
        </div>
      </section>
      <div>
        <Label>Reasoning effort</Label>
        <select
          className="mt-1 flex h-9 w-full rounded-md border bg-transparent px-3 text-sm"
          value={options.reasoning_effort ?? ""}
          onChange={(event) =>
            onChange({
              ...options,
              reasoning_effort: event.target.value
                ? (event.target
                    .value as HalogenFlashOptions["reasoning_effort"])
                : undefined,
            })
          }
        >
          <option value="">Agent default</option>
          <option value="minimal">Minimal</option>
          <option value="low">Low</option>
          <option value="medium">Medium</option>
          <option value="high">High</option>
          <option value="xhigh">XHigh</option>
        </select>
      </div>
    </div>
  )
}
