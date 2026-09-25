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
  cache_entries?: number
  cache_branches?: number
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
    ["cache_entries", "Cache entries", 1],
    ["cache_branches", "Cache branches", 1],
  ] as const

  return (
    <div className="space-y-4 rounded-md border p-4">
      <p className="text-sm text-muted-foreground">
        Flash settings are applied through environment variables when the server
        starts. Model and tokenizer files are managed by the agent.
      </p>
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
