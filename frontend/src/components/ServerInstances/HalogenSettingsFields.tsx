import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"

export type HalogenOptions = {
  model_id?: string
  drafter?: "serial" | "mtp" | "dflash2"
  kv_slots?: number
  slot_ctx?: number
  cache_mb?: number
  cache_reserve_mb?: number
  max_tokens_cap?: number
  queue_timeout?: number
  reasoning_effort?: "high" | "low" | "medium" | "minimal" | "none" | "xhigh"
}

type Props = {
  options: HalogenOptions
  onChange: (options: HalogenOptions) => void
}

export function HalogenSettingsFields({ options, onChange }: Props) {
  const setNumber = (key: keyof HalogenOptions, value: string) => {
    const next = { ...options }
    if (value === "") delete next[key]
    else next[key] = Number(value) as never
    onChange(next)
  }

  return (
    <div className="space-y-4 rounded-md border p-4">
      <p className="text-sm text-muted-foreground">
        Halogen settings are applied when the server process starts. Each server
        runs its own Halogen process and cache directory.
      </p>
      <div className="grid grid-cols-2 gap-4">
        <div>
          <Label>Drafter</Label>
          <Select
            value={options.drafter ?? "default"}
            onValueChange={(value) =>
              onChange({
                ...options,
                drafter:
                  value === "default"
                    ? undefined
                    : (value as HalogenOptions["drafter"]),
              })
            }
          >
            <SelectTrigger className="mt-1">
              <SelectValue placeholder="Use agent default" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="default">Use agent default</SelectItem>
              <SelectItem value="serial">Serial</SelectItem>
              <SelectItem value="mtp">MTP</SelectItem>
              <SelectItem value="dflash2">DFlash2</SelectItem>
            </SelectContent>
          </Select>
        </div>
        <div>
          <Label>Reasoning effort</Label>
          <Select
            value={options.reasoning_effort ?? "default"}
            onValueChange={(value) =>
              onChange({
                ...options,
                reasoning_effort:
                  value === "default"
                    ? undefined
                    : (value as HalogenOptions["reasoning_effort"]),
              })
            }
          >
            <SelectTrigger className="mt-1">
              <SelectValue placeholder="Use model default" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="default">Use model default</SelectItem>
              {(
                ["none", "minimal", "low", "medium", "high", "xhigh"] as const
              ).map((value) => (
                <SelectItem key={value} value={value}>
                  {value}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
      </div>
      <div className="grid grid-cols-2 gap-4">
        <div>
          <Label htmlFor="halogen-kv-slots">KV slots</Label>
          <Input
            id="halogen-kv-slots"
            type="number"
            min={1}
            max={8}
            value={options.kv_slots ?? ""}
            onChange={(event) => setNumber("kv_slots", event.target.value)}
            className="mt-1"
            placeholder="Agent default"
          />
        </div>
        <div>
          <Label htmlFor="halogen-slot-ctx">Context per slot</Label>
          <Input
            id="halogen-slot-ctx"
            type="number"
            min={256}
            step={256}
            value={options.slot_ctx ?? ""}
            onChange={(event) => setNumber("slot_ctx", event.target.value)}
            className="mt-1"
            placeholder="Agent default"
          />
        </div>
        <div>
          <Label htmlFor="halogen-cache-mb">Prompt cache (MiB)</Label>
          <Input
            id="halogen-cache-mb"
            type="number"
            min={0}
            value={options.cache_mb ?? ""}
            onChange={(event) => setNumber("cache_mb", event.target.value)}
            className="mt-1"
            placeholder="Automatic"
          />
        </div>
        <div>
          <Label htmlFor="halogen-max-tokens">Maximum output tokens</Label>
          <Input
            id="halogen-max-tokens"
            type="number"
            min={1}
            value={options.max_tokens_cap ?? ""}
            onChange={(event) =>
              setNumber("max_tokens_cap", event.target.value)
            }
            className="mt-1"
            placeholder="65536"
          />
        </div>
        <div>
          <Label htmlFor="halogen-queue-timeout">Queue timeout (seconds)</Label>
          <Input
            id="halogen-queue-timeout"
            type="number"
            min={1}
            value={options.queue_timeout ?? ""}
            onChange={(event) => setNumber("queue_timeout", event.target.value)}
            className="mt-1"
            placeholder="7200"
          />
        </div>
      </div>
      {options.kv_slots && options.kv_slots > 1 && (
        <p className="text-sm text-amber-600">
          Multiple KV slots improve aggregate throughput but disable speculative
          decoding for individual requests.
        </p>
      )}
    </div>
  )
}
