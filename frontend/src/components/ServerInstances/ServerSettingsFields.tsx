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

export type ServerOptions = {
  threads?: number
  threads_batch?: number
  batch_size?: number
  ubatch_size?: number
  cache_prompt?: boolean
  cache_reuse?: number
  kv_offload?: boolean
  temperature?: number
  top_k?: number
  top_p?: number
  min_p?: number
  repeat_penalty?: number
  seed?: number
  parallel?: number
  cont_batching?: boolean
  warmup?: boolean
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
  return null
}

type Props = {
  options: ServerOptions
  onChange: (options: ServerOptions) => void
}

export function ServerSettingsFields({ options, onChange }: Props) {
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

  return (
    <div className="space-y-5 border-t pt-4">
      <div>
        <h3 className="font-medium">Runtime and batching</h3>
        <p className="text-sm text-muted-foreground">
          Leave fields empty to use llama.cpp defaults.
        </p>
      </div>
      <div className="grid grid-cols-2 gap-4">
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
        {(
          [
            ["threads", "Generation threads"],
            ["threads_batch", "Batch threads"],
            ["parallel", "Parallel slots"],
          ] as const
        ).map(([key, label]) => (
          <div key={key}>
            <Label htmlFor={`option-${key}`}>{label}</Label>
            <Input
              id={`option-${key}`}
              type="number"
              value={options[key] ?? ""}
              onChange={(event) => set(key, event.target.value)}
              className="mt-1"
            />
          </div>
        ))}
      </div>

      <div>
        <h3 className="font-medium">KV cache and sampling</h3>
      </div>
      <div className="grid grid-cols-2 gap-4">
        {(
          [
            ["cache_reuse", "Cache reuse minimum"],
            ["temperature", "Temperature"],
            ["top_k", "Top-k"],
            ["top_p", "Top-p"],
            ["min_p", "Min-p"],
            ["repeat_penalty", "Repeat penalty"],
            ["seed", "Seed"],
          ] as const
        ).map(([key, label]) => (
          <div key={key}>
            <Label htmlFor={`option-${key}`}>{label}</Label>
            <Input
              id={`option-${key}`}
              type="number"
              step={
                key === "top_p" || key === "min_p" || key === "temperature"
                  ? "0.01"
                  : "1"
              }
              value={options[key] ?? ""}
              onChange={(event) => set(key, event.target.value)}
              className="mt-1"
            />
          </div>
        ))}
      </div>
      <div className="grid gap-3 sm:grid-cols-2">
        {(
          [
            ["cache_prompt", "Prompt caching"],
            ["kv_offload", "KV cache offload"],
            ["cont_batching", "Continuous batching"],
            ["warmup", "Warmup"],
          ] as const
        ).map(([key, label]) => (
          <Label key={key} className="flex items-center gap-2">
            <Checkbox
              checked={options[key] ?? true}
              onCheckedChange={(checked) => set(key, checked === true)}
            />
            {label}
          </Label>
        ))}
      </div>
      {validateServerOptions(options) && (
        <p className="text-sm text-destructive">
          {validateServerOptions(options)}
        </p>
      )}
    </div>
  )
}
