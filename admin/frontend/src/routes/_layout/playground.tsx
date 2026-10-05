import { useQuery } from "@tanstack/react-query"
import { createFileRoute } from "@tanstack/react-router"
import { Send, Square } from "lucide-react"
import { useRef, useState } from "react"

import { ModelsService } from "@/client"
import { Button } from "@/components/ui/button"
import { Label } from "@/components/ui/label"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Textarea } from "@/components/ui/textarea"
import { getApiUrl } from "@/utils"

export const Route = createFileRoute("/_layout/playground")({
  component: PlaygroundPage,
  head: () => ({ meta: [{ title: "Playground - Inference Matrix" }] }),
})

// Minimal live-inference demo: posts to the public /v1 endpoints with
// stream=true and renders the SSE deltas as they arrive. The generated
// SDK types these bodies as free-form JSON (the admin passes them to
// litellm), so we use fetch directly for the streaming read.
function PlaygroundPage() {
  const { data: models } = useQuery({
    queryKey: ["playground-models"],
    queryFn: async () =>
      ((await ModelsService.listModels()).data?.data ?? []) as Array<{
        id: string
        owned_by?: string
      }>,
    refetchInterval: 15000,
  })

  const [api, setApi] = useState<"responses" | "chat">("responses")
  const [model, setModel] = useState<string>("")
  const [prompt, setPrompt] = useState("Hello! Say something nice.")
  const [output, setOutput] = useState("")
  const [events, setEvents] = useState<string[]>([])
  const [running, setRunning] = useState(false)
  const abortRef = useRef<AbortController | null>(null)

  const activeModel = model || models?.[0]?.id || "mock-model"

  const stop = () => {
    abortRef.current?.abort()
    setRunning(false)
  }

  const run = async () => {
    setOutput("")
    setEvents([])
    setRunning(true)
    const ctrl = new AbortController()
    abortRef.current = ctrl
    const url =
      api === "responses"
        ? `${getApiUrl()}/v1/responses`
        : `${getApiUrl()}/v1/chat/completions`
    const body =
      api === "responses"
        ? { model: activeModel, input: prompt, stream: true }
        : {
            model: activeModel,
            messages: [{ role: "user", content: prompt }],
            stream: true,
          }
    let text = ""
    const evNames: string[] = []
    try {
      const resp = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
        signal: ctrl.signal,
      })
      if (!resp.ok || !resp.body) {
        const errText = await resp.text()
        setOutput(`HTTP ${resp.status}\n${errText}`)
        setRunning(false)
        return
      }
      const reader = resp.body.getReader()
      const decoder = new TextDecoder()
      let buf = ""
      for (;;) {
        const { done, value } = await reader.read()
        if (done) break
        buf += decoder.decode(value, { stream: true })
        for (;;) {
          const idx = buf.indexOf("\n")
          if (idx < 0) break
          const line = buf.slice(0, idx).trim()
          buf = buf.slice(idx + 1)
          if (!line) continue
          if (line.startsWith("event:")) {
            evNames.push(line.slice(6).trim())
            setEvents([...evNames])
            continue
          }
          if (!line.startsWith("data:")) continue
          const data = line.slice(5).trim()
          if (data === "[DONE]") continue
          try {
            const obj = JSON.parse(data)
            if (api === "responses") {
              if (typeof obj.delta === "string") {
                text += obj.delta
                setOutput(text)
              }
            } else {
              const d = obj.choices?.[0]?.delta?.content
              if (typeof d === "string") {
                text += d
                setOutput(text)
              }
            }
          } catch {
            /* non-JSON keepalive */
          }
        }
      }
    } catch (e) {
      if ((e as Error).name !== "AbortError") {
        setOutput((prev) => `${prev}\n[error] ${(e as Error).message}`)
      }
    } finally {
      setRunning(false)
    }
  }

  return (
    <div className="flex flex-col gap-6">
      <div>
        <h1 className="text-2xl font-bold tracking-tight">Playground</h1>
        <p className="text-muted-foreground">
          Hit the public /v1 inference path live through the scheduler and a
          connected provider.
        </p>
      </div>

      <Tabs
        value={api}
        onValueChange={(v) => setApi(v as "responses" | "chat")}
      >
        <TabsList>
          <TabsTrigger value="responses">/v1/responses</TabsTrigger>
          <TabsTrigger value="chat">/v1/chat/completions</TabsTrigger>
        </TabsList>
      </Tabs>

      <div className="grid gap-6 lg:grid-cols-2">
        <div className="flex flex-col gap-4 rounded-lg border p-4">
          <div className="space-y-2">
            <Label>Model alias</Label>
            <Select value={activeModel} onValueChange={setModel}>
              <SelectTrigger>
                <SelectValue placeholder="Pick a model" />
              </SelectTrigger>
              <SelectContent>
                {(models ?? []).map((m) => (
                  <SelectItem key={m.id} value={m.id}>
                    {m.id}{" "}
                    <span className="text-muted-foreground">
                      ({m.owned_by})
                    </span>
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
            {(models ?? []).length === 0 && (
              <p className="text-xs text-muted-foreground">
                No enabled models — create a definition first.
              </p>
            )}
          </div>
          <div className="space-y-2">
            <Label>Prompt</Label>
            <Textarea
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              rows={6}
            />
          </div>
          <div className="flex gap-2">
            <Button onClick={run} disabled={running}>
              <Send /> Run (stream)
            </Button>
            <Button variant="outline" onClick={stop} disabled={!running}>
              <Square /> Stop
            </Button>
          </div>
        </div>

        <div className="flex flex-col gap-4 rounded-lg border p-4">
          <div className="space-y-2">
            <Label>Output</Label>
            <pre className="min-h-40 flex-1 whitespace-pre-wrap rounded-md bg-muted p-3 font-mono text-sm">
              {output || (running ? "streaming…" : "—")}
            </pre>
          </div>
          {events.length > 0 && (
            <div className="space-y-1">
              <Label className="text-xs text-muted-foreground">
                SSE events ({events.length})
              </Label>
              <div className="flex max-h-24 flex-wrap gap-1 overflow-auto">
                {events.map((e, i) => (
                  <span
                    key={`${e}-${i}`}
                    className="rounded bg-secondary px-1.5 py-0.5 font-mono text-[10px] text-secondary-foreground"
                  >
                    {e}
                  </span>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
