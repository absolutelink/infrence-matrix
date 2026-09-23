import { useSuspenseQuery } from "@tanstack/react-query"
import { createFileRoute } from "@tanstack/react-router"
import {
  ChevronDown,
  ChevronRight,
  ListChecks,
  Plus,
  Send,
  Square,
  Wrench,
} from "lucide-react"
import { useRef, useState } from "react"
import { ServerInstancesService, V15 } from "@/client"
import { Button } from "@/components/ui/button"
import { Card } from "@/components/ui/card"
import {
  Select,
  SelectContent,
  SelectGroup,
  SelectItem,
  SelectLabel,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"
import { Slider } from "@/components/ui/slider"
import { Switch } from "@/components/ui/switch"
import { Textarea } from "@/components/ui/textarea"

export const Route = createFileRoute("/_layout/responses/")({
  component: ResponsesPage,
  head: () => ({
    meta: [{ title: "Responses API - Inference Matrix" }],
  }),
})

type ToolCall = {
  itemId: string
  name: string
  arguments: string
}

type ChatEntry =
  | { kind: "user"; text: string }
  | {
      kind: "assistant"
      text: string
      reasoning: string
      toolCalls: ToolCall[]
      usage: string | null
    }
  | { kind: "error"; text: string }

type RawEvent = {
  seq: number | null
  type: string
  payload: unknown
}

const EVENT_COLORS: Record<string, string> = {
  "response.created": "text-blue-500",
  "response.in_progress": "text-blue-500",
  "response.completed": "text-green-600",
  "response.failed": "text-red-500",
  "response.incomplete": "text-amber-600",
  "response.queued": "text-muted-foreground",
  error: "text-red-500",
}

function eventColor(type: string) {
  if (EVENT_COLORS[type]) return EVENT_COLORS[type]
  if (type.includes("delta")) return "text-emerald-500"
  if (type.includes("added")) return "text-sky-500"
  if (type.includes("done")) return "text-teal-600"
  return "text-muted-foreground"
}

function getServersQueryOptions() {
  return {
    queryFn: async () => {
      const response =
        await ServerInstancesService.instancesListServerInstances()
      const instances = response.data.server_instances || []
      return [...instances].sort((a, b) => {
        const rank = (s: string) =>
          s === "running" ? 0 : s === "starting" ? 1 : 2
        return rank(a.status) - rank(b.status)
      })
    },
    queryKey: ["servers-responses"],
    refetchInterval: 5000,
  }
}

function ToolCallChip({
  name,
  arguments: args,
}: {
  name: string
  arguments: string
}) {
  return (
    <div className="mb-2 rounded-md border border-sky-500/30 bg-sky-500/5 px-3 py-2 text-xs">
      <div className="flex items-center gap-1.5 font-medium text-sky-600 dark:text-sky-400">
        <Wrench className="h-3 w-3" />
        {name}
      </div>
      <pre className="mt-1 overflow-x-auto whitespace-pre-wrap text-muted-foreground">
        {args || "{}"}
      </pre>
    </div>
  )
}

function ThinkingBlock({ content }: { content: string }) {
  return (
    <details className="mb-2 rounded-md border border-black/10 bg-black/5 dark:border-white/10 dark:bg-white/5">
      <summary className="cursor-pointer select-none px-2 py-1 text-xs text-muted-foreground">
        Thinking
      </summary>
      <div className="whitespace-pre-wrap px-3 pb-2 text-xs italic text-muted-foreground">
        {content}
      </div>
    </details>
  )
}

function RawEventRow({ event }: { event: RawEvent }) {
  const [open, setOpen] = useState(false)
  return (
    <div className="border-b border-border/50 last:border-0">
      <button
        type="button"
        onClick={() => setOpen((o) => !o)}
        className="flex w-full items-center gap-2 px-2 py-1.5 text-left text-xs hover:bg-muted/50"
      >
        {open ? (
          <ChevronDown className="h-3 w-3 shrink-0 text-muted-foreground" />
        ) : (
          <ChevronRight className="h-3 w-3 shrink-0 text-muted-foreground" />
        )}
        <span className="w-12 shrink-0 font-mono text-muted-foreground">
          {event.seq ?? "—"}
        </span>
        <span className={`font-mono ${eventColor(event.type)}`}>
          {event.type}
        </span>
      </button>
      {open && (
        <pre className="max-h-48 overflow-auto bg-muted/30 px-3 py-2 font-mono text-[11px] text-muted-foreground">
          {JSON.stringify(event.payload, null, 2)}
        </pre>
      )}
    </div>
  )
}

function ResponsesPage() {
  const { data: servers } = useSuspenseQuery(getServersQueryOptions())
  const [selectedModel, setSelectedModel] = useState<string>("")
  const [entries, setEntries] = useState<ChatEntry[]>([])
  const [input, setInput] = useState("")
  const [isLoading, setIsLoading] = useState(false)
  const abortRef = useRef<AbortController | null>(null)

  // Test affordances
  const [store, setStore] = useState(true)
  const [temperature, setTemperature] = useState(0.7)
  const [maxTokens, setMaxTokens] = useState(0) // 0 = unset
  const [stream, setStream] = useState(true)
  const [lastResponseId, setLastResponseId] = useState<string | null>(null)
  const [lastUsage, setLastUsage] = useState<string | null>(null)
  const [showEvents, setShowEvents] = useState(false)
  const [rawEvents, setRawEvents] = useState<RawEvent[]>([])

  const handleNewConversation = () => {
    setEntries([])
    setInput("")
    setLastResponseId(null)
    setLastUsage(null)
    setRawEvents([])
  }

  const patchLastAssistant = (
    mutate: (entry: Extract<ChatEntry, { kind: "assistant" }>) => void,
  ) => {
    setEntries((prev) => {
      const next = [...prev]
      for (let i = next.length - 1; i >= 0; i--) {
        const entry = next[i]
        if (entry.kind === "assistant") {
          const copy: Extract<ChatEntry, { kind: "assistant" }> = {
            ...entry,
            toolCalls: [...entry.toolCalls],
          }
          mutate(copy)
          next[i] = copy
          break
        }
      }
      return next
    })
  }

  const processEvent = (event: Record<string, unknown>) => {
    const type = event.type as string
    setRawEvents((prev) => [
      {
        seq: (event.sequence_number as number) ?? null,
        type,
        payload: event,
      },
      ...prev,
    ])

    if (type === "response.output_text.delta") {
      patchLastAssistant((entry) => {
        entry.text += event.delta as string
      })
    } else if (type === "response.reasoning.delta") {
      patchLastAssistant((entry) => {
        entry.reasoning += event.delta as string
      })
    } else if (type === "response.output_item.done") {
      const item = event.item as Record<string, unknown> | undefined
      if (item?.type === "function_call") {
        const call: ToolCall = {
          itemId: item.id as string,
          name: (item.name as string) || "unknown",
          arguments: (item.arguments as string) || "{}",
        }
        patchLastAssistant((entry) => {
          if (!entry.toolCalls.some((t) => t.itemId === call.itemId)) {
            entry.toolCalls.push(call)
          }
        })
      } else if (item?.type === "message") {
        // The done item carries the final text; sync in case a delta was missed
        const parts = (item.content as Record<string, unknown>[]) || []
        const finalText = parts
          .filter((p) => p.type === "output_text")
          .map((p) => p.text as string)
          .join("")
        if (finalText) {
          patchLastAssistant((entry) => {
            entry.text = finalText
          })
        }
      }
    } else if (
      type === "response.completed" ||
      type === "response.incomplete" ||
      type === "response.failed"
    ) {
      const response = event.response as Record<string, unknown> | undefined
      if (response) {
        const id = response.id as string
        if (store) setLastResponseId(id)
        const usage = response.usage as Record<string, unknown> | undefined
        if (usage) {
          setLastUsage(
            `${usage.input_tokens} in / ${usage.output_tokens} out / ${usage.total_tokens} total`,
          )
        }
        if (type === "response.incomplete" && response.incomplete_details) {
          setLastUsage(
            `incomplete: ${(response.incomplete_details as Record<string, unknown>).reason}`,
          )
        }
        if (type === "response.failed" && response.error) {
          const err = response.error as Record<string, unknown>
          setEntries((prev) => [
            ...prev,
            {
              kind: "error",
              text: `${err.code || "error"}: ${err.message}`,
            },
          ])
        }
      }
    } else if (type === "error") {
      const err = event.error as Record<string, unknown> | undefined
      if (err) {
        setEntries((prev) => [
          ...prev,
          { kind: "error", text: `${err.code || "error"}: ${err.message}` },
        ])
      }
    }
  }

  const buildRequestBody = (inputItems: unknown) => {
    const body: Record<string, unknown> = {
      model: selectedModel,
      input: inputItems,
      stream,
      store,
      temperature,
    }
    if (maxTokens > 0) body.max_output_tokens = maxTokens
    if (store && lastResponseId) body.previous_response_id = lastResponseId
    return body
  }

  const handleSend = async () => {
    if (!input.trim() || !selectedModel) return

    const userText = input
    const chained = store && lastResponseId
    setEntries((prev) => [...prev, { kind: "user", text: userText }])
    setInput("")
    setIsLoading(true)
    abortRef.current = new AbortController()

    const inputItems = chained
      ? [{ type: "message", role: "user", content: userText }]
      : userText

    try {
      if (!stream) {
        const response = (await V15.createResponse({
          body: buildRequestBody(inputItems) as never,
        })) as unknown as Record<string, unknown>
        const output = (response.output as Record<string, unknown>[]) || []
        let text = ""
        const toolCalls: ToolCall[] = []
        let reasoning = ""
        for (const item of output) {
          if (item.type === "message") {
            const parts = (item.content as Record<string, unknown>[]) || []
            text += parts
              .filter((p) => p.type === "output_text")
              .map((p) => p.text as string)
              .join("")
          } else if (item.type === "reasoning") {
            const parts = (item.content as Record<string, unknown>[]) || []
            reasoning += parts
              .filter((p) => p.type === "reasoning_text")
              .map((p) => p.text as string)
              .join("")
          } else if (item.type === "function_call") {
            toolCalls.push({
              itemId: item.id as string,
              name: item.name as string,
              arguments: (item.arguments as string) || "{}",
            })
          }
        }
        if (store) setLastResponseId(response.id as string)
        const usage = response.usage as Record<string, unknown> | undefined
        if (usage) {
          setLastUsage(
            `${usage.input_tokens} in / ${usage.output_tokens} out / ${usage.total_tokens} total`,
          )
        }
        setEntries((prev) => [
          ...prev,
          {
            kind: "assistant",
            text,
            reasoning,
            toolCalls,
            usage: null,
          },
        ])
      } else {
        const response = await fetch("/v1/responses", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          signal: abortRef.current.signal,
          body: JSON.stringify(buildRequestBody(inputItems)),
        })

        if (!response.ok || !response.body) {
          const detail = await response.text().catch(() => "")
          throw new Error(detail || `Request failed: ${response.status}`)
        }

        setEntries((prev) => [
          ...prev,
          {
            kind: "assistant",
            text: "",
            reasoning: "",
            toolCalls: [],
            usage: null,
          },
        ])

        const reader = response.body.getReader()
        const decoder = new TextDecoder()
        let buffer = ""
        let done = false

        while (!done) {
          const { done: readerDone, value } = await reader.read()
          if (readerDone) break
          buffer += decoder.decode(value, { stream: true })
          const blocks = buffer.split("\n\n")
          buffer = blocks.pop() ?? ""
          for (const block of blocks) {
            // Spec framing: "event: <type>\ndata: {json}" — data line carries the payload
            const dataLine = block
              .split("\n")
              .find((l) => l.startsWith("data: "))
            if (!dataLine) continue
            const data = dataLine.slice(6).trim()
            if (!data) continue
            if (data === "[DONE]") {
              done = true
              continue
            }
            try {
              processEvent(JSON.parse(data))
            } catch {}
          }
        }
      }
    } catch (error) {
      if ((error as Error).name === "AbortError") {
        // User-initiated stop; keep partial output
      } else {
        console.error("Error:", error)
        setEntries((prev) => [
          ...prev,
          {
            kind: "error",
            text: (error as Error).message || "Request failed",
          },
        ])
      }
    } finally {
      setIsLoading(false)
      abortRef.current = null
    }
  }

  return (
    <div className="flex flex-col h-[calc(100vh-8rem)]">
      <div className="flex items-center justify-between mb-4">
        <div>
          <h1 className="text-2xl font-bold tracking-tight">Responses API</h1>
          <p className="text-muted-foreground">
            Test the OpenResponses endpoint (/v1/responses)
          </p>
        </div>
        <div className="flex items-center gap-2">
          <Select value={selectedModel} onValueChange={setSelectedModel}>
            <SelectTrigger className="w-[300px]">
              <SelectValue placeholder="Select a server" />
            </SelectTrigger>
            <SelectContent>
              {servers && servers.length > 0 ? (
                <SelectGroup>
                  <SelectLabel>Servers</SelectLabel>
                  {servers.map((server) => (
                    <SelectItem key={server.id} value={server.alias}>
                      <div className="flex flex-col items-start">
                        <span>
                          {server.alias || server.model_name}
                          {server.status === "stopped" ||
                          server.status === "error" ? (
                            <span className="ml-2 text-xs text-muted-foreground">
                              (will start on first message)
                            </span>
                          ) : null}
                        </span>
                        <span className="text-xs text-muted-foreground">
                          {server.model_name || "Unknown model"}
                          {server.agent_name ? ` · ${server.agent_name}` : ""}
                        </span>
                      </div>
                    </SelectItem>
                  ))}
                </SelectGroup>
              ) : (
                <div className="px-3 py-2 text-sm text-muted-foreground">
                  No servers configured — create one on the Server Instances
                  page
                </div>
              )}
            </SelectContent>
          </Select>
        </div>
      </div>

      <div className="flex flex-1 gap-4 min-h-0">
        <div className="flex flex-col flex-1 min-w-0">
          <Card className="flex-1 overflow-hidden mb-4">
            <div className="h-full overflow-y-auto p-4 space-y-4">
              {entries.length === 0 ? (
                <div className="flex flex-col items-center justify-center h-full text-center text-muted-foreground">
                  <ListChecks className="h-12 w-12 mb-4" />
                  <h3 className="text-lg font-semibold">
                    Test the Responses API
                  </h3>
                  <p>Select a server, adjust options, and send a message</p>
                  <p className="mt-2 text-xs">
                    Events appear live in the inspector panel
                  </p>
                </div>
              ) : (
                entries.map((entry, index) =>
                  entry.kind === "user" ? (
                    <div key={index} className="flex justify-end">
                      <div className="max-w-[80%] rounded-lg px-4 py-2 bg-primary text-primary-foreground">
                        <div className="whitespace-pre-wrap">{entry.text}</div>
                      </div>
                    </div>
                  ) : entry.kind === "error" ? (
                    <div key={index} className="flex justify-start">
                      <div className="max-w-[80%] rounded-lg px-4 py-2 bg-destructive/10 text-destructive text-sm">
                        {entry.text}
                      </div>
                    </div>
                  ) : (
                    <div key={index} className="flex justify-start">
                      <div className="max-w-[80%] rounded-lg px-4 py-2 bg-muted">
                        <div className="font-semibold text-sm mb-1">
                          Assistant
                        </div>
                        {entry.reasoning ? (
                          <ThinkingBlock content={entry.reasoning} />
                        ) : null}
                        {entry.toolCalls.map((call) => (
                          <ToolCallChip
                            key={call.itemId}
                            name={call.name}
                            arguments={call.arguments}
                          />
                        ))}
                        {entry.text ? (
                          <div className="whitespace-pre-wrap">
                            {entry.text}
                          </div>
                        ) : null}
                      </div>
                    </div>
                  ),
                )
              )}
              {isLoading && (
                <div className="flex justify-start">
                  <div className="rounded-lg px-4 py-2 bg-muted text-sm text-muted-foreground">
                    Generating…
                  </div>
                </div>
              )}
            </div>
          </Card>

          <div className="flex gap-2">
            <Textarea
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder={
                lastResponseId
                  ? "Follow-up — will chain via previous_response_id"
                  : "Type your message..."
              }
              className="flex-1 min-h-[80px]"
              onKeyDown={(e: React.KeyboardEvent<HTMLTextAreaElement>) => {
                if (e.key === "Enter" && !e.shiftKey) {
                  e.preventDefault()
                  handleSend()
                }
              }}
              disabled={isLoading}
            />
            <div className="flex flex-col gap-2">
              {isLoading ? (
                <Button
                  variant="destructive"
                  onClick={() => abortRef.current?.abort()}
                >
                  <Square className="h-4 w-4" />
                </Button>
              ) : (
                <Button
                  onClick={handleSend}
                  disabled={!input.trim() || !selectedModel}
                >
                  <Send className="h-4 w-4" />
                </Button>
              )}
              <Button
                variant="outline"
                size="icon"
                title="New conversation (clears previous_response_id)"
                onClick={handleNewConversation}
                disabled={isLoading || !lastResponseId}
              >
                <Plus className="h-4 w-4" />
              </Button>
            </div>
          </div>
        </div>

        <div className="w-[340px] shrink-0 flex flex-col gap-4 min-h-0">
          <Card className="p-4 space-y-4">
            <div className="flex items-center justify-between">
              <span className="text-sm font-medium">Stream events</span>
              <Switch checked={stream} onCheckedChange={setStream} />
            </div>
            <div className="flex items-center justify-between">
              <span className="text-sm font-medium">Store response</span>
              <Switch checked={store} onCheckedChange={setStore} />
            </div>
            <div className="flex items-center justify-between">
              <span className="text-sm font-medium">Event inspector</span>
              <Switch checked={showEvents} onCheckedChange={setShowEvents} />
            </div>
            <div>
              <div className="flex justify-between text-sm mb-1">
                <span className="font-medium">Temperature</span>
                <span className="text-muted-foreground">
                  {temperature.toFixed(2)}
                </span>
              </div>
              <Slider
                value={[temperature]}
                onValueChange={(v: number[]) => setTemperature(v[0])}
                min={0}
                max={2}
                step={0.05}
              />
            </div>
            <div>
              <div className="flex justify-between text-sm mb-1">
                <span className="font-medium">Max output tokens</span>
                <span className="text-muted-foreground">
                  {maxTokens === 0 ? "unset" : maxTokens}
                </span>
              </div>
              <Slider
                value={[maxTokens]}
                onValueChange={(v: number[]) => setMaxTokens(v[0])}
                min={0}
                max={4096}
                step={64}
              />
            </div>
            <div className="rounded-md bg-muted/50 px-3 py-2 text-xs text-muted-foreground space-y-1">
              <div>
                <span className="font-medium">previous_response_id:</span>{" "}
                <span className="font-mono">
                  {store && lastResponseId ? lastResponseId : "—"}
                </span>
              </div>
              {lastUsage ? (
                <div>
                  <span className="font-medium">usage:</span> {lastUsage}
                </div>
              ) : null}
            </div>
          </Card>

          {showEvents ? (
            <Card className="flex-1 min-h-0 flex flex-col overflow-hidden">
              <div className="flex items-center justify-between px-3 py-2 border-b border-border">
                <span className="text-xs font-medium text-muted-foreground">
                  Streaming events ({rawEvents.length})
                </span>
                <Button
                  variant="ghost"
                  size="sm"
                  className="h-6 px-2 text-xs"
                  onClick={() => setRawEvents([])}
                >
                  Clear
                </Button>
              </div>
              <div className="flex-1 overflow-y-auto">
                {rawEvents.length === 0 ? (
                  <div className="p-3 text-xs text-muted-foreground">
                    No events yet — send a message
                  </div>
                ) : (
                  rawEvents.map((event, i) => (
                    <RawEventRow key={`${event.seq}-${i}`} event={event} />
                  ))
                )}
              </div>
            </Card>
          ) : null}
        </div>
      </div>
    </div>
  )
}
