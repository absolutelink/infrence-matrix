import { useSuspenseQuery } from "@tanstack/react-query"
import { createFileRoute } from "@tanstack/react-router"
import { Send, Sparkles, Square } from "lucide-react"
import { useRef, useState } from "react"
import { ServerInstancesService } from "@/client"
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
import { Textarea } from "@/components/ui/textarea"

export const Route = createFileRoute("/_layout/chat/")({
  component: Chat,
  head: () => ({
    meta: [
      {
        title: "Chat - Inference Matrix",
      },
    ],
  }),
})

type Message = {
  role: "user" | "assistant" | "system"
  content: string
  reasoning?: string
}

type ContentSegment =
  | { type: "text"; content: string }
  | { type: "thinking"; content: string }

function parseContent(content: string): ContentSegment[] {
  const segments: ContentSegment[] = []
  let thinking = false
  let buffer = ""
  let i = 0
  while (i < content.length) {
    if (!thinking && content.startsWith("<think>", i)) {
      if (buffer) segments.push({ type: "text", content: buffer })
      buffer = ""
      thinking = true
      i += 7
    } else if (thinking && content.startsWith("</think>", i)) {
      if (buffer) segments.push({ type: "thinking", content: buffer })
      buffer = ""
      thinking = false
      i += 8
    } else {
      buffer += content[i]
      i += 1
    }
  }
  if (buffer) {
    segments.push({ type: thinking ? "thinking" : "text", content: buffer })
  }
  return segments
}

function ThinkingBlock({ content }: { content: string }) {
  return (
    <details
      className="mb-2 rounded-md border border-black/10 bg-black/5 dark:border-white/10 dark:bg-white/5"
      open
    >
      <summary className="cursor-pointer select-none px-2 py-1 text-xs text-muted-foreground">
        Thinking
      </summary>
      <div className="whitespace-pre-wrap px-3 pb-2 text-xs italic text-muted-foreground">
        {content}
      </div>
    </details>
  )
}

function AssistantContent({
  content,
  reasoning,
}: {
  content: string
  reasoning?: string
}) {
  const segments = parseContent(content).filter((s) => s.content.trim())
  const trimmedReasoning = reasoning?.trim()
  if (segments.length === 0 && !trimmedReasoning) return null
  return (
    <>
      {trimmedReasoning ? <ThinkingBlock content={trimmedReasoning} /> : null}
      {segments.map((segment, i) =>
        segment.type === "thinking" ? (
          <details
            key={i}
            className="mb-2 rounded-md border border-black/10 bg-black/5 dark:border-white/10 dark:bg-white/5"
          >
            <summary className="cursor-pointer select-none px-2 py-1 text-xs text-muted-foreground">
              Thinking
            </summary>
            <div className="whitespace-pre-wrap px-3 pb-2 text-xs italic text-muted-foreground">
              {segment.content}
            </div>
          </details>
        ) : (
          <div key={i} className="whitespace-pre-wrap">
            {segment.content}
          </div>
        ),
      )}
    </>
  )
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
    queryKey: ["servers-chat"],
    refetchInterval: 5000,
  }
}

function Chat() {
  const { data: servers } = useSuspenseQuery(getServersQueryOptions())
  const [selectedModel, setSelectedModel] = useState<string>("")
  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState("")
  const [isLoading, setIsLoading] = useState(false)
  const abortRef = useRef<AbortController | null>(null)

  const handleSend = async () => {
    if (!input.trim() || !selectedModel) return

    const userMessage: Message = { role: "user", content: input }
    const history = [...messages, userMessage]
    setMessages(history)
    setInput("")
    setIsLoading(true)
    abortRef.current = new AbortController()

    try {
      const response = await fetch("/v1/chat/completions", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        signal: abortRef.current.signal,
        body: JSON.stringify({
          model: selectedModel,
          messages: history.map((m) => ({ role: m.role, content: m.content })),
          stream: true,
        }),
      })

      if (!response.ok || !response.body) {
        const detail = await response.text().catch(() => "")
        throw new Error(detail || `Request failed: ${response.status}`)
      }

      setMessages((prev) => [...prev, { role: "assistant", content: "" }])

      const reader = response.body.getReader()
      const decoder = new TextDecoder()
      let buffer = ""
      let done = false

      while (!done) {
        const { done: readerDone, value } = await reader.read()
        if (readerDone) break
        buffer += decoder.decode(value, { stream: true })
        const lines = buffer.split("\n\n")
        buffer = lines.pop() ?? ""
        for (const line of lines) {
          const data = line.replace(/^data: /, "").trim()
          if (!data || data === "[DONE]") {
            if (data === "[DONE]") done = true
            continue
          }
          try {
            const chunk = JSON.parse(data)
            if (chunk.error) {
              throw new Error(chunk.error.message ?? "Stream error")
            }
            const delta = chunk.choices?.[0]?.delta ?? {}
            const text = delta.content ?? ""
            const reasoning = delta.reasoning_content ?? ""
            if (text || reasoning) {
              setMessages((prev) => {
                const newMessages = [...prev]
                const lastMessage = newMessages[newMessages.length - 1]
                if (lastMessage.role === "assistant") {
                  if (reasoning) {
                    lastMessage.reasoning =
                      (lastMessage.reasoning ?? "") + reasoning
                  }
                  if (text) {
                    lastMessage.content += text
                  }
                }
                return newMessages
              })
            }
          } catch {}
        }
      }
    } catch (error) {
      if ((error as Error).name === "AbortError") {
        // User-initiated stop; keep partial output
      } else {
        console.error("Error:", error)
        setMessages((prev) => [
          ...prev,
          {
            role: "assistant",
            content: "Error occurred while generating response.",
          },
        ])
      }
    } finally {
      setIsLoading(false)
      abortRef.current = null
    }
  }

  const handleStop = () => {
    abortRef.current?.abort()
  }

  const handleClear = () => {
    setMessages([])
    setInput("")
  }

  return (
    <div className="flex flex-col h-[calc(100vh-8rem)]">
      <div className="flex items-center justify-between mb-4">
        <div>
          <h1 className="text-2xl font-bold tracking-tight">Chat</h1>
          <p className="text-muted-foreground">
            Chat with your running servers
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
          <Button
            variant="outline"
            onClick={handleClear}
            disabled={messages.length === 0}
          >
            Clear Chat
          </Button>
        </div>
      </div>

      <Card className="flex-1 overflow-hidden mb-4">
        <div className="h-full overflow-y-auto p-4 space-y-4">
          {messages.length === 0 ? (
            <div className="flex flex-col items-center justify-center h-full text-center text-muted-foreground">
              <Sparkles className="h-12 w-12 mb-4" />
              <h3 className="text-lg font-semibold">Start a conversation</h3>
              <p>Select a server and type your message below</p>
            </div>
          ) : (
            messages.map((message, index) => (
              <div
                key={index}
                className={`flex ${message.role === "user" ? "justify-end" : "justify-start"}`}
              >
                <div
                  className={`max-w-[80%] rounded-lg px-4 py-2 ${
                    message.role === "user"
                      ? "bg-primary text-primary-foreground"
                      : "bg-muted"
                  }`}
                >
                  <div className="font-semibold text-sm mb-1">
                    {message.role === "user" ? "You" : "Assistant"}
                  </div>
                  {message.role === "assistant" ? (
                    <AssistantContent
                      content={message.content}
                      reasoning={message.reasoning}
                    />
                  ) : (
                    <div className="whitespace-pre-wrap">{message.content}</div>
                  )}
                </div>
              </div>
            ))
          )}
          {isLoading && (
            <div className="flex justify-start">
              <div className="max-w-[80%] rounded-lg px-4 py-2 bg-muted">
                <div className="font-semibold text-sm mb-1">Assistant</div>
                <div className="flex items-center gap-2">
                  <div className="w-2 h-2 bg-muted-foreground rounded-full animate-bounce" />
                  <div className="w-2 h-2 bg-muted-foreground rounded-full animate-bounce [animation-delay:0.2s]" />
                  <div className="w-2 h-2 bg-muted-foreground rounded-full animate-bounce [animation-delay:0.4s]" />
                </div>
              </div>
            </div>
          )}
        </div>
      </Card>

      <div className="flex gap-2">
        <Textarea
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Type your message..."
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
            <Button variant="destructive" onClick={handleStop}>
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
        </div>
      </div>
    </div>
  )
}
