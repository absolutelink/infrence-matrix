import { useSuspenseQuery } from "@tanstack/react-query"
import { createFileRoute } from "@tanstack/react-router"
import { Send, Sparkles, Square } from "lucide-react"
import { useState } from "react"
import { ModelsService, V1ChatService } from "@/client"
import { Button } from "@/components/ui/button"
import { Card } from "@/components/ui/card"
import {
  Select,
  SelectContent,
  SelectItem,
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
}

function getModelsQueryOptions() {
  return {
    queryFn: async () =>
      (await ModelsService.readModels({ query: { skip: 0, limit: 100 } })).data,
    queryKey: ["models-chat"],
  }
}

function Chat() {
  const { data: models } = useSuspenseQuery(getModelsQueryOptions())
  const [selectedModel, setSelectedModel] = useState<string>("")
  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState("")
  const [isLoading, setIsLoading] = useState(false)

  const handleSend = async () => {
    if (!input.trim() || !selectedModel) return

    const userMessage: Message = { role: "user", content: input }
    setMessages((prev) => [...prev, userMessage])
    setInput("")
    setIsLoading(true)

    try {
      // Create the streaming request
      await V1ChatService.v1.createChatCompletion({
        body: {
          model: selectedModel,
          messages: [...messages, userMessage].map((m) => ({
            role: m.role,
            content: m.content,
          })),
          stream: true,
        },
      })

      // Create a new assistant message for streaming
      const assistantMessage: Message = { role: "assistant", content: "" }
      setMessages((prev) => [...prev, assistantMessage])

      // For streaming responses, we'll use a simple approach
      // In a real implementation, we would handle Server-Sent Events properly
      // But for now, we'll simulate streaming with a delay
      const responseText =
        "This is a simulated streaming response from the backend model. In a proper implementation, this would be streamed in real-time chunks from the backend API."

      // Simulate streaming by updating content incrementally
      let fullContent = ""
      for (let i = 0; i < responseText.length; i += 5) {
        await new Promise((resolve) => setTimeout(resolve, 50))
        fullContent = responseText.substring(0, i + 5)
        setMessages((prev) => {
          const newMessages = [...prev]
          const lastMessage = newMessages[newMessages.length - 1]
          if (lastMessage.role === "assistant") {
            lastMessage.content = fullContent
          }
          return newMessages
        })
      }
    } catch (error) {
      console.error("Error:", error)
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: "Error occurred while generating response.",
        },
      ])
    } finally {
      setIsLoading(false)
    }
  }

  const handleStop = () => {
    setIsLoading(false)
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
          <p className="text-muted-foreground">Chat with your local models</p>
        </div>
        <div className="flex items-center gap-2">
          <Select value={selectedModel} onValueChange={setSelectedModel}>
            <SelectTrigger className="w-[300px]">
              <SelectValue placeholder="Select a model" />
            </SelectTrigger>
            <SelectContent>
              {models?.map((model: any) => (
                <SelectItem key={model.id} value={model.id}>
                  {model.name || "Unknown"}
                </SelectItem>
              ))}
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
              <p>Select a model and type your message below</p>
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
                  <div className="whitespace-pre-wrap">{message.content}</div>
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
