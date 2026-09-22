import { useSuspenseQuery } from "@tanstack/react-query"
import { createFileRoute } from "@tanstack/react-router"
import { Send, Sparkles, Square } from "lucide-react"
import { useRef, useState } from "react"
import { ModelsService, V1CompletionsService } from "@/client"
import { Button } from "@/components/ui/button"
import { Card } from "@/components/ui/card"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"
import { Slider } from "@/components/ui/slider"
import { Textarea } from "@/components/ui/textarea"

export const Route = createFileRoute("/_layout/completions/")({
  component: TextCompletion,
  head: () => ({
    meta: [{ title: "Text Completion - Inference Matrix" }],
  }),
})

type CompletionResponse = {
  choices?: { text?: string }[]
}

function getModelsQueryOptions() {
  return {
    queryFn: async () =>
      (await ModelsService.readModels({ query: { skip: 0, limit: 100 } })).data,
    queryKey: ["models-completions"],
  }
}

function TextCompletion() {
  const { data: models } = useSuspenseQuery(getModelsQueryOptions())
  const [selectedModel, setSelectedModel] = useState<string>("")
  const [prompt, setPrompt] = useState("")
  const [output, setOutput] = useState("")
  const [maxTokens, setMaxTokens] = useState(256)
  const [temperature, setTemperature] = useState(0.7)
  const [stream, setStream] = useState(true)
  const [isLoading, setIsLoading] = useState(false)
  const abortRef = useRef<AbortController | null>(null)

  const handleGenerate = async () => {
    if (!prompt.trim() || !selectedModel) return
    setIsLoading(true)
    setOutput("")
    abortRef.current = new AbortController()

    try {
      if (stream) {
        const response = await fetch("/v1/completions", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            model: selectedModel,
            prompt,
            stream: true,
            max_tokens: maxTokens,
            temperature,
          }),
          signal: abortRef.current.signal,
        })

        if (!response.ok || !response.body) {
          throw new Error(`Request failed: ${response.status}`)
        }

        const reader = response.body.getReader()
        const decoder = new TextDecoder()
        let buffer = ""

        for (;;) {
          const { done, value } = await reader.read()
          if (done) break
          buffer += decoder.decode(value, { stream: true })
          const lines = buffer.split("\n\n")
          buffer = lines.pop() ?? ""
          for (const line of lines) {
            const data = line.replace(/^data: /, "").trim()
            if (!data || data === "[DONE]") continue
            try {
              const chunk = JSON.parse(data)
              if (chunk.error) {
                throw new Error(chunk.error.message ?? "Stream error")
              }
              const text = chunk.choices?.[0]?.text ?? ""
              setOutput((prev) => prev + text)
            } catch {}
          }
        }
      } else {
        const response = (await V1CompletionsService.v1.createCompletion({
          body: {
            model: selectedModel,
            prompt,
            stream: false,
            max_tokens: maxTokens,
            temperature,
          },
        })) as unknown as CompletionResponse
        setOutput(response.choices?.[0]?.text ?? "")
      }
    } catch (error) {
      if ((error as Error).name === "AbortError") {
        setOutput((prev) => `${prev}\n\n[Stopped]`)
      } else {
        setOutput(`Error: ${(error as Error).message}`)
      }
    } finally {
      setIsLoading(false)
      abortRef.current = null
    }
  }

  const handleStop = () => {
    abortRef.current?.abort()
  }

  return (
    <div className="flex flex-col h-[calc(100vh-8rem)]">
      <div className="flex items-center justify-between mb-4">
        <div>
          <h1 className="text-2xl font-bold tracking-tight">Text Completion</h1>
          <p className="text-muted-foreground">
            Legacy completions API — raw prompt in, text out
          </p>
        </div>
        <div className="flex items-center gap-2">
          <Button
            variant="outline"
            size="sm"
            onClick={() => setStream((s) => !s)}
          >
            {stream ? "Streaming" : "Blocking"}
          </Button>
          <Select value={selectedModel} onValueChange={setSelectedModel}>
            <SelectTrigger className="w-[300px]">
              <SelectValue placeholder="Select a model" />
            </SelectTrigger>
            <SelectContent>
              {models?.map((model) => (
                <SelectItem key={model.id} value={model.name}>
                  {model.name || "Unknown"}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
      </div>

      <div className="grid grid-cols-[1fr_280px] gap-4 flex-1 min-h-0">
        <div className="flex flex-col gap-4 min-h-0">
          <Card className="p-4">
            <Textarea
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              placeholder="Enter your prompt..."
              className="min-h-[150px] font-mono text-sm"
              disabled={isLoading}
            />
            <div className="flex justify-end mt-3">
              {isLoading ? (
                <Button variant="destructive" onClick={handleStop}>
                  <Square className="h-4 w-4 mr-2" />
                  Stop
                </Button>
              ) : (
                <Button
                  onClick={handleGenerate}
                  disabled={!prompt.trim() || !selectedModel}
                >
                  <Send className="h-4 w-4 mr-2" />
                  Generate
                </Button>
              )}
            </div>
          </Card>

          <Card className="flex-1 min-h-0 overflow-y-auto p-4">
            {output ? (
              <pre className="whitespace-pre-wrap font-mono text-sm">
                {output}
              </pre>
            ) : (
              <div className="flex flex-col items-center justify-center h-full text-center text-muted-foreground">
                <Sparkles className="h-10 w-10 mb-3" />
                <p>Completion output will appear here</p>
              </div>
            )}
          </Card>
        </div>

        <Card className="p-4 space-y-6">
          <div className="space-y-2">
            <div className="flex justify-between text-sm">
              <span className="font-medium">Max tokens</span>
              <span className="text-muted-foreground">{maxTokens}</span>
            </div>
            <Slider
              value={[maxTokens]}
              onValueChange={(v) => setMaxTokens(v[0])}
              min={16}
              max={4096}
              step={16}
            />
          </div>
          <div className="space-y-2">
            <div className="flex justify-between text-sm">
              <span className="font-medium">Temperature</span>
              <span className="text-muted-foreground">
                {temperature.toFixed(2)}
              </span>
            </div>
            <Slider
              value={[temperature]}
              onValueChange={(v) => setTemperature(v[0])}
              min={0}
              max={2}
              step={0.05}
            />
          </div>
          {output && (
            <div className="space-y-2 text-sm">
              <span className="font-medium">Output length</span>
              <div className="text-muted-foreground">{output.length} chars</div>
            </div>
          )}
        </Card>
      </div>
    </div>
  )
}
