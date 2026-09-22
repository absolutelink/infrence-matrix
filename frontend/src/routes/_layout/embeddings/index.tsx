import { useMutation, useSuspenseQuery } from "@tanstack/react-query"
import { createFileRoute } from "@tanstack/react-router"
import { Braces, Copy, Plus, Trash2 } from "lucide-react"
import { useState } from "react"
import {
  type EmbeddingResponse,
  ModelsService,
  V1EmbeddingsService,
} from "@/client"
import { Button } from "@/components/ui/button"
import { Card } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"
import useCustomToast from "@/hooks/useCustomToast"

export const Route = createFileRoute("/_layout/embeddings/")({
  component: Embeddings,
  head: () => ({
    meta: [{ title: "Embeddings - Inference Matrix" }],
  }),
})

function getModelsQueryOptions() {
  return {
    queryFn: async () =>
      (await ModelsService.readModels({ query: { skip: 0, limit: 100 } })).data,
    queryKey: ["models-embeddings"],
  }
}

function Embeddings() {
  const { showSuccessToast, showErrorToast } = useCustomToast()
  const { data: models } = useSuspenseQuery(getModelsQueryOptions())
  const [selectedModel, setSelectedModel] = useState<string>("")
  const [inputs, setInputs] = useState<string[]>([""])
  const [result, setResult] = useState<EmbeddingResponse | null>(null)

  const embedMutation = useMutation({
    mutationFn: async () => {
      const nonEmpty = inputs.filter((i) => i.trim())
      const response = (await V1EmbeddingsService.v1.createEmbedding({
        body: {
          model: selectedModel,
          input: nonEmpty.length === 1 ? nonEmpty[0] : nonEmpty,
        },
      })) as unknown as EmbeddingResponse
      return response
    },
    onSuccess: (data) => {
      setResult(data)
      showSuccessToast(
        `${data.data.length} vector(s), ${data.usage?.total_tokens ?? 0} tokens`,
      )
    },
    onError: (error: Error) => {
      showErrorToast(error.message)
    },
  })

  const dimension = result?.data[0]?.embedding.length ?? 0

  return (
    <div className="flex flex-col h-[calc(100vh-8rem)]">
      <div className="flex items-center justify-between mb-4">
        <div>
          <h1 className="text-2xl font-bold tracking-tight">Embeddings</h1>
          <p className="text-muted-foreground">
            Generate vector embeddings for text
          </p>
        </div>
        <Select value={selectedModel} onValueChange={setSelectedModel}>
          <SelectTrigger className="w-[300px]">
            <SelectValue placeholder="Select an embedding model" />
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

      <div className="grid grid-cols-2 gap-4 flex-1 min-h-0">
        <Card className="p-4 flex flex-col min-h-0">
          <div className="flex items-center justify-between mb-3">
            <span className="font-medium text-sm">Inputs</span>
            <Button
              variant="ghost"
              size="sm"
              onClick={() => setInputs((prev) => [...prev, ""])}
            >
              <Plus className="h-4 w-4 mr-1" />
              Add
            </Button>
          </div>
          <div className="flex-1 overflow-y-auto space-y-2">
            {inputs.map((input, i) => (
              <div key={i} className="flex gap-2 items-start">
                <Input
                  value={input}
                  onChange={(e) =>
                    setInputs((prev) =>
                      prev.map((p, j) => (j === i ? e.target.value : p)),
                    )
                  }
                  placeholder={`Text ${i + 1}`}
                  disabled={embedMutation.isPending}
                />
                <Button
                  variant="ghost"
                  size="icon"
                  disabled={inputs.length === 1}
                  onClick={() =>
                    setInputs((prev) => prev.filter((_, j) => j !== i))
                  }
                >
                  <Trash2 className="h-4 w-4" />
                </Button>
              </div>
            ))}
          </div>
          <Button
            className="mt-3"
            onClick={() => embedMutation.mutate()}
            disabled={
              !selectedModel ||
              embedMutation.isPending ||
              inputs.every((i) => !i.trim())
            }
          >
            <Braces className="h-4 w-4 mr-2" />
            {embedMutation.isPending ? "Generating..." : "Generate Embeddings"}
          </Button>
        </Card>

        <Card className="p-4 flex flex-col min-h-0">
          {result ? (
            <>
              <div className="flex items-center justify-between mb-3">
                <div className="flex gap-2 text-sm text-muted-foreground">
                  <span>{result.data.length} vector(s)</span>
                  <span>·</span>
                  <span>{dimension} dims</span>
                  <span>·</span>
                  <span>{result.usage?.total_tokens ?? 0} tokens</span>
                </div>
                <Button
                  variant="ghost"
                  size="icon"
                  onClick={() =>
                    navigator.clipboard.writeText(
                      JSON.stringify(result.data[0].embedding),
                    )
                  }
                >
                  <Copy className="h-4 w-4" />
                </Button>
              </div>
              <div className="flex-1 overflow-y-auto">
                {result.data.map((d, i) => (
                  <div key={i} className="mb-3">
                    <div className="text-xs font-medium mb-1">
                      Vector {d.index}
                    </div>
                    <div className="flex flex-wrap gap-1">
                      {d.embedding.slice(0, 50).map((v, j) => (
                        <span
                          key={j}
                          className={`px-1.5 py-0.5 rounded text-[10px] font-mono ${
                            v >= 0
                              ? "bg-emerald-500/10 text-emerald-600 dark:text-emerald-400"
                              : "bg-rose-500/10 text-rose-600 dark:text-rose-400"
                          }`}
                        >
                          {v.toFixed(3)}
                        </span>
                      ))}
                      {d.embedding.length > 50 && (
                        <span className="text-[10px] text-muted-foreground self-center">
                          +{d.embedding.length - 50} more
                        </span>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            </>
          ) : (
            <div className="flex flex-col items-center justify-center h-full text-center text-muted-foreground">
              <Braces className="h-10 w-10 mb-3" />
              <p>Embedding vectors will appear here</p>
            </div>
          )}
        </Card>
      </div>
    </div>
  )
}
