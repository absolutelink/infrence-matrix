import { useMutation, useSuspenseQuery } from "@tanstack/react-query"
import { createFileRoute } from "@tanstack/react-router"
import { AudioLines, FileAudio, Upload } from "lucide-react"
import { useRef, useState } from "react"
import {
  ModelsService,
  type TranscriptionResponse,
  V1AudioService,
} from "@/client"
import { Button } from "@/components/ui/button"
import { Card } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"
import useCustomToast from "@/hooks/useCustomToast"

export const Route = createFileRoute("/_layout/audio/")({
  component: AudioTranscription,
  head: () => ({
    meta: [{ title: "Transcriptions - Inference Matrix" }],
  }),
})

const ACCEPTED = ".wav,.mp3,.flac,.ogg,.m4a,.webm"

function getModelsQueryOptions() {
  return {
    queryFn: async () =>
      (await ModelsService.readModels({ query: { skip: 0, limit: 100 } })).data,
    queryKey: ["models-audio"],
  }
}

function AudioTranscription() {
  const { showSuccessToast, showErrorToast } = useCustomToast()
  const { data: models } = useSuspenseQuery(getModelsQueryOptions())
  const [selectedModel, setSelectedModel] = useState<string>("")
  const [file, setFile] = useState<File | null>(null)
  const [language, setLanguage] = useState("")
  const [responseFormat, setResponseFormat] = useState<"json" | "text">("json")
  const [result, setResult] = useState<TranscriptionResponse | null>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)

  const transcribeMutation = useMutation({
    mutationFn: async () => {
      if (!file) throw new Error("No file selected")
      const response = (await V1AudioService.v1.createTranscription({
        body: {
          file,
          model: selectedModel,
          language: language || undefined,
          response_format: responseFormat,
        },
      })) as unknown as TranscriptionResponse
      return response
    },
    onSuccess: (data) => {
      setResult(data)
      showSuccessToast("Transcription complete")
    },
    onError: (error: Error) => {
      showErrorToast(error.message)
    },
  })

  const formatBytes = (bytes: number) => {
    if (bytes < 1024) return `${bytes} B`
    if (bytes < 1048576) return `${(bytes / 1024).toFixed(1)} KB`
    return `${(bytes / 1048576).toFixed(1)} MB`
  }

  return (
    <div className="flex flex-col h-[calc(100vh-8rem)]">
      <div className="flex items-center justify-between mb-4">
        <div>
          <h1 className="text-2xl font-bold tracking-tight">
            Audio Transcription
          </h1>
          <p className="text-muted-foreground">
            Transcribe audio files with speech models
          </p>
        </div>
        <Select value={selectedModel} onValueChange={setSelectedModel}>
          <SelectTrigger className="w-[300px]">
            <SelectValue placeholder="Select a speech model" />
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
        <Card className="p-4 flex flex-col">
          <span className="font-medium text-sm mb-3">Audio file</span>
          <button
            type="button"
            onClick={() => fileInputRef.current?.click()}
            className="flex-1 min-h-[200px] border-2 border-dashed rounded-lg flex flex-col items-center justify-center gap-2 text-muted-foreground hover:border-primary/50 hover:text-foreground transition-colors"
          >
            {file ? (
              <>
                <FileAudio className="h-10 w-10 mb-1" />
                <span className="font-medium text-foreground">{file.name}</span>
                <span className="text-sm">{formatBytes(file.size)}</span>
              </>
            ) : (
              <>
                <Upload className="h-10 w-10 mb-1" />
                <span className="font-medium">Click to upload audio</span>
                <span className="text-sm">wav, mp3, flac, ogg, m4a, webm</span>
              </>
            )}
          </button>
          <Input
            ref={fileInputRef}
            type="file"
            accept={ACCEPTED}
            className="hidden"
            onChange={(e) => setFile(e.target.files?.[0] ?? null)}
          />

          <div className="grid grid-cols-2 gap-3 mt-4">
            <div className="space-y-1.5">
              <Label className="text-sm">Language (optional)</Label>
              <Input
                value={language}
                onChange={(e) => setLanguage(e.target.value)}
                placeholder="e.g. en"
              />
            </div>
            <div className="space-y-1.5">
              <Label className="text-sm">Response format</Label>
              <Select
                value={responseFormat}
                onValueChange={(v) => setResponseFormat(v as "json" | "text")}
              >
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="json">json</SelectItem>
                  <SelectItem value="text">text</SelectItem>
                </SelectContent>
              </Select>
            </div>
          </div>

          <Button
            className="mt-4"
            onClick={() => transcribeMutation.mutate()}
            disabled={!file || !selectedModel || transcribeMutation.isPending}
          >
            <AudioLines className="h-4 w-4 mr-2" />
            {transcribeMutation.isPending ? "Transcribing..." : "Transcribe"}
          </Button>
        </Card>

        <Card className="p-4 flex flex-col min-h-0">
          {result ? (
            <>
              <div className="flex items-center justify-between mb-3">
                <span className="font-medium text-sm">Transcript</span>
              </div>
              <div className="flex-1 overflow-y-auto whitespace-pre-wrap text-sm">
                {result.text}
              </div>
            </>
          ) : (
            <div className="flex flex-col items-center justify-center h-full text-center text-muted-foreground">
              <AudioLines className="h-10 w-10 mb-3" />
              <p>Transcription will appear here</p>
            </div>
          )}
        </Card>
      </div>
    </div>
  )
}
