import { useQuery } from "@tanstack/react-query"
import { createFileRoute } from "@tanstack/react-router"
import { FileAudio, Send, Square, Volume2 } from "lucide-react"
import { useEffect, useRef, useState } from "react"

import { AudioService, ModelsService } from "@/client"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
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
import {
  extractAudioErrorText,
  parseVoicesCatalog,
  summarizeTranscription,
  TTS_RESPONSE_FORMATS,
} from "@/lib/audio"
import { extractError } from "@/lib/errors"
import { getApiUrl } from "@/utils"

export const Route = createFileRoute("/_layout/playground")({
  component: PlaygroundPage,
  head: () => ({ meta: [{ title: "Playground - Inference Matrix" }] }),
})

interface PlaygroundModel {
  id: string
  owned_by?: string
  modality?: string
}

// Resolve a human message from an audio API error. The generated client throws
// an AxiosError on non-2xx (throwOnError: true); because speech uses
// responseType "blob" and transcription "text", err.response.data is a Blob or
// string rather than a parsed object, so unwrap it before falling back to the
// generic extractError().
async function resolveAudioError(e: unknown): Promise<string> {
  const data = (e as { response?: { data?: unknown } })?.response?.data
  if (data instanceof Blob) {
    try {
      return extractAudioErrorText(await data.text()) ?? extractError(e)
    } catch {
      return extractError(e)
    }
  }
  return extractAudioErrorText(data) ?? extractError(e)
}

// Minimal live-inference demo. For `llm` aliases it posts to the public /v1
// chat paths with stream=true and renders the SSE deltas as they arrive (the
// generated SDK types these bodies as free-form JSON, so we use fetch directly
// for the streaming read). Phase 24 adds `tts` (synthesize → <audio>) and `asr`
// (upload → transcript) panels, gated on the selected model's modality so the
// chat path is untouched.
function PlaygroundPage() {
  const { data: models } = useQuery({
    queryKey: ["playground-models"],
    queryFn: async () =>
      ((await ModelsService.listModels()).data?.data ??
        []) as PlaygroundModel[],
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
  const selectedModality =
    models?.find((m) => m.id === activeModel)?.modality ?? "llm"

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
                  ({m.owned_by}
                  {m.modality && m.modality !== "llm" ? ` · ${m.modality}` : ""}
                  )
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

      {selectedModality === "tts" ? (
        <TtsPanel model={activeModel} prompt={prompt} setPrompt={setPrompt} />
      ) : selectedModality === "asr" ? (
        <AsrPanel model={activeModel} />
      ) : (
        <>
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
        </>
      )}
    </div>
  )
}

// TTS: text + voice + response_format → POST /v1/audio/speech (binary) → play
// the returned blob. `pcm` is excluded from the format select (streaming-only,
// no container header to play from a blob).
function TtsPanel({
  model,
  prompt,
  setPrompt,
}: {
  model: string
  prompt: string
  setPrompt: (v: string) => void
}) {
  const { data: voices = [], error: voicesError } = useQuery({
    queryKey: ["playground-voices", model],
    queryFn: async () =>
      parseVoicesCatalog(
        (await AudioService.listVoices({ query: { model } })).data,
      ),
  })
  const [voice, setVoice] = useState<string>("")
  const [format, setFormat] = useState<string>("mp3")
  const [audioUrl, setAudioUrl] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  // Mirror the live object URL so the unmount cleanup can revoke the last one
  // (the synthesize path revokes the previous URL on replace).
  const audioUrlRef = useRef<string | null>(null)
  useEffect(
    () => () => {
      if (audioUrlRef.current) URL.revokeObjectURL(audioUrlRef.current)
    },
    [],
  )

  const synthesize = async () => {
    setError(null)
    setBusy(true)
    try {
      const resp = await AudioService.createSpeech({
        body: {
          model,
          input: prompt,
          voice: voice || undefined,
          response_format: format,
        },
        responseType: "blob",
      } as unknown as Parameters<typeof AudioService.createSpeech>[0])
      const blob = resp.data as unknown as Blob
      if (audioUrlRef.current) URL.revokeObjectURL(audioUrlRef.current)
      const url = URL.createObjectURL(blob)
      audioUrlRef.current = url
      setAudioUrl(url)
    } catch (e) {
      setError(await resolveAudioError(e))
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="grid gap-6 lg:grid-cols-2">
      <div className="flex flex-col gap-4 rounded-lg border p-4">
        <div className="space-y-2">
          <Label>Text</Label>
          <Textarea
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            rows={6}
          />
        </div>
        <div className="space-y-2">
          <Label>Voice</Label>
          <Select value={voice} onValueChange={setVoice}>
            <SelectTrigger>
              <SelectValue
                placeholder={
                  voicesError
                    ? "voices unavailable"
                    : voices.length === 0
                      ? "default voice"
                      : "Pick a voice"
                }
              />
            </SelectTrigger>
            <SelectContent>
              {voices.map((v) => (
                <SelectItem key={v} value={v}>
                  {v}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
          {voicesError && (
            <p className="text-xs text-destructive">
              {extractError(voicesError)}
            </p>
          )}
        </div>
        <div className="space-y-2">
          <Label>Response format</Label>
          <Select value={format} onValueChange={setFormat}>
            <SelectTrigger>
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {TTS_RESPONSE_FORMATS.map((f) => (
                <SelectItem key={f} value={f}>
                  {f}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
          <p className="text-xs text-muted-foreground">
            pcm is streaming-only and not offered here.
          </p>
        </div>
        <div>
          <Button onClick={synthesize} disabled={busy}>
            <Volume2 /> {busy ? "Synthesizing…" : "Synthesize"}
          </Button>
        </div>
        {error && <p className="text-xs text-destructive">{error}</p>}
      </div>

      <div className="flex flex-col gap-4 rounded-lg border p-4">
        <Label>Audio</Label>
        {audioUrl ? (
          // biome-ignore lint/a11y/useMediaCaption: raw TTS preview blob — no caption track exists for synthesized speech.
          <audio controls src={audioUrl} className="w-full" />
        ) : (
          <p className="text-sm text-muted-foreground">
            Synthesize to hear the result.
          </p>
        )}
      </div>
    </div>
  )
}

// ASR: upload a wav/mp3 → POST /v1/audio/transcriptions (json) → show the
// transcript text.
function AsrPanel({ model }: { model: string }) {
  const [file, setFile] = useState<File | null>(null)
  const [output, setOutput] = useState("")
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const transcribe = async () => {
    if (!file) {
      setError("Choose an audio file first.")
      return
    }
    setError(null)
    setOutput("")
    setBusy(true)
    try {
      const fd = new FormData()
      fd.append("model", model)
      fd.append("file", file)
      fd.append("response_format", "json")
      const resp = await AudioService.createTranscription({
        body: fd,
        responseType: "text",
      } as unknown as Parameters<typeof AudioService.createTranscription>[0])
      const raw =
        typeof resp.data === "string" ? resp.data : JSON.stringify(resp.data)
      setOutput(summarizeTranscription(raw))
    } catch (e) {
      setError(await resolveAudioError(e))
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="grid gap-6 lg:grid-cols-2">
      <div className="flex flex-col gap-4 rounded-lg border p-4">
        <div className="space-y-2">
          <Label htmlFor="asr-file">Audio file</Label>
          <Input
            id="asr-file"
            type="file"
            accept=".wav,.mp3,audio/wav,audio/mpeg"
            onChange={(e) => setFile(e.target.files?.[0] ?? null)}
          />
        </div>
        <div>
          <Button onClick={transcribe} disabled={busy}>
            <FileAudio /> {busy ? "Transcribing…" : "Transcribe"}
          </Button>
        </div>
        {error && <p className="text-xs text-destructive">{error}</p>}
      </div>

      <div className="flex flex-col gap-4 rounded-lg border p-4">
        <Label>Transcript</Label>
        <pre className="min-h-40 flex-1 whitespace-pre-wrap rounded-md bg-muted p-3 font-mono text-sm">
          {output || (busy ? "transcribing…" : "—")}
        </pre>
      </div>
    </div>
  )
}
