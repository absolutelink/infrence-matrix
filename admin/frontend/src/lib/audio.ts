// Phase 24 (Audio): pure, framework-free helpers shared by the definitions
// and playground screens. Kept free of React and the generated client so the
// behavior can be locked with light Node tests (see tests/audio.spec.ts).

import { z } from "zod"

// Full modality set. `audio` (the Phase 18 reserved bucket) is retired — speech
// ships as the two concrete modalities `tts` + `asr` (ARCHITECTURE.md §4).
export const MODALITY_VALUES = ["llm", "embedding", "tts", "asr"] as const

export const modalitySchema = z.enum(MODALITY_VALUES)

export type Modality = z.infer<typeof modalitySchema>

export function isModality(value: unknown): value is Modality {
  return (
    typeof value === "string" &&
    (MODALITY_VALUES as readonly string[]).includes(value)
  )
}

// Badge variant per modality, consistent with the pre-Phase-24 scheme
// (llm → outline, embedding → secondary); tts/asr get the filled `default`.
export function modalityBadgeVariant(
  modality?: string | null,
): "default" | "secondary" | "outline" {
  switch (modality ?? "llm") {
    case "embedding":
      return "secondary"
    case "tts":
    case "asr":
      return "default"
    default:
      return "outline"
  }
}

// Human-readable endpoint hint for the definition detail view.
export function modalityEndpointHint(modality?: string | null): string {
  switch (modality ?? "llm") {
    case "embedding":
      return "served by POST /v1/embeddings."
    case "tts":
      return "served by POST /v1/audio/speech (voice catalog: GET /v1/audio/voices)."
    case "asr":
      return "served by POST /v1/audio/transcriptions."
    default:
      return "served by /v1/responses and /v1/chat/completions."
  }
}

// Modalities a provider type can host: its `serves_modalities` intersected
// with the known enum, falling back to ["llm"] when it declares none.
export function servedModalitiesFor(
  typeDetail: { serves_modalities?: string[] } | null | undefined,
): string[] {
  const known = MODALITY_VALUES as readonly string[]
  const raw = typeDetail?.serves_modalities ?? []
  const list = raw.filter((m) => known.includes(m))
  return list.length > 0 ? list : ["llm"]
}

// Modality to reset the editor to when the chosen type no longer serves the
// current one (e.g. after a provider_type switch). Returns null for "leave it
// alone" — including while the provider-type detail is still loading: a cold
// cache makes servedModalitiesFor fall back to ["llm"], and resetting then
// would clobber a stored tts/asr/embedding modality (with instances attached
// the field is disabled → an unsavable PATCH 422 deadlock).
export function nextModalityOnTypeLoad(
  typeDetail: { serves_modalities?: string[] } | null | undefined,
  current: Modality,
): Modality | null {
  if (!typeDetail) return null
  const served = servedModalitiesFor(typeDetail)
  return served.includes(current) ? null : (served[0] as Modality)
}

// Max enrollable voice clip size — mirrors settings.AUDIO_MAX_VOICE_BYTES on
// the admin (32 MiB) so the UI rejects an oversized wav before uploading.
export const MAX_VOICE_BYTES = 32 * 1024 * 1024

// Pull a human message out of an audio error body. The admin surfaces errors
// as {detail: string} (HTTPException) or {error: {message}} (the speech
// passthrough's _error_response); a transcription body may already be a JSON
// string (responseType text) or a Blob (responseType blob — the caller awaits
// .text() first). Returns null when nothing usable is found so the caller can
// fall back to the generic extractError().
export function extractAudioErrorText(body: unknown): string | null {
  if (typeof body === "string") {
    const trimmed = body.trim()
    if (!trimmed) return null
    try {
      return extractAudioErrorText(JSON.parse(trimmed))
    } catch {
      return trimmed
    }
  }
  if (body && typeof body === "object") {
    const o = body as Record<string, unknown>
    if (typeof o.detail === "string" && o.detail) return o.detail
    const err = o.error
    if (err && typeof err === "object") {
      const message = (err as Record<string, unknown>).message
      if (typeof message === "string" && message) return message
    } else if (typeof err === "string" && err) {
      return err
    }
  }
  return null
}

// Buffered, playable speech response formats. `pcm` is intentionally excluded:
// it is the Qwen3 incremental raw-stream format (octet-stream + X-Sample-Rate,
// no container header), so it cannot be played from a single blob — the
// playground offers only the self-describing container formats.
export const TTS_RESPONSE_FORMATS = [
  "mp3",
  "wav",
  "opus",
  "flac",
  "aac",
] as const

export type TtsResponseFormat = (typeof TTS_RESPONSE_FORMATS)[number]

// Turn a /v1/audio/transcriptions body into a display string. json and
// verbose_json carry a `text` field; verbose_json additionally has `segments`
// (used as a fallback summary when `text` is absent). text/srt/vtt (and any
// non-JSON body) pass through untouched.
export function summarizeTranscription(raw: string): string {
  const trimmed = raw.trim()
  if (!trimmed) return ""
  let parsed: unknown
  try {
    parsed = JSON.parse(trimmed)
  } catch {
    return raw
  }
  if (parsed && typeof parsed === "object") {
    const obj = parsed as Record<string, unknown>
    if (typeof obj.text === "string" && obj.text) return obj.text
    const segments = obj.segments
    if (Array.isArray(segments)) {
      const lines = segments
        .map((s) =>
          s && typeof s === "object"
            ? (s as Record<string, unknown>).text
            : undefined,
        )
        .filter((t): t is string => typeof t === "string" && t.length > 0)
      if (lines.length > 0) return lines.join(" ")
    }
  }
  return raw
}

// Parse the live voice catalog proxied from a tts backend (GET
// /v1/audio/voices). The admin passes the agent's body through untouched, so
// the shape is not fixed by the OpenAPI schema — accept a bare array, a
// `{voices: []}` / `{data: []}` envelope, and entries that are either strings
// or objects carrying a name/id/voice field.
export function parseVoicesCatalog(data: unknown): string[] {
  const arr = Array.isArray(data)
    ? data
    : data && typeof data === "object"
      ? ((data as Record<string, unknown>).voices ??
        (data as Record<string, unknown>).data ??
        [])
      : []
  if (!Array.isArray(arr)) return []
  const out: string[] = []
  for (const v of arr) {
    if (typeof v === "string") {
      out.push(v)
    } else if (v && typeof v === "object") {
      const o = v as Record<string, unknown>
      const name = o.name ?? o.id ?? o.voice
      if (typeof name === "string") out.push(name)
    }
  }
  return out
}
