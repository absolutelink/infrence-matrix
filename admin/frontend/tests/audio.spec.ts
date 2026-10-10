import { expect, test } from "@playwright/test"

import {
  extractAudioErrorText,
  isModality,
  MAX_VOICE_BYTES,
  MODALITY_VALUES,
  modalityBadgeVariant,
  modalityEndpointHint,
  modalitySchema,
  nextModalityOnTypeLoad,
  parseVoicesCatalog,
  servedModalitiesFor,
  summarizeTranscription,
  TTS_RESPONSE_FORMATS,
} from "../src/lib/audio"

// Phase 24 (S5) locks the framework-free audio logic the definitions and
// playground screens depend on. These run in Node (no browser, no live
// backend) — the repo has no component/unit harness (Playwright here is
// E2E-only and the generated client is not mockable without a DOM runner), so
// the testable surface is extracted into @/lib/audio and asserted directly.

test("modality enum accepts the Phase 24 values", () => {
  for (const m of ["llm", "embedding", "tts", "asr"]) {
    expect(modalitySchema.safeParse(m).success).toBe(true)
  }
  expect(MODALITY_VALUES).toEqual(["llm", "embedding", "tts", "asr"])
  expect(isModality("tts")).toBe(true)
  expect(isModality("asr")).toBe(true)
})

test("modality enum rejects the retired audio bucket and junk", () => {
  for (const bad of ["audio", "speech", "", "LLM", undefined, null, 42]) {
    expect(modalitySchema.safeParse(bad).success).toBe(false)
  }
  expect(isModality("audio")).toBe(false)
})

test("badge variants stay consistent across modalities", () => {
  expect(modalityBadgeVariant("llm")).toBe("outline")
  expect(modalityBadgeVariant("embedding")).toBe("secondary")
  expect(modalityBadgeVariant("tts")).toBe("default")
  expect(modalityBadgeVariant("asr")).toBe("default")
  // Missing modality falls back to the llm treatment.
  expect(modalityBadgeVariant(undefined)).toBe("outline")
})

test("endpoint hint names the audio routes for the new modalities", () => {
  expect(modalityEndpointHint("tts")).toContain("/v1/audio/speech")
  expect(modalityEndpointHint("asr")).toContain("/v1/audio/transcriptions")
  expect(modalityEndpointHint("embedding")).toContain("/v1/embeddings")
  expect(modalityEndpointHint("llm")).toContain("/v1/responses")
})

test("playground TTS format select excludes pcm (streaming-only)", () => {
  expect(TTS_RESPONSE_FORMATS).not.toContain("pcm")
  for (const f of ["mp3", "wav", "opus", "flac", "aac"]) {
    expect(TTS_RESPONSE_FORMATS).toContain(f)
  }
})

test("summarizeTranscription flattens json / verbose_json / raw bodies", () => {
  expect(summarizeTranscription(JSON.stringify({ text: "hello world" }))).toBe(
    "hello world",
  )
  // verbose_json without a top-level text → join segment texts.
  const verbose = JSON.stringify({
    segments: [{ text: "one" }, { text: "two" }, { text: "" }],
  })
  expect(summarizeTranscription(verbose)).toBe("one two")
  // verbose_json with text wins over segments.
  expect(
    summarizeTranscription(
      JSON.stringify({ text: "full", segments: [{ text: "seg" }] }),
    ),
  ).toBe("full")
  // Non-JSON (text / srt / vtt) passes through untouched.
  expect(summarizeTranscription("plain transcript")).toBe("plain transcript")
  expect(summarizeTranscription("")).toBe("")
})

test("parseVoicesCatalog tolerates the passthrough shapes", () => {
  expect(parseVoicesCatalog(["a", "b"])).toEqual(["a", "b"])
  expect(parseVoicesCatalog({ voices: [{ name: "x" }, { id: "y" }] })).toEqual([
    "x",
    "y",
  ])
  expect(parseVoicesCatalog({ data: ["z"] })).toEqual(["z"])
  expect(parseVoicesCatalog(null)).toEqual([])
  expect(parseVoicesCatalog("nope")).toEqual([])
})

// MEDIUM regression lock: the modality auto-reset must NOT fire while the
// provider-type detail is still loading (cold cache), or editing an asr/tts/
// embedding definition would be clobbered to llm.
test("nextModalityOnTypeLoad does not clobber while the type detail is loading", () => {
  // typeDetail undefined/null (still fetching) → never reset, whatever the
  // stored modality is.
  expect(nextModalityOnTypeLoad(undefined, "asr")).toBeNull()
  expect(nextModalityOnTypeLoad(null, "tts")).toBeNull()
  expect(nextModalityOnTypeLoad(undefined, "embedding")).toBeNull()
})

test("nextModalityOnTypeLoad keeps a served modality and resets an unserved one", () => {
  // Type serves the current modality → leave it alone.
  expect(
    nextModalityOnTypeLoad({ serves_modalities: ["tts", "asr"] }, "asr"),
  ).toBeNull()
  expect(
    nextModalityOnTypeLoad({ serves_modalities: ["llm"] }, "llm"),
  ).toBeNull()
  // Switched to a type that no longer serves the current modality → reset to
  // the first served value.
  expect(nextModalityOnTypeLoad({ serves_modalities: ["tts"] }, "asr")).toBe(
    "tts",
  )
  expect(
    nextModalityOnTypeLoad({ serves_modalities: ["embedding"] }, "llm"),
  ).toBe("embedding")
  // Permissive/empty declaration falls back to ["llm"].
  expect(nextModalityOnTypeLoad({}, "tts")).toBe("llm")
})

test("servedModalitiesFor intersects the enum and falls back to llm", () => {
  expect(
    servedModalitiesFor({ serves_modalities: ["llm", "tts", "bogus"] }),
  ).toEqual(["llm", "tts"])
  expect(servedModalitiesFor({ serves_modalities: [] })).toEqual(["llm"])
  expect(servedModalitiesFor(undefined)).toEqual(["llm"])
})

test("extractAudioErrorText unwraps the admin error bodies", () => {
  expect(extractAudioErrorText({ detail: "no connected provider agent" })).toBe(
    "no connected provider agent",
  )
  expect(
    extractAudioErrorText({
      error: { type: "server_error", message: "upstream unreachable" },
    }),
  ).toBe("upstream unreachable")
  // A JSON string body (transcription responseType text).
  expect(
    extractAudioErrorText(
      JSON.stringify({ detail: "not a speech (tts) model" }),
    ),
  ).toBe("not a speech (tts) model")
  // A plain non-JSON string passes through; empty/nothing usable → null.
  expect(extractAudioErrorText("boom")).toBe("boom")
  expect(extractAudioErrorText("")).toBeNull()
  expect(extractAudioErrorText({})).toBeNull()
  expect(extractAudioErrorText(null)).toBeNull()
})

test("voice clip cap matches the admin's 32 MiB", () => {
  expect(MAX_VOICE_BYTES).toBe(32 * 1024 * 1024)
})
