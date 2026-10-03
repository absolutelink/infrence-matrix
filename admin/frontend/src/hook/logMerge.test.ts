import { describe, expect, it } from "bun:test"

import {
  type FeedLine,
  MAX_LINES,
  mergeEntries,
  parseHistoryResponse,
  toLines,
} from "./logMerge"

const line = (seq: number, stream = "stdout", text = `l${seq}`): FeedLine => ({
  seq,
  stream,
  line: text,
})

describe("mergeEntries", () => {
  it("appends fresh entries and advances the cursor", () => {
    const { tail, cursor } = mergeEntries([], 0, [line(0), line(1)])
    expect(tail.map((entry) => entry.seq)).toEqual([0, 1])
    expect(cursor).toBe(2)
  })

  it("drops entries at or below the cursor (no duplicates on reconnect)", () => {
    const tail = [line(0), line(1), line(2)]
    const result = mergeEntries(tail, 3, [line(1), line(2), line(3)])
    expect(result.tail).toEqual([...tail, line(3)])
    expect(result.cursor).toBe(4)
  })

  it("returns the same reference when nothing is fresh", () => {
    const tail = [line(0)]
    const result = mergeEntries(tail, 1, [line(0)])
    expect(result.tail).toBe(tail)
    expect(result.cursor).toBe(1)
  })

  it("identical repeated lines never merge as duplicates", () => {
    const repeated = [line(0), line(1), line(2)]
    const merged = mergeEntries(repeated, 3, [
      line(3, "stdout", "l1"),
      line(4, "stdout", "l1"),
    ])
    expect(merged.tail.map((entry) => entry.seq)).toEqual([0, 1, 2, 3, 4])
  })

  it("caps the tail at MAX_LINES", () => {
    const entries = Array.from({ length: MAX_LINES + 50 }, (_, i) => line(i))
    const { tail, cursor } = mergeEntries([], 0, entries)
    expect(tail).toHaveLength(MAX_LINES)
    expect(tail[tail.length - 1].seq).toBe(MAX_LINES + 49)
    expect(cursor).toBe(MAX_LINES + 50)
  })
})

describe("toLines", () => {
  it("falls back to startSeq when entries lack seq", () => {
    const entries = toLines([{ line: "a" }, { line: "b" }], 7)
    expect(entries).toEqual([
      { seq: 7, stream: "stdout", line: "a" },
      { seq: 8, stream: "stdout", line: "b" },
    ])
  })
})

describe("parseHistoryResponse", () => {
  it("parses the cursor shape", () => {
    const history = parseHistoryResponse({
      entries: [
        { seq: 4, stream: "stderr", line: "x" },
        { seq: 5, stream: "stdout", line: "y" },
      ],
      next_cursor: 6,
      gap: true,
    })
    expect(history.nextCursor).toBe(6)
    expect(history.gap).toBe(true)
    expect(history.lines).toEqual([
      { seq: 4, stream: "stderr", line: "x" },
      { seq: 5, stream: "stdout", line: "y" },
    ])
  })

  it("synthesizes cursors for the legacy stdout/stderr shape", () => {
    const history = parseHistoryResponse({
      stdout: ["a", "b"],
      stderr: ["c"],
    })
    expect(history.lines).toEqual([
      { seq: 0, stream: "stdout", line: "a" },
      { seq: 1, stream: "stdout", line: "b" },
      { seq: 2, stream: "stderr", line: "c" },
    ])
    expect(history.nextCursor).toBe(3)
    expect(history.gap).toBe(false)
  })
})

import { unionLines } from "./logMerge"

describe("unionLines", () => {
  it("merges overlapping sequences without duplicates", () => {
    const history = [line(0), line(1), line(2)]
    const live = [line(2), line(3)]
    expect(unionLines(history, live).map((e) => e.seq)).toEqual([0, 1, 2, 3])
  })

  it("keeps sorted order when live leads", () => {
    const live = [line(5)]
    const history = [line(0), line(3)]
    expect(unionLines(history, live).map((e) => e.seq)).toEqual([0, 3, 5])
  })

  it("caps at MAX_LINES", () => {
    const a = Array.from({ length: MAX_LINES }, (_, i) => line(i))
    const b = [line(MAX_LINES), line(MAX_LINES + 1)]
    const out = unionLines(a, b)
    expect(out).toHaveLength(MAX_LINES)
    expect(out[out.length - 1].seq).toBe(MAX_LINES + 1)
  })
})
