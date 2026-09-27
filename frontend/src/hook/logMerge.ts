/**
 * Pure cursor-merge helpers shared by log feeds.
 *
 * Live agent events and polled history carry agent-side sequence numbers;
 * merging is done purely by cursor, never by content matching, so repeated
 * log lines can never duplicate.
 */

export type FeedLine = { seq: number; stream: string; line: string }

export const MAX_LINES = 2000

export type RawEntry = { seq?: number; stream?: string; line?: string }

export function toLines(
  entries: RawEntry[] | undefined,
  startSeq: number,
): FeedLine[] {
  return (entries ?? []).map((entry, index) => ({
    seq: entry.seq ?? startSeq + index,
    stream: entry.stream ?? "stdout",
    line: entry.line ?? "",
  }))
}

/**
 * Merge sequenced entries into the tail, dropping anything at or below the
 * cursor. Returns the same array reference when nothing was appended.
 */
export function mergeEntries(
  tail: FeedLine[],
  cursor: number,
  entries: FeedLine[],
): { tail: FeedLine[]; cursor: number } {
  const fresh = entries.filter((entry) => entry.seq >= cursor)
  if (fresh.length === 0) return { tail, cursor }
  const next = [...tail, ...fresh]
  return {
    tail: next.length > MAX_LINES ? next.slice(-MAX_LINES) : next,
    cursor: fresh[fresh.length - 1].seq + 1,
  }
}

/**
 * Union two sequences of lines that are each ascending by seq, dropping
 * duplicates. Used when live events arrive while the initial history
 * bootstrap is in flight.
 */
export function unionLines(a: FeedLine[], b: FeedLine[]): FeedLine[] {
  const out: FeedLine[] = []
  let i = 0
  let j = 0
  while (i < a.length && j < b.length) {
    if (a[i].seq < b[j].seq) {
      out.push(a[i++])
    } else if (a[i].seq > b[j].seq) {
      out.push(b[j++])
    } else {
      out.push(a[i])
      i += 1
      j += 1
    }
  }
  while (i < a.length) out.push(a[i++])
  while (j < b.length) out.push(b[j++])
  return out.length > MAX_LINES ? out.slice(-MAX_LINES) : out
}

/**
 * Interpret a history payload. Handles the cursor shape (entries +
 * next_cursor) and the legacy unsequenced stdout/stderr arrays.
 */
export type History = { lines: FeedLine[]; nextCursor: number; gap: boolean }

export function parseHistoryResponse(data: {
  entries?: RawEntry[]
  lines?: RawEntry[]
  stdout?: string[]
  stderr?: string[]
  next_cursor?: number
  gap?: boolean
}): History {
  if (data.entries) {
    return {
      lines: toLines(data.entries, 0),
      nextCursor: data.next_cursor ?? 0,
      gap: Boolean(data.gap),
    }
  }
  if (data.lines) {
    return {
      lines: toLines(data.lines, 0),
      nextCursor: data.next_cursor ?? 0,
      gap: Boolean(data.gap),
    }
  }
  const stdout = data.stdout ?? []
  const legacy: FeedLine[] = [
    ...stdout.map((line, i) => ({ seq: i, stream: "stdout", line })),
    ...(data.stderr ?? []).map((line, i) => ({
      seq: stdout.length + i,
      stream: "stderr",
      line,
    })),
  ]
  return { lines: legacy, nextCursor: legacy.length, gap: false }
}
