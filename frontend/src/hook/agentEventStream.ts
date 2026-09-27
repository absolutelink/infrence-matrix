/**
 * Shared, refcounted WebSocket streams for agent events.
 *
 * One socket per agent_id is opened no matter how many components subscribe.
 * Sequence numbers are assigned by the stream (not the consumer) and keep
 * increasing across reconnects, so consumers can detect dropped events.
 */

export type AgentEvent = {
  event: string
  data: Record<string, unknown>
  timestamp: string
  /** Monotonic per-agent sequence, continuous across reconnects */
  seq: number
}

export type StreamStatus = "connecting" | "connected" | "reconnecting"

const MAX_EVENTS = 500
const RECONNECT_BASE_MS = 1000
const RECONNECT_MAX_MS = 15000

type Listener = (events: AgentEvent[]) => void
type StatusListener = (status: StreamStatus, attempt: number) => void

class AgentEventStream {
  readonly agentId: string
  private ws: WebSocket | null = null
  private events: AgentEvent[] = []
  private seq = 0
  private status: StreamStatus = "connecting"
  private attempt = 0
  private listeners = new Set<Listener>()
  private statusListeners = new Set<StatusListener>()
  private reconnectTimer: ReturnType<typeof setTimeout> | null = null
  private disposed = false

  constructor(agentId: string) {
    this.agentId = agentId
    this.connect()
  }

  private url(): string {
    const baseUrl =
      (window as unknown as { APP_CONFIG?: { API_URL?: string } }).APP_CONFIG
        ?.API_URL ||
      (import.meta.env.VITE_API_URL as string | undefined) ||
      window.location.origin
    return `${baseUrl.replace(/^http/, "ws")}/api/ws/events/${this.agentId}`
  }

  private setStatus(status: StreamStatus) {
    if (this.status === status) return
    this.status = status
    for (const listener of this.statusListeners) {
      listener(status, this.attempt)
    }
  }

  private connect() {
    if (this.disposed) return
    this.setStatus(this.attempt === 0 ? "connecting" : "reconnecting")
    const ws = new WebSocket(this.url())
    this.ws = ws

    ws.onopen = () => {
      this.attempt = 0
      this.setStatus("connected")
    }
    ws.onmessage = (message) => {
      try {
        const parsed = JSON.parse(message.data)
        if (parsed.event === "heartbeat") return
        const batch: AgentEvent[] = []
        for (const item of Array.isArray(parsed) ? parsed : [parsed]) {
          batch.push({ ...item, seq: this.seq++ })
        }
        this.events = [...this.events, ...batch].slice(-MAX_EVENTS)
        for (const listener of this.listeners) {
          listener(batch)
        }
      } catch {
        // ignore malformed messages
      }
    }
    ws.onclose = () => {
      this.ws = null
      if (this.disposed) return
      this.setStatus("reconnecting")
      this.scheduleReconnect()
    }
    ws.onerror = () => {
      ws.close()
    }
  }

  private scheduleReconnect() {
    if (this.reconnectTimer) return
    this.attempt += 1
    const delay = Math.min(
      RECONNECT_BASE_MS * 2 ** (this.attempt - 1),
      RECONNECT_MAX_MS,
    )
    this.reconnectTimer = setTimeout(() => {
      this.reconnectTimer = null
      this.connect()
    }, delay)
  }

  addListener(listener: Listener) {
    this.listeners.add(listener)
  }

  removeListener(listener: Listener) {
    this.listeners.delete(listener)
  }

  addStatusListener(listener: StatusListener) {
    this.statusListeners.add(listener)
    listener(this.status, this.attempt)
  }

  removeStatusListener(listener: StatusListener) {
    this.statusListeners.delete(listener)
  }

  recentEvents(sinceSeq = -1): AgentEvent[] {
    return this.events.filter((event) => event.seq > sinceSeq)
  }

  getStatus(): StreamStatus {
    return this.status
  }

  getAttempt(): number {
    return this.attempt
  }

  hasSubscribers(): boolean {
    return this.listeners.size > 0 || this.statusListeners.size > 0
  }

  dispose() {
    this.disposed = true
    if (this.reconnectTimer) clearTimeout(this.reconnectTimer)
    this.listeners.clear()
    this.statusListeners.clear()
    this.ws?.close()
    this.ws = null
  }
}

const streams = new Map<string, AgentEventStream>()

function getStream(agentId: string): AgentEventStream {
  let stream = streams.get(agentId)
  if (!stream) {
    stream = new AgentEventStream(agentId)
    streams.set(agentId, stream)
  }
  return stream
}

/**
 * Acquire the shared stream for an agent. Returns the stream and a release
 * function; the socket closes when the last holder releases.
 */
export function acquireStream(agentId: string): {
  stream: AgentEventStream
  release: () => void
} {
  const stream = getStream(agentId)
  let released = false
  const release = () => {
    if (released) return
    released = true
    if (stream.hasSubscribers()) return
    stream.dispose()
    if (streams.get(agentId) === stream) {
      streams.delete(agentId)
    }
  }
  return { stream, release }
}

export type { AgentEventStream }
