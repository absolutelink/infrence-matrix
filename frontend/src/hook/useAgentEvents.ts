import { useEffect, useRef, useState } from "react"

export type AgentEvent = {
  event: string
  data: Record<string, unknown>
  timestamp: string
  /** Monotonic per-hook sequence for consumers to dedupe processed events */
  seq: number
}

const MAX_EVENTS = 200

/**
 * Subscribe to an agent's live event stream via the backend WebSocket
 * (mounted at /api/ws/agents/{agent_id}).
 */
export function useAgentEvents(agentId: string, enabled: boolean) {
  const [events, setEvents] = useState<AgentEvent[]>([])
  const [connected, setConnected] = useState(false)
  const wsRef = useRef<WebSocket | null>(null)
  const seqRef = useRef(0)

  useEffect(() => {
    if (!enabled || !agentId) {
      return
    }

    const baseUrl =
      (window as any).APP_CONFIG?.API_URL ||
      import.meta.env.VITE_API_URL ||
      window.location.origin
    const wsUrl = `${baseUrl.replace(/^http/, "ws")}/api/ws/events/${agentId}`

    let ws: WebSocket | null = null
    let reconnectTimer: ReturnType<typeof setTimeout> | null = null
    let closed = false

    const connect = () => {
      if (closed) {
        return
      }
      ws = new WebSocket(wsUrl)
      wsRef.current = ws

      ws.onopen = () => {
        setConnected(true)
      }
      ws.onmessage = (message) => {
        try {
          const parsed = JSON.parse(message.data)
          if (parsed.event === "heartbeat") {
            return
          }
          setEvents((prev) => [
            ...prev.slice(-(MAX_EVENTS - 1)),
            { ...parsed, seq: seqRef.current++ },
          ])
        } catch {
          // ignore malformed messages
        }
      }
      ws.onclose = () => {
        setConnected(false)
        if (!closed) {
          reconnectTimer = setTimeout(connect, 5000)
        }
      }
      ws.onerror = () => {
        ws?.close()
      }
    }

    connect()

    return () => {
      closed = true
      if (reconnectTimer) {
        clearTimeout(reconnectTimer)
      }
      ws?.close()
      wsRef.current = null
    }
  }, [agentId, enabled])

  return { events, connected }
}
