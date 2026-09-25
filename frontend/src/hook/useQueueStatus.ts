import { useEffect, useState } from "react"

export type QueueServerStatus = {
  id: string
  alias: string | null
  model_id: string
  capacity: number
  active: number
  available: number
  telemetry_known: boolean
}

export type QueueStatus = {
  queued: number
  active: number
  available: number
  capacity: number
  servers: QueueServerStatus[]
}

export function useQueueStatus() {
  const [status, setStatus] = useState<QueueStatus | null>(null)
  const [connected, setConnected] = useState(false)

  useEffect(() => {
    const baseUrl =
      (window as any).APP_CONFIG?.API_URL ||
      import.meta.env.VITE_API_URL ||
      window.location.origin
    const wsUrl = `${baseUrl.replace(/^http/, "ws")}/api/ws/queue-status`
    let socket: WebSocket | null = null
    let reconnectTimer: ReturnType<typeof setTimeout> | null = null
    let closed = false

    const connect = () => {
      if (closed) return
      socket = new WebSocket(wsUrl)
      socket.onopen = () => setConnected(true)
      socket.onmessage = (message) => {
        try {
          const parsed = JSON.parse(message.data)
          if (parsed.event === "queue.status" && parsed.data) {
            setStatus(parsed.data as QueueStatus)
          }
        } catch {
          // Ignore malformed status frames.
        }
      }
      socket.onclose = () => {
        setConnected(false)
        if (!closed) reconnectTimer = setTimeout(connect, 5000)
      }
      socket.onerror = () => socket?.close()
    }

    connect()
    return () => {
      closed = true
      if (reconnectTimer) clearTimeout(reconnectTimer)
      socket?.close()
    }
  }, [])

  return { status, connected }
}
