import { useCallback, useEffect, useState } from "react"

import {
  type AgentEvent,
  acquireStream,
  type StreamStatus,
} from "@/hook/agentEventStream"

export type { AgentEvent, StreamStatus }

/**
 * Subscribe to an agent's live event stream via the backend WebSocket
 * (mounted at /api/ws/events/{agent_id}).
 *
 * The underlying socket is shared per agent across all consumers and
 * reconnects with exponential backoff. `status` distinguishes the initial
 * connect from a reconnect so UIs can show a "reconnecting" state.
 */
export function useAgentEvents(agentId: string, enabled: boolean) {
  const [events, setEvents] = useState<AgentEvent[]>([])
  const [status, setStatus] = useState<StreamStatus>("connecting")
  const [attempt, setAttempt] = useState(0)

  const handleBatch = useCallback((batch: AgentEvent[]) => {
    if (batch.length === 0) return
    setEvents((prev) => [...prev, ...batch].slice(-500))
  }, [])

  const handleStatus = useCallback(
    (nextStatus: StreamStatus, nextAttempt: number) => {
      setStatus(nextStatus)
      setAttempt(nextAttempt)
    },
    [],
  )

  useEffect(() => {
    if (!enabled || !agentId) return

    const { stream, release } = acquireStream(agentId)
    // Seed from the shared buffer so a newly mounted consumer (or one
    // remounting after a disconnect) keeps the visible history.
    setEvents(stream.recentEvents())
    stream.addListener(handleBatch)
    stream.addStatusListener(handleStatus)

    return () => {
      stream.removeListener(handleBatch)
      stream.removeStatusListener(handleStatus)
      release()
    }
  }, [agentId, enabled, handleBatch, handleStatus])

  return {
    events,
    connected: status === "connected",
    status,
    attempt,
  }
}
