import { createContext, useContext, useMemo, useState } from "react"

export interface LogInstance {
  id: string
  kind?: "server" | "benchmark"
  run_id?: string | null
  model_name?: string | null
  agent_id: string
  agent_name?: string | null
  agent_host?: string | null
  agent_port?: number | null
}

interface LogPanelContextValue {
  tabs: LogInstance[]
  activeTabId: string | null
  isOpen: boolean
  height: number
  openLogs: (instance: LogInstance) => void
  closeLogs: (instanceId: string) => void
  setActiveTab: (instanceId: string) => void
  closePanel: () => void
  setHeight: (height: number) => void
}

const LogPanelContext = createContext<LogPanelContextValue | null>(null)

export function LogPanelProvider({ children }: { children: React.ReactNode }) {
  const [tabs, setTabs] = useState<LogInstance[]>([])
  const [activeTabId, setActiveTabId] = useState<string | null>(null)
  const [isOpen, setIsOpen] = useState(false)
  const [height, setHeight] = useState(320)

  const value = useMemo<LogPanelContextValue>(
    () => ({
      tabs,
      activeTabId,
      isOpen,
      height,
      openLogs: (instance) => {
        setTabs((current) =>
          current.some((tab) => tab.id === instance.id)
            ? current
            : [...current, instance],
        )
        setActiveTabId(instance.id)
        setIsOpen(true)
      },
      closeLogs: (instanceId) => {
        setTabs((current) => {
          const next = current.filter((tab) => tab.id !== instanceId)
          if (activeTabId === instanceId) {
            setActiveTabId(next[next.length - 1]?.id ?? null)
            if (next.length === 0) setIsOpen(false)
          }
          return next
        })
      },
      setActiveTab: (instanceId) => {
        setActiveTabId(instanceId)
        setIsOpen(true)
      },
      closePanel: () => setIsOpen(false),
      setHeight,
    }),
    [activeTabId, height, isOpen, tabs],
  )

  return (
    <LogPanelContext.Provider value={value}>
      {children}
    </LogPanelContext.Provider>
  )
}

export function useLogPanel() {
  const context = useContext(LogPanelContext)
  if (!context) {
    throw new Error("useLogPanel must be used inside LogPanelProvider")
  }
  return context
}
