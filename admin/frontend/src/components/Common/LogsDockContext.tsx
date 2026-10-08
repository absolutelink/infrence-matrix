import {
  createContext,
  type ReactNode,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
} from "react"

import type { ProviderInstance } from "@/types/admin"

// One open logs tab. The full instance snapshot rides along so the panel
// can render status badges; the dock re-derives live status from the
// shared instances query inside LogsPanel, so a stale snapshot here only
// affects the tab label.
export interface LogsDockTab {
  instanceId: string
  label: string
  instance: ProviderInstance
}

// Dock sizing bounds. Default is ~40vh (resolved to px on first render);
// drag resize clamps between MIN_HEIGHT_PX and 70% of the viewport.
export const MIN_HEIGHT_PX = 200
const DEFAULT_VH = 0.4
const MAX_VH = 0.7

function defaultHeight(): number {
  const vh = typeof window !== "undefined" ? window.innerHeight : 800
  return Math.round(vh * DEFAULT_VH)
}

export function maxHeightPx(): number {
  const vh = typeof window !== "undefined" ? window.innerHeight : 800
  return Math.round(vh * MAX_VH)
}

interface LogsDockContextValue {
  tabs: LogsDockTab[]
  activeTabId: string | null
  // Dock is shown only when there is at least one tab AND the operator has
  // not collapsed it via the chevron.
  visible: boolean
  collapsed: boolean
  height: number
  setHeight: (px: number) => void
  openTab: (instance: ProviderInstance) => void
  closeTab: (instanceId: string) => void
  setActive: (instanceId: string) => void
  collapse: () => void
  expand: () => void
}

const LogsDockContext = createContext<LogsDockContextValue | null>(null)

function instanceLabel(instance: ProviderInstance): string {
  return instance.alias ?? instance.id.slice(0, 8)
}

export function LogsDockProvider({ children }: { children: ReactNode }) {
  const [tabs, setTabs] = useState<LogsDockTab[]>([])
  const [activeTabId, setActiveTabId] = useState<string | null>(null)
  const [collapsed, setCollapsed] = useState(false)
  const [height, setHeightState] = useState<number>(() => defaultHeight())

  const setHeight = useCallback((px: number) => {
    const max = maxHeightPx()
    setHeightState(Math.max(MIN_HEIGHT_PX, Math.min(max, Math.round(px))))
  }, [])

  // Re-clamp the dock height when the viewport shrinks so it never exceeds
  // 70vh (or dips under the 200px floor) after a resize.
  useEffect(() => {
    const onResize = () => {
      const max = maxHeightPx()
      setHeightState((prev) => Math.max(MIN_HEIGHT_PX, Math.min(max, prev)))
    }
    window.addEventListener("resize", onResize)
    return () => window.removeEventListener("resize", onResize)
  }, [])

  const openTab = useCallback((instance: ProviderInstance) => {
    setCollapsed(false)
    setTabs((prev) => {
      if (prev.some((t) => t.instanceId === instance.id)) {
        // Refresh the snapshot for an already-open tab without reordering.
        return prev.map((t) =>
          t.instanceId === instance.id
            ? { ...t, instance, label: instanceLabel(instance) }
            : t,
        )
      }
      return [
        ...prev,
        {
          instanceId: instance.id,
          label: instanceLabel(instance),
          instance,
        },
      ]
    })
    setActiveTabId(instance.id)
  }, [])

  const closeTab = useCallback(
    (instanceId: string) => {
      const idx = tabs.findIndex((t) => t.instanceId === instanceId)
      if (idx === -1) return
      const next = tabs.filter((t) => t.instanceId !== instanceId)
      setTabs(next)
      // Compute the successor from current state (not inside a setTabs
      // updater) so the reducer stays pure under StrictMode double-invoke.
      if (activeTabId === instanceId) {
        setActiveTabId(
          next.length === 0
            ? null
            : next[Math.min(idx, next.length - 1)].instanceId,
        )
      }
    },
    [tabs, activeTabId],
  )

  const setActive = useCallback((instanceId: string) => {
    setActiveTabId(instanceId)
    setCollapsed(false)
  }, [])

  const collapse = useCallback(() => setCollapsed(true), [])
  const expand = useCallback(() => setCollapsed(false), [])

  const value = useMemo<LogsDockContextValue>(
    () => ({
      tabs,
      activeTabId,
      visible: tabs.length > 0 && !collapsed,
      collapsed,
      height,
      setHeight,
      openTab,
      closeTab,
      setActive,
      collapse,
      expand,
    }),
    [
      tabs,
      activeTabId,
      collapsed,
      height,
      setHeight,
      openTab,
      closeTab,
      setActive,
      collapse,
      expand,
    ],
  )

  return (
    <LogsDockContext.Provider value={value}>
      {children}
    </LogsDockContext.Provider>
  )
}

export function useLogsDock(): LogsDockContextValue {
  const ctx = useContext(LogsDockContext)
  if (!ctx) {
    throw new Error("useLogsDock must be used within a LogsDockProvider")
  }
  return ctx
}
