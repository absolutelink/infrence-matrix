import { ChevronDown, X } from "lucide-react"
import { useCallback, useEffect, useRef } from "react"
import {
  MIN_HEIGHT_PX,
  maxHeightPx,
  useLogsDock,
} from "@/components/Common/LogsDockContext"
import { LogsPanel } from "@/components/Common/LogsPanel"
import { Button } from "@/components/ui/button"
import { cn } from "@/lib/utils"

function tabId(instanceId: string): string {
  return `logs-tab-${instanceId}`
}
function panelId(instanceId: string): string {
  return `logs-tabpanel-${instanceId}`
}

export function LogsDock() {
  const {
    tabs,
    activeTabId,
    visible,
    height,
    setHeight,
    closeTab,
    setActive,
    collapse,
  } = useLogsDock()

  const dragRef = useRef<{ startY: number; startHeight: number } | null>(null)
  // Holds the active window listener teardown so both pointercancel/
  // pointerup and component unmount mid-drag clean up identically.
  const cleanupDragRef = useRef<(() => void) | null>(null)

  useEffect(() => {
    return () => cleanupDragRef.current?.()
  }, [])

  const onPointerDown = useCallback(
    (e: React.PointerEvent<HTMLElement>) => {
      e.preventDefault()
      dragRef.current = { startY: e.clientY, startHeight: height }
      const max = maxHeightPx()
      const target = e.currentTarget
      const pointerId = e.pointerId
      target.setPointerCapture(pointerId)

      const onMove = (ev: PointerEvent) => {
        if (!dragRef.current) return
        // Dragging up (clientY decreases) grows the dock.
        const delta = dragRef.current.startY - ev.clientY
        const next = Math.max(
          MIN_HEIGHT_PX,
          Math.min(max, dragRef.current.startHeight + delta),
        )
        setHeight(next)
      }
      const cleanup = () => {
        dragRef.current = null
        cleanupDragRef.current = null
        try {
          target.releasePointerCapture(pointerId)
        } catch {
          // Capture may already be gone (e.g. pointercancel).
        }
        window.removeEventListener("pointermove", onMove)
        window.removeEventListener("pointerup", cleanup)
        window.removeEventListener("pointercancel", cleanup)
      }
      cleanupDragRef.current = cleanup
      window.addEventListener("pointermove", onMove)
      window.addEventListener("pointerup", cleanup)
      window.addEventListener("pointercancel", cleanup)
    },
    [height, setHeight],
  )

  if (!visible) return null

  return (
    <section
      className="flex shrink-0 flex-col border-t bg-background"
      style={{ height }}
      aria-label="Logs panel"
    >
      {/* Drag-resize handle */}
      <button
        type="button"
        aria-label={`Resize logs panel, currently ${height} pixels tall`}
        onKeyDown={(e) => {
          if (e.key === "ArrowUp") {
            e.preventDefault()
            setHeight(height + 24)
          } else if (e.key === "ArrowDown") {
            e.preventDefault()
            setHeight(height - 24)
          }
        }}
        onPointerDown={onPointerDown}
        className="group flex h-2 shrink-0 cursor-row-resize items-center justify-center border-b border-transparent bg-background outline-none hover:bg-muted/60 focus-visible:bg-muted/60"
        title="Drag to resize"
      >
        <div className="h-1 w-10 rounded-full bg-muted-foreground/40 group-hover:bg-muted-foreground/70" />
      </button>

      {/* Tab strip */}
      <div
        role="tablist"
        aria-label="Open instance logs"
        className="flex shrink-0 items-stretch gap-1 overflow-x-auto border-b bg-muted/30 px-2"
      >
        {tabs.map((tab) => {
          const isActive = tab.instanceId === activeTabId
          return (
            <div
              key={tab.instanceId}
              className={cn(
                "flex items-center gap-1 border-b-2 pl-1 text-sm",
                isActive
                  ? "border-primary bg-background text-foreground"
                  : "border-transparent text-muted-foreground hover:text-foreground",
              )}
            >
              <button
                type="button"
                role="tab"
                id={tabId(tab.instanceId)}
                aria-selected={isActive}
                aria-controls={panelId(tab.instanceId)}
                onClick={() => setActive(tab.instanceId)}
                className="max-w-48 truncate px-2 py-1.5 font-mono text-xs"
                title={tab.label}
              >
                {tab.label}
              </button>
              <button
                type="button"
                aria-label={`Close logs for ${tab.label}`}
                onClick={() => closeTab(tab.instanceId)}
                className="rounded-sm p-1 text-muted-foreground hover:bg-muted hover:text-foreground"
              >
                <X className="size-3.5" />
              </button>
            </div>
          )
        })}
        <div className="ml-auto flex items-center">
          <Button
            variant="ghost"
            size="icon"
            className="size-8"
            onClick={collapse}
            title="Hide logs panel"
          >
            <ChevronDown />
          </Button>
        </div>
      </div>

      {/* Panels: every open tab stays mounted so its scrollback and cursor
          survive a tab switch; only the active one polls (see LogsPanel). */}
      <div className="relative min-h-0 flex-1">
        {tabs.map((tab) => (
          <div
            key={tab.instanceId}
            role="tabpanel"
            id={panelId(tab.instanceId)}
            aria-labelledby={tabId(tab.instanceId)}
            hidden={tab.instanceId !== activeTabId}
            className={cn(
              "absolute inset-0 flex-col",
              tab.instanceId === activeTabId ? "flex" : "hidden",
            )}
          >
            <LogsPanel
              instance={tab.instance}
              active={tab.instanceId === activeTabId}
            />
          </div>
        ))}
      </div>
    </section>
  )
}
