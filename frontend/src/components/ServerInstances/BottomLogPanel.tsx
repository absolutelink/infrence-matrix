import { ChevronDown, GripHorizontal, X } from "lucide-react"
import { useState } from "react"

import { useLogPanel } from "@/components/ServerInstances/LogPanelContext"
import { ServerLogsSheet } from "@/components/ServerInstances/ServerLogsSheet"
import { Button } from "@/components/ui/button"
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs"

export function BottomLogPanel() {
  const { tabs, activeTabId, isOpen, closeLogs, closePanel, setActiveTab } =
    useLogPanel()
  const [height, setHeight] = useState(320)

  if (!isOpen || !activeTabId || tabs.length === 0) return null

  const activeTab = tabs.find((tab) => tab.id === activeTabId) ?? tabs[0]

  const resize = (event: React.PointerEvent<HTMLButtonElement>) => {
    const startY = event.clientY
    const startHeight = height
    event.currentTarget.setPointerCapture(event.pointerId)
    const move = (moveEvent: PointerEvent) => {
      setHeight(
        Math.min(720, Math.max(180, startHeight + startY - moveEvent.clientY)),
      )
    }
    const stop = () => {
      window.removeEventListener("pointermove", move)
      window.removeEventListener("pointerup", stop)
    }
    window.addEventListener("pointermove", move)
    window.addEventListener("pointerup", stop)
  }

  return (
    <section
      className="fixed inset-x-0 bottom-0 z-40 flex flex-col border-t bg-background shadow-2xl"
      style={{ height }}
      aria-label="Logs"
    >
      <button
        type="button"
        aria-label="Resize logs panel"
        className="flex h-2 shrink-0 cursor-ns-resize items-center justify-center bg-muted/60 hover:bg-primary/20"
        onPointerDown={resize}
      >
        <GripHorizontal className="h-3 w-8 text-muted-foreground" />
      </button>
      <div className="flex min-h-12 items-center gap-2 border-b px-3">
        <Tabs
          value={activeTab.id}
          onValueChange={setActiveTab}
          className="min-w-0 flex-1"
        >
          <TabsList className="h-10 max-w-full justify-start overflow-x-auto rounded-none bg-transparent p-0">
            {tabs.map((tab) => (
              <div key={tab.id} className="flex h-10 items-center">
                <TabsTrigger
                  value={tab.id}
                  className="h-10 max-w-56 shrink-0 rounded-none border-b-2 border-transparent px-3 data-[state=active]:border-primary data-[state=active]:bg-transparent"
                >
                  <span className="truncate">{tab.model_name || tab.id}</span>
                </TabsTrigger>
                <button
                  type="button"
                  aria-label={`Close logs for ${tab.model_name || tab.id}`}
                  className="mr-1 rounded p-1 hover:bg-muted"
                  onClick={() => closeLogs(tab.id)}
                >
                  <X className="h-3 w-3" />
                </button>
              </div>
            ))}
          </TabsList>
        </Tabs>
        <Button
          variant="ghost"
          size="icon"
          className="shrink-0"
          onClick={closePanel}
        >
          <ChevronDown className="h-4 w-4" />
          <span className="sr-only">Collapse logs</span>
        </Button>
      </div>
      <div className="min-h-0 flex-1">
        <ServerLogsSheet key={activeTab.id} isOpen instance={activeTab} />
      </div>
    </section>
  )
}
