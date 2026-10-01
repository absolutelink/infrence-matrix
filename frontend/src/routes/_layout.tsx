import { createFileRoute, Outlet } from "@tanstack/react-router"

import { Footer } from "@/components/Common/Footer"
import { QueueStatusBar } from "@/components/Queue/QueueStatusBar"
import { BottomLogPanel } from "@/components/ServerInstances/BottomLogPanel"
import {
  LogPanelProvider,
  useLogPanel,
} from "@/components/ServerInstances/LogPanelContext"
import AppSidebar from "@/components/Sidebar/AppSidebar"
import {
  SidebarInset,
  SidebarProvider,
  SidebarTrigger,
} from "@/components/ui/sidebar"

export const Route = createFileRoute("/_layout")({
  component: Layout,
})

function Layout() {
  return (
    <LogPanelProvider>
      <LayoutContent />
    </LogPanelProvider>
  )
}

function LayoutContent() {
  const { height, isOpen } = useLogPanel()
  return (
    <SidebarProvider>
      <AppSidebar />
      <SidebarInset
        style={
          {
            "--log-panel-h": isOpen ? `${height}px` : "0px",
          } as React.CSSProperties
        }
      >
        <header className="sticky top-0 z-10 flex h-16 shrink-0 items-center gap-2 border-b bg-background px-4">
          <SidebarTrigger className="-ml-1 text-muted-foreground" />
          <QueueStatusBar />
        </header>
        <main className="min-h-0 flex-1 overflow-y-auto p-6 pb-[calc(1.5rem+var(--log-panel-h))] md:p-8 md:pb-[calc(2rem+var(--log-panel-h))]">
          <div className="mx-auto h-full max-w-7xl">
            <Outlet />
          </div>
        </main>
        <Footer />
      </SidebarInset>
      <BottomLogPanel />
    </SidebarProvider>
  )
}
