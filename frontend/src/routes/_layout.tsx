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
      <SidebarInset>
        <header className="sticky top-0 z-10 flex h-16 shrink-0 items-center gap-2 border-b bg-background px-4">
          <SidebarTrigger className="-ml-1 text-muted-foreground" />
          <QueueStatusBar />
        </header>
        <main
          className="flex-1 p-6 md:p-8"
          style={{ paddingBottom: isOpen ? height + 24 : undefined }}
        >
          <div className="mx-auto max-w-7xl">
            <Outlet />
          </div>
        </main>
        <Footer />
      </SidebarInset>
      <BottomLogPanel />
    </SidebarProvider>
  )
}
