import { createFileRoute, Outlet } from "@tanstack/react-router"

import StatsBar from "@/components/Common/StatsBar"
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
    <SidebarProvider>
      <AppSidebar />
      <SidebarInset>
        <header className="sticky top-0 z-10 flex h-16 shrink-0 items-center gap-2 overflow-x-auto border-b bg-background px-4">
          <SidebarTrigger className="-ml-1 shrink-0 text-muted-foreground" />
          <StatsBar />
        </header>
        <main className="min-h-0 flex-1 overflow-y-auto p-6 md:p-8">
          <div className="mx-auto h-full max-w-7xl">
            <Outlet />
          </div>
        </main>
      </SidebarInset>
    </SidebarProvider>
  )
}
