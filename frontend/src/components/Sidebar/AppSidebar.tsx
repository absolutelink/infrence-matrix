import { Box, Cpu, Home, MessageSquare, Server } from "lucide-react"

import { SidebarAppearance } from "@/components/Common/Appearance"
import { Logo } from "@/components/Common/Logo"
import {
  Sidebar,
  SidebarContent,
  SidebarFooter,
  SidebarHeader,
} from "@/components/ui/sidebar"
import { type Item, Main } from "./Main"

const baseItems: Item[] = [
  { icon: Home, title: "Dashboard", path: "/" },
  { icon: Server, title: "Models", path: "/models" },
  { icon: Cpu, title: "Agents", path: "/agents" },
  { icon: Box, title: "Server Instances", path: "/server-instances" },
  { icon: MessageSquare, title: "Chat", path: "/chat" },
]

export function AppSidebar() {
  return (
    <Sidebar collapsible="icon">
      <SidebarHeader className="px-4 py-6 group-data-[collapsible=icon]:px-0 group-data-[collapsible=icon]:items-center">
        <Logo variant="responsive" />
      </SidebarHeader>
      <SidebarContent>
        <Main items={baseItems} />
      </SidebarContent>
      <SidebarFooter>
        <SidebarAppearance />
      </SidebarFooter>
    </Sidebar>
  )
}

export default AppSidebar
