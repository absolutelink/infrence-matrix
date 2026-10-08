import {
  Boxes,
  Cpu,
  LayoutDashboard,
  MessagesSquare,
  Network,
  Package,
  Play,
  Server,
  Settings,
} from "lucide-react"

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
  { icon: LayoutDashboard, title: "Dashboard", path: "/" },
  { icon: Server, title: "Machines", path: "/machines" },
  { icon: Network, title: "Agents", path: "/agents" },
  { icon: Package, title: "Definitions", path: "/definitions" },
  { icon: Boxes, title: "Provider Types", path: "/provider-types" },
  { icon: Cpu, title: "Instances", path: "/instances" },
  { icon: MessagesSquare, title: "Responses / Usage", path: "/responses" },
  { icon: Play, title: "Playground", path: "/playground" },
  { icon: Settings, title: "Settings", path: "/settings" },
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
