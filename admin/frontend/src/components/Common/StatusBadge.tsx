import { Badge } from "@/components/ui/badge"
import { cn } from "@/lib/utils"

const INSTANCE_COLORS: Record<string, string> = {
  running:
    "bg-emerald-500/15 text-emerald-600 dark:text-emerald-400 border-emerald-500/30",
  awaiting_config:
    "bg-violet-500/15 text-violet-600 dark:text-violet-400 border-violet-500/30",
  registering: "bg-sky-500/15 text-sky-600 dark:text-sky-400 border-sky-500/30",
  initializing:
    "bg-amber-500/15 text-amber-600 dark:text-amber-400 border-amber-500/30",
  starting:
    "bg-amber-500/15 text-amber-600 dark:text-amber-400 border-amber-500/30",
  in_use: "bg-teal-500/15 text-teal-600 dark:text-teal-400 border-teal-500/30",
  stopping:
    "bg-orange-500/15 text-orange-600 dark:text-orange-400 border-orange-500/30",
  unhealthy: "bg-red-500/15 text-red-600 dark:text-red-400 border-red-500/30",
  error: "bg-red-500/15 text-red-600 dark:text-red-400 border-red-500/30",
  failed: "bg-red-500/15 text-red-600 dark:text-red-400 border-red-500/30",
  disconnected:
    "bg-zinc-500/15 text-zinc-500 dark:text-zinc-400 border-zinc-500/30",
  stopped: "bg-gray-500/10 text-gray-500 dark:text-gray-400 border-gray-500/20",
  completed:
    "bg-emerald-500/15 text-emerald-600 dark:text-emerald-400 border-emerald-500/30",
  in_progress:
    "bg-amber-500/15 text-amber-600 dark:text-amber-400 border-amber-500/30",
  incomplete:
    "bg-orange-500/15 text-orange-600 dark:text-orange-400 border-orange-500/30",
}

export function StatusBadge({
  status,
  className,
}: {
  status: string | null | undefined
  className?: string
}) {
  const value = status ?? "unknown"
  return (
    <Badge
      variant="outline"
      className={cn(
        "font-mono",
        INSTANCE_COLORS[value] ??
          "bg-muted text-muted-foreground border-border",
        className,
      )}
    >
      {value}
    </Badge>
  )
}

export function ConnectionBadge({ connected }: { connected: boolean }) {
  return (
    <span className="inline-flex items-center gap-1.5 text-xs text-muted-foreground">
      <span
        className={cn(
          "size-2 rounded-full",
          connected ? "bg-emerald-500" : "bg-zinc-400 dark:bg-zinc-600",
        )}
      />
      {connected ? "ws connected" : "ws down"}
    </span>
  )
}
