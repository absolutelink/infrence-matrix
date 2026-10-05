import { Link } from "@tanstack/react-router"
import { ArrowRight, Server } from "lucide-react"

import { Button } from "@/components/ui/button"

export function EmptyNudge({
  text,
  actionLabel,
  to,
  onAction,
}: {
  text: string
  actionLabel: string
  to?: string
  onAction?: () => void
}) {
  return (
    <div className="flex flex-col items-center justify-center py-10 text-center">
      <div className="mb-4 rounded-full bg-muted p-4">
        <Server className="h-8 w-8 text-muted-foreground" />
      </div>
      <h3 className="text-lg font-semibold">{text}</h3>
      <p className="mb-3 text-muted-foreground">
        Get started by creating your first machine, then a provider definition
        for it.
      </p>
      {onAction ? (
        <Button variant="link" className="text-sm" onClick={onAction}>
          {actionLabel} <ArrowRight className="size-3" />
        </Button>
      ) : (
        <Link
          to={to ?? "/"}
          className="inline-flex items-center gap-1 text-sm text-primary hover:underline"
        >
          {actionLabel} <ArrowRight className="size-3" />
        </Link>
      )}
    </div>
  )
}
