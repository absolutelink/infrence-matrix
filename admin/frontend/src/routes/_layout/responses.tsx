import { createFileRoute } from "@tanstack/react-router"
import { useState } from "react"

import { StatusBadge } from "@/components/Common/StatusBadge"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { useResponses, useUsageStats } from "@/hooks/useAdminData"
import type { UsageSample } from "@/types/admin"

export const Route = createFileRoute("/_layout/responses")({
  component: ResponsesPage,
  head: () => ({ meta: [{ title: "Responses - Inference Matrix" }] }),
})

const PAGE_SIZE = 25

function ResponsesPage() {
  const [page, setPage] = useState(0)
  const { data: list, isLoading } = useResponses(PAGE_SIZE, page * PAGE_SIZE)
  const { data: usage } = useUsageStats(200)

  const responses = list?.responses ?? []
  const total = list?.total ?? 0
  const pages = Math.max(1, Math.ceil(total / PAGE_SIZE))

  return (
    <div className="flex flex-col gap-6">
      <div>
        <h1 className="text-2xl font-bold tracking-tight">Responses / Usage</h1>
        <p className="text-muted-foreground">
          Stored inference turns (responses + chat) and per-request token
          telemetry.
        </p>
      </div>

      <Tabs defaultValue="responses">
        <TabsList>
          <TabsTrigger value="responses">Responses</TabsTrigger>
          <TabsTrigger value="usage">Token usage</TabsTrigger>
        </TabsList>

        <TabsContent value="responses" className="mt-4 space-y-4">
          {isLoading ? (
            <p className="text-muted-foreground">Loading…</p>
          ) : responses.length === 0 ? (
            <p className="text-muted-foreground py-8 text-center">
              No stored responses yet — run something through /v1 or the
              Playground.
            </p>
          ) : (
            <>
              <div className="rounded-md border">
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead>Response ID</TableHead>
                      <TableHead>Model</TableHead>
                      <TableHead>Format</TableHead>
                      <TableHead>Status</TableHead>
                      <TableHead>In</TableHead>
                      <TableHead>Out</TableHead>
                      <TableHead>Total</TableHead>
                      <TableHead>Prev</TableHead>
                      <TableHead>Created</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {responses.map((r) => (
                      <TableRow key={r.id}>
                        <TableCell>
                          <code className="font-mono text-xs">
                            {r.response_id.slice(0, 18)}…
                          </code>
                        </TableCell>
                        <TableCell className="font-medium">
                          {r.model_alias}
                        </TableCell>
                        <TableCell>
                          <Badge
                            variant="outline"
                            className="font-mono text-xs"
                          >
                            {r.api_format === "chat_completions"
                              ? "chat"
                              : "responses"}
                          </Badge>
                        </TableCell>
                        <TableCell>
                          <StatusBadge status={r.status} />
                          {r.error_code && (
                            <span className="ml-1 text-xs text-destructive">
                              {r.error_code}
                            </span>
                          )}
                        </TableCell>
                        <TableCell className="font-mono text-xs">
                          {r.input_tokens}
                        </TableCell>
                        <TableCell className="font-mono text-xs">
                          {r.output_tokens}
                        </TableCell>
                        <TableCell className="font-mono text-xs">
                          {r.total_tokens}
                        </TableCell>
                        <TableCell>
                          {r.previous_response_id ? (
                            <code className="font-mono text-xs text-muted-foreground">
                              {r.previous_response_id.slice(0, 14)}…
                            </code>
                          ) : (
                            <span className="text-xs text-muted-foreground">
                              —
                            </span>
                          )}
                        </TableCell>
                        <TableCell className="text-xs text-muted-foreground">
                          {new Date(r.created_at).toLocaleString()}
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </div>
              <div className="flex items-center justify-between text-sm text-muted-foreground">
                <span>
                  Page {page + 1} of {pages} ({total} total)
                </span>
                <div className="flex gap-2">
                  <Button
                    variant="outline"
                    size="sm"
                    disabled={page === 0}
                    onClick={() => setPage((p) => Math.max(0, p - 1))}
                  >
                    Previous
                  </Button>
                  <Button
                    variant="outline"
                    size="sm"
                    disabled={page + 1 >= pages}
                    onClick={() => setPage((p) => p + 1)}
                  >
                    Next
                  </Button>
                </div>
              </div>
            </>
          )}
        </TabsContent>

        <TabsContent value="usage" className="mt-4 space-y-4">
          <UsageSummary samples={usage?.samples ?? []} totals={usage?.totals} />
        </TabsContent>
      </Tabs>
    </div>
  )
}

function UsageSummary({
  samples,
  totals,
}: {
  samples: UsageSample[]
  totals?: {
    prompt_tokens: number
    cached_tokens: number
    completion_tokens: number
  }
}) {
  const recent = samples.slice(0, 40)
  const maxTokens = Math.max(
    1,
    ...recent.map((s) => s.prompt_tokens + s.completion_tokens),
  )
  const withRates = recent.filter((s) => s.predicted_per_second > 0)
  const avgGenRate =
    withRates.length > 0
      ? withRates.reduce((a, s) => a + s.predicted_per_second, 0) /
        withRates.length
      : 0
  const avgPromptRate =
    withRates.length > 0
      ? withRates.reduce((a, s) => a + s.prompt_per_second, 0) /
        withRates.length
      : 0

  return (
    <div className="flex flex-col gap-4">
      <div className="grid gap-4 sm:grid-cols-3">
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm text-muted-foreground">
              Prompt tokens (all time)
            </CardTitle>
          </CardHeader>
          <CardContent className="text-2xl font-bold">
            {totals?.prompt_tokens ?? 0}
            <span className="ml-2 text-xs font-normal text-muted-foreground">
              cached {totals?.cached_tokens ?? 0}
            </span>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm text-muted-foreground">
              Completion tokens
            </CardTitle>
          </CardHeader>
          <CardContent className="text-2xl font-bold">
            {totals?.completion_tokens ?? 0}
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm text-muted-foreground">
              Avg speed (recent)
            </CardTitle>
          </CardHeader>
          <CardContent className="text-2xl font-bold">
            {avgGenRate.toFixed(1)} t/s
            <span className="ml-2 text-xs font-normal text-muted-foreground">
              prompt {avgPromptRate.toFixed(0)} t/s
            </span>
          </CardContent>
        </Card>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>
            Last {recent.length} requests — prompt vs completion
          </CardTitle>
        </CardHeader>
        <CardContent>
          {recent.length === 0 ? (
            <p className="text-sm text-muted-foreground">
              No usage samples recorded yet.
            </p>
          ) : (
            <div className="space-y-2">
              {recent.map((s) => {
                const pW = (s.prompt_tokens / maxTokens) * 100
                const cW = (s.completion_tokens / maxTokens) * 100
                const cacheW =
                  s.prompt_tokens > 0
                    ? (s.cached_tokens / Math.max(1, s.prompt_tokens)) * pW
                    : 0
                return (
                  <div key={s.id} className="flex items-center gap-2">
                    <span className="w-36 shrink-0 truncate text-xs text-muted-foreground">
                      {new Date(s.created_at).toLocaleTimeString()}
                    </span>
                    <div className="flex h-4 flex-1 overflow-hidden rounded-sm bg-muted">
                      <div
                        className="bg-sky-500/70 dark:bg-sky-400/50"
                        style={{ width: `${pW}%` }}
                        title={`prompt ${s.prompt_tokens} (cached ${s.cached_tokens})`}
                      >
                        <div
                          className="h-full bg-emerald-500/70 dark:bg-emerald-400/60"
                          style={{ width: `${cacheW}%` }}
                          title={`cached ${s.cached_tokens}`}
                        />
                      </div>
                      <div
                        className="bg-violet-500/70 dark:bg-violet-400/50"
                        style={{ width: `${cW}%` }}
                        title={`completion ${s.completion_tokens}`}
                      />
                    </div>
                    <span className="w-24 shrink-0 text-right font-mono text-xs text-muted-foreground">
                      {s.prompt_tokens}+{s.completion_tokens}
                    </span>
                  </div>
                )
              })}
              <div className="flex gap-4 pt-2 text-xs text-muted-foreground">
                <span className="inline-flex items-center gap-1">
                  <span className="size-2.5 rounded-sm bg-sky-500/70" />
                  prompt
                </span>
                <span className="inline-flex items-center gap-1">
                  <span className="size-2.5 rounded-sm bg-emerald-500/70" />
                  cached
                </span>
                <span className="inline-flex items-center gap-1">
                  <span className="size-2.5 rounded-sm bg-violet-500/70" />
                  completion
                </span>
              </div>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  )
}
