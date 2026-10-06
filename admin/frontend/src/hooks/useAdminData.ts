import { useQuery } from "@tanstack/react-query"

import { AdminService } from "@/client"
import type {
  LogKind,
  LogsResponse,
  Machine,
  OverviewStats,
  ProviderDefinition,
  ProviderInstance,
  ProviderTypeDetail,
  ProviderTypeSummary,
  ResponsesList,
  UsageStats,
} from "@/types/admin"

export const machineKeys = { all: ["machines"] as const }
export const definitionKeys = { all: ["definitions"] as const }
export const instanceKeys = { all: ["instances"] as const }
export const responseKeys = (limit: number, offset: number) =>
  ["responses", limit, offset] as const
export const usageKeys = {
  all: ["stats", "usage"] as const,
  withLimit: (limit: number) => ["stats", "usage", limit] as const,
}
export const overviewKeys = { all: ["stats", "overview"] as const }
export const providerTypeKeys = {
  all: ["provider-types"] as const,
  detail: (name: string) => ["provider-types", name] as const,
}
export const logsKeys = {
  all: ["instance-logs"] as const,
  tail: (instanceId: string, kind: LogKind, since: number, limit: number) =>
    ["instance-logs", instanceId, kind, since, limit] as const,
}

// The generated SDK types the admin dict responses as loose JSON maps;
// the hand-written shapes in @/types/admin mirror the backend dicts.
function cast<T>(value: unknown): T {
  return value as unknown as T
}

export function useMachines(refetchInterval = 5000) {
  return useQuery({
    queryKey: machineKeys.all,
    queryFn: async () =>
      cast<Machine[]>((await AdminService.listMachines()).data),
    refetchInterval,
  })
}

export function useDefinitions(refetchInterval = 5000) {
  return useQuery({
    queryKey: definitionKeys.all,
    queryFn: async () =>
      cast<ProviderDefinition[]>((await AdminService.listDefinitions()).data),
    refetchInterval,
  })
}

export function useInstances(refetchInterval = 4000) {
  return useQuery({
    queryKey: instanceKeys.all,
    queryFn: async () =>
      cast<ProviderInstance[]>((await AdminService.listInstances()).data),
    refetchInterval,
  })
}

export function useResponses(limit = 50, offset = 0, refetchInterval = 5000) {
  return useQuery({
    queryKey: responseKeys(limit, offset),
    queryFn: async () =>
      cast<ResponsesList>(
        (
          await AdminService.listResponses({
            query: { limit, offset },
          })
        ).data,
      ),
    refetchInterval,
  })
}

export function useUsageStats(limit = 200, refetchInterval = 5000) {
  return useQuery({
    queryKey: usageKeys.withLimit(limit),
    queryFn: async () =>
      cast<UsageStats>(
        (await AdminService.usageStats({ query: { limit } })).data,
      ),
    refetchInterval,
  })
}

export function useOverview(refetchInterval = 4000) {
  return useQuery({
    queryKey: overviewKeys.all,
    queryFn: async () =>
      cast<OverviewStats>((await AdminService.overviewStats()).data),
    refetchInterval,
  })
}

// Phase 13: per-instance log tail (Redis-backed). The API returns
// entries NEWEST FIRST; callers reverse for chronological display.
// ``since`` means "seq > since"; poll with the last cursor for live
// tailing. ``live`` drives the ~2s refetchInterval (paused when false;
// react-query also pauses intervals when the tab is not focused).
export function useInstanceLogs(
  instanceId: string | null,
  opts: {
    kind: LogKind
    since?: number
    limit?: number
    live?: boolean
    enabled?: boolean
  },
) {
  const since = opts.since ?? 0
  const limit = opts.limit ?? 500
  return useQuery({
    queryKey: logsKeys.tail(instanceId ?? "", opts.kind, since, limit),
    queryFn: async () =>
      cast<LogsResponse>(
        (
          await AdminService.getInstanceLogs({
            path: { instance_id: instanceId as string },
            query: { kind: opts.kind, since, limit },
          })
        ).data,
      ),
    enabled: (opts.enabled ?? true) && instanceId !== null && instanceId !== "",
    refetchInterval: opts.live ? 2000 : false,
    // M2: the key carries `since` so each poll is a fresh key; sweep
    // superseded snapshots immediately instead of caching ~150 x 500
    // entries per 5-minute live session.
    gcTime: 1000,
  })
}

export function useProviderTypes(refetchInterval = 10000) {
  return useQuery({
    queryKey: providerTypeKeys.all,
    queryFn: async () =>
      cast<ProviderTypeSummary[]>(
        (await AdminService.listProviderTypes()).data,
      ),
    refetchInterval,
  })
}

export function useProviderType(name: string | null, refetchInterval = 10000) {
  return useQuery({
    queryKey: providerTypeKeys.detail(name ?? ""),
    queryFn: async () =>
      cast<ProviderTypeDetail>(
        (await AdminService.getProviderType({ path: { name: name as string } }))
          .data,
      ),
    enabled: name !== null && name !== "",
    refetchInterval,
  })
}
