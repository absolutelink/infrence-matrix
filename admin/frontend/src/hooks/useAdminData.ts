import { useQuery } from "@tanstack/react-query"

import { AdminService } from "@/client"
import type {
  FleetMetrics,
  InstanceStats,
  LogKind,
  LogsResponse,
  Machine,
  MachineMetrics,
  OverviewStats,
  ProviderAgent,
  ProviderDefinition,
  ProviderInstance,
  ProviderTypeDetail,
  ProviderTypeSummary,
  ResponsesList,
  SchedulerStats,
  UsageStats,
} from "@/types/admin"

export const machineKeys = {
  all: ["machines"] as const,
  metrics: (machineId: string) => ["machine-metrics", machineId] as const,
}
export const definitionKeys = { all: ["definitions"] as const }
export const instanceKeys = { all: ["instances"] as const }
export const agentKeys = { all: ["agents"] as const }
export const responseKeys = (limit: number, offset: number) =>
  ["responses", limit, offset] as const
export const usageKeys = {
  all: ["stats", "usage"] as const,
  withLimit: (limit: number) => ["stats", "usage", limit] as const,
}
export const overviewKeys = { all: ["stats", "overview"] as const }
// Phase 19: live stats bar — scheduler queue snapshot + fleet VRAM/GPU rollup.
export const schedulerStatsKeys = { all: ["stats", "scheduler"] as const }
export const fleetMetricsKeys = { all: ["stats", "metrics"] as const }
export const instanceStatsKeys = { all: ["stats", "instances"] as const }
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

// Phase 17: merged live machine metrics (per-GPU union across every agent
// partial + the owner-gated machine-wide snapshot). Callers gate the query on
// row expansion via ``enabled`` so the panel only polls while visible.
export function useMachineMetrics(
  machineId: string | undefined,
  opts: { enabled?: boolean; refetchInterval?: number } = {},
) {
  return useQuery({
    queryKey: machineKeys.metrics(machineId ?? ""),
    queryFn: async () =>
      cast<MachineMetrics>(
        (
          await AdminService.getMachineMetrics({
            path: { machine_id: machineId as string },
          })
        ).data,
      ),
    enabled: (opts.enabled ?? true) && machineId != null,
    refetchInterval: opts.refetchInterval ?? 5000,
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

// Phase 16: provider agents (hardware-local containers). Backends nest under
// each agent in the API payload; the Agents page reads them from here.
// ``enabled`` lets callers (e.g. the Definitions dialog) mount the query only
// while it is actually needed instead of polling unconditionally.
export function useAgents(refetchInterval = 4000, enabled = true) {
  return useQuery({
    queryKey: agentKeys.all,
    queryFn: async () =>
      cast<ProviderAgent[]>((await AdminService.listAgents()).data),
    refetchInterval,
    enabled,
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

// Slice 2: per-instance token/throughput rollups (live tps + 24h/7d/30d
// windows) keyed by provider instance id, for the Instances throughput
// column + stats block.
export function useInstanceStats(refetchInterval = 10000) {
  return useQuery({
    queryKey: instanceStatsKeys.all,
    queryFn: async () =>
      cast<Record<string, InstanceStats>>(
        (await AdminService.instanceStats()).data,
      ),
    refetchInterval,
  })
}

// Phase 19: authoritative in-process scheduler queue snapshot (per-definition
// queued/active + totals) for the live stats bar. Polls faster than the
// overview mirror because it drives the operator queue-clear affordance.
export function useSchedulerStats(refetchInterval = 4000) {
  return useQuery({
    queryKey: schedulerStatsKeys.all,
    queryFn: async () =>
      cast<SchedulerStats>((await AdminService.schedulerStats()).data),
    refetchInterval,
  })
}

// Phase 19: fleet VRAM (summed) + GPU utilization (unweighted mean of
// per-machine means) rollup for the live stats bar.
export function useFleetMetrics(refetchInterval = 5000) {
  return useQuery({
    queryKey: fleetMetricsKeys.all,
    queryFn: async () =>
      cast<FleetMetrics>((await AdminService.fleetMetrics()).data),
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
