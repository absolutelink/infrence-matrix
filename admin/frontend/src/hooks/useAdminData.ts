import { useQuery } from "@tanstack/react-query"

import { AdminService } from "@/client"
import type {
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
