import { useQuery } from "@tanstack/react-query"

import { StatsService } from "@/client"
import type { TokenStatsResponse } from "@/client/types.gen"

export function useTokenStats(refetchInterval = 5000) {
  return useQuery({
    queryKey: ["token-stats"],
    queryFn: async () => (await StatsService.getTokenStats()).data,
    refetchInterval,
    staleTime: 1000,
  })
}

export type { TokenStatsResponse }
