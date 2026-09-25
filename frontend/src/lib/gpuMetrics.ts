export type AgentGpuSnapshot = {
  id?: number | string
  name?: string
  vram_total?: number
  vram_used?: number
  utilization?: number
}

type AgentWithGpu = {
  id?: string
  host?: string | null
  gpu_info?: AgentGpuSnapshot | null
}

/**
 * Multiple agent processes can expose the same physical GPU. Keep one
 * telemetry sample per host/GPU instead of summing those duplicate reports.
 */
export function uniqueGpuSnapshots(agents: AgentWithGpu[]): AgentGpuSnapshot[] {
  const seen = new Set<string>()
  const snapshots: AgentGpuSnapshot[] = []

  for (const agent of agents) {
    const gpu = agent.gpu_info
    if (!gpu?.vram_total) continue

    const host = agent.host || agent.id || "agent"
    const gpuIdentity = `${gpu.id ?? gpu.name ?? "gpu"}:${gpu.vram_total}`
    const key = `${host}:${gpuIdentity}`
    if (seen.has(key)) continue

    seen.add(key)
    snapshots.push(gpu)
  }

  return snapshots
}
