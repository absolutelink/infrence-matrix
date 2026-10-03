export type AgentGpuSnapshot = {
  id?: number | string
  uuid?: string
  name?: string
  vram_total?: number
  vram_used?: number
  utilization?: number
  gpus?: AgentGpuSnapshot[]
}

type AgentWithGpu = {
  id?: string
  host?: string | null
  gpu_info?: AgentGpuSnapshot | null
}

/**
 * Multiple agent processes can expose the same physical GPU. Prefer a
 * hardware UUID when available; older agents fall back to host-scoped GPU
 * identity so identical cards on separate hosts are not merged.
 */
export function uniqueGpuSnapshots(agents: AgentWithGpu[]): AgentGpuSnapshot[] {
  const seen = new Set<string>()
  const snapshots: AgentGpuSnapshot[] = []

  for (const agent of agents) {
    const gpu = agent.gpu_info
    if (!gpu?.vram_total) continue

    const host = agent.host || agent.id || "agent"
    const deviceSnapshots = gpu.gpus?.length ? gpu.gpus : [gpu]
    for (const device of deviceSnapshots) {
      if (!device.vram_total) continue
      const gpuIdentity = device.uuid
        ? `uuid:${device.uuid}`
        : `${host}:${device.id ?? device.name ?? "gpu"}:${device.vram_total}`
      if (seen.has(gpuIdentity)) continue

      seen.add(gpuIdentity)
      snapshots.push(device)
    }
  }

  return snapshots
}
