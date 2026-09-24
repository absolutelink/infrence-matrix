// Temporary adapter until the benchmark endpoints are present in the generated OpenAPI client.
// Keep the wire-format assumptions here so regeneration only requires replacing this module.

export interface BenchmarkDefinition {
  id: string
  name: string
  description?: string | null
  config?: Record<string, unknown> | null
  source_server_instance_id?: string | null
  server_alias?: string | null
  created_at?: string | null
  updated_at?: string | null
}

export interface BenchmarkRun {
  id: string
  definition_id?: string | null
  definition_name?: string | null
  status: string
  created_at?: string | null
  started_at?: string | null
  finished_at?: string | null
  error?: string | null
  results?: unknown
  agent_id?: string | null
  agent_name?: string | null
  server_id?: string | null
}

export interface BenchmarkDefinitionInput {
  name: string
  description?: string
  source_server_instance_id: string
  config?: Record<string, unknown>
}

const basePath = "/api/v1/benchmarks"

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${basePath}${path}`, {
    ...init,
    headers: { "Content-Type": "application/json", ...init?.headers },
  })
  if (!response.ok) {
    const detail = await response.text()
    throw new Error(detail || `${response.status} ${response.statusText}`)
  }
  if (response.status === 204) return undefined as T
  return (await response.json()) as T
}

function collection<T>(value: T[] | { items?: T[]; data?: T[] }): T[] {
  return Array.isArray(value) ? value : (value.items ?? value.data ?? [])
}

export const benchmarkApi = {
  async listDefinitions() {
    return collection(
      await request<
        | BenchmarkDefinition[]
        | { items?: BenchmarkDefinition[]; data?: BenchmarkDefinition[] }
      >("/definitions"),
    )
  },
  createDefinition(body: BenchmarkDefinitionInput) {
    return request<BenchmarkDefinition>("/definitions", {
      method: "POST",
      body: JSON.stringify(body),
    })
  },
  updateDefinition(id: string, body: BenchmarkDefinitionInput) {
    return request<BenchmarkDefinition>(`/definitions/${id}`, {
      method: "PATCH",
      body: JSON.stringify(body),
    })
  },
  deleteDefinition(id: string) {
    return request<void>(`/definitions/${id}`, { method: "DELETE" })
  },
  async listRuns() {
    return collection(
      await request<
        BenchmarkRun[] | { items?: BenchmarkRun[]; data?: BenchmarkRun[] }
      >("/runs"),
    )
  },
  runDefinition(definitionId: string) {
    return request<BenchmarkRun>("/runs", {
      method: "POST",
      body: JSON.stringify({ definition_id: definitionId }),
    })
  },
  cancelRun(id: string) {
    return request<BenchmarkRun>(`/runs/${id}/cancel`, { method: "POST" })
  },
  abortRun(id: string) {
    return request<BenchmarkRun>(`/runs/${id}/abort`, { method: "POST" })
  },
  forceStopRun(id: string) {
    return request<BenchmarkRun>(`/runs/${id}/force-stop`, { method: "POST" })
  },
  getResults(id: string) {
    return request<unknown>(`/runs/${id}/results`)
  },
}
