// Hand-written shapes for the admin API dicts (the generated client types
// the Phase 9/10 endpoints as loose JSON maps; these mirror
// admin/backend/app/api/admin/{machines,definitions,instances,responses}.py).

// Phase 12: provider types come from the admin registry
// (AdminService.listProviderTypes) — no hardcoded list.
export type ProviderType = string

export interface GpuInfo {
  uuid?: string
  vendor?: string
  name?: string
  total_vram_bytes?: number
}

export interface MachineHardware {
  gpus?: GpuInfo[]
  cpu?: Record<string, unknown>
  ram?: Record<string, unknown>
  total_vram_bytes?: number
  [key: string]: unknown
}

// Phase 17: merged live machine metrics (mirrors
// admin/backend/app/services/metrics_service.py::read_machine_metrics). The
// per-GPU ``vram``/``gpu_usage`` sections union across every agent partial;
// ``os_ram``/``cpu``/``storage`` come from the single machine-wide owner.
// Sections are omitted when no agent has reported them yet.
export interface MetricsGpuEntry {
  uuid?: string
  id?: number
  name?: string
  vendor?: string
  backend?: string
  vram_total?: number
  vram_used?: number
  vram_free?: number
  utilization?: number
  temperature?: number
  memory_type?: string
  /** Agent that contributed this GPU (attribution from the merge-on-read). */
  agent_id?: string
  [key: string]: unknown
}

export interface MetricsVramSection {
  total_bytes?: number
  used_bytes?: number
  free_bytes?: number
  gpu_count?: number
  gpus?: MetricsGpuEntry[]
}

export interface MetricsGpuUsageEntry {
  uuid?: string
  id?: number
  utilization?: number
  agent_id?: string
}

export interface MetricsGpuUsageSection {
  utilization_percent?: number
  gpus?: MetricsGpuUsageEntry[]
}

export interface MetricsOsRamSection {
  total_bytes?: number
  available_bytes?: number
  used_bytes?: number
  [key: string]: unknown
}

export interface MetricsCpuSection {
  cores?: number
  load_1m?: number
  load_5m?: number
  load_15m?: number
  [key: string]: unknown
}

export interface MachineMetrics {
  machine_uid: string
  vram?: MetricsVramSection
  gpu_usage?: MetricsGpuUsageSection
  os_ram?: MetricsOsRamSection
  cpu?: MetricsCpuSection
  storage?: Record<string, unknown>
  owner_agent_id?: string
  reporting_agents?: string[]
}

export interface Machine {
  id: string
  uid: string
  name: string
  host: string | null
  dns: string | null
  ip: string | null
  total_vram_bytes: number
  hardware: MachineHardware
  created_at: string
  updated_at: string | null
  /** Phase 16: shared agent registration secret (trusted-LAN plaintext). */
  registration_secret?: string
  /** Phase 16: number of ProviderAgents attached to this machine. */
  agent_count?: number
}

// Phase 16: one backend (ProviderInstance) hosted by a ProviderAgent.
export interface AgentBackend {
  id: string
  provider_definition_id: string
  backend_status: string
  port: number
  config_fingerprint: string | null
}

// Phase 16: a hardware-local container bound to (machine, provider_type,
// agent_id). Mirrors admin/backend/app/api/admin/agents.py::_agent_dict.
export interface ProviderAgent {
  id: string
  machine_id: string
  machine_uid: string | null
  provider_type: string
  agent_id: string
  base_port: number
  version: string
  agent_status: string
  websocket_connected: boolean
  epoch: number
  reported_schema_fingerprint: string | null
  assigned_gpus: string[]
  error_message: string | null
  last_seen: string | null
  created_at: string
  updated_at: string | null
  backends?: AgentBackend[]
  definition_count?: number
}

export interface InstanceSummary {
  id: string
  machine_uid: string | null
  agent_status: string
  backend_status: string
  websocket_connected: boolean
  config_fingerprint: string | null
  port: number
  version: string
}

export interface ConfigUpdateResult {
  instance_id: string
  ok: boolean
  noop: boolean
  error: string | null
  step: string | null
  config_fingerprint: string | null
}

export interface ProviderDefinition {
  id: string
  alias: string
  /** Phase 16: provider_type is required (no shells). */
  provider_type: ProviderType | string
  /** Phase 16: backend_config is required (no shells). */
  backend_config: Record<string, unknown>
  config_fingerprint: string | null
  vram_required_bytes: number
  idle_timeout_seconds: number
  capacity: number
  model_metadata: Record<string, unknown>
  enabled: boolean
  status: string
  /** Phase 16: 'any_of_type' | 'specific'. */
  agent_placement: string
  /** Phase 16: ProviderAgent ids linked for 'specific' placement (empty for any_of_type). */
  agents?: string[]
  created_at: string
  updated_at: string | null
  instances?: InstanceSummary[]
  connected_instance_count?: number
  config_update_results?: ConfigUpdateResult[]
}

export interface ProviderInstance {
  id: string
  machine_id: string
  machine_uid: string | null
  machine_name: string | null
  provider_definition_id: string
  alias: string | null
  provider_type: string | null
  port: number
  version: string
  agent_status: string
  backend_status: string
  websocket_connected: boolean
  epoch: number
  last_seen: string | null
  last_request_at: string | null
  config_fingerprint: string | null
  reported_schema_fingerprint?: string | null
  assigned_gpus: string[]
  error_message: string | null
  created_at: string
}

export interface ResponseRecord {
  id: string
  response_id: string
  previous_response_id: string | null
  model_alias: string | null
  provider_type: string | null
  provider_instance_id: string | null
  api_format: string
  status: string
  error_code: string | null
  error_message: string | null
  input_tokens: number
  output_tokens: number
  total_tokens: number
  store: boolean
  created_at: string
  completed_at: string | null
}

export interface ResponsesList {
  total: number
  limit: number
  offset: number
  responses: ResponseRecord[]
}

export interface UsageSample {
  id: string
  provider_instance_id: string | null
  provider_definition_id: string | null
  prompt_tokens: number
  cached_tokens: number
  completion_tokens: number
  prompt_ms: number
  predicted_ms: number
  prompt_per_second: number
  predicted_per_second: number
  created_at: string
}

export interface UsageStats {
  totals: {
    prompt_tokens: number
    cached_tokens: number
    completion_tokens: number
  }
  samples: UsageSample[]
}

export interface OverviewStats {
  machines: number
  definitions: number
  definitions_enabled: number
  instances: number
  instances_connected: number
  backends_running: number
  queued_requests: number
  active_requests: number
  input_tokens: number
  output_tokens: number
  total_tokens: number
}

export interface StorageActionResult {
  ok: boolean
  instance_id: string
  dry_run?: boolean
  deleted?: string[]
  bytes_freed?: number
  kept?: string[]
}

// Manual backend control (provider_lib.ops over the WS). Start/restart/
// initialize ack "accepted" by default — the boot runs in the background
// because a cold engine may download its weights first — so `accepted` and
// `backend_status` describe the transition, not a finished boot.
export interface BackendActionResult {
  ok: boolean
  instance_id: string
  backend?: string
  backend_status?: string
  capacity?: number
  accepted?: boolean
  waited_for_running?: boolean
  no_config?: boolean
  api_port?: number
  engine_port?: number
  backend_port?: number
  effective_capacity?: number
}

// Phase 13: log tails (mirrors admin/backend/app/services/log_store.read_logs).
// ``entries`` arrive NEWEST FIRST from the API; the UI reverses them for
// chronological (oldest-top → newest-bottom) display.
export type LogKind = "backend" | "provider" | "all"

export interface LogEntry {
  seq: number
  ts: string
  stream: string
  text: string
}

export interface LogsResponse {
  entries: LogEntry[]
  cursor: number
  dropped: number
  gap: boolean
  oldest_seq: number
  unseen_total: number
}

// Phase 12: provider type registry (mirrors
// admin/backend/app/api/admin/provider_types.py dicts).
export interface ProviderTypeConsensus {
  status: string
  committed_fingerprint: string
  universe: string[]
  universe_count: number
  on_committed: number
  pending_fingerprint: string | null
  voters: string[]
  voter_count: number
  waiting_on: string[]
}

export interface ProviderTypeSummary {
  name: string
  status: string
  schema_fingerprint: string
  /** Phase 16: per-agent running cap (0 = unlimited). */
  max_running_backends?: number
  consensus: ProviderTypeConsensus
}

export interface ProviderTypeDetail {
  name: string
  status: string
  schema: Record<string, unknown>
  schema_fingerprint: string
  /** Phase 16: per-agent running cap (0 = unlimited). */
  max_running_backends?: number
  pending_schema: Record<string, unknown> | null
  consensus: ProviderTypeConsensus
  created_at: string
  updated_at: string | null
}
