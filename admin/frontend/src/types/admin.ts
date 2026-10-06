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
  instance_count?: number
}

export interface InstanceSummary {
  id: string
  machine_uid: string | null
  instance_status: string
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
  provider_type: ProviderType | string
  backend_config: Record<string, unknown>
  config_fingerprint: string
  vram_required_bytes: number
  idle_timeout_seconds: number
  capacity: number
  registration_token: string
  model_metadata: Record<string, unknown>
  enabled: boolean
  status: string
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
  instance_status: string
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
  consensus: ProviderTypeConsensus
}

export interface ProviderTypeDetail {
  name: string
  status: string
  schema: Record<string, unknown>
  schema_fingerprint: string
  pending_schema: Record<string, unknown> | null
  consensus: ProviderTypeConsensus
  created_at: string
  updated_at: string | null
}
