export type Scope = 'all' | 'custom' | 'single'
export type ViewName = 'models' | 'data' | 'task' | 'progress' | 'result' | 'archive'
export type ThemeName = 'light' | 'dark'

export interface CaseOption {
  id: string
  label: string
  models: string[]
  model_groups: { case: string[]; benchmark: string[] }
  default_bbox: number[]
  scopes: Scope[]
}

export interface ProductMetadata {
  name: string
  kind: string
  format: string
  grid: string
  clock: string
  cadence: string
  domain: string
}

export interface DataMetadata {
  llm: { provider: string; model: string; role: string }
  products: ProductMetadata[]
  cases: { id: string; label: string; availability: string }[]
  rules: string[]
}

export interface Initialization {
  id: string
  time_utc: string
  time_bjt: string
  label: string
  available_leads: number[]
  complete: boolean
}

export interface TaskPayload {
  case_id: string
  models: string[]
  scope: Scope
  bbox: number[]
  complete_only: boolean
  initialization_ids: string[]
  comparison_valid_time: string | null
  prompt: string
  use_default_plan: boolean
}

export interface JobEvent {
  index: number
  stage: string
  message: string
  progress: number
  timestamp: string
  elapsed_seconds: number
}

export interface Artifact {
  name: string
  label: string
  mime: string
  url: string
  svg_url?: string
}

export interface CurveSeries {
  model: string
  threshold_dbz: number
  metric: string
  points: [number, number | null][]
}

export interface SpatialFrame extends Artifact {
  lead_minutes: number
}

export interface VerificationResult {
  kind: 'single' | 'custom' | 'all' | 'benchmark'
  request: Record<string, unknown>
  context: Record<string, unknown>
  coverage: Record<string, unknown>
  curves: CurveSeries[]
  summary_table: Record<string, unknown>[]
  spatial_frames: SpatialFrame[]
  gallery: Artifact[]
  insight: string
  insight_source: string
  downloads: Artifact[]
}

export interface Job {
  id: string
  status: 'queued' | 'running' | 'complete' | 'failed'
  stage: string
  progress: number
  message: string
  created_at: string
  request?: TaskPayload
  error: string | null
  result: VerificationResult | null
}
