export type ClientConfig = {
  url: string
  token: string
  thread_id: string
  credential_storage?: string
}

export type AgentEvent = {
  type: string
  run_id: string
  sequence: number
  timestamp: string
  data: Record<string, unknown>
}

export type Approval = {
  approval_id: string
  tool_name: string
  description: string
  arguments: Record<string, unknown>
  allowed_decisions: Array<"approve" | "edit" | "reject">
  run_id: string
}

export type ToolCall = {
  callId: string
  name: string
  arguments: unknown
  result?: unknown
  error?: unknown
  ok?: boolean
  startedAt: string
  completedAt?: string
}

export type TurnPart =
  | { type: "assistant"; text: string }
  | { type: "tool"; callId: string }

export type Turn = {
  runId: string
  user: string
  fileIds: string[]
  assistant: string
  tools: ToolCall[]
  parts: TurnPart[]
  approvals: Approval[]
  artifacts: Record<string, unknown>[]
  error?: string
  status: "idle" | "running" | "approval" | "completed" | "failed"
  lastSequence: number
  startedAt: string
}

export type ThreadSummary = {
  thread_id: string
  title: string
  channel: string
  archived: boolean
  created_at: string
  updated_at: string
  last_run_id: string | null
  status: string
  preview: string
}

export type TimelineEntry = {
  id: string
  type: string
  run_id: string
  timestamp: string
  text?: string
  file_ids?: string[]
  event_type?: string
  sequence?: number
  data?: Record<string, unknown>
}

export type InferenceProvider = "api" | "codex"

export type InferenceSelection = {
  provider: InferenceProvider
  model: string
  reasoning_effort: string | null
  fallback_to_api: boolean
}

export type InferencePreference = {
  revision: number
  selection: InferenceSelection | null
  effective: InferenceSelection
}

export type InferenceModel = {
  provider: InferenceProvider
  id: string
  reasoning_efforts: string[]
  default_reasoning_effort: string | null
}

export type InferenceStatus = {
  api_default: InferenceSelection
  codex: {
    connected: boolean
    credential_revision: number
    experimental: boolean
  }
}

export type CodexDeviceLogin = {
  login_id: string
  status: "pending" | "authorized" | "expired" | "failed"
  verification_url: string
  user_code: string
  interval_seconds: number
  expires_at: string
  error: string | null
}
