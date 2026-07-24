import type { AgentEvent, Approval, TimelineEntry, ToolCall, Turn } from "./types.js"

export function emptyTurn(runId: string, user = "", startedAt = new Date().toISOString()): Turn {
  return {
    runId,
    user,
    fileIds: [],
    assistant: "",
    tools: [],
    parts: [],
    approvals: [],
    artifacts: [],
    status: "running",
    lastSequence: 0,
    startedAt,
  }
}

export function reduceEvent(turn: Turn, event: AgentEvent): Turn {
  if (event.run_id !== turn.runId && turn.runId) return turn
  if (event.sequence <= turn.lastSequence) return turn
  const next: Turn = {
    ...turn,
    runId: event.run_id,
    lastSequence: event.sequence,
    tools: [...turn.tools],
    parts: turn.parts.map((part) => ({ ...part })),
    approvals: [...turn.approvals],
    artifacts: [...turn.artifacts],
  }
  if (event.type === "run.started") {
    next.status = "running"
    next.approvals = []
  }
  if (event.type === "message.delta") {
    const text = typeof event.data.text === "string" ? event.data.text : ""
    next.assistant += text
    appendAssistant(next, text)
  }
  if (event.type === "tool.started") {
    const callId = String(event.data.call_id ?? "").trim()
    const name = String(event.data.name ?? "").trim()
    if (!callId || callId.toLocaleLowerCase() === "none" || !name) return next
    const existing = next.tools.findIndex((tool) => tool.callId === callId)
    const tool: ToolCall = {
      callId,
      name,
      arguments: event.data.arguments ?? {},
      startedAt: event.timestamp,
    }
    if (existing >= 0) next.tools[existing] = { ...next.tools[existing], ...tool }
    else {
      next.tools.push(tool)
      next.parts.push({ type: "tool", callId })
    }
  }
  if (event.type === "tool.completed") {
    const callId = String(event.data.call_id ?? "").trim()
    if (!callId || callId.toLocaleLowerCase() === "none") return next
    const existing = next.tools.findIndex((tool) => tool.callId === callId)
    const completion: ToolCall = {
      callId,
      name: String(event.data.name ?? "tool"),
      arguments: existing >= 0 ? next.tools[existing].arguments : {},
      result: event.data.result,
      error: event.data.error,
      ok: event.data.ok !== false,
      startedAt: existing >= 0 ? next.tools[existing].startedAt : event.timestamp,
      completedAt: event.timestamp,
    }
    if (existing >= 0) next.tools[existing] = completion
    else {
      next.tools.push(completion)
      next.parts.push({ type: "tool", callId })
    }
  }
  if (event.type === "approval.required") {
    const approval = approvalFrom(event.run_id, event.data)
    if (approval && !next.approvals.some((item) => item.approval_id === approval.approval_id)) {
      next.approvals.push(approval)
    }
    next.status = "approval"
  }
  if (event.type === "artifact.ready") next.artifacts.push(event.data)
  if (event.type === "run.completed") {
    if (!next.assistant && typeof event.data.text === "string") {
      next.assistant = event.data.text
      appendAssistant(next, event.data.text)
    }
    next.status = "completed"
    next.approvals = []
    next.tools = finishTools(next.tools, event.timestamp)
  }
  if (event.type === "run.failed") {
    const cancelled = event.data.code === "agent_run_cancelled"
    next.status = cancelled ? "cancelled" : "failed"
    next.error = cancelled
      ? undefined
      : String(event.data.message ?? "The agent run could not be completed.")
    next.approvals = []
    next.tools = finishTools(next.tools, event.timestamp)
  }
  return next
}

export function consumeApproval(turn: Turn, approvalId: string): Turn {
  const approvals = turn.approvals.filter((item) => item.approval_id !== approvalId)
  if (approvals.length === turn.approvals.length) return turn
  return {
    ...turn,
    approvals,
    status: approvals.length ? "approval" : "running",
  }
}

export function turnsFromTimeline(entries: TimelineEntry[]): Turn[] {
  const turns = new Map<string, Turn>()
  for (const entry of entries) {
    let turn = turns.get(entry.run_id)
    if (!turn) {
      turn = emptyTurn(entry.run_id, "", entry.timestamp)
      turn.status = "completed"
      turns.set(entry.run_id, turn)
    }
    if (entry.type === "user") {
      turn.user = entry.text ?? ""
      turn.fileIds = entry.file_ids ?? []
      continue
    }
    if (entry.type === "assistant") {
      if (!turn.assistant) {
        turn.assistant = entry.text ?? ""
        appendAssistant(turn, turn.assistant)
      }
      turn.status = "completed"
      turn.tools = finishTools(turn.tools, entry.timestamp)
      continue
    }
    if (!entry.event_type || !entry.data) continue
    const event: AgentEvent = {
      type: entry.event_type,
      run_id: entry.run_id,
      sequence: entry.sequence ?? turn.lastSequence + 1,
      timestamp: entry.timestamp,
      data: entry.data,
    }
    turn = reduceEvent(turn, event)
    turns.set(entry.run_id, turn)
  }
  return [...turns.values()]
}

function appendAssistant(turn: Turn, text: string): void {
  if (!text) return
  const last = turn.parts.at(-1)
  if (last?.type === "assistant") last.text += text
  else turn.parts.push({ type: "assistant", text })
}

function finishTools(tools: ToolCall[], timestamp: string): ToolCall[] {
  return tools.map((tool) => tool.completedAt ? tool : { ...tool, completedAt: timestamp })
}

function approvalFrom(runId: string, data: Record<string, unknown>): Approval | undefined {
  const approvalId = String(data.approval_id ?? "")
  if (!approvalId) return undefined
  const values = Array.isArray(data.allowed_decisions) ? data.allowed_decisions : []
  const allowed = values.filter(
    (value): value is "approve" | "edit" | "reject" =>
      value === "approve" || value === "edit" || value === "reject",
  )
  return {
    approval_id: approvalId,
    tool_name: String(data.tool_name ?? "action"),
    description: String(data.description ?? ""),
    arguments: isRecord(data.arguments) ? data.arguments : {},
    allowed_decisions: allowed,
    run_id: runId,
  }
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return !!value && typeof value === "object" && !Array.isArray(value)
}

export function compactJson(value: unknown, limit = 500): string {
  let rendered: string
  try {
    rendered = JSON.stringify(value, null, 2) ?? "null"
  } catch {
    rendered = String(value)
  }
  return rendered.length <= limit ? rendered : `${rendered.slice(0, limit - 3)}...`
}

export function approvalSummary(approval: Approval): string[] {
  const args = approval.arguments
  const parameters = isRecord(args.parameters) ? args.parameters : args
  const lines: string[] = []
  const action = shortText(args.action_name)
  if (action) lines.push(`Action: ${action}`)

  const owner = shortText(parameters.owner)
  const repo = shortText(parameters.repo)
  const branch = shortText(parameters.branch)
  const baseBranch = shortText(parameters.base_branch)
  const repository = owner && repo ? `${owner}/${repo}` : repo
  const target = [
    repository,
    branch ? `${branch}${baseBranch ? ` <- ${baseBranch}` : ""}` : "",
  ].filter(Boolean).join(" · ")
  if (target) lines.push(`Target: ${target}`)

  const paths = [
    ...pathsFrom(parameters.upserts),
    ...pathsFrom(parameters.deletes),
  ]
  if (paths.length) {
    const visible = paths.slice(0, 3)
    const remaining = paths.length - visible.length
    lines.push(
      `${paths.length} ${paths.length === 1 ? "file" : "files"}: ${visible.join(", ")}${remaining ? ` +${remaining}` : ""}`,
    )
  }

  const message = shortText(parameters.message)
  if (message && lines.length < 3) lines.push(`Message: ${message}`)
  return lines.slice(0, 3)
}

function pathsFrom(value: unknown): string[] {
  if (!Array.isArray(value)) return []
  return value.flatMap((item) => {
    if (typeof item === "string") return [shortText(item)].filter(Boolean)
    if (!isRecord(item)) return []
    const path = shortText(item.path)
    return path ? [path] : []
  })
}

function shortText(value: unknown, limit = 120): string {
  if (typeof value !== "string") return ""
  const text = value.trim().replaceAll(/\s+/g, " ")
  return text.length <= limit ? text : `${text.slice(0, limit - 3)}...`
}

export function toolLabel(tool: ToolCall): string {
  const name = tool.name.replaceAll("_", " ")
  if (!tool.arguments || typeof tool.arguments !== "object" || Array.isArray(tool.arguments)) return name
  const args = tool.arguments as Record<string, unknown>
  for (const key of ["query", "path", "url", "command", "instruction", "description", "action"]) {
    const value = args[key]
    if (typeof value !== "string" || !value.trim()) continue
    const summary = value.trim().replaceAll(/\s+/g, " ")
    return `${name}  ${summary.length > 72 ? `${summary.slice(0, 69)}...` : summary}`
  }
  return name
}
