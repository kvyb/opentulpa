import type {
  AgentEvent,
  ClientConfig,
  CodexDeviceLogin,
  InferenceModel,
  InferencePreference,
  InferenceProvider,
  InferenceSelection,
  InferenceStatus,
  ThreadSummary,
  TimelineEntry,
} from "./types.js"

export class ApiError extends Error {
  constructor(message: string, readonly status?: number) {
    super(message)
  }
}

export class OpenTulpaApi {
  constructor(readonly config: ClientConfig) {}

  async hostStatus(): Promise<Record<string, unknown>> {
    return this.json("/_host/api/status")
  }

  async createThread(title?: string): Promise<ThreadSummary> {
    return this.json("/v2/agent/threads", {
      method: "POST",
      body: JSON.stringify(title ? { title } : {}),
    })
  }

  async threads(): Promise<ThreadSummary[]> {
    const result = await this.json<{ threads: ThreadSummary[] }>("/v2/agent/threads?limit=100")
    return result.threads
  }

  async timeline(threadId: string): Promise<{ entries: TimelineEntry[]; thread: ThreadSummary }> {
    const entries: TimelineEntry[] = []
    let cursor: number | null = 0
    let thread: ThreadSummary | undefined
    while (cursor !== null) {
      const page: {
        entries: TimelineEntry[]
        thread: ThreadSummary
        next_cursor: number | null
      } = await this.json(
        `/v2/agent/threads/${encodeURIComponent(threadId)}/timeline?limit=100&cursor=${cursor}`,
      )
      entries.push(...page.entries)
      thread = page.thread
      cursor = page.next_cursor
    }
    if (!thread) throw new ApiError("OpenTulpa returned an empty thread timeline.")
    return { entries, thread }
  }

  async updateThread(threadId: string, body: { title?: string; archived?: boolean }): Promise<ThreadSummary> {
    return this.json(`/v2/agent/threads/${encodeURIComponent(threadId)}`, {
      method: "PATCH",
      body: JSON.stringify(body),
    })
  }

  async inferenceStatus(): Promise<InferenceStatus> {
    return this.json("/v2/inference")
  }

  async inferenceModels(provider: InferenceProvider, query = ""): Promise<InferenceModel[]> {
    const result = await this.json<{ models: InferenceModel[] }>(
      `/v2/inference/models?provider=${provider}&query=${encodeURIComponent(query)}`,
    )
    return result.models
  }

  async threadInference(threadId: string): Promise<InferencePreference> {
    return this.json(`/v2/agent/threads/${encodeURIComponent(threadId)}/inference`)
  }

  async updateThreadInference(
    threadId: string,
    expectedRevision: number,
    selection: InferenceSelection | null,
  ): Promise<InferencePreference> {
    return this.json(`/v2/agent/threads/${encodeURIComponent(threadId)}/inference`, {
      method: "PATCH",
      body: JSON.stringify({ expected_revision: expectedRevision, selection }),
    })
  }

  async startCodexLogin(): Promise<CodexDeviceLogin> {
    return this.json("/v2/inference/codex/device-logins", { method: "POST" })
  }

  async codexLogin(loginId: string): Promise<CodexDeviceLogin> {
    return this.json(`/v2/inference/codex/device-logins/${encodeURIComponent(loginId)}`)
  }

  async cancelCodexLogin(loginId: string): Promise<void> {
    const response = await fetch(
      `${this.config.url}/v2/inference/codex/device-logins/${encodeURIComponent(loginId)}`,
      this.withHeaders({ method: "DELETE" }),
    )
    if (!response.ok) throw await this.error(response)
  }

  async logoutCodex(resetThreads = false): Promise<{ disconnected: boolean; reset_threads: number }> {
    return this.json(`/v2/inference/codex/credential?reset_threads=${resetThreads}`, {
      method: "DELETE",
    })
  }

  run(threadId: string, text: string, fileIds: string[]): AsyncGenerator<AgentEvent> {
    return this.stream("/v2/agent/runs", {
      method: "POST",
      headers: { "Idempotency-Key": `tui-run:${crypto.randomUUID()}` },
      body: JSON.stringify({ thread_id: threadId, text, file_ids: fileIds }),
    })
  }

  events(runId: string, afterSequence: number): AsyncGenerator<AgentEvent> {
    return this.stream(
      `/v2/agent/runs/${encodeURIComponent(runId)}/events?after_sequence=${Math.max(0, afterSequence)}`,
      { headers: { "Last-Event-ID": String(Math.max(0, afterSequence)) } },
    )
  }

  resume(
    runId: string,
    approvalId: string,
    decision: "approve" | "edit" | "reject",
    editedArguments?: Record<string, unknown>,
  ): AsyncGenerator<AgentEvent> {
    return this.stream(`/v2/agent/runs/${encodeURIComponent(runId)}/resume`, {
      method: "POST",
      body: JSON.stringify({
        approval_id: approvalId,
        decision,
        edited_arguments: decision === "edit" ? editedArguments : null,
      }),
    })
  }

  async cancel(runId: string): Promise<void> {
    await this.json(`/v2/agent/runs/${encodeURIComponent(runId)}/cancel`, { method: "POST" })
  }

  async runSnapshot(runId: string): Promise<Record<string, unknown>> {
    return this.json(`/v2/agent/runs/${encodeURIComponent(runId)}`)
  }

  async upload(path: string): Promise<string> {
    const file = Bun.file(path)
    const form = new FormData()
    form.set("kind", file.type.startsWith("image/") ? "image" : "document")
    form.set("upload", file, path.split(/[\\/]/).pop() ?? "attachment")
    const result = await this.json<{ file: { id: string } }>("/v2/files", {
      method: "POST",
      headers: { "Idempotency-Key": `tui-file:${crypto.randomUUID()}` },
      body: form,
    })
    return result.file.id
  }

  async logs(): Promise<Array<Record<string, unknown>>> {
    const result = await this.json<{ logs: Array<Record<string, unknown>> }>("/_host/api/logs")
    return result.logs
  }

  async notifications(afterId: number): Promise<{
    notifications: Array<Record<string, unknown>>
    next_after_id: number
  }> {
    return this.json(`/v2/notifications?after_id=${Math.max(0, afterId)}&limit=100&wait_seconds=20`)
  }

  async acknowledgeNotification(notificationId: number): Promise<void> {
    const response = await fetch(
      `${this.config.url}/v2/notifications/${notificationId}/ack`,
      this.withHeaders({ method: "POST" }),
    )
    if (!response.ok) throw await this.error(response)
  }

  private async json<T = Record<string, unknown>>(path: string, init: RequestInit = {}): Promise<T> {
    let response: Response
    try {
      response = await fetch(`${this.config.url}${path}`, this.withHeaders(init))
    } catch {
      throw new ApiError("Could not reach the OpenTulpa server.")
    }
    if (!response.ok) throw await this.error(response)
    try {
      return (await response.json()) as T
    } catch {
      throw new ApiError("OpenTulpa returned an invalid response.", response.status)
    }
  }

  private async *stream(path: string, init: RequestInit = {}): AsyncGenerator<AgentEvent> {
    let runId = ""
    let sequence = 0
    let reconnectAttempts = 0
    while (true) {
      let response: Response
      try {
        response = await fetch(`${this.config.url}${path}`, this.withHeaders(init))
      } catch {
        if (!runId || reconnectAttempts >= 6) {
          throw new ApiError("The OpenTulpa event stream disconnected.")
        }
        reconnectAttempts += 1
        await new Promise((resolve) => setTimeout(resolve, Math.min(4_000, reconnectAttempts * 500)))
        continue
      }
      if (!response.ok) throw await this.error(response)
      if (!response.body) throw new ApiError("OpenTulpa returned an empty event stream.")

      const reader = response.body.getReader()
      const decoder = new TextDecoder()
      let buffer = ""
      let terminal = false
      let disconnected = false
      try {
        while (true) {
          const chunk = await reader.read()
          buffer += decoder.decode(chunk.value, { stream: !chunk.done }).replaceAll("\r\n", "\n")
          let boundary = buffer.indexOf("\n\n")
          while (boundary >= 0) {
            const frame = buffer.slice(0, boundary)
            buffer = buffer.slice(boundary + 2)
            const data = frame
              .split("\n")
              .filter((line) => line.startsWith("data:"))
              .map((line) => line.slice(5).trimStart())
              .join("\n")
            if (data) {
              const event = this.event(data)
              runId = event.run_id
              if (event.sequence > sequence) {
                sequence = event.sequence
                reconnectAttempts = 0
              }
              terminal = ["run.completed", "run.failed", "approval.required"].includes(event.type)
              yield event
            }
            boundary = buffer.indexOf("\n\n")
          }
          if (chunk.done) break
        }
      } catch {
        disconnected = true
      }
      if (!disconnected && buffer.trim()) {
        const data = buffer
          .split("\n")
          .filter((line) => line.startsWith("data:"))
          .map((line) => line.slice(5).trimStart())
          .join("\n")
        if (data) {
          const event = this.event(data)
          runId = event.run_id
          if (event.sequence > sequence) {
            sequence = event.sequence
            reconnectAttempts = 0
          }
          terminal = ["run.completed", "run.failed", "approval.required"].includes(event.type)
          yield event
        }
      }
      if (terminal || !runId) {
        if (disconnected && !runId) {
          throw new ApiError("The OpenTulpa event stream disconnected.")
        }
        return
      }
      if (reconnectAttempts >= 6) {
        throw new ApiError("The OpenTulpa event stream disconnected.")
      }
      reconnectAttempts += 1
      path = `/v2/agent/runs/${encodeURIComponent(runId)}/events?after_sequence=${sequence}`
      init = { headers: { "Last-Event-ID": String(sequence) } }
      await new Promise((resolve) => setTimeout(resolve, Math.min(4_000, reconnectAttempts * 500)))
    }
  }

  private event(raw: string): AgentEvent {
    try {
      const value = JSON.parse(raw) as AgentEvent
      if (!value.type || !value.run_id || !Number.isInteger(value.sequence)) throw new Error()
      return value
    } catch {
      throw new ApiError("OpenTulpa returned an invalid stream event.")
    }
  }

  private withHeaders(init: RequestInit): RequestInit {
    const headers = new Headers(init.headers)
    headers.set("Accept", "application/json, text/event-stream")
    if (this.config.token) headers.set("Authorization", `Bearer ${this.config.token}`)
    if (init.body && !(init.body instanceof FormData)) headers.set("Content-Type", "application/json")
    return { ...init, headers }
  }

  private async error(response: Response): Promise<ApiError> {
    let message = `OpenTulpa request failed (HTTP ${response.status}).`
    try {
      const body = (await response.json()) as { detail?: unknown }
      if (typeof body.detail === "string" && body.detail.trim()) message = body.detail.trim()
    } catch {}
    return new ApiError(message, response.status)
  }
}
