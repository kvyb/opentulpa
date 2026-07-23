/** @jsxImportSource @opentui/solid */
import {
  SyntaxStyle,
  TextAttributes,
  decodePasteBytes,
  type CliRenderer,
  type Selection,
  type TextareaRenderable,
} from "@opentui/core"
import { For, Index, Show, createMemo, createSignal, onCleanup, onMount } from "solid-js"
import { useKeyboard, useRenderer } from "@opentui/solid"
import { spawn } from "node:child_process"
import { basename } from "node:path"
import { droppedFiles } from "./attachments.js"
import { ApiError, OpenTulpaApi } from "./api.js"
import {
  approvalSummary,
  compactJson,
  consumeApproval,
  emptyTurn,
  reduceEvent,
  toolLabel,
  turnsFromTimeline,
} from "./state.js"
import type {
  AgentEvent,
  Approval,
  ClientConfig,
  CodexDeviceLogin,
  InferenceModel,
  InferencePreference,
  InferenceProvider,
  InferenceSelection,
  RepositoryWorkspace,
  ThreadSummary,
  ToolCall,
  Turn,
} from "./types.js"

const COLORS = {
  bg: "#000000",
  panel: "#111111",
  raised: "#1A1A1A",
  selected: "#2A2A2A",
  text: "#E0E0E0",
  muted: "#666666",
  dim: "#444444",
  blue: "#5C9CF5",
  green: "#8ADF8A",
  red: "#DF8A8A",
  amber: "#E5C07B",
}

const SPINNER = ["⬒", "⬔", "⬓", "⬕"]
const LOADING_CELLS = ["■··", "■■·", "·■■", "··■", "·■■", "■■·"]
const SPLIT = {
  topLeft: "",
  bottomLeft: "",
  vertical: "┃",
  topRight: "",
  bottomRight: "",
  horizontal: " ",
  bottomT: "",
  topT: "",
  cross: "",
  leftT: "",
  rightT: "",
}
const MARKDOWN_STYLE = SyntaxStyle.fromStyles({
  default: { fg: COLORS.text },
  "markup.heading": { fg: COLORS.text, bold: true },
  "markup.bold": { fg: "#E8A465", bold: true },
  "markup.italic": { fg: COLORS.amber, italic: true },
  "markup.list": { fg: COLORS.muted },
  "markup.quote": { fg: COLORS.muted, italic: true },
  "markup.raw": { fg: COLORS.green },
  "markup.raw.inline": { fg: COLORS.green },
  "markup.link": { fg: COLORS.blue, underline: true },
  "markup.link.url": { fg: COLORS.blue, underline: true },
})

type PickerItem = {
  id: string
  label: string
  detail: string
  select: () => void
}

type PickerState = {
  title: string
  hint: string
  items: PickerItem[]
}

export type SlashCommand = {
  value: string
  description: string
  acceptsArgument?: boolean
}

export const SLASH_COMMANDS: readonly SlashCommand[] = [
  { value: "/new", description: "Start a new session" },
  { value: "/sessions", description: "Browse previous sessions" },
  { value: "/session", description: "Open a session by name or number", acceptsArgument: true },
  { value: "/model", description: "Choose the provider and model" },
  { value: "/reasoning", description: "Choose the reasoning effort" },
  { value: "/speed", description: "Choose normal or fast Codex inference" },
  { value: "/repo", description: "Open or inspect a repository workspace", acceptsArgument: true },
  { value: "/repos", description: "List repository workspaces" },
  { value: "/login codex", description: "Connect a ChatGPT Codex subscription" },
  { value: "/logout codex", description: "Disconnect the Codex subscription" },
  { value: "/regenerate", description: "Regenerate the latest response" },
  { value: "/attach", description: "Attach a local file", acceptsArgument: true },
  { value: "/tools", description: "Expand or collapse tool history" },
  { value: "/logs", description: "Show recent server logs" },
  { value: "/cancel", description: "Cancel the active run" },
  { value: "/help", description: "Show command help" },
  { value: "/quit", description: "Exit OpenTulpa" },
]

export function filterSlashCommands(query: string): SlashCommand[] {
  const normalized = query.trim().toLocaleLowerCase()
  if (!normalized) return [...SLASH_COMMANDS]
  return SLASH_COMMANDS.filter((item) =>
    `${item.value.slice(1)} ${item.description}`.toLocaleLowerCase().includes(normalized),
  )
}

export function App(props: { config: ClientConfig; onConnectionChange: (threadId: string) => void }) {
  const renderer = useRenderer()
  const api = new OpenTulpaApi(props.config)
  let composer: TextareaRenderable | undefined
  const [threads, setThreads] = createSignal<ThreadSummary[]>([])
  const [thread, setThread] = createSignal<ThreadSummary>()
  const [turns, setTurns] = createSignal<Turn[]>([])
  const [draft, setDraft] = createSignal("")
  const [attachments, setAttachments] = createSignal<string[]>([])
  const [status, setStatus] = createSignal("Connecting")
  const [error, setError] = createSignal("")
  const [busy, setBusy] = createSignal(false)
  const [uploading, setUploading] = createSignal(false)
  const [sessionDialog, setSessionDialog] = createSignal(false)
  const [sessionQuery, setSessionQuery] = createSignal("")
  const [sessionIndex, setSessionIndex] = createSignal(0)
  const [editApproval, setEditApproval] = createSignal<Approval>()
  const [spinner, setSpinner] = createSignal(0)
  const [clock, setClock] = createSignal(Date.now())
  const [copiedUntil, setCopiedUntil] = createSignal(0)
  const [terminalWidth, setTerminalWidth] = createSignal(renderer.width)
  const [expandedToolsGroup, setExpandedToolsGroup] = createSignal("")
  const [inference, setInference] = createSignal<InferencePreference>()
  const [repository, setRepository] = createSignal<RepositoryWorkspace>()
  const [picker, setPicker] = createSignal<PickerState>()
  const [pickerQuery, setPickerQuery] = createSignal("")
  const [pickerIndex, setPickerIndex] = createSignal(0)
  const [codexLogin, setCodexLogin] = createSignal<CodexDeviceLogin>()
  const [slashIndex, setSlashIndex] = createSignal(0)
  const [dismissedSlash, setDismissedSlash] = createSignal("")
  const approvalNotifications = new Map<string, number>()
  let notificationCursor = 0
  let alive = true

  const activeTurn = createMemo(() => turns().at(-1))
  const pendingApproval = createMemo(() => activeTurn()?.approvals.at(-1))
  const filteredThreads = createMemo(() => {
    const query = sessionQuery().trim().toLocaleLowerCase()
    return query
      ? threads().filter((item) => `${item.title} ${item.preview}`.toLocaleLowerCase().includes(query))
      : threads()
  })
  const filteredPicker = createMemo(() => {
    const value = picker()
    const query = pickerQuery().trim().toLocaleLowerCase()
    if (!value) return []
    return query
      ? value.items.filter((item) => `${item.label} ${item.detail}`.toLocaleLowerCase().includes(query))
      : value.items
  })
  const modalOpen = createMemo(() => sessionDialog() || !!picker() || !!codexLogin())
  const slashCommands = createMemo(() => {
    const value = draft()
    if (
      !value.startsWith("/")
      || /\s/.test(value.slice(1))
      || value === dismissedSlash()
      || modalOpen()
      || pendingApproval()
      || editApproval()
      || busy()
    ) return []
    return filterSlashCommands(value.slice(1))
  })

  const refreshThreads = async () => {
    const values = await api.threads()
    setThreads(values)
    return values
  }

  const refreshRepository = async () => {
    const selected = thread()
    if (!selected) {
      setRepository(undefined)
      return
    }
    setRepository((await api.activeRepository(selected.thread_id)) ?? undefined)
  }

  const loadThread = async (threadId: string) => {
    setSessionDialog(false)
    setError("")
    setStatus("Loading session")
    const [timeline, preference, activeRepository] = await Promise.all([
      api.timeline(threadId),
      api.threadInference(threadId),
      api.activeRepository(threadId),
    ])
    let restored = turnsFromTimeline(timeline.entries)
    setThread(timeline.thread)
    setInference(preference)
    setRepository(activeRepository ?? undefined)
    props.onConnectionChange(threadId)
    const active = ["running", "interrupted", "resume_pending"].includes(timeline.thread.status)
    if (active && timeline.thread.last_run_id) {
      const previous = restored.find((item) => item.runId === timeline.thread.last_run_id)
      restored = restored.filter((item) => item.runId !== timeline.thread.last_run_id)
      const rebuilding = emptyTurn(
        timeline.thread.last_run_id,
        previous?.user ?? "",
        previous?.startedAt ?? new Date().toISOString(),
      )
      rebuilding.fileIds = previous?.fileIds ?? []
      restored.push(rebuilding)
      setTurns(restored)
      setBusy(true)
      setStatus("Reconnecting")
      await consume(api.events(rebuilding.runId, 0), rebuilding.runId)
    } else {
      setTurns(restored)
      setBusy(false)
      setStatus("Ready")
    }
  }

  const boot = async () => {
    try {
      const values = await refreshThreads()
      let selected = values.find((item) => item.thread_id === props.config.thread_id)
      if (!selected) {
        selected = await api.createThread()
        setThreads((current) => [selected!, ...current])
      }
      await loadThread(selected.thread_id)
    } catch (cause) {
      setError(message(cause))
      setStatus("Disconnected")
    }
  }

  onMount(() => {
    void boot()
    void pollNotifications()
    const timer = setInterval(() => {
      setSpinner((value) => (value + 1) % SPINNER.length)
      setClock(Date.now())
    }, 120)
    const resized = (width: number) => setTerminalWidth(width)
    const removeSelectionCopy = installSelectionClipboard(
      renderer,
      copyToHostClipboard,
      () => setCopiedUntil(Date.now() + 1400),
    )
    renderer.on("resize", resized)
    renderer.once("destroy", () => {
      alive = false
      clearInterval(timer)
      renderer.off("resize", resized)
      removeSelectionCopy()
    })
  })

  onCleanup(() => {
    alive = false
  })

  const pollNotifications = async () => {
    while (alive) {
      try {
        const page = await api.notifications(notificationCursor)
        notificationCursor = Math.max(notificationCursor, page.next_after_id)
        for (const item of page.notifications) {
          const id = Number(item.id ?? 0)
          const approvals = Array.isArray(item.approvals) ? item.approvals : []
          for (const approval of approvals) {
            if (approval && typeof approval === "object") {
              const approvalId = String((approval as Record<string, unknown>).approval_id ?? "")
              if (approvalId && id > 0) approvalNotifications.set(approvalId, id)
            }
          }
          if (!approvals.length && id > 0) await api.acknowledgeNotification(id)
          const notificationThread = String(item.thread_id ?? "")
          if (notificationThread && notificationThread === thread()?.thread_id && !busy()) {
            await loadThread(notificationThread)
          } else {
            await refreshThreads()
          }
          if (typeof item.text === "string") setError(`Notification: ${item.text}`)
        }
      } catch {
        await Bun.sleep(1500)
      }
    }
  }

  const selectSlashCommand = (item: SlashCommand) => {
    setDismissedSlash(item.value)
    if (item.acceptsArgument) {
      const value = `${item.value} `
      setDraft(value)
      composer?.setText(value)
      composer?.focus()
      return
    }
    void send(item.value)
  }

  useKeyboard((key) => {
    if (key.ctrl && key.name === "d") renderer.destroy()
    if (key.ctrl && key.name === "n") {
      void createSession()
      key.preventDefault()
      return
    }
    if (key.ctrl && key.name === "p") {
      setSessionDialog((value) => !value)
      setSessionIndex(0)
      key.preventDefault()
      return
    }
    if (key.ctrl && key.name === "t") {
      const groupId = lastToolGroupId(activeTurn())
      setExpandedToolsGroup((value) => value === groupId ? "" : groupId)
      key.preventDefault()
      return
    }
    if (key.ctrl && key.name === "c") {
      if (busy()) void cancelActive()
      else renderer.destroy()
      key.preventDefault()
      return
    }
    if (codexLogin()) {
      if (key.name === "escape") void closeCodexLogin()
      key.preventDefault()
      return
    }
    if (picker()) {
      if (key.name === "escape") setPicker(undefined)
      else if (key.name === "down") {
        setPickerIndex((value) => Math.min(filteredPicker().length - 1, value + 1))
      } else if (key.name === "up") {
        setPickerIndex((value) => Math.max(0, value - 1))
      } else if (key.name === "return") {
        filteredPicker()[pickerIndex()]?.select()
      } else return
      key.preventDefault()
      return
    }
    const availableSlashCommands = slashCommands()
    if (availableSlashCommands.length) {
      if (key.name === "escape") setDismissedSlash(draft())
      else if (key.name === "down") {
        setSlashIndex((value) => Math.min(availableSlashCommands.length - 1, value + 1))
      } else if (key.name === "up") {
        setSlashIndex((value) => Math.max(0, value - 1))
      } else if (key.name === "return") {
        const selected = availableSlashCommands[slashIndex()]
        if (selected) selectSlashCommand(selected)
      } else return
      key.preventDefault()
      return
    }
    const approval = pendingApproval()
    if (approval && !busy() && !modalOpen()) {
      const decision = key.name === "a" ? "approve" : key.name === "e" ? "edit" : key.name === "r" ? "reject" : undefined
      if (decision && approval.allowed_decisions.includes(decision)) {
        void decide(approval, decision)
        key.preventDefault()
        return
      }
    }
    if (!sessionDialog()) return
    if (key.name === "escape") {
      setSessionDialog(false)
      key.preventDefault()
    } else if (key.name === "down") {
      setSessionIndex((value) => Math.min(filteredThreads().length - 1, value + 1))
      key.preventDefault()
    } else if (key.name === "up") {
      setSessionIndex((value) => Math.max(0, value - 1))
      key.preventDefault()
    } else if (key.name === "return") {
      const selected = filteredThreads()[sessionIndex()]
      if (selected) void loadThread(selected.thread_id)
      key.preventDefault()
    }
  })

  const consume = async (events: AsyncGenerator<AgentEvent>, runId: string): Promise<boolean> => {
    try {
      for await (const event of events) {
        setTurns((current) =>
          current.map((turn) => (turn.runId === runId ? reduceEvent(turn, event) : turn)),
        )
        if (event.type === "approval.required") {
          setBusy(false)
          setStatus("Approval required")
        } else if (event.type === "run.completed") {
          setBusy(false)
          setStatus("Ready")
        } else if (event.type === "run.failed") {
          setBusy(false)
          setStatus("Failed")
        } else {
          setBusy(true)
          setStatus("Working")
        }
      }
      await refreshThreads()
      await refreshRepository()
      return true
    } catch (cause) {
      setError(message(cause))
      setBusy(false)
      setStatus(cause instanceof ApiError && cause.status ? "Failed" : "Disconnected")
      return false
    }
  }

  const send = async (raw: string) => {
    const text = raw.trim()
    if (!text || busy() || uploading()) return
    if (editApproval()) {
      await submitEditedApproval(text)
      return
    }
    if (text.startsWith("/") && text !== "/regenerate") {
      try {
        if (await command(text)) clearComposer()
      } catch (cause) {
        setStatus("Failed")
        setError(message(cause))
      }
      return
    }
    const selected = thread()
    if (!selected) return
    setError("")
    setUploading(attachments().length > 0)
    const fileIds: string[] = []
    try {
      for (const path of attachments()) fileIds.push(await api.upload(path))
    } catch (cause) {
      setUploading(false)
      setError(`Attachment upload failed: ${message(cause)}`)
      return
    }
    setUploading(false)
    const pending = emptyTurn("", text)
    pending.fileIds = fileIds
    setTurns((current) => [...current, pending])
    clearComposer()
    setAttachments([])
    setBusy(true)
    setStatus("Thinking")
    try {
      const stream = api.run(selected.thread_id, text, fileIds)
      const first = await stream.next()
      if (first.done) throw new ApiError("The agent run returned no events.")
      pending.runId = first.value.run_id
      let current = reduceEvent(pending, first.value)
      setTurns((values) => [...values.slice(0, -1), current])
      for await (const event of stream) {
        current = reduceEvent(current, event)
        setTurns((values) => [...values.slice(0, -1), current])
        if (event.type === "approval.required") {
          setBusy(false)
          setStatus("Approval required")
        }
      }
      if (current.status === "running") {
        await consume(api.events(current.runId, current.lastSequence), current.runId)
      } else {
        setBusy(false)
        setStatus(current.status === "failed" ? "Failed" : current.status === "approval" ? "Approval required" : "Ready")
        await refreshThreads()
        await refreshRepository()
      }
    } catch (cause) {
      setBusy(false)
      setStatus("Failed")
      setError(message(cause))
    }
  }

  const applyInferenceSelection = async (selection: InferenceSelection | null) => {
    const selectedThread = thread()
    const current = inference()
    if (!selectedThread || !current || busy()) return
    setPicker(undefined)
    setError("")
    setStatus("Updating model")
    try {
      const updated = await api.updateThreadInference(
        selectedThread.thread_id,
        current.revision,
        selection,
      )
      setInference(updated)
      setStatus("Ready")
    } catch (cause) {
      setStatus("Failed")
      setError(message(cause))
      try {
        setInference(await api.threadInference(selectedThread.thread_id))
      } catch {}
    }
  }

  const chooseModel = async (
    provider?: InferenceProvider,
    requestedModel?: string,
    fallbackToApi = false,
  ) => {
    const current = inference()
    if (!current) return
    const targetProvider = provider ?? current.effective.provider
    const statusValue = await api.inferenceStatus()
    if (targetProvider === "codex" && !statusValue.codex.connected) {
      await startCodexLogin()
      return
    }
    const models = await api.inferenceModels(targetProvider)
    if (requestedModel) {
      const selected = models.find((item) => item.id === requestedModel)
      if (!selected && targetProvider === "codex") {
        throw new ApiError(`${requestedModel} is not available from ${targetProvider}.`)
      }
      await applyInferenceSelection({
        provider: targetProvider,
        model: selected?.id ?? requestedModel,
        reasoning_effort: selected?.default_reasoning_effort ?? null,
        service_tier: selected?.default_service_tier ?? null,
        fallback_to_api: targetProvider === "codex" && fallbackToApi,
      })
      return
    }
    let visibleModels: InferenceModel[] = models
    if (provider === undefined) {
      const apiModels = targetProvider === "api" ? models : await api.inferenceModels("api")
      const codexModels = statusValue.codex.connected
        ? targetProvider === "codex" ? models : await api.inferenceModels("codex")
        : []
      visibleModels = [...apiModels, ...codexModels]
    }
    const items: PickerItem[] = [
      {
        id: "default",
        label: "Server default",
        detail: `${statusValue.api_default.model} · ${statusValue.api_default.reasoning_effort ?? "provider default"}`,
        select: () => void applyInferenceSelection(null),
      },
      ...visibleModels.map((model) => ({
        id: `${model.provider}:${model.id}`,
        label: model.id,
        detail: `${model.provider}${model.reasoning_efforts.length ? ` · ${model.reasoning_efforts.join(" / ")}` : ""}${model.service_tiers.length ? " · normal / fast" : ""}`,
        select: () => void applyInferenceSelection({
          provider: model.provider,
          model: model.id,
          reasoning_effort: model.default_reasoning_effort,
          service_tier: model.default_service_tier,
          fallback_to_api: false,
        }),
      })),
    ]
    if (provider === undefined && !statusValue.codex.connected) {
      items.push({
        id: "connect-codex",
        label: "Connect ChatGPT Codex",
        detail: "Use a ChatGPT subscription for this conversation",
        select: () => void startCodexLogin(),
      })
    }
    setPicker({ title: provider ? `${targetProvider === "codex" ? "Codex" : "API"} models` : "Models", hint: "Search providers and models", items })
    setPickerQuery("")
    setPickerIndex(0)
  }

  const chooseSpeed = async (requested?: string) => {
    const current = inference()
    if (!current) return
    const selection = current.effective
    if (selection.provider !== "codex") {
      throw new ApiError("Normal/Fast selection is available for Codex models.")
    }
    const models = await api.inferenceModels("codex")
    const model = models.find((item) => item.id === selection.model)
    if (!model) throw new ApiError(`${selection.model} is not available from Codex.`)
    const apply = (serviceTier: string | null) => void applyInferenceSelection(
      { ...selection, service_tier: serviceTier },
    )
    if (requested !== undefined) {
      const normalized = requested.toLocaleLowerCase()
      if (normalized === "normal" || normalized === "default") {
        apply(null)
        return
      }
      const tier = model.service_tiers.find(
        (item) => item.id.toLocaleLowerCase() === normalized
          || item.name.toLocaleLowerCase() === normalized,
      )
      if (!tier) throw new ApiError(`${requested} is not supported by ${selection.model}.`)
      apply(tier.id)
      return
    }
    const items: PickerItem[] = [
      {
        id: "normal",
        label: "Normal",
        detail: selection.service_tier === null ? "Current" : "Standard usage and latency",
        select: () => apply(null),
      },
      ...model.service_tiers.map((tier) => ({
        id: tier.id,
        label: tier.name,
        detail: selection.service_tier === tier.id ? "Current" : tier.description || selection.model,
        select: () => apply(tier.id),
      })),
    ]
    setPicker({
      title: `Speed · ${selection.model}`,
      hint: model.service_tiers.length
        ? "Choose normal or fast"
        : "This model supports normal speed only",
      items,
    })
    setPickerQuery("")
    setPickerIndex(0)
  }

  const chooseReasoning = async (requested?: string) => {
    const current = inference()
    if (!current) return
    const selection = current.effective
    const models = await api.inferenceModels(selection.provider)
    const model = models.find((item) => item.id === selection.model)
    const efforts = model?.reasoning_efforts ?? []
    const apply = (effort: string | null) => void applyInferenceSelection({
      ...selection,
      reasoning_effort: effort,
    })
    if (requested !== undefined) {
      const effort = requested === "default" ? null : requested
      if (effort && efforts.length && !efforts.includes(effort)) {
        throw new ApiError(`${effort} is not supported by ${selection.model}.`)
      }
      apply(effort)
      return
    }
    const items: PickerItem[] = [
      { id: "default", label: "Provider default", detail: "Do not force an effort", select: () => apply(null) },
      ...efforts.map((effort) => ({
        id: effort,
        label: effort,
        detail: effort === selection.reasoning_effort ? "Current" : selection.model,
        select: () => apply(effort),
      })),
    ]
    setPicker({ title: `Reasoning · ${selection.model}`, hint: "Search efforts", items })
    setPickerQuery("")
    setPickerIndex(0)
  }

  const startCodexLogin = async () => {
    setPicker(undefined)
    setError("")
    setStatus("Starting Codex login")
    try {
      const login = await api.startCodexLogin()
      setCodexLogin(login)
      setStatus("Codex login")
      void pollCodexLogin(login.login_id)
    } catch (cause) {
      setStatus("Failed")
      setError(message(cause))
    }
  }

  const pollCodexLogin = async (loginId: string) => {
    while (codexLogin()?.login_id === loginId && codexLogin()?.status === "pending") {
      await Bun.sleep(Math.max(1000, (codexLogin()?.interval_seconds ?? 2) * 1000))
      try {
        const next = await api.codexLogin(loginId)
        if (codexLogin()?.login_id !== loginId) return
        setCodexLogin(next)
        if (next.status === "authorized") {
          setStatus("Ready")
          await Bun.sleep(500)
          setCodexLogin(undefined)
          await chooseModel("codex")
          return
        }
        if (next.status !== "pending") {
          setStatus("Failed")
          return
        }
      } catch (cause) {
        setError(message(cause))
        return
      }
    }
  }

  const closeCodexLogin = async () => {
    const login = codexLogin()
    setCodexLogin(undefined)
    if (login?.status === "pending") {
      try {
        await api.cancelCodexLogin(login.login_id)
      } catch {}
    }
    setStatus("Ready")
  }

  const command = async (input: string): Promise<boolean> => {
    const [name, ...parts] = input.split(/\s+/)
    if (name === "/quit") renderer.destroy()
    else if (name === "/help") {
      setError("/new  /sessions  /model  /reasoning  /speed  /repo  /repos  /login codex  /logout codex  /regenerate  /attach  /cancel  /quit")
    } else if (name === "/sessions") {
      setSessionDialog(true)
      setSessionQuery("")
    } else if (name === "/new") {
      await createSession(parts.join(" ") || undefined)
    } else if (name === "/session") {
      const selector = parts.join(" ").toLocaleLowerCase()
      const index = Number.parseInt(selector, 10)
      const selected = Number.isFinite(index)
        ? threads()[index - 1]
        : threads().find((item) => item.title.toLocaleLowerCase() === selector)
      if (!selected) setError("Unknown session. Use /sessions to choose one.")
      else await loadThread(selected.thread_id)
    } else if (name === "/attach") {
      const files = droppedFiles(parts.join(" "))
      if (!files.length) setError("Usage: /attach PATH")
      else setAttachments((current) => [...new Set([...current, ...files])])
    } else if (name === "/tool" || name === "/tools") {
      const groupId = lastToolGroupId(activeTurn())
      setExpandedToolsGroup((value) => value === groupId ? "" : groupId)
    } else if (name === "/model") {
      if (parts[0] === "default") await applyInferenceSelection(null)
      else {
        const explicitProvider = parts[0] === "api" || parts[0] === "codex" ? parts.shift() as InferenceProvider : undefined
        const model = parts.shift()
        const fallback = parts.includes("fallback") || parts.includes("--fallback")
        await chooseModel(explicitProvider, model, fallback)
      }
    } else if (name === "/reasoning") await chooseReasoning(parts[0])
    else if (name === "/speed") await chooseSpeed(parts[0])
    else if (name === "/repo") {
      const selectedThread = thread()
      if (!selectedThread) return true
      if (parts[0] === "close") {
        const active = repository()
        if (!active) setError("No repository workspace is active.")
        else {
          await api.closeRepository(selectedThread.thread_id, active.id)
          setRepository(undefined)
          setError("Repository workspace stopped.")
        }
      } else if (parts[0] === "status" || !parts.length) {
        const active = repository()
        if (!active) setError("No repository is active. Use /repo open https://github.com/owner/repo")
        else {
          const current = await api.repositoryStatus(selectedThread.thread_id, active.id)
          setRepository(current)
          setError(`${current.repository_url.replace(/\.git$/, "")} · ${current.branch} · ${current.clean ? "clean" : `${current.changes.length} changed`} · ${current.commits_ahead} commits ahead`)
        }
      } else {
        const values = parts[0] === "open" ? parts.slice(1) : parts
        const repositoryUrl = values[0]
        if (!repositoryUrl) setError("Usage: /repo open https://github.com/owner/repo [base]")
        else {
          setStatus("Opening repository")
          const opened = await api.openRepository(
            selectedThread.thread_id,
            repositoryUrl,
            values[1] ?? "main",
          )
          setRepository(opened)
          setStatus("Ready")
          setError(`Repository ready: ${opened.branch}`)
        }
      }
    } else if (name === "/repos") {
      const workspaces = await api.repositories()
      setError(
        workspaces.length
          ? workspaces.map((item) => `${item.status} · ${item.branch} · ${item.repository_url.replace(/\.git$/, "")}`).join("\n")
          : "No repository workspaces.",
      )
    }
    else if (name === "/login" && parts[0] === "codex") await startCodexLogin()
    else if (name === "/logout" && parts[0] === "codex") {
      try {
        const result = await api.logoutCodex(parts.includes("confirm"))
        if (result.reset_threads && thread()) setInference(await api.threadInference(thread()!.thread_id))
        setError(result.disconnected ? "Codex signed out." : "Codex was not connected.")
      } catch (cause) {
        setError(`${message(cause)} Use /logout codex confirm to reset Codex conversations.`)
      }
    } else if (name === "/cancel") await cancelActive()
    else if (name === "/logs") {
      const logs = await api.logs()
      setError(logs.slice(-8).map((item) => `${item.stream ?? "log"} ${item.text ?? ""}`).join("\n"))
    } else return false
    return true
  }

  const createSession = async (name?: string) => {
    if (busy()) {
      setError("Cancel the current run before starting a new session.")
      return
    }
    setSessionDialog(false)
    setError("")
    setStatus("Creating session")
    try {
      const created = await api.createThread(name)
      await refreshThreads()
      await loadThread(created.thread_id)
    } catch (cause) {
      setStatus("Failed")
      setError(message(cause))
    }
  }

  const cancelActive = async () => {
    const current = activeTurn()
    if (!current || !busy()) return
    try {
      await api.cancel(current.runId)
      setBusy(false)
      setStatus("Cancelled")
    } catch (cause) {
      setError(message(cause))
    }
  }

  const decide = async (approval: Approval, decision: "approve" | "edit" | "reject") => {
    if (busy()) return
    if (decision === "edit") {
      consumeVisibleApproval(approval)
      setEditApproval(approval)
      const value = compactJson(approval.arguments, 20_000)
      setDraft(value)
      setTimeout(() => {
        composer?.setText(value)
        composer?.focus()
      }, 0)
      return
    }
    consumeVisibleApproval(approval)
    setBusy(true)
    setStatus(decision === "approve" ? "Approving" : "Rejecting")
    if (await consume(api.resume(approval.run_id, approval.approval_id, decision), approval.run_id)) {
      await acknowledgeApproval(approval.approval_id)
    } else {
      await recoverCurrentThread()
    }
  }

  const submitEditedApproval = async (value: string) => {
    const approval = editApproval()
    if (!approval) return
    let argumentsValue: Record<string, unknown>
    try {
      const parsed = JSON.parse(value) as unknown
      if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) throw new Error()
      argumentsValue = parsed as Record<string, unknown>
    } catch {
      setError("Edited approval arguments must be a JSON object.")
      return
    }
    setEditApproval(undefined)
    clearComposer()
    consumeVisibleApproval(approval)
    setBusy(true)
    setStatus("Applying edited action")
    if (
      await consume(
        api.resume(approval.run_id, approval.approval_id, "edit", argumentsValue),
        approval.run_id,
      )
    ) {
      await acknowledgeApproval(approval.approval_id)
    } else {
      await recoverCurrentThread()
    }
  }

  const consumeVisibleApproval = (approval: Approval) => {
    setTurns((current) =>
      current.map((turn) =>
        turn.runId === approval.run_id
          ? consumeApproval(turn, approval.approval_id)
          : turn,
      ),
    )
  }

  const recoverCurrentThread = async () => {
    const current = thread()
    if (!current) return
    try {
      await loadThread(current.thread_id)
    } catch (cause) {
      setError(message(cause))
      setBusy(false)
      setStatus(cause instanceof ApiError && cause.status ? "Failed" : "Disconnected")
    }
  }

  const acknowledgeApproval = async (approvalId: string) => {
    const notificationId = approvalNotifications.get(approvalId)
    if (!notificationId) return
    approvalNotifications.delete(approvalId)
    try {
      await api.acknowledgeNotification(notificationId)
    } catch {}
  }

  const clearComposer = () => {
    setDraft("")
    composer?.setText("")
  }

  const pasted = (value: string) => {
    const files = droppedFiles(value)
    if (!files.length) return false
    setAttachments((current) => [...new Set([...current, ...files])])
    return true
  }

  return (
    <box width="100%" height="100%" flexDirection="column" backgroundColor={COLORS.bg}>
      <Show when={clock() < copiedUntil()}>
        <CopyFlash width={terminalWidth()} />
      </Show>
      <HeaderBar title={thread()?.title ?? "Connecting"} status={status()} width={terminalWidth()} />
      <box flexGrow={1} minHeight={0} flexDirection="column" paddingLeft={2} paddingRight={2} paddingBottom={1} gap={1}>
        <scrollbox
          flexGrow={1}
          minHeight={0}
          stickyScroll
          stickyStart="bottom"
          style={{
            rootOptions: { backgroundColor: COLORS.bg },
            viewportOptions: { backgroundColor: COLORS.bg },
            contentOptions: { backgroundColor: COLORS.bg, paddingTop: 1, paddingBottom: 1 },
            scrollbarOptions: { visible: false },
          }}
        >
          <Show when={!turns().length && status() === "Ready"}>
            <box paddingTop={2} paddingLeft={3}>
              <text fg={COLORS.muted}>Tell me what you want to build, connect, automate, or improve.</text>
            </box>
          </Show>
          <TurnList
            turns={turns()}
            busy={busy()}
            activeRunId={activeTurn()?.runId ?? ""}
            spinner={SPINNER[spinner()]}
            now={clock()}
            expandedToolsGroup={expandedToolsGroup()}
            onToggleTools={(groupId) => setExpandedToolsGroup((value) => value === groupId ? "" : groupId)}
          />
        </scrollbox>
        <Show when={error()}>
          <box flexShrink={0} maxHeight={4} overflow="hidden" paddingLeft={2} paddingRight={2}>
            <text fg={COLORS.amber} wrapMode="word">{error()}</text>
          </box>
        </Show>
        <Show when={!editApproval() ? pendingApproval() : undefined}>
          {(approval) => <ApprovalView approval={approval()} onDecision={decide} />}
        </Show>
        <Show when={slashCommands().length}>
          <SlashCommandPalette
            items={slashCommands()}
            selected={slashIndex()}
            onSelect={selectSlashCommand}
          />
        </Show>
        <Show when={attachments().length}>
          <box flexShrink={0} flexDirection="row" flexWrap="wrap" gap={1}>
            <For each={attachments()}>
              {(path) => (
                <box backgroundColor={COLORS.raised} paddingLeft={1} paddingRight={1} onMouseUp={() => setAttachments((items) => items.filter((item) => item !== path))}>
                  <text fg={COLORS.text}>{basename(path)} <span style={{ fg: COLORS.muted }}>×</span></text>
                </box>
              )}
            </For>
          </box>
        </Show>
        <box flexShrink={0} flexDirection="column" backgroundColor={COLORS.panel}>
          <Show when={!pendingApproval() || editApproval()}>
            <box
              flexShrink={0}
              paddingLeft={2}
              paddingRight={2}
              paddingTop={1}
              paddingBottom={1}
              backgroundColor={COLORS.raised}
              flexDirection="row"
              gap={2}
              alignItems="flex-start"
            >
              <text flexShrink={0} fg={COLORS.blue} attributes={TextAttributes.BOLD}>
                {busy() ? LOADING_CELLS[spinner() % LOADING_CELLS.length] : ">"}
              </text>
              <box flexGrow={1}>
                <textarea
                  ref={(value) => (composer = value)}
                  focused={!modalOpen()}
                  minHeight={1}
                  maxHeight={10}
                  wrapMode="word"
                  placeholder={editApproval() ? "Edit the JSON arguments, then press Enter" : "Ask OpenTulpa anything"}
                  placeholderColor={COLORS.muted}
                  textColor={COLORS.text}
                  focusedTextColor={COLORS.text}
                  backgroundColor={COLORS.raised}
                  cursorColor={COLORS.blue}
                  keyBindings={[
                    { name: "return", action: "submit" },
                    { name: "return", shift: true, action: "newline" },
                  ]}
                  onContentChange={() => {
                    setDraft(composer?.plainText ?? "")
                    setSlashIndex(0)
                  }}
                  onSubmit={() => void send(draft())}
                  onPaste={(event) => {
                    const value = decodePasteBytes(event.bytes)
                    if (pasted(value)) event.preventDefault()
                  }}
                />
              </box>
            </box>
          </Show>
          <StatusBar
            busy={busy()}
            uploading={uploading()}
            status={status()}
            width={terminalWidth()}
            onNewSession={() => void createSession()}
            onSessions={() => { setSessionDialog(true); setSessionQuery(""); setSessionIndex(0) }}
            inference={inference()?.effective}
            repository={repository()}
          />
        </box>
      </box>
      <Show when={sessionDialog()}>
        <SessionDialog
          threads={filteredThreads()}
          selected={sessionIndex()}
          query={sessionQuery()}
          onQuery={(value) => { setSessionQuery(value); setSessionIndex(0) }}
          onSelect={(value) => void loadThread(value.thread_id)}
          onNew={() => void createSession()}
          onClose={() => setSessionDialog(false)}
        />
      </Show>
      <Show when={picker()}>
        {(value) => (
          <PickerDialog
            state={value()}
            items={filteredPicker()}
            selected={pickerIndex()}
            query={pickerQuery()}
            onQuery={(query) => { setPickerQuery(query); setPickerIndex(0) }}
            onClose={() => setPicker(undefined)}
          />
        )}
      </Show>
      <Show when={codexLogin()}>
        {(login) => (
          <CodexLoginDialog
            login={login()}
            onOpen={() => openExternal(login().verification_url)}
            onCopy={() => {
              renderer.copyToClipboardOSC52(login().user_code)
              void copyToHostClipboard(login().user_code)
              setCopiedUntil(Date.now() + 1400)
            }}
            onClose={() => void closeCodexLogin()}
            onRetry={() => void startCodexLogin()}
          />
        )}
      </Show>
    </box>
  )
}

export function HeaderBar(props: { title: string; status: string; width: number }) {
  return (
    <box
      flexShrink={0}
      width="100%"
      paddingTop={1}
      paddingBottom={1}
      paddingLeft={2}
      paddingRight={2}
      flexDirection="row"
      alignItems="center"
    >
      <text flexShrink={0} fg={COLORS.blue} attributes={TextAttributes.BOLD}>OpenTulpa</text>
      <text flexGrow={1} fg={COLORS.text} attributes={TextAttributes.BOLD} wrapMode="none" truncate>: {props.title}</text>
      <Show when={props.width >= 76}>
        <text flexShrink={0} fg={COLORS.dim}>{props.status.toLocaleLowerCase()}</text>
      </Show>
    </box>
  )
}

export function TurnList(props: {
  turns: Turn[]
  busy: boolean
  activeRunId: string
  spinner: string
  now: number
  expandedToolsGroup: string
  onToggleTools: (groupId: string) => void
}) {
  const view = (turn: () => Turn) => (
    <TurnView
      turn={turn()}
      active={props.busy && turn().runId === props.activeRunId}
      spinner={props.spinner}
      now={props.now}
      expandedToolsGroup={props.expandedToolsGroup}
      onToggleTools={props.onToggleTools}
    />
  )
  return (
    <Index each={props.turns}>
      {(turn) => (
        <Show when={turn().runId} keyed fallback={view(turn)}>
          {(_runId) => view(turn)}
        </Show>
      )}
    </Index>
  )
}

export function TurnView(props: {
  turn: Turn
  active: boolean
  spinner: string
  now: number
  expandedToolsGroup: string
  onToggleTools: (groupId: string) => void
}) {
  const hasActiveTool = createMemo(() => props.turn.tools.some((tool) => !tool.completedAt))
  const groups = createMemo(() => turnGroups(props.turn))
  const waiting = createMemo(() => {
    const last = groups().at(-1)
    return props.active && !hasActiveTool() && (!last || last.type === "tools")
  })
  return (
    <box flexDirection="column" flexShrink={0}>
      <Show when={props.turn.user}>
        <box
          border={["left"]}
          customBorderChars={SPLIT}
          borderColor={COLORS.blue}
          marginTop={1}
          marginBottom={1}
          flexShrink={0}
        >
          <box
            paddingTop={1}
            paddingBottom={1}
            paddingLeft={2}
            paddingRight={1}
            backgroundColor={COLORS.panel}
            flexShrink={0}
          >
            <text fg={COLORS.text} wrapMode="word">{props.turn.user}</text>
          </box>
        </box>
      </Show>
      <Show when={props.turn.assistant || props.turn.tools.length || props.turn.status === "running"}>
        <Index each={groups()}>
          {(group, index) => {
            if (group().type === "assistant") {
              const assistantText = () => {
                const value = group()
                return value.type === "assistant" ? value.text : ""
              }
              return (
                <box paddingLeft={3} marginTop={1} flexShrink={0}>
                  <markdown
                    id={`assistant-${props.turn.runId}-${index}`}
                    content={assistantText()}
                    syntaxStyle={MARKDOWN_STYLE}
                    streaming
                    internalBlockMode="top-level"
                    conceal
                    fg={COLORS.text}
                  />
                </box>
              )
            }
            const toolGroup = () => {
              const value = group()
              return value.type === "tools" ? value : { type: "tools" as const, id: "", tools: [] }
            }
            const groupId = () => toolGroupId(props.turn.runId, toolGroup().id)
            return (
              <ToolActivity
                tools={toolGroup().tools}
                running={props.active && props.turn.status === "running"}
                spinner={props.spinner}
                now={props.now}
                expanded={props.expandedToolsGroup === groupId()}
                onToggle={() => props.onToggleTools(groupId())}
              />
            )
          }}
        </Index>
        <Show when={waiting()}>
          <box paddingLeft={3} marginTop={1} flexDirection="row" gap={1} flexShrink={0}>
            <text fg={COLORS.blue}>{props.spinner}</text>
            <text fg={COLORS.muted}>{waitingLabel(props.turn.startedAt, props.now)}</text>
            <text fg={COLORS.dim}>{elapsed(props.turn.startedAt, props.now)}</text>
          </box>
        </Show>
      </Show>
      <For each={props.turn.artifacts}>
        {(artifact) => <text fg={COLORS.green}>  artifact  {String(artifact.name ?? artifact.id ?? "ready")}</text>}
      </For>
      <Show when={props.turn.error}><box paddingLeft={3}><text fg={COLORS.red}>{props.turn.error}</text></box></Show>
    </box>
  )
}

export function ToolActivity(props: {
  tools: ToolCall[]
  running?: boolean
  spinner?: string
  now?: number
  expanded: boolean
  onToggle: () => void
}) {
  const renderer = useRenderer()
  const [selected, setSelected] = createSignal("")
  const active = createMemo(() => props.running ? props.tools.filter((tool) => !tool.completedAt) : [])
  const completed = createMemo(() => props.tools.filter((tool) => tool.completedAt))
  const latest = createMemo(() => active().at(-1))
  const click = (action: () => void) => {
    if (renderer.getSelection()?.getSelectedText()) return
    renderer.clearSelection()
    action()
  }
  const toggleTool = (tool: ToolCall) => {
    click(() => setSelected((value) => value === tool.callId ? "" : tool.callId))
  }
  return (
    <Show when={props.tools.length}>
      <box flexDirection="column" marginTop={1} paddingLeft={3} flexShrink={0}>
        <Show when={latest()}>
          {(tool) => (
            <box flexDirection="column">
              <box flexDirection="row" gap={1} onMouseUp={() => toggleTool(tool())}>
                <text flexShrink={0} fg={COLORS.blue}>{props.spinner ?? "⠋"}</text>
                <text flexShrink={0} fg={COLORS.dim}>{selected() === tool().callId ? "▾" : "▸"}</text>
                <text flexGrow={1} fg={COLORS.muted} wrapMode="none" truncate>{activityLabel(tool())}</text>
                <Show when={active().length > 1}>
                  <text flexShrink={0} fg={COLORS.muted}>+{active().length - 1}</text>
                </Show>
                <text flexShrink={0} fg={COLORS.dim}>{elapsed(tool().startedAt, props.now ?? Date.now())}</text>
              </box>
              <Show when={selected() === tool().callId}>
                <ToolDetails tool={tool()} />
              </Show>
            </box>
          )}
        </Show>
        <Show when={completed().length === 1}>
          {(present) => {
            const tool = () => present() ? completed()[0] : undefined
            return (
              <Show when={tool()}>
                {(item) => (
                  <box flexDirection="column">
                    <box flexDirection="row" gap={1} onMouseUp={() => toggleTool(item())}>
                      <text flexShrink={0} fg={item().ok === false ? COLORS.red : COLORS.muted}>{item().ok === false ? "×" : "→"}</text>
                      <text flexShrink={0} fg={COLORS.dim}>{selected() === item().callId ? "▾" : "▸"}</text>
                      <text flexGrow={1} fg={selected() === item().callId ? COLORS.text : COLORS.muted} wrapMode="none" truncate>{activityLabel(item())}</text>
                      <text flexShrink={0} fg={COLORS.dim}>{duration(item())}</text>
                    </box>
                    <Show when={selected() === item().callId}>
                      <ToolDetails tool={item()} />
                    </Show>
                  </box>
                )}
              </Show>
            )
          }}
        </Show>
        <Show when={completed().length > 1}>
          <box onMouseUp={() => click(props.onToggle)} flexDirection="row" gap={1}>
            <text flexShrink={0} fg={COLORS.muted}>→</text>
            <text flexShrink={0} fg={COLORS.dim}>{props.expanded ? "▾" : "▸"}</text>
            <text fg={COLORS.muted}>{completed().length} tools completed</text>
          </box>
        </Show>
        <Show when={props.expanded}>
          <For each={completed()}>
            {(tool) => (
              <box flexDirection="column">
                <box flexDirection="row" gap={1} onMouseUp={() => toggleTool(tool)}>
                  <text flexShrink={0} fg={tool.ok === false ? COLORS.red : COLORS.muted}>{tool.ok === false ? "×" : "→"}</text>
                  <text flexShrink={0} fg={COLORS.dim}>{selected() === tool.callId ? "▾" : "▸"}</text>
                  <text flexGrow={1} fg={selected() === tool.callId ? COLORS.text : COLORS.muted} wrapMode="none" truncate>{activityLabel(tool)}</text>
                  <text flexShrink={0} fg={COLORS.dim}>{duration(tool)}</text>
                </box>
                <Show when={selected() === tool.callId}>
                  <ToolDetails tool={tool} />
                </Show>
              </box>
            )}
          </For>
        </Show>
      </box>
    </Show>
  )
}

function ToolDetails(props: { tool: ToolCall }) {
  return (
    <box
      border={["left"]}
      customBorderChars={SPLIT}
      borderColor={props.tool.ok === false ? COLORS.red : COLORS.dim}
      backgroundColor={COLORS.panel}
      paddingLeft={2}
      paddingRight={1}
      paddingTop={1}
      paddingBottom={1}
      marginTop={1}
      flexDirection="column"
    >
      <text fg={COLORS.blue} attributes={TextAttributes.BOLD}>Request</text>
      <text fg={COLORS.muted} wrapMode="word">{compactJson(props.tool.arguments)}</text>
      <Show when={props.tool.completedAt}>
        <text fg={props.tool.ok === false ? COLORS.red : COLORS.green} attributes={TextAttributes.BOLD}>
          {props.tool.ok === false ? "Error" : "Response"}
        </text>
        <text fg={props.tool.ok === false ? COLORS.red : COLORS.muted} wrapMode="word">
          {compactJson(props.tool.ok === false ? props.tool.error : props.tool.result)}
        </text>
      </Show>
    </box>
  )
}

export function ApprovalView(props: { approval: Approval; onDecision: (approval: Approval, decision: "approve" | "edit" | "reject") => void }) {
  const summary = createMemo(() => approvalSummary(props.approval))
  return (
    <box
      flexShrink={0}
      maxHeight={10}
      overflow="hidden"
      border={["left"]}
      customBorderChars={SPLIT}
      borderColor={COLORS.amber}
      backgroundColor={COLORS.panel}
      paddingLeft={2}
      paddingRight={2}
      paddingTop={1}
      paddingBottom={1}
      flexDirection="column"
    >
      <text fg={COLORS.amber} attributes={TextAttributes.BOLD}>Approval required</text>
      <text fg={COLORS.text} wrapMode="none" truncate>{props.approval.tool_name.replaceAll("_", " ")}</text>
      <Show
        when={summary().length}
        fallback={(
          <Show when={props.approval.description}>
            <text fg={COLORS.muted} wrapMode="none" truncate>{props.approval.description}</text>
          </Show>
        )}
      >
        <For each={summary()}>
          {(line) => <text fg={COLORS.muted} wrapMode="none" truncate>{line}</text>}
        </For>
      </Show>
      <box flexDirection="row" gap={1} marginTop={1}>
        <Show when={props.approval.allowed_decisions.includes("approve")}>
          <box backgroundColor="#194B36" paddingLeft={1} paddingRight={1} onMouseUp={() => props.onDecision(props.approval, "approve")}><text fg={COLORS.text}>a  Approve</text></box>
        </Show>
        <Show when={props.approval.allowed_decisions.includes("edit")}>
          <box backgroundColor="#3D3520" paddingLeft={1} paddingRight={1} onMouseUp={() => props.onDecision(props.approval, "edit")}><text fg={COLORS.text}>e  Edit</text></box>
        </Show>
        <Show when={props.approval.allowed_decisions.includes("reject")}>
          <box backgroundColor="#54252B" paddingLeft={1} paddingRight={1} onMouseUp={() => props.onDecision(props.approval, "reject")}><text fg={COLORS.text}>r  Reject</text></box>
        </Show>
      </box>
    </box>
  )
}

export function SlashCommandPalette(props: {
  items: SlashCommand[]
  selected: number
  onSelect: (item: SlashCommand) => void
}) {
  return (
    <box
      flexShrink={0}
      height={Math.min(17, props.items.length + 2)}
      border={["left"]}
      customBorderChars={SPLIT}
      borderColor={COLORS.blue}
      backgroundColor={COLORS.panel}
      paddingLeft={1}
      paddingRight={1}
      flexDirection="column"
    >
      <text fg={COLORS.text} attributes={TextAttributes.BOLD}>
        Commands <span style={{ fg: COLORS.dim }}>up/down select · enter run · esc close</span>
      </text>
      <For each={props.items}>
        {(item, index) => (
          <box
            height={1}
            flexShrink={0}
            flexDirection="row"
            backgroundColor={index() === props.selected ? COLORS.selected : COLORS.panel}
            onMouseUp={() => props.onSelect(item)}
          >
            <text width={16} flexShrink={0} fg={index() === props.selected ? COLORS.blue : COLORS.muted} wrapMode="none" truncate>
              {index() === props.selected ? "> " : "  "}{item.value}
            </text>
            <text flexGrow={1} fg={COLORS.dim} wrapMode="none" truncate>{item.description}</text>
          </box>
        )}
      </For>
    </box>
  )
}

export function StatusBar(props: {
  busy: boolean
  uploading: boolean
  status: string
  width: number
  onNewSession?: () => void
  onSessions?: () => void
  inference?: InferenceSelection
  repository?: RepositoryWorkspace
}) {
  const label = createMemo(() => {
    const state = props.uploading ? "Uploading attachments" : props.status
    if (props.busy) return props.uploading ? state : ""
    if (["Failed", "Disconnected", "Cancelled", "Approval required"].includes(state)) return state
    return ""
  })
  return (
    <box height={1} flexShrink={0} paddingLeft={2} paddingRight={2} flexDirection="row" gap={1}>
      <text
        flexGrow={1}
        fg={props.busy ? COLORS.blue : props.status === "Failed" ? COLORS.red : COLORS.muted}
        wrapMode="none"
        truncate
      >
        {label()}
      </text>
      <Show when={props.inference && props.width >= 58}>
        <text
          flexShrink={1}
          maxWidth={Math.max(18, Math.floor(props.width * 0.45))}
          fg={COLORS.dim}
          wrapMode="none"
          truncate
        >
          {props.inference!.provider} · {props.inference!.model} · {props.inference!.reasoning_effort ?? "default"} · {props.inference!.service_tier === "priority" ? "fast" : props.inference!.service_tier ?? "normal"}
        </text>
      </Show>
      <Show when={props.repository && props.width >= 76}>
        <text
          flexShrink={1}
          maxWidth={Math.max(16, Math.floor(props.width * 0.24))}
          fg={COLORS.dim}
          wrapMode="none"
          truncate
        >
          repo · {props.repository!.branch}
        </text>
      </Show>
      <Show when={props.width >= 40}>
        <text flexShrink={0} fg={COLORS.dim} onMouseUp={props.onNewSession}>ctrl+n new</text>
      </Show>
      <Show when={props.width >= 64}>
        <text flexShrink={0} fg={COLORS.dim} onMouseUp={props.onSessions}>ctrl+p sessions</text>
      </Show>
      <Show when={props.width >= 92}>
        <text flexShrink={0} fg={COLORS.dim}>shift+enter newline</text>
      </Show>
    </box>
  )
}

function CopyFlash(props: { width: number }) {
  const panelWidth = () => Math.min(31, Math.max(22, props.width - 4))
  return (
    <box
      position="absolute"
      zIndex={500}
      top={1}
      left={Math.max(1, Math.floor((props.width - panelWidth()) / 2))}
      width={panelWidth()}
      height={3}
      alignItems="center"
      justifyContent="center"
      backgroundColor={COLORS.selected}
    >
      <text fg={COLORS.text}>Copied to clipboard</text>
    </box>
  )
}

function PickerDialog(props: {
  state: PickerState
  items: PickerItem[]
  selected: number
  query: string
  onQuery: (value: string) => void
  onClose: () => void
}) {
  return (
    <box
      position="absolute"
      zIndex={120}
      left="12%"
      top="12%"
      width="76%"
      height="70%"
      backgroundColor={COLORS.panel}
      border
      borderColor={COLORS.dim}
      padding={1}
      flexDirection="column"
      gap={1}
    >
      <box flexDirection="row" justifyContent="space-between">
        <text fg={COLORS.text} attributes={TextAttributes.BOLD}>{props.state.title}</text>
        <text fg={COLORS.muted} onMouseUp={props.onClose}>esc</text>
      </box>
      <box backgroundColor={COLORS.raised} paddingLeft={1} paddingRight={1}>
        <input
          focused
          value={props.query}
          placeholder={props.state.hint}
          placeholderColor={COLORS.muted}
          textColor={COLORS.text}
          backgroundColor={COLORS.raised}
          onInput={props.onQuery}
        />
      </box>
      <scrollbox flexGrow={1}>
        <For each={props.items}>
          {(item, index) => (
            <box
              backgroundColor={index() === props.selected ? COLORS.selected : COLORS.panel}
              paddingLeft={1}
              paddingRight={1}
              paddingTop={1}
              paddingBottom={1}
              onMouseUp={item.select}
              flexDirection="column"
            >
              <text fg={index() === props.selected ? COLORS.text : COLORS.muted}>
                {index() === props.selected ? "> " : "  "}{item.label}
              </text>
              <text fg={COLORS.dim}>  {item.detail}</text>
            </box>
          )}
        </For>
      </scrollbox>
      <text fg={COLORS.dim}>up/down navigate  enter select  esc close</text>
    </box>
  )
}

function CodexLoginDialog(props: {
  login: CodexDeviceLogin
  onOpen: () => void
  onCopy: () => void
  onClose: () => void
  onRetry: () => void
}) {
  const pending = () => props.login.status === "pending"
  return (
    <box
      position="absolute"
      zIndex={140}
      left="16%"
      top="18%"
      width="68%"
      backgroundColor={COLORS.panel}
      border
      borderColor={pending() ? COLORS.blue : props.login.status === "authorized" ? COLORS.green : COLORS.red}
      padding={2}
      flexDirection="column"
      gap={1}
    >
      <text fg={COLORS.text} attributes={TextAttributes.BOLD}>Connect ChatGPT Codex</text>
      <Show when={pending()} fallback={
        <text fg={props.login.status === "authorized" ? COLORS.green : COLORS.red}>
          {props.login.status === "authorized" ? "Connected" : "Login expired or failed"}
        </text>
      }>
        <text fg={COLORS.muted}>Open the sign-in page, then enter this one-time code.</text>
        <text fg={COLORS.blue} attributes={TextAttributes.UNDERLINE} onMouseUp={props.onOpen}>
          {props.login.verification_url}
        </text>
        <box backgroundColor={COLORS.raised} paddingLeft={2} paddingRight={2} paddingTop={1} paddingBottom={1} onMouseUp={props.onCopy}>
          <text fg={COLORS.text} attributes={TextAttributes.BOLD}>{props.login.user_code}  <span style={{ fg: COLORS.muted }}>click to copy</span></text>
        </box>
        <text fg={COLORS.muted}>Waiting for authorization...</text>
      </Show>
      <box flexDirection="row" gap={2} marginTop={1}>
        <Show when={!pending() && props.login.status !== "authorized"}>
          <text fg={COLORS.blue} onMouseUp={props.onRetry}>Retry</text>
        </Show>
        <text fg={COLORS.muted} onMouseUp={props.onClose}>{pending() ? "Cancel" : "Close"}</text>
      </box>
    </box>
  )
}

function SessionDialog(props: {
  threads: ThreadSummary[]
  selected: number
  query: string
  onQuery: (value: string) => void
  onSelect: (thread: ThreadSummary) => void
  onNew: () => void
  onClose: () => void
}) {
  return (
    <box
      position="absolute"
      zIndex={100}
      left="10%"
      top="10%"
      width="80%"
      height="80%"
      backgroundColor={COLORS.panel}
      border
      borderColor={COLORS.dim}
      padding={1}
      flexDirection="column"
      gap={1}
    >
      <box flexDirection="row" justifyContent="space-between">
        <text fg={COLORS.text} attributes={TextAttributes.BOLD}>Sessions</text>
        <box flexDirection="row" gap={2}>
          <text fg={COLORS.blue} onMouseUp={props.onNew}>New session</text>
          <text fg={COLORS.muted} onMouseUp={props.onClose}>esc</text>
        </box>
      </box>
      <box backgroundColor={COLORS.raised} paddingLeft={1} paddingRight={1}>
        <input
          focused
          value={props.query}
          placeholder="Search sessions"
          placeholderColor={COLORS.muted}
          textColor={COLORS.text}
          backgroundColor={COLORS.raised}
          onInput={props.onQuery}
        />
      </box>
      <scrollbox flexGrow={1}>
        <For each={props.threads}>
          {(item, index) => (
            <box backgroundColor={index() === props.selected ? COLORS.selected : COLORS.panel} paddingLeft={1} paddingRight={1} paddingTop={1} paddingBottom={1} onMouseUp={() => props.onSelect(item)} flexDirection="column">
              <text fg={index() === props.selected ? COLORS.text : COLORS.muted}>{index() === props.selected ? "> " : "  "}{item.title}  <span style={{ fg: COLORS.dim }}>{item.channel}</span></text>
              <Show when={item.preview}><text fg={COLORS.dim}>{item.preview.slice(0, 100)}</text></Show>
            </box>
          )}
        </For>
      </scrollbox>
      <text fg={COLORS.dim}>up/down navigate  enter open  ctrl+n new  esc close</text>
    </box>
  )
}

function duration(tool: ToolCall): string {
  if (!tool.completedAt) return ""
  const elapsed = Date.parse(tool.completedAt) - Date.parse(tool.startedAt)
  return Number.isFinite(elapsed) && elapsed >= 0 ? `${(elapsed / 1000).toFixed(1)}s` : ""
}

const TOOL_ACTIONS: Record<string, string> = {
  web_search: "Searching web",
  content_fetch: "Fetching content",
  file_search: "Searching files",
  file_get: "Reading file",
  file_analyze: "Analyzing file",
  file_inspect: "Inspecting file",
  artifact_deliver: "Delivering artifact",
  knowledge_list: "Listing knowledge",
  knowledge_find: "Finding knowledge",
  knowledge_query: "Querying knowledge",
  browser_start: "Starting browser",
  browser_get: "Reading browser",
  browser_act: "Using browser",
  browser_stop: "Stopping browser",
  integration_action_search: "Searching integration actions",
  integration_invoke: "Calling integration",
  repository_open: "Opening repository",
  repository_list: "Listing repositories",
  repository_status: "Checking repository",
  repository_close: "Stopping repository",
  repository_publish_pr: "Publishing pull request",
  job_get: "Checking job",
  job_events: "Reading job events",
  job_artifacts: "Reading job artifacts",
  execute: "Running command",
}

type TurnGroup =
  | { type: "assistant"; text: string }
  | { type: "tools"; id: string; tools: ToolCall[] }

export function turnGroups(turn: Turn): TurnGroup[] {
  const groups: TurnGroup[] = []
  const parts = turn.parts.length
    ? turn.parts
    : [
        ...turn.tools.map((tool) => ({ type: "tool" as const, callId: tool.callId })),
        ...(turn.assistant ? [{ type: "assistant" as const, text: turn.assistant }] : []),
      ]
  for (const part of parts) {
    if (part.type === "assistant") {
      if (part.text) groups.push({ type: "assistant", text: part.text })
      continue
    }
    const tool = turn.tools.find((item) => item.callId === part.callId)
    if (!tool) continue
    const previous = groups.at(-1)
    if (previous?.type === "tools") previous.tools.push(tool)
    else groups.push({ type: "tools", id: tool.callId, tools: [tool] })
  }
  return groups
}

function toolGroupId(runId: string, groupId: string): string {
  return `${runId}:${groupId}`
}

function lastToolGroupId(turn: Turn | undefined): string {
  if (!turn) return ""
  const group = turnGroups(turn).findLast((item) => item.type === "tools")
  return group?.type === "tools" ? toolGroupId(turn.runId, group.id) : ""
}

function activityLabel(tool: ToolCall): string {
  const plainName = tool.name.replaceAll("_", " ")
  const detail = toolLabel(tool)
  const action = TOOL_ACTIONS[tool.name] ?? plainName
  return detail === plainName ? action : `${action}${detail.slice(plainName.length)}`
}

export function elapsed(startedAt: string | undefined, now = Date.now()): string {
  const milliseconds = now - Date.parse(startedAt ?? "")
  if (!Number.isFinite(milliseconds) || milliseconds < 0) return "0.0s"
  const tenths = Math.floor(milliseconds / 100) / 10
  if (tenths < 60) return `${tenths.toFixed(1)}s`
  const seconds = Math.floor(tenths)
  return `${Math.floor(seconds / 60)}m ${String(seconds % 60).padStart(2, "0")}s`
}

export function waitingLabel(startedAt: string | undefined, now = Date.now()): string {
  const milliseconds = now - Date.parse(startedAt ?? "")
  if (!Number.isFinite(milliseconds) || milliseconds < 8_000) return "Planning next moves"
  if (milliseconds < 30_000) return "Waiting for model response"
  return "Model is responding slowly"
}

function message(cause: unknown): string {
  return cause instanceof Error ? cause.message : String(cause)
}

export function installSelectionClipboard(
  renderer: CliRenderer,
  writeHostClipboard: (text: string) => Promise<void> = copyToHostClipboard,
  onCopied?: (text: string) => void,
): () => void {
  const selected = (selection: Selection) => {
    const text = selection.getSelectedText()
    if (!text) return
    renderer.copyToClipboardOSC52(text)
    void writeHostClipboard(text)
    onCopied?.(text)
  }
  renderer.on("selection", selected)
  return () => renderer.off("selection", selected)
}

async function copyToHostClipboard(text: string): Promise<void> {
  const candidates: Array<[string, string[]]> = process.platform === "darwin"
    ? [["pbcopy", []]]
    : process.platform === "linux"
      ? [
          ...(process.env.WAYLAND_DISPLAY ? [["wl-copy", []] as [string, string[]]] : []),
          ["xclip", ["-selection", "clipboard"]],
          ["xsel", ["--clipboard", "--input"]],
        ]
      : []
  for (const [command, argumentsValue] of candidates) {
    if (await pipeToCommand(command, argumentsValue, text)) return
  }
}

function openExternal(url: string): void {
  try {
    const target = new URL(url)
    if (target.protocol !== "https:" || !["auth.openai.com", "chatgpt.com"].includes(target.hostname)) return
  } catch {
    return
  }
  const command = process.platform === "darwin" ? "open" : process.platform === "linux" ? "xdg-open" : ""
  if (!command) return
  const child = spawn(command, [url], { detached: true, stdio: "ignore" })
  child.unref()
}

function pipeToCommand(command: string, argumentsValue: string[], text: string): Promise<boolean> {
  return new Promise((resolve) => {
    let settled = false
    const finish = (value: boolean) => {
      if (settled) return
      settled = true
      resolve(value)
    }
    try {
      const child = spawn(command, argumentsValue, { stdio: ["pipe", "ignore", "ignore"] })
      child.once("error", () => finish(false))
      child.once("close", (code) => finish(code === 0))
      child.stdin?.end(text)
    } catch {
      finish(false)
    }
  })
}
