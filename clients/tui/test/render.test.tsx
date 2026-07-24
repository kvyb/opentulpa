/** @jsxImportSource @opentui/solid */
import { describe, expect, test } from "bun:test"
import { testRender } from "@opentui/solid"
import { createSignal } from "solid-js"
import {
  ApprovalView,
  HeaderBar,
  SlashCommandPalette,
  StatusBar,
  ToolActivity,
  TurnList,
  TurnView,
  elapsed,
  filterSlashCommands,
  installSelectionClipboard,
  runningComposerAction,
  waitingLabel,
} from "../src/app.js"
import type { Approval, ToolCall } from "../src/types.js"
import { emptyTurn, reduceEvent } from "../src/state.js"

const tools: ToolCall[] = [
  {
    callId: "search",
    name: "web_search",
    arguments: { query: "OpenTulpa terminal interface" },
    result: { items: 4 },
    ok: true,
    startedAt: "2026-01-01T00:00:00Z",
    completedAt: "2026-01-01T00:00:01Z",
  },
  {
    callId: "inspect",
    name: "file_inspect",
    arguments: { path: "/workspace/report.pdf" },
    result: { pages: 3 },
    ok: true,
    startedAt: "2026-01-01T00:00:01Z",
    completedAt: "2026-01-01T00:00:02Z",
  },
  {
    callId: "fetch",
    name: "content_fetch",
    arguments: { url: "https://example.com/a/long/path" },
    startedAt: "2026-01-01T00:00:02Z",
  },
]

describe("slash command palette", () => {
  test("filters commands by command and description", () => {
    expect(filterSlashCommands("mo").map((item) => item.value)).toContain("/model")
    expect(filterSlashCommands("fast").map((item) => item.value)).toContain("/speed")
    expect(filterSlashCommands("subscription").map((item) => item.value)).toEqual([
      "/login codex",
      "/logout codex",
    ])
  })

  for (const width of [40, 80]) {
    test(`renders compact selectable commands at ${width} columns`, async () => {
      const selected: string[] = []
      const items = filterSlashCommands("model")
      const setup = await testRender(
        () => (
          <SlashCommandPalette
            items={items}
            selected={0}
            onSelect={(item) => selected.push(item.value)}
          />
        ),
        { width, height: 8 },
      )
      await setup.renderOnce()
      const lines = setup.captureCharFrame().split("\n")
      expect(lines.every((line) => line.length <= width)).toBe(true)
      expect(lines.join("\n")).toContain("/model")
      const y = lines.findIndex((line) => line.includes("/model"))
      await setup.mockMouse.click(3, y)
      expect(selected).toEqual(["/model"])
      setup.renderer.destroy()
    })
  }
})

describe("running composer controls", () => {
  test("maps escape, enter, and shift-enter to stop, queue, and steer", () => {
    expect(runningComposerAction({ name: "escape" })).toBe("cancel")
    expect(runningComposerAction({ name: "return" })).toBe("queue")
    expect(runningComposerAction({ name: "return", shift: true })).toBe("steer")
    expect(runningComposerAction({ name: "a" })).toBeUndefined()
  })
})

describe("tool activity rendering", () => {
  for (const width of [60, 120]) {
    test(`keeps active work visible and history compact at ${width} columns`, async () => {
      const setup = await testRender(
        () => <ToolActivity tools={tools} running spinner="⬒" now={Date.parse(tools[2]!.startedAt) + 400} expanded={false} onToggle={() => {}} />,
        { width, height: 10 },
      )
      await setup.renderOnce()
      const frame = setup.captureCharFrame()
      expect(frame).toContain("Fetching content")
      expect(frame).toContain("example.com/a/long/path")
      expect(frame).toContain("2 tools completed")
      expect(frame).not.toContain("input")
      expect(frame).not.toContain("items")
      setup.renderer.destroy()
    })
  }

  test("does not render an unfinished tool as active after a run stops", async () => {
    const setup = await testRender(
      () => <ToolActivity tools={tools} running={false} spinner="⠋" expanded={false} onToggle={() => {}} />,
      { width: 60, height: 8 },
    )
    await setup.renderOnce()
    const frame = setup.captureCharFrame()
    expect(frame).not.toContain("Fetching content")
    expect(frame).toContain("2 tools completed")
    setup.renderer.destroy()
  })

  test("opens request parameters when the active tool row is clicked", async () => {
    const setup = await testRender(
      () => <ToolActivity tools={[tools[2]!]} running spinner="⬒" now={Date.parse(tools[2]!.startedAt) + 400} expanded={false} onToggle={() => {}} />,
      { width: 80, height: 12 },
    )
    await setup.renderOnce()
    const initial = setup.captureCharFrame().split("\n")
    const y = initial.findIndex((line) => line.includes("Fetching content"))
    const x = initial[y]?.indexOf("Fetching content") ?? -1
    expect(x).toBeGreaterThanOrEqual(0)
    await setup.mockMouse.click(x, y)
    await setup.renderOnce()
    const expanded = setup.captureCharFrame()
    expect(expanded).toContain("Request")
    expect(expanded).toContain("https://example.com/a/long/path")
    setup.renderer.destroy()
  })

  test("opens a completed tool group and then its request parameters", async () => {
    const setup = await testRender(
      () => {
        const [expanded, setExpanded] = createSignal(false)
        return (
          <ToolActivity
            tools={tools.slice(0, 2)}
            running={false}
            spinner="⠋"
            expanded={expanded()}
            onToggle={() => setExpanded((value) => !value)}
          />
        )
      },
      { width: 80, height: 16 },
    )
    await setup.renderOnce()
    let lines = setup.captureCharFrame().split("\n")
    let y = lines.findIndex((line) => line.includes("2 tools completed"))
    let x = lines[y]?.indexOf("2 tools completed") ?? -1
    expect(x).toBeGreaterThanOrEqual(0)
    await setup.mockMouse.click(x, y)
    await setup.renderOnce()
    lines = setup.captureCharFrame().split("\n")
    y = lines.findIndex((line) => line.includes("Searching web"))
    x = lines[y]?.indexOf("Searching web") ?? -1
    expect(x).toBeGreaterThanOrEqual(0)
    await setup.mockMouse.click(x, y)
    await setup.renderOnce()
    const frame = setup.captureCharFrame()
    expect(frame).toContain("Request")
    expect(frame).toContain("OpenTulpa terminal interface")
    setup.renderer.destroy()
  })
})

describe("terminal chrome", () => {
  test("shows the effective Codex reasoning and speed", async () => {
    const setup = await testRender(
      () => (
        <StatusBar
          busy={false}
          uploading={false}
          status="Ready"
          width={90}
          inference={{
            provider: "codex",
            model: "gpt-5.6-sol",
            reasoning_effort: "ultra",
            service_tier: "priority",
            fallback_to_api: false,
          }}
        />
      ),
      { width: 90, height: 3 },
    )
    await setup.renderOnce()
    const frame = setup.captureCharFrame()
    expect(frame).toContain("gpt-5.6-sol")
    expect(frame).toContain("ultra")
    expect(frame).toContain("fast")
    setup.renderer.destroy()
  })

  test("shows visible thinking activity before the first token or tool", async () => {
    const turn = emptyTurn("run-1", "Help me inspect this project")
    const setup = await testRender(
      () => <TurnView turn={turn} active spinner="⬒" now={Date.parse(turn.startedAt) + 400} expandedToolsGroup="" onToggleTools={() => {}} />,
      { width: 60, height: 10 },
    )
    await setup.renderOnce()
    const frame = setup.captureCharFrame()
    expect(frame).toContain("Planning next moves")
    expect(frame).toContain("0.4s")
    setup.renderer.destroy()
  })

  test("renders assistant text and tools in event lineage order", async () => {
    let turn = emptyTurn("run-lineage", "Find the answer")
    turn = reduceEvent(turn, {
      type: "message.delta",
      run_id: "run-lineage",
      sequence: 1,
      timestamp: "2026-01-01T00:00:00Z",
      data: { text: "I will check the source." },
    })
    turn = reduceEvent(turn, {
      type: "tool.started",
      run_id: "run-lineage",
      sequence: 2,
      timestamp: "2026-01-01T00:00:01Z",
      data: { call_id: "search", name: "web_search", arguments: { query: "answer" } },
    })
    turn = reduceEvent(turn, {
      type: "tool.completed",
      run_id: "run-lineage",
      sequence: 3,
      timestamp: "2026-01-01T00:00:02Z",
      data: { call_id: "search", name: "web_search", ok: true, result: { items: 1 } },
    })
    turn = reduceEvent(turn, {
      type: "message.delta",
      run_id: "run-lineage",
      sequence: 4,
      timestamp: "2026-01-01T00:00:03Z",
      data: { text: "The answer is 42." },
    })
    const setup = await testRender(
      () => <TurnView turn={turn} active={false} spinner="⬒" now={Date.parse(turn.startedAt) + 400} expandedToolsGroup="" onToggleTools={() => {}} />,
      { width: 80, height: 20 },
    )
    const frame = await setup.waitForFrame((value) => value.includes("The answer is 42."))
    const before = frame.indexOf("I will check the source.")
    const tool = frame.indexOf("Searching web")
    const after = frame.indexOf("The answer is 42.")
    expect(before).toBeGreaterThanOrEqual(0)
    expect(tool).toBeGreaterThan(before)
    expect(after).toBeGreaterThan(tool)
    setup.renderer.destroy()
  })

  test("keeps one markdown renderable mounted while response text streams", async () => {
    let turn = emptyTurn("run-stream", "Hello")
    turn = reduceEvent(turn, {
      type: "message.delta",
      run_id: "run-stream",
      sequence: 1,
      timestamp: "2026-01-01T00:00:00Z",
      data: { text: "First" },
    })
    const [current, setCurrent] = createSignal([turn])
    const setup = await testRender(
      () => <TurnList turns={current()} busy activeRunId="run-stream" spinner="⬒" now={Date.parse(turn.startedAt) + 400} expandedToolsGroup="" onToggleTools={() => {}} />,
      { width: 80, height: 14 },
    )
    await setup.waitForFrame((value) => value.includes("First"))
    const markdown = setup.renderer.root.findDescendantById("assistant-run-stream-0")
    expect(markdown).toBeDefined()

    turn = reduceEvent(turn, {
      type: "message.delta",
      run_id: "run-stream",
      sequence: 2,
      timestamp: "2026-01-01T00:00:01Z",
      data: { text: " second token" },
    })
    setCurrent([turn])
    await setup.waitForFrame((value) => value.includes("First second token"))

    expect(setup.renderer.root.findDescendantById("assistant-run-stream-0")).toBe(markdown)
    await Bun.sleep(500)
    setup.renderer.destroy()
  })

  test("copies highlighted text as soon as selection finishes", async () => {
    const copied: string[] = []
    const setup = await testRender(
      () => <text>copy this phrase</text>,
      { width: 40, height: 3 },
    )
    const remove = installSelectionClipboard(setup.renderer, async (text) => { copied.push(text) })
    await setup.renderOnce()
    await setup.mockMouse.drag(0, 0, 8, 0)
    await setup.renderOnce()
    expect(copied).toHaveLength(1)
    expect(copied[0]).toContain("copy")
    remove()
    setup.renderer.destroy()
  })

  for (const width of [40, 80, 120]) {
    test(`stays compact without overlap at ${width} columns`, async () => {
      const setup = await testRender(
        () => (
          <box width="100%" height="100%" flexDirection="column">
            <HeaderBar
              title="A very long session title that must not take over the terminal"
              status="Working"
              width={width}
            />
            <box flexGrow={1} minHeight={0} />
            <StatusBar
              busy
              uploading={false}
              status="Working"
              width={width}
              inference={{
                provider: "codex",
                model: "a-very-long-model-identifier-that-must-be-truncated-cleanly",
                reasoning_effort: "high",
                service_tier: "priority",
                fallback_to_api: false,
              }}
            />
          </box>
        ),
        { width, height: 8 },
      )
      await setup.renderOnce()
      const lines = setup.captureCharFrame().split("\n")
      expect(lines.every((line) => line.length <= width)).toBe(true)
      expect(lines.join("\n")).toContain("OpenTulpa")
      expect(lines.join("\n").includes("working")).toBe(width >= 76)
      expect(lines.join("\n")).not.toContain("http")
      expect(lines.join("\n").includes("ctrl+n new")).toBe(width >= 40)
      expect(lines.join("\n").includes("ctrl+p sessions")).toBe(width >= 64)
      expect(lines.join("\n").includes("codex")).toBe(width >= 58)
      setup.renderer.destroy()
    })
  }

  test("clears transient loading text when the client becomes ready", async () => {
    let setStatus: (value: string) => void = () => {}
    const setup = await testRender(
      () => {
        const [status, updateStatus] = createSignal("Loading session")
        setStatus = updateStatus
        return (
          <StatusBar
            busy={false}
            uploading={false}
            status={status()}
            width={24}
          />
        )
      },
      { width: 24, height: 3 },
    )
    await setup.renderOnce()
    setStatus("Ready")
    await setup.renderOnce()
    expect(setup.captureCharFrame().split("\n")[0]?.trim()).toBe("")
    setup.renderer.destroy()
  })

  test("updates elapsed time in tenths from the supplied render clock", () => {
    const started = "2026-01-01T00:00:00.000Z"
    expect(elapsed(started, Date.parse(started))).toBe("0.0s")
    expect(elapsed(started, Date.parse(started) + 1400)).toBe("1.4s")
    expect(elapsed(started, Date.parse(started) + 61_900)).toBe("1m 01s")
  })

  test("explains when the model has not produced its first token", () => {
    const started = "2026-01-01T00:00:00.000Z"
    const timestamp = Date.parse(started)
    expect(waitingLabel(started, timestamp + 2_000)).toBe("Planning next moves")
    expect(waitingLabel(started, timestamp + 8_000)).toBe("Waiting for model response")
    expect(waitingLabel(started, timestamp + 30_000)).toBe("Model is responding slowly")
  })

  test("approval controls fit a narrow terminal", async () => {
    const approval: Approval = {
      approval_id: "approval-1",
      tool_name: "integration_invoke",
      description: "Send a message to an external service",
      arguments: {},
      allowed_decisions: ["approve", "edit", "reject"],
      run_id: "run-1",
    }
    const setup = await testRender(
      () => <ApprovalView approval={approval} onDecision={() => {}} />,
      { width: 40, height: 10 },
    )
    await setup.renderOnce()
    const frame = setup.captureCharFrame()
    expect(frame).toContain("Approval required")
    expect(frame).toContain("Approve")
    expect(frame).toContain("Edit")
    expect(frame).toContain("Reject")
    setup.renderer.destroy()
  })

  test("large external writes keep decisions visible without rendering file contents", async () => {
    const approval: Approval = {
      approval_id: "approval-large",
      tool_name: "integration_invoke",
      description: "Tool execution requires approval",
      arguments: {
        action_name: "GITHUB_COMMIT_MULTIPLE_FILES",
        parameters: {
          owner: "kvyb",
          repo: "opentulpa",
          branch: "readme/beautify-hero",
          base_branch: "main",
          upserts: [
            { path: "docs/assets/opentulpa-hero.svg", content: `<svg>${"x".repeat(50_000)}</svg>` },
            { path: "README.md", content: "![OpenTulpa](docs/assets/opentulpa-hero.svg)" },
          ],
        },
      },
      allowed_decisions: ["approve", "edit", "reject"],
      run_id: "run-large",
    }
    const setup = await testRender(
      () => <ApprovalView approval={approval} onDecision={() => {}} />,
      { width: 100, height: 10 },
    )
    await setup.renderOnce()
    const frame = setup.captureCharFrame()
    expect(frame).toContain("GITHUB_COMMIT_MULTIPLE_FILES")
    expect(frame).toContain("kvyb/opentulpa")
    expect(frame).toContain("2 files")
    expect(frame).toContain("Approve")
    expect(frame).toContain("Edit")
    expect(frame).toContain("Reject")
    expect(frame).not.toContain("<svg>")
    setup.renderer.destroy()
  })
})
