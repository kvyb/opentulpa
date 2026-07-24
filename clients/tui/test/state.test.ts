import { describe, expect, test } from "bun:test"
import { consumeApproval, emptyTurn, reduceEvent, turnsFromTimeline } from "../src/state.js"
import type { AgentEvent } from "../src/types.js"

function event(sequence: number, type: string, data: Record<string, unknown>): AgentEvent {
  return { type, run_id: "run-1", sequence, timestamp: `time-${sequence}`, data }
}

describe("run event reducer", () => {
  test("streams text and deduplicates replayed sequences", () => {
    const event = {
      type: "message.delta",
      run_id: "run-1",
      sequence: 2,
      timestamp: "now",
      data: { text: "hello" },
    }
    const first = reduceEvent(emptyTurn("run-1"), event)
    expect(first.assistant).toBe("hello")
    expect(reduceEvent(first, event)).toEqual(first)
  })

  test("tracks tool start and completion as one call", () => {
    const started = reduceEvent(emptyTurn("run-1"), {
      type: "tool.started",
      run_id: "run-1",
      sequence: 2,
      timestamp: "start",
      data: { call_id: "call-1", name: "web_search", arguments: { query: "news" } },
    })
    const completed = reduceEvent(started, {
      type: "tool.completed",
      run_id: "run-1",
      sequence: 3,
      timestamp: "end",
      data: { call_id: "call-1", name: "web_search", ok: true, result: { items: 2 } },
    })
    expect(completed.tools).toHaveLength(1)
    expect(completed.tools[0]?.completedAt).toBe("end")
    expect(completed.tools[0]?.result).toEqual({ items: 2 })
    expect(completed.parts).toEqual([{ type: "tool", callId: "call-1" }])
  })

  test("preserves assistant and tool lineage across streamed events", () => {
    let turn = emptyTurn("run-1")
    turn = reduceEvent(turn, event(1, "message.delta", { text: "Before. " }))
    turn = reduceEvent(turn, event(2, "tool.started", {
      call_id: "search",
      name: "web_search",
      arguments: { query: "answer" },
    }))
    turn = reduceEvent(turn, event(3, "tool.completed", {
      call_id: "search",
      name: "web_search",
      ok: true,
      result: { items: 1 },
    }))
    turn = reduceEvent(turn, event(4, "message.delta", { text: "After." }))

    expect(turn.parts).toEqual([
      { type: "assistant", text: "Before. " },
      { type: "tool", callId: "search" },
      { type: "assistant", text: "After." },
    ])
  })

  test("does not duplicate the assistant projection after replaying deltas", () => {
    const turns = turnsFromTimeline([
      { id: "u", type: "user", run_id: "run-1", timestamp: "1", text: "Hi" },
      {
        id: "d",
        type: "event",
        event_type: "message.delta",
        sequence: 1,
        run_id: "run-1",
        timestamp: "2",
        data: { text: "Hello" },
      },
      { id: "a", type: "assistant", run_id: "run-1", timestamp: "3", text: "Hello" },
    ])
    expect(turns[0]?.assistant).toBe("Hello")
    expect(turns[0]?.parts).toEqual([{ type: "assistant", text: "Hello" }])
  })

  test("reconstructs a transcript without replaying message deltas", () => {
    const turns = turnsFromTimeline([
      { id: "u", type: "user", run_id: "run-1", timestamp: "1", text: "Hi" },
      { id: "a", type: "assistant", run_id: "run-1", timestamp: "2", text: "Hello" },
    ])
    expect(turns).toHaveLength(1)
    expect(turns[0]?.user).toBe("Hi")
    expect(turns[0]?.assistant).toBe("Hello")
  })

  test("restores uploaded attachments from durable timeline entries", () => {
    const turns = turnsFromTimeline([
      {
        id: "u",
        type: "user",
        run_id: "run-1",
        timestamp: "1",
        text: "What is in this image?",
        file_ids: ["file-1"],
        attachments: [
          {
            id: "file-1",
            kind: "image",
            original_filename: "screen.png",
            mime_type: "image/png",
            size_bytes: 2048,
            available: true,
          },
        ],
      },
    ])

    expect(turns[0]?.fileIds).toEqual(["file-1"])
    expect(turns[0]?.attachments[0]?.original_filename).toBe("screen.png")
  })

  test("ignores malformed tool fragments and leaves no active tool after completion", () => {
    let turn = emptyTurn("run-1")
    turn = reduceEvent(turn, event(1, "tool.started", {
      call_id: "execute_1",
      name: "execute",
      arguments: {},
    }))
    turn = reduceEvent(turn, event(2, "tool.started", {
      call_id: "None",
      name: "",
      arguments: { command: "" },
    }))
    turn = reduceEvent(turn, event(3, "run.completed", { text: "Finished" }))

    expect(turn.tools).toHaveLength(1)
    expect(turn.tools[0].callId).toBe("execute_1")
    expect(turn.tools[0].completedAt).toBeTruthy()
    expect(turn.status).toBe("completed")
  })

  test("replaces a consumed approval when a resumed run interrupts again", () => {
    let turn = emptyTurn("run-1")
    turn = reduceEvent(turn, event(1, "approval.required", {
      approval_id: "approval-1",
      tool_name: "capability_activate",
      allowed_decisions: ["approve", "edit", "reject"],
    }))
    turn = consumeApproval(turn, "approval-1")

    expect(turn.approvals).toEqual([])
    expect(turn.status).toBe("running")

    turn = reduceEvent(turn, event(2, "run.started", { resumed: true }))
    turn = reduceEvent(turn, event(3, "approval.required", {
      approval_id: "approval-2",
      tool_name: "capability_activate",
      allowed_decisions: ["approve", "edit", "reject"],
    }))

    expect(turn.approvals.map((approval) => approval.approval_id)).toEqual(["approval-2"])
    expect(turn.status).toBe("approval")
  })

  test("treats an explicit cancellation as terminal without a failure banner", () => {
    const turn = reduceEvent(
      emptyTurn("run-1", "Stop"),
      event(1, "run.failed", {
        code: "agent_run_cancelled",
        message: "The agent run was cancelled before completion.",
      }),
    )

    expect(turn.status).toBe("cancelled")
    expect(turn.error).toBeUndefined()
  })

  test("timeline replay drops an earlier consumed approval after resume", () => {
    const turns = turnsFromTimeline([
      {
        id: "a1",
        type: "event",
        event_type: "approval.required",
        sequence: 1,
        run_id: "run-1",
        timestamp: "1",
        data: {
          approval_id: "approval-1",
          tool_name: "first",
          allowed_decisions: ["approve"],
        },
      },
      {
        id: "resume",
        type: "event",
        event_type: "run.started",
        sequence: 2,
        run_id: "run-1",
        timestamp: "2",
        data: { resumed: true },
      },
      {
        id: "a2",
        type: "event",
        event_type: "approval.required",
        sequence: 3,
        run_id: "run-1",
        timestamp: "3",
        data: {
          approval_id: "approval-2",
          tool_name: "second",
          allowed_decisions: ["approve"],
        },
      },
    ])

    expect(turns[0]?.approvals.map((approval) => approval.approval_id)).toEqual(["approval-2"])
  })
})
