import { describe, expect, test } from "bun:test"
import { emptyTurn, reduceEvent, turnsFromTimeline } from "../src/state.js"
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
})
