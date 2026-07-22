import { afterEach, describe, expect, test } from "bun:test"
import { OpenTulpaApi } from "../src/api.js"

const originalFetch = globalThis.fetch

afterEach(() => {
  globalThis.fetch = originalFetch
})

describe("V2 event transport", () => {
  test("parses split SSE frames and sends the replay cursor", async () => {
    const encoder = new TextEncoder()
    const chunks = [
      'id: 3\ndata: {"type":"message.delta","run_id":"run-1",',
      '"sequence":3,"timestamp":"2026-01-01T00:00:00Z","data":{"text":"hi"}}\n\n',
      'id: 4\ndata: {"type":"run.completed","run_id":"run-1","sequence":4,',
      '"timestamp":"2026-01-01T00:00:01Z","data":{}}\n\n',
    ]
    let request: Request | undefined
    globalThis.fetch = (async (input, init) => {
      request = new Request(input, init)
      return new Response(
        new ReadableStream({
          start(controller) {
            for (const chunk of chunks) controller.enqueue(encoder.encode(chunk))
            controller.close()
          },
        }),
        { status: 200, headers: { "content-type": "text/event-stream" } },
      )
    }) as typeof fetch

    const api = new OpenTulpaApi({ url: "https://tulpa.test", token: "owner", thread_id: "thread-1" })
    const events = []
    for await (const event of api.events("run-1", 2)) events.push(event)

    expect(events.map((event) => [event.sequence, event.type])).toEqual([
      [3, "message.delta"],
      [4, "run.completed"],
    ])
    expect(request?.url).toBe("https://tulpa.test/v2/agent/runs/run-1/events?after_sequence=2")
    expect(request?.headers.get("last-event-id")).toBe("2")
    expect(request?.headers.get("authorization")).toBe("Bearer owner")
  })

  test("updates only the current thread inference preference", async () => {
    let request: Request | undefined
    globalThis.fetch = (async (input, init) => {
      request = new Request(input, init)
      return Response.json({
        revision: 3,
        selection: {
          provider: "codex",
          model: "gpt-test",
          reasoning_effort: "high",
          service_tier: "priority",
          fallback_to_api: false,
        },
        effective: {
          provider: "codex",
          model: "gpt-test",
          reasoning_effort: "high",
          service_tier: "priority",
          fallback_to_api: false,
        },
      })
    }) as typeof fetch
    const api = new OpenTulpaApi({ url: "https://tulpa.test", token: "owner", thread_id: "thread-1" })

    const result = await api.updateThreadInference("thread-1", 2, {
      provider: "codex",
      model: "gpt-test",
      reasoning_effort: "high",
      service_tier: "priority",
      fallback_to_api: false,
    })

    expect(result.revision).toBe(3)
    expect(request?.url).toBe("https://tulpa.test/v2/agent/threads/thread-1/inference")
    expect(request?.method).toBe("PATCH")
    expect(await request?.json()).toEqual({
      expected_revision: 2,
      selection: {
        provider: "codex",
        model: "gpt-test",
        reasoning_effort: "high",
        service_tier: "priority",
        fallback_to_api: false,
      },
    })
  })
})
