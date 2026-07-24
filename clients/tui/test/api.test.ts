import { afterEach, describe, expect, test } from "bun:test"
import { mkdtemp, rm, writeFile } from "node:fs/promises"
import { tmpdir } from "node:os"
import { join } from "node:path"
import { OpenTulpaApi } from "../src/api.js"

const originalFetch = globalThis.fetch

afterEach(() => {
  globalThis.fetch = originalFetch
})

describe("V2 event transport", () => {
  test("returns durable attachment metadata after upload", async () => {
    const directory = await mkdtemp(join(tmpdir(), "opentulpa-tui-upload-"))
    const path = join(directory, "screen.png")
    await writeFile(path, "image")
    let request: Request | undefined
    globalThis.fetch = (async (input, init) => {
      request = new Request(input, init)
      return Response.json(
        {
          file: {
            id: "file-1",
            kind: "image",
            original_filename: "screen.png",
            mime_type: "image/png",
            size_bytes: 5,
          },
        },
        { status: 201 },
      )
    }) as typeof fetch
    try {
      const api = new OpenTulpaApi({
        url: "https://tulpa.test",
        token: "owner",
        thread_id: "thread-1",
      })

      const attachment = await api.upload(path)

      expect(attachment.original_filename).toBe("screen.png")
      expect(request?.method).toBe("POST")
      const form = await request?.formData()
      expect((form?.get("upload") as File | null)?.name).toBe("screen.png")
    } finally {
      await rm(directory, { recursive: true, force: true })
    }
  })

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

  test("reconnects a dropped run stream from the last durable sequence", async () => {
    const encoder = new TextEncoder()
    const requests: Request[] = []
    globalThis.fetch = (async (input, init) => {
      requests.push(new Request(input, init))
      if (requests.length === 1) {
        let reads = 0
        return new Response(
          new ReadableStream({
            pull(controller) {
              if (reads === 0) {
                reads += 1
                controller.enqueue(
                  encoder.encode(
                    'id: 1\ndata: {"type":"run.started","run_id":"run-1","sequence":1,' +
                      '"timestamp":"2026-01-01T00:00:00Z","data":{}}\n\n',
                  ),
                )
                return
              }
              controller.error(new Error("socket closed"))
            },
          }),
          { status: 200, headers: { "content-type": "text/event-stream" } },
        )
      }
      if (requests.length === 2) {
        return new Response("runtime is switching", { status: 503 })
      }
      return new Response(
        'id: 2\ndata: {"type":"run.completed","run_id":"run-1","sequence":2,' +
          '"timestamp":"2026-01-01T00:00:01Z","data":{"text":"done"}}\n\n',
        { status: 200, headers: { "content-type": "text/event-stream" } },
      )
    }) as typeof fetch

    const api = new OpenTulpaApi({
      url: "https://tulpa.test",
      token: "owner",
      thread_id: "thread-1",
    })
    const events = []
    for await (const event of api.run("thread-1", "work", [])) events.push(event)

    expect(events.map((event) => [event.sequence, event.type])).toEqual([
      [1, "run.started"],
      [2, "run.completed"],
    ])
    expect(requests).toHaveLength(3)
    expect(requests[1]?.url).toBe(
      "https://tulpa.test/v2/agent/runs/run-1/events?after_sequence=1",
    )
    expect(requests[1]?.headers.get("last-event-id")).toBe("1")
    expect(requests[2]?.url).toBe(
      "https://tulpa.test/v2/agent/runs/run-1/events?after_sequence=1",
    )
  })

  test("replays a known approval run when cutover happens before its first event", async () => {
    const requests: Request[] = []
    globalThis.fetch = (async (input, init) => {
      requests.push(new Request(input, init))
      if (requests.length === 1) {
        return new Response("runtime is switching", { status: 503 })
      }
      return new Response(
        'id: 8\ndata: {"type":"run.completed","run_id":"run-1","sequence":8,' +
          '"timestamp":"2026-01-01T00:00:01Z","data":{"text":"active"}}\n\n',
        { status: 200, headers: { "content-type": "text/event-stream" } },
      )
    }) as typeof fetch

    const api = new OpenTulpaApi({
      url: "https://tulpa.test",
      token: "owner",
      thread_id: "thread-1",
    })
    const events = []
    for await (const event of api.resume("run-1", "approval-1", "approve")) events.push(event)

    expect(events.map((event) => [event.sequence, event.type])).toEqual([[8, "run.completed"]])
    expect(requests).toHaveLength(2)
    expect(requests[0]?.url).toBe("https://tulpa.test/v2/agent/runs/run-1/resume")
    expect(requests[1]?.url).toBe(
      "https://tulpa.test/v2/agent/runs/run-1/events?after_sequence=0",
    )
    expect(requests[1]?.headers.get("last-event-id")).toBe("0")
  })

  test("steers an active run through the continuation endpoint", async () => {
    let request: Request | undefined
    globalThis.fetch = (async (input, init) => {
      request = new Request(input, init)
      return new Response(
        'id: 1\ndata: {"type":"run.completed","run_id":"run-2","sequence":1,' +
          '"timestamp":"2026-01-01T00:00:01Z","data":{"text":"adjusted"}}\n\n',
        { status: 200, headers: { "content-type": "text/event-stream" } },
      )
    }) as typeof fetch
    const api = new OpenTulpaApi({
      url: "https://tulpa.test",
      token: "owner",
      thread_id: "thread-1",
    })

    const events = []
    for await (const event of api.steer("run-1", "Focus on tests", ["file-1"])) {
      events.push(event)
    }

    expect(events.map((event) => event.run_id)).toEqual(["run-2"])
    expect(request?.url).toBe("https://tulpa.test/v2/agent/runs/run-1/steer")
    expect(request?.method).toBe("POST")
    expect(await request?.json()).toEqual({
      text: "Focus on tests",
      file_ids: ["file-1"],
    })
  })

  test("stops reading and does not reconnect after the stream is aborted", async () => {
    const requests: Request[] = []
    globalThis.fetch = (async (input, init) => {
      const request = new Request(input, init)
      requests.push(request)
      return new Response(
        new ReadableStream({
          start(controller) {
            request.signal.addEventListener(
              "abort",
              () => controller.error(new DOMException("aborted", "AbortError")),
              { once: true },
            )
          },
        }),
        { status: 200, headers: { "content-type": "text/event-stream" } },
      )
    }) as typeof fetch
    const api = new OpenTulpaApi({
      url: "https://tulpa.test",
      token: "owner",
      thread_id: "thread-1",
    })
    const controller = new AbortController()
    const consume = (async () => {
      for await (const _event of api.run("thread-1", "work", [], controller.signal)) {}
    })()

    await Bun.sleep(10)
    controller.abort()
    await consume

    expect(requests).toHaveLength(1)
    expect(requests[0]?.signal.aborted).toBe(true)
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
