import { afterEach, describe, expect, test } from "bun:test"
import { mkdtempSync, rmSync, writeFileSync } from "node:fs"
import { tmpdir } from "node:os"
import { join } from "node:path"
import { pathToFileURL } from "node:url"
import { droppedFiles } from "../src/attachments.js"

let root = ""
afterEach(() => {
  if (root) rmSync(root, { recursive: true, force: true })
  root = ""
})

describe("terminal attachment paste", () => {
  test("accepts escaped paths and file URLs", () => {
    root = mkdtempSync(join(tmpdir(), "opentulpa-tui-"))
    const path = join(root, "hello world.png")
    writeFileSync(path, "image")
    expect(droppedFiles(path.replaceAll(" ", "\\ "))).toEqual([path])
    expect(droppedFiles(pathToFileURL(path).toString())).toEqual([path])
  })

  test("does not turn ordinary text into attachments", () => {
    expect(droppedFiles("hello from the terminal")).toEqual([])
  })
})
