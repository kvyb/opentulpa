#!/usr/bin/env bun
import { chmodSync, mkdirSync, rmSync } from "node:fs"
import { join } from "node:path"
import { fileURLToPath } from "node:url"
import { createSolidTransformPlugin } from "@opentui/solid/bun-plugin"

type Target = { platform: "darwin" | "linux"; arch: "arm64" | "x64" }
const all: Target[] = [
  { platform: "darwin", arch: "arm64" },
  { platform: "darwin", arch: "x64" },
  { platform: "linux", arch: "arm64" },
  { platform: "linux", arch: "x64" },
]
const args = new Set(process.argv.slice(2))
const targets = args.has("--all")
  ? all
  : all.filter((target) => target.platform === process.platform && target.arch === process.arch)
if (!targets.length) throw new Error(`Unsupported build host: ${process.platform}-${process.arch}`)

const parserWorkerPath = "opentui-tree-sitter-worker.js"
const parserWorker = await Bun.file(fileURLToPath(import.meta.resolve("@opentui/core/parser.worker"))).text()
rmSync("dist", { recursive: true, force: true })
mkdirSync("dist", { recursive: true })

for (const target of targets) {
  const output = join("dist", `opentulpa-tui-${target.platform}-${target.arch}`)
  const bunfsRoot = target.platform === "darwin" || target.platform === "linux" ? "/$bunfs/root/" : "B:/~BUN/root/"
  const result = await Bun.build({
    entrypoints: ["src/index.tsx", parserWorkerPath],
    files: { [parserWorkerPath]: parserWorker },
    tsconfig: "tsconfig.json",
    plugins: [createSolidTransformPlugin()],
    conditions: ["bun", "node"],
    define: { OTUI_TREE_SITTER_WORKER_PATH: bunfsRoot + parserWorkerPath },
    format: "esm",
    splitting: true,
    minify: true,
    sourcemap: "none",
    compile: {
      autoloadBunfig: false,
      autoloadDotenv: false,
      autoloadPackageJson: true,
      autoloadTsconfig: true,
      target: `bun-${target.platform}-${target.arch}`,
      outfile: output,
      execArgv: ["--use-system-ca", "--"],
    },
  })
  if (!result.success) {
    for (const log of result.logs) console.error(log)
    process.exit(1)
  }
  chmodSync(output, 0o755)
  console.log(output)
}
