/** @jsxImportSource @opentui/solid */
import { readFileSync, writeFileSync } from "node:fs"
import type { ClientConfig } from "./types.js"

// Most terminals render punctuation and symbols with wcwidth semantics. Forcing the
// matching OpenTUI path prevents glyphs such as em dashes from occupying two cells.
if (!process.env.OPENTUI_FORCE_UNICODE) process.env.OPENTUI_FORCE_WCWIDTH = "1"

if (process.argv.includes("--version")) {
  console.log("opentulpa-tui 0.2.0")
  process.exit(0)
}

if (process.argv.includes("--protocol-version")) {
  console.log("2")
  process.exit(0)
}

const [{ render }, { App }] = await Promise.all([import("@opentui/solid"), import("./app.js")])

function readConfig(): ClientConfig {
  const descriptor = Number.parseInt(process.env.OPENTULPA_CONNECTION_FD ?? "", 10)
  if (!Number.isInteger(descriptor) || descriptor < 3) {
    throw new Error("OpenTulpa did not provide the terminal client connection.")
  }
  let parsed: unknown
  try {
    parsed = JSON.parse(readFileSync(descriptor, "utf8"))
  } catch {
    throw new Error("OpenTulpa provided an invalid terminal client connection.")
  }
  if (!parsed || typeof parsed !== "object") throw new Error("OpenTulpa client configuration is invalid.")
  const value = parsed as Partial<ClientConfig>
  if (!/^https?:\/\//.test(value.url ?? "") || typeof value.token !== "string" || !value.thread_id) {
    throw new Error("OpenTulpa client configuration is invalid.")
  }
  return value as ClientConfig
}

const config = readConfig()
let activeThread = config.thread_id

await render(
  () => <App config={config} onConnectionChange={(threadId) => (activeThread = threadId)} />,
  {
    targetFps: 60,
    exitOnCtrlC: false,
    useMouse: true,
    autoFocus: true,
    openConsoleOnError: false,
    onDestroy: () => {
      const descriptor = Number.parseInt(process.env.OPENTULPA_STATE_FD ?? "", 10)
      if (Number.isInteger(descriptor) && descriptor >= 3) {
        try {
          writeFileSync(descriptor, JSON.stringify({ thread_id: activeThread }))
        } catch {}
      }
      process.exit(0)
    },
  },
)
