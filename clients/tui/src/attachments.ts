import { accessSync, constants, existsSync, statSync } from "node:fs"
import { fileURLToPath } from "node:url"

export function droppedFiles(value: string): string[] {
  const tokens = shellTokens(value.trim())
  if (!tokens.length) return []
  const files: string[] = []
  for (const token of tokens) {
    let candidate = token
    if (candidate.startsWith("file://")) {
      try {
        candidate = fileURLToPath(candidate)
      } catch {
        return []
      }
    }
    try {
      if (!existsSync(candidate) || !statSync(candidate).isFile()) return []
      accessSync(candidate, constants.R_OK)
    } catch {
      return []
    }
    if (!files.includes(candidate)) files.push(candidate)
  }
  return files
}

function shellTokens(value: string): string[] {
  if (!value) return []
  const result: string[] = []
  let token = ""
  let quote = ""
  let escaped = false
  for (const character of value) {
    if (escaped) {
      token += character
      escaped = false
      continue
    }
    if (character === "\\" && quote !== "'") {
      escaped = true
      continue
    }
    if (quote) {
      if (character === quote) quote = ""
      else token += character
      continue
    }
    if (character === "'" || character === '"') {
      quote = character
      continue
    }
    if (/\s/.test(character)) {
      if (token) {
        result.push(token)
        token = ""
      }
      continue
    }
    token += character
  }
  if (escaped) token += "\\"
  if (quote) return []
  if (token) result.push(token)
  return result
}
