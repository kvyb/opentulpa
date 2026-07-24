# OpenTulpa TUI

The terminal client is a focused OpenTUI/SolidJS implementation derived from the interaction
patterns of OpenCode's `--mini` TUI. It speaks only OpenTulpa's V2 HTTP/SSE protocol and contains
no OpenCode runtime, SDK, provider, project, MCP, or agent code.

Use `/repo open https://github.com/owner/repo` to bind an isolated repository checkout to
the current session, `/repo status` to inspect it, and `/repos` to list workspaces.
During a run, `Esc` stops immediately, `Enter` queues a follow-up, and `Shift+Enter`
steers the active turn. When idle, `Shift+Enter` remains the multiline shortcut.

```sh
bun install --frozen-lockfile
bun test
bun run typecheck
bun run build
```
