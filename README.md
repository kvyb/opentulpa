<p align="center">
  <picture>
    <source media="(max-width: 600px)" srcset="./assets/readme/hero-mobile.svg">
    <img src="./assets/readme/hero.svg" width="100%" alt="OpenTulpa, the self-evolving agent">
  </picture>
</p>

No static agent harness can be great at every job. OpenTulpa aims to solve this by
evolving its own code for your specific use cases.

## Get started

OpenTulpa needs Git, curl, and a model API key. The installer handles the rest.

### Run locally

```bash
git clone https://github.com/kvyb/opentulpa.git
cd opentulpa
./install.sh
opentulpa
```

Choose **Run here** and enter your model API key. OpenTulpa starts a private server on
your computer and opens the terminal interface.

### Use a hosted server

OpenTulpa is self-hosted. Run the server on Railway, with Docker, or on your own VM.

**Railway:** deploy this repository, attach a persistent volume at
`/app/opentulpa_data`, and set
`OPENTULPA_DATA_ROOT=/app/opentulpa_data`. The included
[Railway config](railway.toml) supplies the start command and health check.

**Docker:**

```bash
git clone https://github.com/kvyb/opentulpa.git
cd opentulpa
cp .env.example .env
docker compose up --build
```

Keep the `opentulpa_data` volume. It stores conversations, settings, and the agent's
release history.

Once the server is running, copy the one-time pairing code from its logs. Install the
terminal client on your computer using the local steps above, then connect:

```bash
opentulpa connect https://your-opentulpa.example
```

Paste the pairing code when prompted. See the
[deployment guide](docs/DEPLOYMENT.md) for domains, environment variables, and managed
VM setup.

## Details

### Capabilities

- The terminal, Telegram, schedules, and Agent API all talk to the same agent.
- Conversations keep their messages, memory, skills, files, and workspaces.
- Repository work runs in isolated checkouts. OpenTulpa can edit code, run tests, commit
  the result, and publish that exact commit as a pull request.
- Optional adapters add browser access, web search, Composio integrations, document
  tools, and hosted sandboxes.
- OpenTulpa works with OpenAI-compatible model providers and supports ChatGPT Codex
  device login.

### How self-evolution works

OpenTulpa can inspect its own source code and redacted execution traces. When a change
could help with your work:

1. The agent makes the change in an isolated Git worktree.
2. The fixed host tests the exact candidate commit.
3. The host starts the candidate and checks that it stays healthy.
4. If it fails, the host restores the previous healthy release.

The agent can change its runtime, prompts, tools, integrations, and interfaces. It
cannot access host credentials or release controls. The fixed host remains responsible
for evaluation, activation, and rollback.

## Documentation

- [Architecture](docs/ARCHITECTURE.md)
- [Deployment](docs/DEPLOYMENT.md)
- [Tool contract](docs/tool-contract.md)
- [E2E testing](docs/E2E_TESTING.md)
- [Prompt cookbook](docs/CHAT_COOKBOOK.md)

MIT licensed.
