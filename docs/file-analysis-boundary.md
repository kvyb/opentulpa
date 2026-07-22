# File Analysis Boundary

`opentulpa.files.FileAnalysisService` is the deterministic application boundary for uploaded-file inspection and analysis.

- `file_inspect` reads tenant-owned bytes and returns bounded structural evidence synchronously.
- `file_analyze` submits a registered durable job and returns a job handle.
- File ownership is checked before parsing or job creation.
- Parsers receive bytes and metadata, not credentials or host paths.
- Results expose source references and bounded previews; storage paths remain internal.

The service may delegate format extraction to `opentulpa.business_knowledge.extraction`, but it must not own chat prompts, agent orchestration, Telegram routing, workflow activation, or external delivery. New formats belong behind this application boundary and need size, timeout, and malformed-input tests.
