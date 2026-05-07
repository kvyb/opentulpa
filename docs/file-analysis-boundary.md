# File Analysis Boundary

## Decision

Keep `src/opentulpa/agent/file_analysis.py` as a cohesive runtime-facing module for now.

The file is large, but it still has one clear responsibility: turn uploaded file bytes into text or concise analysis that the agent runtime can store, cite, or use to answer a user. Splitting it today would mostly move private helpers around without reducing a current behavioral risk.

## Current Boundary

`file_analysis.py` owns uploaded-file analysis helpers that need direct access to raw bytes:

* document and spreadsheet text extraction
* short audio transcription
* image summarization
* video segmentation and synthesis
* the runtime compatibility functions imported by `runtime.py`

The runtime should keep calling these public helpers instead of reaching into media-specific internals:

* `extract_uploaded_text(...)`
* `summarize_uploaded_blob(...)`
* `transcribe_audio_blob(...)`
* `analyze_uploaded_file(...)`

Private helpers, including video segment helpers and media-model selection helpers, are implementation details of this module.

## What Does Not Belong Here

Do not add workflow policy, Telegram routing, storage writes, tool registration, or user-context persistence to `file_analysis.py`.

Those concerns should stay with the runtime, Telegram attachment handling, tool modules, or the relevant service layer. This module should only receive bytes plus file metadata and return extracted text or analysis.

## Split Trigger

Split the module only when one of these becomes true:

* document extraction and media analysis need separate test fixtures or dependency gates
* media-specific code grows enough that reviewing document extraction requires loading unrelated audio/video prompts
* another caller needs a media-specific helper as a public API
* optional dependencies or provider behavior diverge enough that a single module makes startup or testing brittle

When splitting, keep `file_analysis.py` as a compatibility layer until callers are migrated. `runtime.py` imports should continue to work through the existing public helper names.

## Preferred Future Shape

If a split becomes necessary, use narrow modules:

* `file_text_extraction.py` for text, PDF, DOCX, and XLSX extraction
* `file_audio_analysis.py` for audio format inference and transcription
* `file_media_analysis.py` for image and video analysis
* `file_analysis.py` as the public aggregator during migration

Avoid behavior changes in the same patch as the split unless a test first exposes the behavior gap.
