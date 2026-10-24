# Architecture

## Boundaries

- `backend/api/main.py`: app factory, upload/status/library/chat/export API and static UI.
- `core/config.py`: dotenv/environment, loopback-only Ollama, limits, private paths.
- `core/audio.py`: ffprobe metadata and ffmpeg decode validation; no shell interpolation.
- `core/storage.py`: SQLite JSON metadata and strict canonical UUID paths.
- `core/jobs.py` / `worker.py`: one admission lock, one executor orchestration thread,
  one killable subprocess. The same lock bounds synchronous chat inference.
- `core/transcription`: lazy Faster-Whisper; CPU compute validation; original ASR timestamps.
- `core/diarization`: optional lazy Pyannote, maximum-overlap assignment, stable labels.
- `core/llm`: dedicated prompts, HTTP timeouts, structured Pydantic output, reference
  checks, bounded chunking/tree synthesis and the preserved BM25 baseline.
- `core/exports.py`: report/transcript serializers, CSV formula protection, escaped PDF text.
- `frontend`: no framework/CDN; localization dictionary, safe DOM text, generated SVG nodes.

## Lifecycle

A valid upload receives a random UUID; its original name is a display title only.
Admission occurs before application-level streaming and decoding. ASGI body limits
also bound multipart spooling, including requests without Content-Length. ffprobe
and full decode validation run off the event loop. Successful admission returns 202.

Job states: queued → processing → complete; partial means ASR succeeded but report
analysis failed; failed includes timeouts and interrupted processing. Stage percentages
are milestones, not calibrated estimates. Progress and partial transcript are saved
in SQLite. Startup marks queued/processing records interrupted. Reports survive restarts.

Worker subprocesses release model memory on exit. Whisper is discarded before optional
speaker analysis; Pyannote is discarded before Ollama. Ollama uses keep_alive=0.
Cold model loads happen on subsequent jobs: bounded memory and reliable cancellation
are prioritized over a persistent model cache on a 16 GB machine. No cross-meeting
inference cache or unbounded conversation history is maintained.

Chat retrieves up to eight positive-score transcript segments, caps context size,
then asks Ollama. No match returns an explicit local abstention without calling the
model. Generated source IDs must belong to retrieved segments. This does not prove
semantic entailment. Speaker aliases affect retrieval/display, never original segments.

## API

| Method | Path | Result |
|---|---|---|
| GET | `/api/health` | Local models, backend settings and ffmpeg readiness |
| POST | `/api/process` | multipart audio + audio_language + summary_language; 202 |
| GET | `/api/meetings` | Metadata library |
| GET | `/api/meetings/{id}` | Meeting, stages, transcript and structured report |
| POST | `/api/meetings/{id}/retry` | Reanalyze saved transcript; otherwise retained audio |
| POST | `/api/meetings/{id}/chat` | question + language → answer, references, sources |
| PATCH | `/api/meetings/{id}/speakers` | Validated label-to-name aliases |
| DELETE | `/api/meetings/{id}` | Delete owned meeting files and metadata |
| GET | `/api/meetings/{id}/audio` | Retained source only, when configured |
| GET | `/api/meetings/{id}/export/{kind}` | txt/md/json/csv/pdf |

Original global chat/export endpoints were replaced with explicit meeting IDs to
prevent cross-meeting state leakage. The upload endpoint remains `/api/process`,
with asynchronous job acceptance instead of a long blocking result response.
Old third-party clients need to poll the meeting endpoint and use scoped routes.

## Security and retention

Loopback host binding, trusted hosts, same-origin checks, no permissive CORS,
CSP, no untrusted HTML, UUID storage and bounded upload/job limits. No authentication:
local software is inside the trust boundary. Do not run multiple server workers or
expose this service on a shared host without further work.

Source audio is deleted after the job finishes unless retention is enabled. Converted
copies are not routinely made. Metadata and transcripts remain until deletion. Exports
are streamed from memory and not kept on the server. Models stay in an ignored cache.
Original tracked generated artifacts were untracked while their local files were
preserved; earlier upstream history was not rewritten.

## Intentional limitations

No embeddings/vector service, no identity recognition, no persistent model-serving
pool, no PNG rendering dependency. The original unsafe LLM Mermaid path was replaced
with bounded SVG from validated report fields. The visual map shows up to eight items
per category; the complete report/export remains available. Long-report synthesis has
bounded context and schema cardinalities, so it can condense details. Quality requires
human review. Optional Pyannote dependencies are isolated from base installation.
