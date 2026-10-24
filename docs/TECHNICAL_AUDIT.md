# Technical audit

Baseline: `a6d7b9d` (main), inspected 2026-10-24. Clean newly cloned checkout;
no user changes. Reviewed all tracked Python, HTML/CSS/JS, scripts, configuration,
dependency declarations, tracked artifact structure, and recent Git history.
No license file or AGENTS.md is present. Do not assume redistribution permission.

## Baseline execution

- `bash run_api.sh`: exit 1, requires `.env` and HF_TOKEN before starting.
- `python3 -c 'import fastapi, uvicorn'`: ModuleNotFoundError (FastAPI).
- `python3 -m pytest --version`: pytest not installed.
- Scripts also assume `backend/venv`; no such environment exists.
- ffmpeg and Ollama executables are installed; model availability not yet verified.
- Existing `test_*.py` are manual model runners, not assertion-based unit tests.

## Findings

| Severity | Evidence and problem | Proposed repair |
|---|---|---|
| Critical | `api/main.py` joins `file.filename` directly to audio directory; arbitrary overwrite through traversal/absolute filenames | UUID internal paths, bounded streaming, media validation |
| High | `frontend/index.html` inserts `marked.parse(ai_summary)` into innerHTML without sanitization | Text/structured rendering, safe local diagram construction |
| High | `main.py` constructs MeetingProcessor at import; processor eagerly loads large-v3 and gated Pyannote | Lazy imports/models; optional diarization; configurable CPU small model |
| High | async process/chat/diagram handlers perform synchronous heavy work | One bounded worker, persisted jobs, responsive status endpoints |
| High | No upload size/duration limits or validated decode; retained source and converted copies | Stream limits, ffprobe/ffmpeg timeouts, deterministic cleanup |
| High | Global last_result and shared RAG overwrite previous meeting; restart loses all state | SQLite meeting library and recovery for interrupted jobs |
| High | run_api.sh binds 0.0.0.0; no origin/host safeguards | Loopback binding, explicit local host/origin checks |
| High | Pyannote failure prevents all transcription; silent per-chunk ASR exceptions lose content | Transcribe independently; report optional diarization failures explicitly |
| High | Unvalidated Markdown/JSON/diagram LLM output, no explicit timeouts, errors returned as successful summaries | Validated schemas, bounded contexts/timeouts and useful errors |
| Medium | Processor decodes audio twice, saves WAV, materializes all chunks and submits all futures | Whole-recording ASR with original timestamps; sequential expensive stages |
| Medium | `chat.py` contains actual BM25 retrieval, but no-match fallback injects unrelated segments; full generated summary and unlimited saved history | Preserve BM25; explicit abstention, bounded context and references |
| Medium | Whole transcript sent to 8192-token context; no chunking | Bounded analysis chunks and hierarchical synthesis |
| Medium | Raw LLM Mermaid syntax, CDN resources despite offline claims | Generate diagram from validated data locally |
| Medium | Frontend simulates progress at 5/15 seconds; Russian only | Real stage status; separate UI/audio/output languages |
| Medium | CSV exports transcript rather than tasks; PDF relies on CDN rasterization | Verified UTF-8 task CSV and Unicode PDF |
| Medium | Tracked `data/protocol.json/.csv` are generated meeting artifacts | Untrack without deleting local originals; ignore runtime artifacts |
| Medium | Dependencies unpinned; torch/Pyannote mandatory; no automated suite | Separate optional AI dependencies; reproducible pins and mock tests |
| Low | README claims absolute confidentiality, unsupported speedups and large-v3-turbo (code uses large-v3) | Describe measured behavior and privacy limits honestly |
| Low | Scripts source shell code from .env and hardcode Homebrew/venv paths | dotenv config, portable root-aware scripts |

## Preserved foundations and implementation order

Keep FastAPI, vanilla frontend, Faster-Whisper, optional Pyannote, local Ollama,
BM25 scoring and timestamped transcript representation. Preserve source history.
Stabilize configuration/storage/jobs/uploads first, then structured inference and
retrieval, UI, exports, tests, benchmarks, bilingual documentation and release checks.
A passing mock suite is not evidence that real models work. Live English/Russian
transcription, optional gated diarization and Ollama must be reported separately.
