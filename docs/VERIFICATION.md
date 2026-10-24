# Release verification

Last release verification: **2026-10-25**, macOS arm64, Python 3.12.
The real AI evidence below was captured on **2026-10-24** and reused because the
model pipeline was not changed during release recovery. Screenshots were refreshed
against the running application; all displayed meetings use synthesized test speech.

## Automated checks

| Command | Result |
|---|---|
| `.venv/bin/python -m pytest -q` | **51 passed, 0 failed, 0 skipped**; final dependency-update run: 0.58 s |
| `npm test --prefix frontend` | **5 passed, 0 failed, 0 skipped**; real application JavaScript in jsdom, mocked HTTP |
| `.venv/bin/python -m ruff check backend tests scripts` | Passed |
| `node --check frontend/app.js` | Passed |
| `.venv/bin/python -m pip check` | No broken requirements |
| `git diff --check` | Passed |
| `python -m pip_audit --progress-spinner off` | No known vulnerabilities in the installed environment after updates |
| `npm install --ignore-scripts --prefix frontend` | npm audit reported zero vulnerabilities |

Backend tests cover real ffmpeg validation for all five supported audio formats,
corrupt/empty/oversized uploads, streamed-body limits, duration limits, path safety,
origin/host checks, API errors, timestamp parsing, speaker alignment, optional-model
fallback, structured reports and mandatory nullable action fields, Ollama transport
and malformed-output failures, bounded retries, reference validation, BM25 retrieval
and abstention, persistence/recovery, retention/deletion, UTF-8/Unicode exports,
CSV formula protection and bilingual labels. Expensive AI is mocked in this suite.

Frontend tests execute the shipped script against the shipped HTML. They cover
navigation/settings, independent UI language persistence, action-table fields,
reference navigation, search/copy, escaped chat output and pending state, confirmed
deletion, mind-map zoom, Cyrillic SVG serialization and attached download links.
The SVG regression test caught duplicate xmlns attributes; the released serializer
now produces valid XML. These tests are DOM integration tests, not visual browser tests.

GitHub Actions runs the same lightweight suites on Ubuntu/Python 3.12/Node 22,
with ffmpeg and DejaVu Sans installed, and no AI packages or model downloads.
The first hosted run passed: [run 40130398408](https://github.com/muratkareem/hackathon-audio/actions/runs/40130398408).
Use the PR's current checks for the latest head; a historical run does not certify
later commits. The release procedure waits for the final head before merging.

## Real local AI workflows

Models: Faster-Whisper **small**, CTranslate2 **4.6.0**, CPU **int8**;
Ollama **qwen2.5:7b**. PyAV **15.1.0** is pinned after a real failure with PyAV 19,
which removed the metadata_errors argument used by Faster-Whisper 1.2.1.

Safe fixtures were generated with installed macOS Samantha and Milena voices and
converted with ffmpeg. No private recording or transcript was committed.

| Input / report | Audio duration | Upload-to-report elapsed | Segments | Result |
|---|---:|---:|---:|---|
| English / English | 8.90 s | 28.21 s | 3 | Complete; owner Anna and deadline “by Friday”; grounded answer cites 3.44 s |
| Russian / Russian | 10.34 s | 26.23 s | 4 | Complete; owner Анна and deadline “пятница”; grounded answer cites 3.94 s |
| Auto-detected English / Russian | 8.90 s | 20.17 s | 3 | Complete; Russian report independent of English speech and UI |

Commands, from repository root, using explicitly supplied temporary fixtures:

```bash
.venv/bin/python scripts/live_check.py /path/to/english.wav en
.venv/bin/python scripts/live_check.py /path/to/russian.wav ru
.venv/bin/python scripts/live_check.py /path/to/english.wav auto ru
```

Each successful live check exercised upload → ASR → structured report → retrieved
Q&A with valid references → TXT/Markdown/JSON/CSV/PDF. PDF text was parsed and a
Russian PDF was rendered with Poppler and visually inspected for Cyrillic/layout.
Report language and owner/deadline omissions found in earlier runs were corrected
and rerun. Do not infer word-error rates or factual guarantees from these short samples.
Whisper rendered SQLite as “SQ Lite”/“склайт”; screenshots show actual output.

## Browser and restart checks

- Backend startup and real static frontend loading passed after dependency updates.
- Saved completed meetings remained available after stopping/restarting the backend.
- Dashboard, meetings, new-recording and settings navigation worked.
- Actual browser file selection/submission completed a real auto-language job.
- EN/RU switching persisted across reload while audio/report language stayed independent.
- Transcript search, clipboard copy and answer/report links to highlighted segments worked.
- Real Russian Q&A returned an answer with a clickable 00:00:03 source.
- Action tables exposed the actual owner and deadline, with stacked cells on small screens.
- Page scroll width equaled viewport width at **1440, 1280, 768 and 390 px** after
  fixing mobile grid overflow; Russian mobile labels were also checked.
- Cyrillic mind maps rendered and zoomed. The final **3,212-byte browser SVG download**
  was parsed as valid XML and contained no script nodes.
- Meeting deletion and retained-audio access were verified through API tests; the
  frontend confirmation/DELETE transition is covered by DOM integration tests.
- No browser console errors were observed in the checked workflows.

## Performance observations

BM25 baseline: 1,000 synthetic segments, 30 queries, median **2.804 ms**, max **3.341 ms**.
A repeat after removing unused chat state measured **2.902 ms** median: this is ordinary
run variation, not evidence of a speedup.

The English job worker reported **26.932 s** and **1,139,720,192 bytes peak RSS**.
The Russian worker reported **24.467 s** and **1,145,946,112 bytes peak RSS**.
Ollama is a separate process and is **not included** in these memory figures.
Upload-to-report times include validation and polling; they exclude subsequent Q&A/export.
No original-version speed comparison, multi-hour load test, or whole-system RAM claim
is made. One heavy job at a time and process-level timeouts bound contention/lifetime.

## Security review and remaining limits

The release audit found 73 advisories across old pins for pypdf, pytest, python-dotenv,
python-multipart and Starlette. These were upgraded, with compatible FastAPI, and all
checks rerun. No known advisories were reported in the follow-up scan. This is a
point-in-time advisory lookup, not a guarantee of security.

- Optional gated Pyannote was not tested live; fallback and alignment have automated coverage.
- Multi-hour recordings and overlapping real speakers were not tested live.
- BM25 is lexical; paraphrases, morphology and cross-language retrieval can miss evidence.
- Valid schemas and source IDs do not prove semantic correctness. Review AI reports.
- Runtime files are unencrypted and the service has no multi-user authentication.
- No original project license was found; redistribution rights need owner review.
