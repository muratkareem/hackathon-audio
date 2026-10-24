# Development

Use Python 3.12 on macOS for the verified dependency set. See the README for setup.
Base requirements run API/UI/tests without AI packages; install `requirements-ai.txt`
for speech, and `requirements-diarization.txt` separately for gated speaker models.

```bash
./run_api.sh
./run_test.sh -q
.venv/bin/python -m ruff check backend tests scripts
.venv/bin/python -m ruff format --check backend tests scripts
npm ci --ignore-scripts --prefix frontend
npm test --prefix frontend
node --check frontend/app.js
.venv/bin/python -m pip check
```

Unit/API tests use temporary directories and generated audio. They never discover
personal recordings or require downloading AI weights. The old manual runner entry
points remain, but now require an explicit file argument instead of choosing an
arbitrary recording from disk.

## Safe live verification

Start the app and Ollama. macOS can synthesize public test text with installed voices:

```bash
mkdir -p /tmp/steppe-demo
say -v Samantha -o /tmp/steppe-demo/en.aiff 'Anna will test the backend by Friday.'
ffmpeg -i /tmp/steppe-demo/en.aiff /tmp/steppe-demo/en.wav
.venv/bin/python scripts/live_check.py /tmp/steppe-demo/en.wav en
```

For Russian use an installed Russian voice (`say -v '?'`) and language `ru`.
The script asserts upload, completed processing, grounded Q&A and valid nonempty
exports. Its sample question expects backend testing content. It leaves the synthetic
meeting in the library so the UI can be examined. Delete it through the app when done.
For other sample content, adjust the question deliberately.

`./run_api.sh` uses one worker and localhost; `.env` is parsed as data, never sourced as
shell commands. Model errors intentionally avoid returning filesystem paths, tokens
or raw exceptions to clients. Diagnose against explicitly safe fixtures with
`.venv/bin/python backend/test_whisper.py /path/to/safe.wav`.

## Benchmarks

`.venv/bin/python scripts/benchmark.py` runs 30 BM25 queries over 1,000 synthetic
segments. It reports median/max latency, not a claimed speedup. Live jobs record worker
elapsed time and peak resident memory. On macOS RSS is already bytes; on Linux it is
converted from KiB. Ollama memory is outside worker RSS and must be measured separately
before drawing whole-system conclusions. No multi-hour performance claim is made.

## Browser checks

Inspect the running application, switch EN/RU, reload, upload a generated sample,
open a saved report, follow references, search transcript, open map, ask a question
and download artifacts. Verify desktop and mobile widths and browser console errors.
Screenshots in `docs/screenshots/` must come from the actual running application.
No fabricated mockup is accepted as implementation evidence.

## Contribution and release

Keep private `.env`, audio, model caches, runtime DBs and exports out of Git. Use
`git diff --check`, inspect staged changes and rerun relevant tests before each commit.
The modernization branch is `feature/steppe-hub-modernization`; never force-push or
merge it automatically. There was no license file in the baseline; ask the owner to
resolve licensing before redistribution rather than inventing terms.

Suggested repository description: “Local AI meeting assistant with FastAPI,
Faster-Whisper, Ollama, timestamp-grounded Q&A and English/Russian support.”

Frontend interaction tests use jsdom only as a development dependency. The shipped
application has no npm runtime dependency. GitHub Actions installs no AI models.
