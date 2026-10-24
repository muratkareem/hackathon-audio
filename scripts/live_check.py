"""Upload an explicitly supplied non-sensitive WAV to a running server.

Usage: .venv/bin/python scripts/live_check.py path/to/synthetic.wav en
Never discovers or reads personal recordings. Outputs measured results as JSON.
"""

import io
import json
import sys
import time
from pathlib import Path
import httpx
from pypdf import PdfReader

path = Path(sys.argv[1])
language = sys.argv[2]
summary_language = sys.argv[3] if len(sys.argv) > 3 else language
started = time.perf_counter()
with httpx.Client(
    base_url="http://127.0.0.1:8000", timeout=240, trust_env=False
) as client:
    with path.open("rb") as audio:
        response = client.post(
            "/api/process",
            files={"file": (path.name, audio, "audio/wav")},
            data={"audio_language": language, "summary_language": summary_language},
        )
    response.raise_for_status()
    meeting = response.json()
    while meeting["status"] in ("queued", "processing"):
        if time.perf_counter() - started > 7200:
            raise TimeoutError("live test")
        time.sleep(2)
        meeting = client.get("/api/meetings/" + meeting["id"]).json()
    seconds = time.perf_counter() - started
    if meeting["status"] != "complete":
        print(
            json.dumps(
                {
                    "status": meeting["status"],
                    "error": meeting.get("error"),
                    "seconds": seconds,
                }
            )
        )
        raise SystemExit(1)
    question = (
        "Who will test the backend?"
        if language in ("en", "auto")
        else "Кто проверит серверную часть?"
    )
    answer = client.post(
        "/api/meetings/" + meeting["id"] + "/chat",
        json={
            "question": question,
            "language": "en" if language == "auto" else language,
        },
    )
    answer.raise_for_status()
    answer = answer.json()
    assert answer["references"] and not answer["insufficient_evidence"], answer
    exports = {}
    for kind in ("txt", "md", "csv", "json", "pdf"):
        response = client.get("/api/meetings/" + meeting["id"] + "/export/" + kind)
        response.raise_for_status()
        assert response.content
        exports[kind] = len(response.content)
        if kind == "json":
            json.loads(response.content)
        if kind == "pdf":
            assert PdfReader(io.BytesIO(response.content)).pages[0].extract_text()
    print(
        json.dumps(
            {
                "language": language,
                "meeting_id": meeting["id"],
                "status": meeting["status"],
                "duration": meeting["duration"],
                "processing_seconds": round(seconds, 2),
                "segments": len(meeting["raw_data"]),
                "report": meeting["report"],
                "metrics": meeting.get("metrics"),
                "answer": answer,
                "export_bytes": exports,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
