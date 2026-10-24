"""Reproducible, non-sensitive retrieval baseline; no model downloads."""

import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
from core.llm.chat import MeetingRAG

rag = MeetingRAG()
rag.load(
    {
        "raw_data": [
            {
                "id": i,
                "start": i * 5,
                "end": i * 5 + 5,
                "speaker": None,
                "text": f"Synthetic meeting task {i}: test backend SQLite persistence.",
            }
            for i in range(1000)
        ]
    }
)
times = []
for _ in range(30):
    start = time.perf_counter()
    assert rag.retrieve("SQLite")
    times.append((time.perf_counter() - start) * 1000)
print(
    json.dumps(
        {
            "segments": 1000,
            "runs": 30,
            "median_ms": round(statistics.median(times), 3),
            "max_ms": round(max(times), 3),
        },
        indent=2,
    )
)
