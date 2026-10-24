"""Private subprocess entry point. Receives only a UUID, never arbitrary file paths."""

import sys
import resource
import time
from core.config import Settings
from core.storage import Store
from core.processor import MeetingProcessor


def main():
    started = time.perf_counter()
    settings = Settings()
    store = Store(settings.data_dir)
    meeting = store.get(sys.argv[1])

    def progress(stage, percent):
        meeting.update(stage=stage, progress=percent)
        store.save(meeting)

    processor = MeetingProcessor(settings)
    if sys.argv[2] == "analyze":
        progress("analyzing", 75)
        processor.analyze(meeting)
    else:
        processor.process(
            store.directory(meeting["id"]) / ("audio" + meeting["extension"]),
            meeting,
            progress,
        )
    meeting["metrics"] = {
        "total_seconds": round(time.perf_counter() - started, 3),
        "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        * (1 if sys.platform == "darwin" else 1024),
    }
    meeting.update(stage="done", progress=100)
    store.save(meeting)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        settings = Settings()
        store = Store(settings.data_dir)
        meeting = store.get(sys.argv[1])
        safe = {"empty_transcript", "unsupported_compute_type"}
        meeting.update(
            status="failed",
            stage="failed",
            error=str(exc) if str(exc) in safe else "processing_failed",
        )
        store.save(meeting)
