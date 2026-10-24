import os
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from core.config import ROOT


class Jobs:
    def __init__(self, store, settings):
        self.store, self.settings = store, settings
        self.slot = threading.Lock()
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.process = None

    def submit(self, meeting, mode="process"):
        # Caller must acquire slot before accepting upload, bounding memory/disk work too.
        self.executor.submit(self.run, meeting["id"], mode)

    def run(self, meeting_id, mode):
        settings = self.settings
        try:
            meeting = self.store.get(meeting_id)
            meeting.update(status="processing", stage="starting")
            self.store.save(meeting)
            env = {
                **os.environ,
                "STEPPE_DATA_DIR": str(settings.data_dir),
                "WHISPER_MODEL": settings.whisper_model,
                "WHISPER_COMPUTE_TYPE": settings.compute_type,
                "OLLAMA_MODEL": settings.llm_model,
                "ENABLE_DIARIZATION": str(settings.diarization).lower(),
                "LLM_TIMEOUT_SECONDS": str(settings.llm_timeout),
            }
            self.process = subprocess.Popen(
                [sys.executable, "-m", "core.worker", meeting_id, mode],
                cwd=ROOT / "backend",
                env=env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            try:
                code = self.process.wait(timeout=settings.job_timeout)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
                raise RuntimeError("job_timeout")
            if code:
                raise RuntimeError("processing_failed")
        except Exception as exc:
            meeting = self.store.get(meeting_id)
            if meeting:
                meeting.update(
                    status="failed",
                    stage="failed",
                    error=str(exc)
                    if str(exc) in ("job_timeout", "processing_failed")
                    else "processing_failed",
                )
                self.store.save(meeting)
        finally:
            self.process = None
            try:
                if not settings.retain_audio:
                    for path in self.store.directory(meeting_id).glob("audio.*"):
                        path.unlink(missing_ok=True)
            finally:
                self.slot.release()

    def close(self):
        if self.process and self.process.poll() is None:
            self.process.terminate()
        self.executor.shutdown(wait=True, cancel_futures=True)
