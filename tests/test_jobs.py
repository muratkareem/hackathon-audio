import subprocess
from core.jobs import Jobs
from core.storage import Store


def test_timeout_releases_slot_and_cleans_audio(settings, meeting, monkeypatch):
    store = Store(settings.data_dir)
    store.save(meeting)
    directory = store.directory(meeting["id"])
    directory.mkdir()
    (directory / "audio.wav").write_bytes(b"synthetic")

    class Process:
        killed = False

        def __init__(self, *a, **kw):
            pass

        def wait(self, timeout=None):
            if timeout:
                raise subprocess.TimeoutExpired("worker", timeout)
            return 0

        def kill(self):
            self.killed = True

    monkeypatch.setattr(subprocess, "Popen", Process)
    jobs = Jobs(store, settings)
    jobs.slot.acquire()
    jobs.run(meeting["id"], "process")
    assert store.get(meeting["id"])["error"] == "job_timeout"
    assert not jobs.slot.locked()
    assert not (directory / "audio.wav").exists()
    jobs.close()
