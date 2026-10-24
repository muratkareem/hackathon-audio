import io
import threading
import wave
from uuid import uuid4
import pytest
from fastapi.testclient import TestClient
from core.config import Settings
from api.main import create_app


@pytest.fixture
def settings(tmp_path):
    return Settings(data_dir=tmp_path / "runtime", max_upload_bytes=50000)


@pytest.fixture
def wav():
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(16000)
        stream.writeframes(b"\0\0" * 1600)
    return buffer.getvalue()


class FakeJobs:
    def __init__(self, store, settings):
        self.slot = threading.Lock()
        self.store = store
        self.submitted = []

    def submit(self, meeting, mode="process"):
        self.submitted.append((meeting["id"], mode))
        self.slot.release()

    def close(self):
        pass


@pytest.fixture
def app(settings):
    return create_app(settings, jobs_factory=FakeJobs)


@pytest.fixture
def client(app):
    with TestClient(app, base_url="http://localhost") as client:
        yield client


@pytest.fixture
def meeting():
    return {
        "id": str(uuid4()),
        "title": "Synthetic meeting / Учебная встреча",
        "status": "complete",
        "stage": "done",
        "progress": 100,
        "extension": ".wav",
        "audio_language": "en",
        "summary_language": "ru",
        "speaker_names": {},
        "warnings": [],
        "error": None,
        "raw_data": [
            {
                "id": 0,
                "start": 0,
                "end": 5,
                "text": "Анна проверит backend в пятницу. Anna will test the backend Friday.",
                "speaker": "Speaker 1",
                "language": "ru",
            },
            {
                "id": 1,
                "start": 5,
                "end": 9,
                "text": "We decided to use SQLite.",
                "speaker": "Speaker 2",
                "language": "en",
            },
        ],
        "report": {
            "overview": "Учебная встреча о backend.",
            "topics": ["Backend"],
            "discussion": [],
            "decisions": [{"text": "Use SQLite", "references": [1]}],
            "actions": [
                {
                    "text": "Проверить backend",
                    "owner": "Анна",
                    "deadline": "пятница",
                    "references": [0],
                }
            ],
            "open_questions": [],
            "takeaways": ["Проверить backend"],
        },
    }
