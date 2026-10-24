from types import SimpleNamespace
import pytest
from pydantic import ValidationError
from core.schemas import Segment, Report, Answer
from core.diarization.diarizer import align_speakers
from core.transcription.transcriber import MeetingTranscriber
from core.llm.chat import MeetingRAG
from core.llm.client import InferenceError, LocalLLM
from core.llm.llm_processor import ProtocolGenerator, check_references, chunks


@pytest.mark.parametrize(
    "start,end", [(-1, 2), (3, 2), (float("nan"), 2), (0, float("inf"))]
)
def test_invalid_timestamps(start, end):
    with pytest.raises(ValidationError):
        Segment(id=0, start=start, end=end, text="x")


def test_transcription_parsing():
    engine = MeetingTranscriber()
    engine.model = SimpleNamespace(
        transcribe=lambda *a, **k: (
            iter(
                [
                    SimpleNamespace(start=1, end=3, text=" hello ", no_speech_prob=0.1),
                    SimpleNamespace(start=4, end=5, text=" ", no_speech_prob=0.1),
                    SimpleNamespace(start=6, end=7, text="noise", no_speech_prob=0.99),
                ]
            ),
            SimpleNamespace(language="en"),
        )
    )
    result = engine.transcribe("fake.wav")
    assert len(result["segments"]) == 1
    assert result["segments"][0] == {
        "id": 0,
        "start": 1,
        "end": 3,
        "text": "hello",
        "speaker": None,
        "language": "en",
    }


def test_speaker_overlap_and_unknown(meeting):
    turns = [
        {"start": 0, "end": 4, "speaker": "A"},
        {"start": 3, "end": 8, "speaker": "B"},
    ]
    result = align_speakers(meeting["raw_data"], turns)
    assert [s["speaker"] for s in result] == ["Speaker 1", "Speaker 2"]
    assert align_speakers(meeting["raw_data"], [])[0]["speaker"] is None


def test_retrieval(meeting):
    rag = MeetingRAG()
    rag.load(meeting)
    assert rag.retrieve("SQLite")[0]["id"] == 1
    assert rag.retrieve("пятницу")[0]["id"] == 0
    assert rag.retrieve("galaxies") == []
    assert rag.retrieve("the and") == []


def test_grounded_answer(monkeypatch, settings, meeting):
    rag = MeetingRAG()
    rag.load(meeting)
    monkeypatch.setattr(
        LocalLLM,
        "generate",
        lambda *a: Answer(answer="SQLite", references=[1], insufficient_evidence=False),
    )
    result = rag.ask("SQLite", settings)
    assert result["sources"][0]["start"] == 5
    monkeypatch.setattr(
        LocalLLM,
        "generate",
        lambda *a: Answer(
            answer="invented", references=[99], insufficient_evidence=False
        ),
    )
    with pytest.raises(InferenceError):
        rag.ask("SQLite", settings)


def test_summary_schema_and_references(settings, meeting, monkeypatch):
    report = Report.model_validate(meeting["report"])
    monkeypatch.setattr(LocalLLM, "generate", lambda *a: report)
    assert (
        ProtocolGenerator(settings).generate_protocol(meeting["raw_data"], "ru")[
            "actions"
        ][0]["owner"]
        == "Анна"
    )
    with pytest.raises(InferenceError):
        check_references(report, {0})
    bad = meeting["report"] | {"decisions": [{"text": "Decision", "references": []}]}
    with pytest.raises(ValidationError):
        Report.model_validate(bad)


def test_chunking_preserves_every_segment(meeting):
    groups = list(chunks(meeting["raw_data"], budget=100))
    assert len(groups) == 2
    assert [s["id"] for group in groups for s in group] == [0, 1]


def test_optional_diarization_degrades(monkeypatch, settings, meeting):
    from core.processor import MeetingProcessor
    import core.processor as module

    settings.diarization = True
    monkeypatch.setattr(
        MeetingTranscriber,
        "transcribe",
        lambda *a: {"segments": meeting["raw_data"], "language": "ru"},
    )
    monkeypatch.setattr(
        module, "SpeakerDiarizer", lambda **k: (_ for _ in ()).throw(RuntimeError())
    )
    monkeypatch.setattr(
        ProtocolGenerator, "generate_protocol", lambda *a: meeting["report"]
    )
    result = MeetingProcessor(settings).process("fake", meeting, lambda *a: None)
    assert result["status"] == "complete"
    assert result["diarization_status"] == "unavailable"
    assert result["raw_data"]


def test_missing_ollama_preserves_transcript(monkeypatch, settings, meeting):
    from core.processor import MeetingProcessor

    monkeypatch.setattr(
        ProtocolGenerator,
        "generate_protocol",
        lambda *a: (_ for _ in ()).throw(InferenceError("ollama_unavailable")),
    )
    MeetingProcessor(settings).analyze(meeting)
    assert meeting["status"] == "partial"
    assert meeting["raw_data"]


def test_empty_audio_explicit(monkeypatch, settings, meeting):
    from core.processor import MeetingProcessor

    monkeypatch.setattr(
        MeetingTranscriber, "transcribe", lambda *a: {"segments": [], "language": "en"}
    )
    with pytest.raises(InferenceError, match="empty_transcript"):
        MeetingProcessor(settings).process("fake", meeting, lambda *a: None)


def test_action_requires_explicit_owner_and_deadline(meeting):
    from core.schemas import Action
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        Action(text="Test backend", references=[0])
    action = Action(text="Test backend", references=[0], owner=None, deadline=None)
    assert action.owner is None


def test_summary_language_is_explicit_in_system_prompt(monkeypatch, settings, meeting):
    prompts = []

    def generate(self, schema, system, content):
        prompts.append(system)
        return Report.model_validate(meeting["report"])

    monkeypatch.setattr(LocalLLM, "generate", generate)
    ProtocolGenerator(settings).generate_protocol(meeting["raw_data"], "ru")
    assert "in Russian" in prompts[0]
    assert "even when the transcript is English" in prompts[0]
