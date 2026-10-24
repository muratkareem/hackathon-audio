"""Sequential, lazy processing inside a killable job subprocess."""

import gc
from core.transcription.transcriber import MeetingTranscriber
from core.diarization.diarizer import SpeakerDiarizer, align_speakers
from core.llm.llm_processor import ProtocolGenerator
from core.llm.client import InferenceError


class MeetingProcessor:
    def __init__(self, settings):
        self.settings = settings

    def process(self, audio_path, meeting, progress):
        progress("transcribing", 15)
        transcriber = MeetingTranscriber(
            self.settings.whisper_model, self.settings.compute_type
        )
        result = transcriber.transcribe(
            audio_path,
            meeting["audio_language"] if meeting["audio_language"] != "auto" else None,
        )
        meeting["raw_data"] = result["segments"]
        meeting["detected_language"] = result["language"]
        del transcriber
        gc.collect()
        if not meeting["raw_data"]:
            raise InferenceError("empty_transcript")
        progress("diarizing", 60)
        meeting["diarization_status"] = "disabled"
        if self.settings.diarization:
            try:
                diarizer = SpeakerDiarizer(self.settings.hf_token)
                meeting["raw_data"] = align_speakers(
                    meeting["raw_data"], diarizer.diarize(audio_path)
                )
                meeting["diarization_status"] = "available"
                del diarizer
            except Exception:
                meeting["diarization_status"] = "unavailable"
                meeting["warnings"].append("diarization_unavailable")
            gc.collect()
        progress("analyzing", 75)
        self.analyze(meeting)
        return meeting

    def analyze(self, meeting):
        try:
            meeting["report"] = ProtocolGenerator(self.settings).generate_protocol(
                meeting["raw_data"], meeting["summary_language"]
            )
            meeting["status"] = "complete"
            meeting["error"] = None
        except InferenceError as exc:
            meeting["status"] = "partial"
            meeting["error"] = str(exc)
