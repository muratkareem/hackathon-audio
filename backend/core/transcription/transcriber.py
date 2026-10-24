"""Faster-Whisper CPU backend; no claim of Metal support."""

from pathlib import Path
from core.schemas import Segment
from core.config import ROOT


class MeetingTranscriber:
    def __init__(self, model_size="small", compute_type="int8"):
        self.model_size = model_size
        self.compute_type = compute_type
        self.model = None

    def transcribe(self, audio_path, language=None):
        if self.model is None:
            import onnxruntime

            onnxruntime.disable_telemetry_events()
            import ctranslate2
            from faster_whisper import WhisperModel

            if self.compute_type not in ctranslate2.get_supported_compute_types("cpu"):
                raise ValueError("unsupported_compute_type")
            self.model = WhisperModel(
                self.model_size,
                device="cpu",
                compute_type=self.compute_type,
                cpu_threads=4,
                num_workers=1,
                download_root=str(ROOT / "data" / "models" / "huggingface" / "hub"),
            )
        segments, info = self.model.transcribe(
            (str(audio_path) if isinstance(audio_path, (str, Path)) else audio_path),
            language=language,
            beam_size=5,
            vad_filter=True,
            condition_on_previous_text=False,
        )
        parsed = []
        for segment in segments:
            text = segment.text.strip()
            if not text or getattr(segment, "no_speech_prob", 0) > 0.8:
                continue
            parsed.append(
                Segment(
                    id=len(parsed),
                    start=segment.start,
                    end=segment.end,
                    text=text,
                    language=info.language,
                ).model_dump()
            )
        return {
            "language": info.language,
            "full_text": " ".join(s["text"] for s in parsed),
            "segments": parsed,
        }

    def transcribe_array(self, audio_array, sample_rate=16000):
        if sample_rate != 16000:
            raise ValueError("Expected mono 16kHz samples")
        # Kept for callers of the original transcriber API.
        return self.transcribe(audio_array)
