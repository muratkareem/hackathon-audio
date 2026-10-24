"""Optional CPU diarization. Labels describe turns, never real identities."""


class SpeakerDiarizer:
    def __init__(self, hf_token):
        if not hf_token:
            raise ValueError("diarization_token_missing")
        from pyannote.audio import Pipeline
        import torch

        self.pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1", token=hf_token
        )
        if self.pipeline is None:
            raise RuntimeError("diarization_unavailable")
        self.pipeline.to(torch.device("cpu"))

    def diarize(self, audio_path):
        result = self.pipeline(str(audio_path))
        annotation = getattr(result, "speaker_diarization", result)
        return [
            {"start": turn.start, "end": turn.end, "speaker": speaker}
            for turn, _, speaker in annotation.itertracks(yield_label=True)
        ]


def align_speakers(segments, turns):
    """Choose maximum overlapping turn; ties stable. No overlap stays unknown."""
    labels = {}
    for turn in sorted(turns, key=lambda t: t["start"]):
        labels.setdefault(turn["speaker"], f"Speaker {len(labels) + 1}")
    result = []
    for segment in segments:
        scores = [
            (
                max(
                    0, min(segment["end"], t["end"]) - max(segment["start"], t["start"])
                ),
                t["speaker"],
            )
            for t in turns
        ]
        best = max(scores, key=lambda item: item[0], default=(0, None))
        result.append({**segment, "speaker": labels[best[1]] if best[0] > 0 else None})
    return result
