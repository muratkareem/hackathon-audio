"""Manual transcription smoke test; does not scan personal audio directories."""

import argparse
from core.config import Settings
from core.transcription.transcriber import MeetingTranscriber


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("audio", help="Path to a non-sensitive audio fixture")
    args = parser.parse_args()
    settings = Settings()
    result = MeetingTranscriber(
        settings.whisper_model, settings.compute_type
    ).transcribe(args.audio)
    print("Language:", result["language"], "Segments:", len(result["segments"]))


if __name__ == "__main__":
    main()
