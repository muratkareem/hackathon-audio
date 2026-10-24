"""Manual compatibility entry point; use only explicitly supplied safe samples."""

import argparse
from core.config import Settings
from core.processor import MeetingProcessor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("audio", help="Path to a non-sensitive audio fixture")
    parser.add_argument("--language", choices=["auto", "en", "ru"], default="auto")
    args = parser.parse_args()
    meeting = {
        "audio_language": args.language,
        "summary_language": "en",
        "warnings": [],
    }
    result = MeetingProcessor(Settings()).process(
        args.audio, meeting, lambda stage, percent: print(stage, percent)
    )
    print("Status:", result["status"], "Segments:", len(result["raw_data"]))


if __name__ == "__main__":
    main()
