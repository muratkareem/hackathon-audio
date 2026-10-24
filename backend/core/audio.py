import json
import math
import subprocess
from pathlib import Path

EXTENSIONS = {".wav", ".mp3", ".m4a", ".flac", ".ogg"}


class AudioError(ValueError):
    pass


def validate_audio(path: Path, max_duration: int) -> float:
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-protocol_whitelist",
                "file,pipe",
                "-show_entries",
                "format=duration:stream=codec_type",
                "-of",
                "json",
                str(path),
            ],
            capture_output=True,
            timeout=20,
            check=True,
        )
        data = json.loads(result.stdout)
        duration = float(data.get("format", {}).get("duration", 0))
        if not any(s.get("codec_type") == "audio" for s in data.get("streams", [])):
            raise AudioError("invalid_audio")
        if not math.isfinite(duration) or duration <= 0:
            raise AudioError("invalid_audio")
        if duration > max_duration:
            raise AudioError("audio_too_long")
        # Decode validation catches damaged packets which probing alone can accept.
        subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-xerror",
                "-nostdin",
                "-protocol_whitelist",
                "file,pipe",
                "-i",
                str(path),
                "-map",
                "0:a:0",
                "-f",
                "null",
                "-",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=120,
            check=True,
        )
        return duration
    except FileNotFoundError as exc:
        raise AudioError("ffmpeg_missing") from exc
    except (
        subprocess.SubprocessError,
        json.JSONDecodeError,
        ValueError,
        TypeError,
    ) as exc:
        if isinstance(exc, AudioError):
            raise
        raise AudioError("invalid_audio") from exc
