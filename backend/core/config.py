"""Local, explicit configuration. Importing this module never loads AI models."""

import os
from dataclasses import dataclass, field
from pathlib import Path
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")
# Disable optional model-library telemetry before importing those libraries.
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["DO_NOT_TRACK"] = "1"


@dataclass
class Settings:
    data_dir: Path = field(
        default_factory=lambda: Path(
            os.getenv("STEPPE_DATA_DIR", str(ROOT / "data" / "runtime"))
        ).resolve()
    )
    whisper_model: str = field(
        default_factory=lambda: os.getenv("WHISPER_MODEL", "small")
    )
    compute_type: str = field(
        default_factory=lambda: os.getenv("WHISPER_COMPUTE_TYPE", "int8")
    )
    llm_model: str = field(
        default_factory=lambda: os.getenv("OLLAMA_MODEL", "qwen2.5:7b")
    )
    ollama_url: str = "http://127.0.0.1:11434"
    max_upload_bytes: int = field(
        default_factory=lambda: int(os.getenv("MAX_UPLOAD_MB", "200")) * 1024 * 1024
    )
    max_duration: int = field(
        default_factory=lambda: int(os.getenv("MAX_AUDIO_SECONDS", "7200"))
    )
    job_timeout: int = field(
        default_factory=lambda: int(os.getenv("JOB_TIMEOUT_SECONDS", "7200"))
    )
    llm_timeout: int = field(
        default_factory=lambda: int(os.getenv("LLM_TIMEOUT_SECONDS", "180"))
    )
    retain_audio: bool = field(
        default_factory=lambda: os.getenv("RETAIN_AUDIO", "false").lower() == "true"
    )
    diarization: bool = field(
        default_factory=lambda: os.getenv("ENABLE_DIARIZATION", "false").lower()
        == "true"
    )
    hf_token: str | None = field(
        default_factory=lambda: os.getenv("HF_TOKEN") or None, repr=False
    )
    pdf_font: str | None = field(
        default_factory=lambda: os.getenv("PDF_FONT_PATH") or None
    )

    def __post_init__(self):
        if (
            min(
                self.max_upload_bytes,
                self.max_duration,
                self.job_timeout,
                self.llm_timeout,
            )
            <= 0
        ):
            raise ValueError("Limits and timeouts must be positive")
