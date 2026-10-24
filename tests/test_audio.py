import subprocess
import pytest
from core.audio import AudioError, validate_audio


@pytest.mark.parametrize("extension", ["wav", "mp3", "m4a", "flac", "ogg"])
def test_supported_formats_decode(tmp_path, wav, extension):
    source = tmp_path / "source.wav"
    source.write_bytes(wav)
    target = tmp_path / ("fixture." + extension)
    subprocess.run(
        ["ffmpeg", "-v", "error", "-nostdin", "-y", "-i", str(source), str(target)],
        check=True,
    )
    assert 0 < validate_audio(target, 10) < 1


def test_duration_limit(tmp_path, wav):
    source = tmp_path / "source.wav"
    source.write_bytes(wav)
    with pytest.raises(AudioError, match="audio_too_long"):
        validate_audio(source, 0.01)


def test_probe_without_audio_stream_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *a, **k: subprocess.CompletedProcess(
            a, 0, stdout=b'{"format":{"duration":2},"streams":[{"codec_type":"video"}]}'
        ),
    )
    with pytest.raises(AudioError, match="invalid_audio"):
        validate_audio(tmp_path / "video.wav", 10)
