import os

# ПАТЧ ДЛЯ MAC: Чтобы torchcodec находил системный ffmpeg из Homebrew
if os.path.exists("/opt/homebrew/lib"):
    os.environ["DYLD_FALLBACK_LIBRARY_PATH"] = "/opt/homebrew/lib:" + os.environ.get(
        "DYLD_FALLBACK_LIBRARY_PATH", ""
    )

from pathlib import Path
from core.processor import MeetingProcessor

# ==========================================
# Читаем токен из окружения для безопасности
HF_TOKEN = os.environ.get("HF_TOKEN", "ВАШ_ТОКЕН_ЗДЕСЬ")
# ==========================================


def main():
    if HF_TOKEN == "ВАШ_ТОКЕН_ЗДЕСЬ":
        print("❌ ОШИБКА: Пожалуйста, вставьте ваш HF_TOKEN в файле test_pipeline.py!")
        return

    base_dir = Path(__file__).resolve().parent.parent
    audio_dir = base_dir / "data" / "audio"
    audio_files = [
        f for f in audio_dir.iterdir() if f.suffix in [".mp3", ".wav", ".m4a"]
    ]

    if not audio_files:
        print(f"❌ Ошибка: Положите аудиофайл в {audio_dir}")
        return

    test_audio = str(audio_files[0])

    print("\n🚀 Инициализация полного пайплайна (Whisper + Pyannote)...")
    processor = MeetingProcessor(hf_token=HF_TOKEN)

    print("\n🎧 Начинаем обработку аудио...")
    result = processor.process(test_audio)

    print("\n" + "=" * 50)
    print("📝 СЫРОЙ ПРОТОКОЛ (Whisper + Pyannote):")
    print("=" * 50)
    print(result["formatted_transcript"])

    print("\n" + "=" * 50)
    print("🤖 ИТОГОВЫЙ ИИ-ПРОТОКОЛ (Ollama):")
    print("=" * 50)
    print(result["ai_summary"])
    print("=" * 50)


if __name__ == "__main__":
    main()
