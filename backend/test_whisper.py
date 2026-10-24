import os
from core.transcription.transcriber import MeetingTranscriber
from pathlib import Path


def main():
    # 1. Проверяем, есть ли аудиофайлы в папке data/audio
    base_dir = Path(__file__).resolve().parent.parent
    audio_dir = base_dir / "data" / "audio"

    # Берем первый попавшийся mp3 или wav файл для теста
    audio_files = [
        f for f in audio_dir.iterdir() if f.suffix in [".mp3", ".wav", ".m4a"]
    ]

    if not audio_files:
        print(
            f"❌ Ошибка: Положите любой тестовый аудиофайл (mp3/wav) в папку {audio_dir}"
        )
        return

    test_audio = str(audio_files[0])

    # 2. Инициализируем наш транскрибатор
    # Можно использовать "base" (очень быстрая) или "small" (точнее)
    transcriber = MeetingTranscriber(model_size="small")

    # 3. Запускаем распознавание
    print(f"\n🚀 Запуск обработки для файла: {os.path.basename(test_audio)}")
    result = transcriber.transcribe(test_audio)

    # 4. Выводим итоговый текст, который потом полетит в LLM
    print("\n" + "=" * 50)
    print("ИТОГОВЫЙ ТЕКСТ (ДЛЯ LLM):")
    print("=" * 50)
    print(result["full_text"])
    print("=" * 50)


if __name__ == "__main__":
    main()
