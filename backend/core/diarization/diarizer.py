import os
from pyannote.audio import Pipeline
from pathlib import Path

# Чтобы модель сохранилась локально и не скачивалась каждый раз
BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
MODELS_DIR = BASE_DIR / "data" / "models"
os.environ["PYANNOTE_CACHE"] = str(MODELS_DIR)

class SpeakerDiarizer:
    def __init__(self, hf_token: str):
        """
        Инициализация модели диаризации от Pyannote.
        Требуется токен от Hugging Face, так как модель закрыта лицензией.
        """
        print("Загрузка модели Pyannote (Диаризация)...")
        try:
            self.pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                token=hf_token
            )
        except Exception as e:
            print("\n❌ ОШИБКА: Не удалось загрузить модель Pyannote.")
            print("Убедитесь, что вы передали правильный HF_TOKEN и приняли соглашения на Hugging Face!")
            raise e
            
        print("✅ Модель диаризации готова!")

    def diarize(self, audio_path: str):
        """
        Определяет кто, когда и сколько говорил.
        """
        print(f"Запуск разделения по спикерам для файла: {audio_path}")
        out = self.pipeline(audio_path)
        # В pyannote.audio 4.0 pipeline возвращает объект DiarizeOutput
        diarization = out.speaker_diarization if hasattr(out, "speaker_diarization") else out
        
        segments = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            segments.append({
                "start": turn.start,
                "end": turn.end,
                "speaker": speaker # Будет выдавать "SPEAKER_00", "SPEAKER_01" и т.д.
            })
            
        return segments
