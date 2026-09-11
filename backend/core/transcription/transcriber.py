from pathlib import Path
from faster_whisper import WhisperModel

# Вычисляем корень проекта (чтобы пути работали откуда угодно)
BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
MODELS_DIR = BASE_DIR / "data" / "models"

class MeetingTranscriber:
    def __init__(self, model_size="small"):
        """
        Инициализация модели. 
        model_size: "tiny", "base", "small", "medium", "large-v3"
        Для хакатона и тестов на MacBook лучше всего подойдет "small" или "base".
        """
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        
        print(f"Загрузка модели '{model_size}' в {MODELS_DIR}...")
        # Используем CPU и int8. Это безопасно работает на всех Mac и экономит память.
        self.model = WhisperModel(
            model_size_or_path=model_size,
            device="cpu", 
            compute_type="int8",
            download_root=str(MODELS_DIR)
        )
        print("✅ Модель готова к работе!")

    def transcribe(self, audio_path: str):
        """
        Транскрибация аудиофайла в текст.
        """
        print(f"Начало транскрибации файла: {audio_path}")
        
        segments, info = self.model.transcribe(
            audio_path, 
            beam_size=5
        )
        
        print(f"Определен язык: '{info.language}' с вероятностью {info.language_probability:.2f}")
        
        full_text = ""
        segments_data = []
        
        # Внимание: segments — это генератор. Вычисления происходят во время итерации!
        for segment in segments:
            print(f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}")
            full_text += segment.text + " "
            segments_data.append({
                "start": segment.start,
                "end": segment.end,
                "text": segment.text.strip()
            })
            
        return {
            "language": info.language,
            "full_text": full_text.strip(),
            "segments": segments_data
        }
