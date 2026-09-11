import numpy as np
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
        # compute_type="auto" — ctranslate2 сам выберет лучший тип для платформы
        # На Apple Silicon использует Accelerate/NEON оптимизации
        self.model = WhisperModel(
            model_size_or_path=model_size,
            device="cpu", 
            compute_type="auto",
            download_root=str(MODELS_DIR)
        )
        print("✅ Модель готова к работе!")

    def transcribe(self, audio_path: str):
        """
        Транскрибация аудиофайла в текст.
        Оптимизирован для коротких чанков после диаризации.
        """
        segments, info = self.model.transcribe(
            audio_path, 
            beam_size=1,                        # greedy decoding — 3× быстрее на коротких фразах
            vad_filter=True,                    # отсекает тишину внутри чанка
            condition_on_previous_text=False     # предотвращает галлюцинации-петли
        )
        
        full_text = ""
        segments_data = []
        
        # segments — генератор, вычисления происходят во время итерации
        for segment in segments:
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

    def transcribe_array(self, audio_array: np.ndarray, sample_rate: int = 16000):
        """
        Транскрибация из NumPy массива напрямую (без записи на диск).
        audio_array: float32 ndarray, моно, нормализован в [-1, 1]
        sample_rate: частота дискретизации (Whisper ожидает 16kHz)
        """
        # faster-whisper принимает ndarray напрямую (с версии 0.9+)
        segments, info = self.model.transcribe(
            audio_array,
            beam_size=1,
            vad_filter=True,
            condition_on_previous_text=False
        )
        
        full_text = ""
        segments_data = []
        
        for segment in segments:
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
