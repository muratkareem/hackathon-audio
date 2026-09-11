import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from pydub import AudioSegment
from core.transcription.transcriber import MeetingTranscriber
from core.diarization.diarizer import SpeakerDiarizer
from core.llm.llm_processor import ProtocolGenerator
import time

class MeetingProcessor:
    def __init__(self, hf_token: str, llm_model="llama3.1", max_workers: int = 2):
        self.transcriber = MeetingTranscriber(model_size="large-v3")
        self.diarizer = SpeakerDiarizer(hf_token=hf_token)
        self.protocol_generator = ProtocolGenerator(model_name=llm_model)
        self.max_workers = max_workers
        
    @staticmethod
    def _audio_chunk_to_numpy(chunk: AudioSegment) -> np.ndarray:
        """
        Конвертирует AudioSegment в float32 numpy array для Whisper.
        Whisper ожидает: mono, 16kHz, float32 в [-1, 1].
        """
        # Приводим к mono 16kHz
        chunk = chunk.set_channels(1).set_frame_rate(16000)
        # Получаем сырые int16 сэмплы
        samples = np.frombuffer(chunk.raw_data, dtype=np.int16)
        # Нормализуем в float32 [-1, 1]
        return samples.astype(np.float32) / 32768.0

    @staticmethod
    def _merge_segments(segments: list, max_gap: float = 0.5, max_duration: float = 15.0) -> list:
        """
        Мерджит соседние сегменты одного спикера с паузой ≤ max_gap секунд.
        Ограничивает длину мерджнутого сегмента до max_duration секунд.
        Это уменьшает количество вызовов Whisper.
        """
        if not segments:
            return []
        
        merged = [segments[0].copy()]
        
        for seg in segments[1:]:
            prev = merged[-1]
            gap = seg["start"] - prev["end"]
            merged_duration = seg["end"] - prev["start"]
            
            # Мерджим если тот же спикер, маленькая пауза, и не превышаем лимит длины
            if (seg["speaker"] == prev["speaker"] 
                and gap <= max_gap 
                and merged_duration <= max_duration):
                prev["end"] = seg["end"]
            else:
                merged.append(seg.copy())
        
        return merged

    def _transcribe_chunk(self, audio_array: np.ndarray, seg_info: dict) -> dict | None:
        """
        Транскрибирует один чанк. Вызывается из ThreadPoolExecutor.
        """
        try:
            result = self.transcriber.transcribe_array(audio_array)
            text = result["full_text"]
            lang = result["language"]
        except Exception as e:
            print(f"⚠️ Ошибка транскрипции [{seg_info['start']:.1f}s]: {e}")
            return None
        
        if text.strip():
            return {
                "start": seg_info["start"],
                "end": seg_info["end"],
                "speaker": seg_info["speaker"],
                "language": lang,
                "text": text
            }
        return None

    def process(self, audio_path: str):
        t_total = time.time()
        
        # 0. Конвертация аудио (защита от багов Torchaudio с VBR MP3)
        print(f"\n0. Конвертация аудио в стандартный WAV 16kHz...")
        audio_seg = AudioSegment.from_file(audio_path)
        audio_seg = audio_seg.set_frame_rate(16000).set_channels(1)
        wav_path = str(audio_path) + "_converted.wav"
        audio_seg.export(wav_path, format="wav")
        audio_path = wav_path
        
        # 1. Диаризация
        print("\n1. Запуск Pyannote (Диаризация) для нарезки аудио...")
        t0 = time.time()
        diarization_segments = self.diarizer.diarize(audio_path)
        print(f"   ✅ Диаризация: {time.time() - t0:.1f}s, найдено {len(diarization_segments)} сегментов")
        
        # 2. Загрузка аудио + мерджинг сегментов
        print("\n2. Загрузка аудио и подготовка чанков...")
        t0 = time.time()
        audio = AudioSegment.from_file(audio_path)
        
        # Сортируем и мерджим соседние сегменты одного спикера
        diarization_segments = sorted(diarization_segments, key=lambda x: x["start"])
        merged_segments = self._merge_segments(diarization_segments)
        print(f"   ✅ Загрузка: {time.time() - t0:.1f}s")
        print(f"   📊 Мерджинг: {len(diarization_segments)} → {len(merged_segments)} сегментов")
        
        # 3. Подготовка чанков в памяти (NumPy) — без записи на диск
        chunks = []
        for seg in merged_segments:
            start_ms = int(seg["start"] * 1000)
            end_ms = int(seg["end"] * 1000)
            chunk = audio[start_ms:end_ms]
            
            # Игнорируем слишком короткие звуки (менее 0.5 секунды)
            if len(chunk) < 500:
                continue
            
            audio_array = self._audio_chunk_to_numpy(chunk)
            seg_info = {
                "start": seg["start"],
                "end": seg["end"],
                "speaker": seg["speaker"].replace("SPEAKER_", "Спикер ")
            }
            chunks.append((audio_array, seg_info))
        
        # 4. Параллельная транскрипция через ThreadPoolExecutor
        print(f"\n3. Транскрибация {len(chunks)} чанков (×{self.max_workers} потока, in-memory)...")
        t0 = time.time()
        
        final_protocol = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Отправляем все чанки на транскрипцию
            future_to_idx = {}
            for idx, (audio_array, seg_info) in enumerate(chunks):
                future = executor.submit(self._transcribe_chunk, audio_array, seg_info)
                future_to_idx[future] = idx
            
            # Собираем результаты по мере готовности
            results = [None] * len(chunks)
            done_count = 0
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                result = future.result()
                results[idx] = result
                done_count += 1
                if done_count % 10 == 0 or done_count == len(chunks):
                    print(f"   📝 Транскрибировано: {done_count}/{len(chunks)}")
        
        # Собираем в правильном порядке (по таймстампу)
        for result in results:
            if result is not None:
                # Округляем таймстемпы до 1 знака после запятой
                result['start'] = round(result['start'], 1)
                result['end'] = round(result['end'], 1)
                if 'segments' in result:
                    for s in result['segments']:
                        s['start'] = round(s['start'], 1)
                        s['end'] = round(s['end'], 1)
                final_protocol.append(result)
                
        print(f"   ✅ Транскрипция: {time.time() - t0:.1f}s")
        
        # 5. Формируем текст для LLM
        formatted_text = ""
        for entry in final_protocol:
            print(f"[{entry['start']:.1f}s -> {entry['end']:.1f}s] {entry['speaker']} ({entry['language']}): {entry['text']}")
            formatted_text += f"[{entry['start']:.1f} - {entry['end']:.1f}] {entry['speaker']}: {entry['text']}\n"

        # 6. LLM суммаризация
        print("\n4. Генерация итогового протокола через Ollama...")
        t0 = time.time()
        ai_protocol = self.protocol_generator.generate_protocol(formatted_text)
        print(f"   ✅ LLM: {time.time() - t0:.1f}s")
        
        print(f"\n⏱️ ИТОГО: {time.time() - t_total:.1f}s")
            
        return {
            "formatted_transcript": formatted_text,
            "ai_summary": ai_protocol,
            "raw_data": final_protocol
        }
