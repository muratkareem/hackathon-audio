import os
import tempfile
from pydub import AudioSegment
from core.transcription.transcriber import MeetingTranscriber
from core.diarization.diarizer import SpeakerDiarizer
from core.llm.llm_processor import ProtocolGenerator

class MeetingProcessor:
    def __init__(self, hf_token: str, llm_model="llama3.1"):
        self.transcriber = MeetingTranscriber(model_size="large-v3")
        self.diarizer = SpeakerDiarizer(hf_token=hf_token)
        self.protocol_generator = ProtocolGenerator(model_name=llm_model)
        
    def process(self, audio_path: str):
        # 1. Сначала запускаем Pyannote для разделения аудио на фразы спикеров
        print("\n1. Запуск Pyannote (Диаризация) для нарезки аудио...")
        diarization_segments = self.diarizer.diarize(audio_path)
        
        print("\n2. Загрузка аудио в память для покадровой обработки...")
        audio = AudioSegment.from_file(audio_path)
        
        final_protocol = []
        formatted_text = ""
        
        print("\n3. Транскрибация каждой фразы отдельно (Smart Code-Switching)...")
        # Сортируем сегменты по времени на всякий случай
        diarization_segments = sorted(diarization_segments, key=lambda x: x["start"])
        
        for i, seg in enumerate(diarization_segments):
            speaker = seg["speaker"].replace("SPEAKER_", "Спикер ")
            start_ms = int(seg["start"] * 1000)
            end_ms = int(seg["end"] * 1000)
            
            chunk = audio[start_ms:end_ms]
            
            # Игнорируем слишком короткие звуки (менее 0.5 секунды), это обычно вздохи/шум
            if len(chunk) < 500:
                continue
                
            # Сохраняем фразу во временный файл
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
                chunk.export(temp_audio.name, format="wav")
                temp_path = temp_audio.name
            
            # Передаем этот маленький кусок в Whisper. 
            # Теперь Whisper обязан переопределять язык для КАЖДОЙ отдельной фразы!
            try:
                whisper_result = self.transcriber.transcribe(temp_path)
                text = whisper_result["full_text"]
                lang = whisper_result["language"]
            finally:
                os.remove(temp_path)
            
            if text.strip():
                print(f"[{seg['start']:.1f}s -> {seg['end']:.1f}s] {speaker} ({lang}): {text}")
                final_protocol.append({
                    "start": seg["start"],
                    "end": seg["end"],
                    "speaker": speaker,
                    "language": lang,
                    "text": text
                })
                formatted_text += f"[{seg['start']:.1f} - {seg['end']:.1f}] {speaker}: {text}\n"

        print("\n4. Генерация итогового протокола через Ollama...")
        ai_protocol = self.protocol_generator.generate_protocol(formatted_text)
            
        return {
            "formatted_transcript": formatted_text,
            "ai_summary": ai_protocol,
            "raw_data": final_protocol
        }
