import os
import shutil
from fastapi import FastAPI, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pathlib import Path
import pandas as pd

# ПАТЧ ДЛЯ MAC: Чтобы torchcodec находил системный ffmpeg из Homebrew
if os.path.exists("/opt/homebrew/lib"):
    os.environ["DYLD_FALLBACK_LIBRARY_PATH"] = "/opt/homebrew/lib:" + os.environ.get("DYLD_FALLBACK_LIBRARY_PATH", "")

from core.processor import MeetingProcessor

app = FastAPI(title="AI Meeting Protocol API")

# Токен читаем из переменных окружения для безопасности (GitHub Push Protection)
HF_TOKEN = os.environ.get("HF_TOKEN", "ВАШ_ТОКЕН_ЗДЕСЬ")

print("⏳ Инициализация ИИ-моделей (Whisper + Pyannote + Ollama)...")
processor = MeetingProcessor(hf_token=HF_TOKEN, llm_model="llama3.1")
print("✅ Все модели успешно загружены в память!")

# Храним последний результат в памяти для простого экспорта в CSV
last_result = None

@app.post("/api/process")
async def process_audio(file: UploadFile = File(...)):
    global last_result
    
    base_dir = Path(__file__).resolve().parent.parent.parent
    audio_dir = base_dir / "data" / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    
    file_path = audio_dir / file.filename
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    print(f"\n🎧 Получен файл на обработку: {file.filename}")
    result = processor.process(str(file_path))
    last_result = result
    
    return result

@app.get("/api/export/csv")
async def export_csv():
    if not last_result:
        return {"error": "Нет данных для экспорта. Сначала загрузите аудио."}
        
    # Преобразуем сырой лог диалога в таблицу Pandas
    df = pd.DataFrame(last_result["raw_data"])
    
    # Сохраняем во временный CSV
    csv_path = Path(__file__).resolve().parent.parent.parent / "data" / "protocol.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig") # utf-8-sig для Excel
    
    return FileResponse(
        path=csv_path, 
        filename="meeting_protocol.csv", 
        media_type="text/csv"
    )

# Раздаем статические файлы фронтенда
frontend_path = Path(__file__).resolve().parent.parent.parent / "frontend"
app.mount("/", StaticFiles(directory=str(frontend_path), html=True), name="frontend")
