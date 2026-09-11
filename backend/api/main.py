import os
import shutil
from fastapi import FastAPI, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pathlib import Path
from pydantic import BaseModel
import pandas as pd

# ПАТЧ ДЛЯ MAC: Чтобы torchcodec находил системный ffmpeg из Homebrew
if os.path.exists("/opt/homebrew/lib"):
    os.environ["DYLD_FALLBACK_LIBRARY_PATH"] = "/opt/homebrew/lib:" + os.environ.get("DYLD_FALLBACK_LIBRARY_PATH", "")

from core.processor import MeetingProcessor
from core.llm.chat import MeetingRAG

app = FastAPI(title="AI Meeting Protocol API")

# Токен читаем из переменных окружения для безопасности (GitHub Push Protection)
HF_TOKEN = os.environ.get("HF_TOKEN", "ВАШ_ТОКЕН_ЗДЕСЬ")

LLM_MODEL = "llama3.1"

print("⏳ Инициализация ИИ-моделей (Whisper + Pyannote + Ollama)...")
processor = MeetingProcessor(hf_token=HF_TOKEN, llm_model=LLM_MODEL)
print("✅ Все модели успешно загружены в память!")

# RAG-чат по встрече
rag = MeetingRAG(model_name=LLM_MODEL)

# Храним последний результат в памяти для простого экспорта в CSV
last_result = None


class ChatRequest(BaseModel):
    question: str


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
    
    # Загружаем результат в RAG для чата
    rag.load(result)
    print("💬 RAG-чат готов к вопросам по встрече")
    
    return result


@app.post("/api/chat")
async def chat(req: ChatRequest):
    """RAG-чат: задавайте вопросы по содержимому встречи."""
    if not rag.is_loaded:
        return {"error": "Сначала загрузите и обработайте аудиофайл."}
    
    print(f"\n💬 Вопрос: {req.question}")
    answer = rag.ask(req.question)
    print(f"💬 Ответ: {answer[:100]}...")
    
    return {"answer": answer}


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

@app.get("/api/diagram")
async def generate_diagram():
    if not last_result or "ai_summary" not in last_result:
        return {"error": "Нет данных для генерации диаграммы."}
    
    # Используем ProtocolGenerator для вызова Ollama
    diagram_code = processor.protocol_generator.generate_diagram(last_result["ai_summary"])
    return {"mermaid_code": diagram_code}


@app.get("/api/export/json")
async def export_json():
    if not last_result:
        return {"error": "Нет данных для экспорта."}
    
    import json
    json_path = Path(__file__).resolve().parent.parent.parent / "data" / "protocol.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(last_result, f, ensure_ascii=False, indent=2)
        
    return FileResponse(
        path=json_path, 
        filename="meeting_protocol.json", 
        media_type="application/json"
    )

@app.get("/api/export/md")
async def export_md():
    if not last_result:
        return {"error": "Нет данных для экспорта."}
        
    md_path = Path(__file__).resolve().parent.parent.parent / "data" / "protocol.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(last_result.get("ai_summary", ""))
        
    return FileResponse(
        path=md_path, 
        filename="meeting_protocol.md", 
        media_type="text/markdown"
    )


@app.get("/api/generate_qa")
async def generate_qa():
    if not last_result or "ai_summary" not in last_result:
        return {"qa_pairs": []}
    
    # Генерация в фоне после основного ответа
    qa = processor.protocol_generator.generate_qa_pairs(last_result["ai_summary"])
    return {"qa_pairs": qa}

# Раздаем статические файлы фронтенда
frontend_path = Path(__file__).resolve().parent.parent.parent / "frontend"
app.mount("/", StaticFiles(directory=str(frontend_path), html=True), name="frontend")
