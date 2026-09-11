#!/bin/bash
if [ -f .env ]; then
  source .env
else
  echo "❌ ОШИБКА: Файл .env не найден! Скопируйте .env.example в .env и вставьте ваш HF_TOKEN."
  exit 1
fi
export DYLD_LIBRARY_PATH=/opt/homebrew/lib:$DYLD_LIBRARY_PATH
source backend/venv/bin/activate
cd backend && uvicorn api.main:app --host 0.0.0.0 --port 8000
