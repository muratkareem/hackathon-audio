#!/bin/bash
if [ -f .env ]; then
  source .env
else
  echo "❌ ОШИБКА: Файл .env не найден! Скопируйте .env.example в .env и вставьте ваш HF_TOKEN."
  exit 1
fi
export DYLD_LIBRARY_PATH=/opt/homebrew/lib:$DYLD_LIBRARY_PATH
source backend/venv/bin/activate
python3 backend/test_pipeline.py
