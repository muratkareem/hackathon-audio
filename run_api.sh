#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
if [[ ! -x .venv/bin/python ]]; then
  echo 'Create .venv and install backend/requirements-ai.txt; see README.' >&2
  exit 1
fi
exec .venv/bin/python -m uvicorn api.main:app --app-dir backend --host 127.0.0.1 --port "${PORT:-8000}" --workers 1
