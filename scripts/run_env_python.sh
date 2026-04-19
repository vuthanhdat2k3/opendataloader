#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ $# -eq 0 ]]; then
  echo "Usage: bash scripts/run_env_python.sh '<python code>'"
  echo "Example: bash scripts/run_env_python.sh 'import opendataloader_pdf; print(opendataloader_pdf.__version__)'"
  exit 1
fi

exec uv run --project "${ROOT_DIR}" python -c "$1"