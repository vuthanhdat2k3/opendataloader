#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "[1/1] Sync project environment"
uv sync --project "${ROOT_DIR}" --python 3.10

echo
echo "Done."
echo "- Environment: ${ROOT_DIR}/.venv"