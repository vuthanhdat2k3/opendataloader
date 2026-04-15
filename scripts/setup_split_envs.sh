#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OCR_ENV_DIR="${ROOT_DIR}/.venv-ocr"
DOC_ENV_DIR="${ROOT_DIR}/.venv-doc"

echo "[1/4] Sync OCR environment -> ${OCR_ENV_DIR}"
UV_PROJECT_ENVIRONMENT="${OCR_ENV_DIR}" uv sync --project "${ROOT_DIR}" --python 3.10 --no-dev

echo "[2/4] Sync DOC environment from pyproject.doc.toml -> ${DOC_ENV_DIR}"
"${DOC_ENV_DIR}/bin/python" --version >/dev/null 2>&1 || uv venv "${DOC_ENV_DIR}" --python 3.10
uv pip install --python "${DOC_ENV_DIR}/bin/python" "opendataloader-pdf[hybrid]==2.2.1"

echo "[3/4] Smoke check OCR env"
"${OCR_ENV_DIR}/bin/python" -c "import paddle, paddleocr; print('OCR env OK:', paddle.__version__)"

echo "[4/4] Smoke check DOC env"
"${DOC_ENV_DIR}/bin/python" -c "import opendataloader_pdf; print('DOC env OK')"

echo
echo "Done."
echo "- OCR env: ${OCR_ENV_DIR}"
echo "- DOC env: ${DOC_ENV_DIR}"
