#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OCR_ENV_DIR="${ROOT_DIR}/.venv-ocr"

if [[ ! -x "${OCR_ENV_DIR}/bin/python" ]]; then
  echo "Missing OCR environment: ${OCR_ENV_DIR}"
  echo "Run: bash scripts/setup_split_envs.sh"
  exit 1
fi

# Avoid proxy interception for local Gradio self-check.
export NO_PROXY="${NO_PROXY:-},127.0.0.1,localhost"
export no_proxy="${no_proxy:-},127.0.0.1,localhost"
export GRADIO_SERVER_NAME="${GRADIO_SERVER_NAME:-127.0.0.1}"
export GRADIO_ANALYTICS_ENABLED="False"

exec "${OCR_ENV_DIR}/bin/python" "${ROOT_DIR}/main.py"
