#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DOC_ENV_DIR="${ROOT_DIR}/.venv-doc"

if [[ ! -x "${DOC_ENV_DIR}/bin/python" ]]; then
  echo "Missing DOC environment: ${DOC_ENV_DIR}"
  echo "Run: bash scripts/setup_split_envs.sh"
  exit 1
fi

if [[ $# -eq 0 ]]; then
  echo "Usage: bash scripts/run_doc_env_python.sh '<python code>'"
  echo "Example: bash scripts/run_doc_env_python.sh 'import opendataloader_pdf; print(opendataloader_pdf.__version__)'"
  exit 1
fi

exec "${DOC_ENV_DIR}/bin/python" -c "$1"
