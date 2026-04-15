# Split Environments (Option A)

This repository now supports two isolated Python environments to avoid dependency conflicts:

- `.venv-ocr`: `paddleocr` / `paddlex` / `paddlepaddle-gpu` stack for OCR pipeline and UI.
- `.venv-doc`: `opendataloader-pdf[hybrid]` / `docling` stack.

## Setup

```bash
bash scripts/setup_split_envs.sh
```

## Run OCR UI

```bash
bash scripts/run_ui_ocr.sh
```

## Run code in DOC environment

```bash
bash scripts/run_doc_env_python.sh "import opendataloader_pdf; print('ok')"
```

## Why this exists

`paddlex==3.0.0` and `opendataloader-pdf[hybrid]==2.2.1` pull conflicting `pandas` ranges, so a single environment is unstable.
