# Native Table Detector Architecture

## Overview
- `src/pipeline/`: orchestration, contracts, factory, config.
- `src/stages/`: stage implementations for ODL extraction, rotation detection, OCR, merge.
- `src/utils/`: shared operational utilities.

## Runtime Flow
1. Stage 1 extracts tables from OpenDataLoader output.
2. Stage 2 runs detector-first rotated table matching.
3. Stage 3 crops/deskews and OCRs rotated detections.
4. Stage 4 merges JSON and patches Stage-1 markdown blocks.

