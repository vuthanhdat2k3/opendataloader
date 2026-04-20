from __future__ import annotations

import argparse
from pathlib import Path

from src.pipeline.orchestrator import run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Run native table detector pipeline")
    parser.add_argument("--pdf", type=str, default="data/rotated_tables_clean.pdf")
    parser.add_argument("--out", type=str, default="native_table_detector/output/test_pipeline/")
    parser.add_argument("--threshold", type=float, default=2.0)
    parser.add_argument("--overlap-threshold", type=float, default=0.08)
    args = parser.parse_args()

    result = run_pipeline(
        pdf_path=Path(args.pdf),
        output_dir=Path(args.out),
        angle_threshold=args.threshold,
        overlap_threshold=args.overlap_threshold,
    )
    print(
        f"done: detected={result.stage2.detector_rotated_total}, "
        f"replaced={result.stage4.tables_replaced}, metrics={result.metrics}"
    )


if __name__ == "__main__":
    main()

