import argparse
import json
from pathlib import Path
from typing import Dict, List

import cv2
import fitz
import numpy as np

from paddle_ocr_processor import PaddleOCRProcessor


def _resolve_pdf_path(cli_pdf: str | None) -> Path:
    if cli_pdf:
        path = Path(cli_pdf)
        if path.exists() and path.is_file():
            return path.resolve()
        raise FileNotFoundError(f"PDF not found: {path}")

    candidates = [
        Path("../data/tet_rotated.pdf"),
        Path("../data/rotated_tables_clean.pdf"),
        Path("../../data/complex_tables_rotated.pdf"),
    ]
    for path in candidates:
        if path.exists() and path.is_file():
            return path.resolve()

    raise FileNotFoundError(
        "Could not find a default PDF. Pass one with --pdf /path/to/file.pdf"
    )


def _render_page_to_bgr(page: fitz.Page, zoom: float = 2.0) -> np.ndarray:
    matrix = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=matrix, alpha=False)
    image = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)

    if pix.n == 1:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if pix.n == 3:
        return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if pix.n == 4:
        return cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
    raise ValueError(f"Unsupported pixmap channel count: {pix.n}")


def _count_types(items: List[Dict]) -> Dict[str, int]:
    stats: Dict[str, int] = {"text": 0, "table": 0, "figure": 0}
    for item in items:
        typ = str(item.get("type", "text")).lower()
        if typ not in stats:
            stats[typ] = 0
        stats[typ] += 1
    return stats


def run_real_pdf_test(
    pdf_path: Path,
    out_dir: Path,
    max_pages: int,
    use_gpu: bool,
    use_tensorrt: bool,
    precision: str,
) -> Dict:
    out_dir.mkdir(parents=True, exist_ok=True)

    processor = PaddleOCRProcessor(
        lang="vi",
        use_gpu=use_gpu,
        use_tensorrt=use_tensorrt,
        precision=precision,
    )

    summary: Dict = {
        "pdf_path": str(pdf_path),
        "max_pages": int(max_pages),
        "pages": [],
        "totals": {"text": 0, "table": 0, "figure": 0, "all": 0},
    }

    doc = fitz.open(str(pdf_path))
    try:
        total_pages = len(doc)
        pages_to_run = min(total_pages, max_pages)

        print(f"Processing PDF: {pdf_path}")
        print(f"Total pages: {total_pages}, run pages: {pages_to_run}")

        for page_idx in range(pages_to_run):
            page_no = page_idx + 1
            page = doc[page_idx]
            image_bgr = _render_page_to_bgr(page, zoom=2.0)

            print(f" -> Running PaddleOCRProcessor on page {page_no}...")
            items = processor.process_document(image_bgr)
            type_stats = _count_types(items)

            page_result = {
                "page": page_no,
                "item_count": len(items),
                "type_stats": type_stats,
                "items": items,
            }
            summary["pages"].append(page_result)

            summary["totals"]["all"] += len(items)
            for k, v in type_stats.items():
                if k not in summary["totals"]:
                    summary["totals"][k] = 0
                summary["totals"][k] += v

            per_page_path = out_dir / f"page_{page_no}_paddle_ocr.json"
            per_page_path.write_text(
                json.dumps(page_result, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print(f"    [+] Saved page result: {per_page_path}")

    finally:
        doc.close()

    summary_path = out_dir / "paddle_ocr_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\nDone real PDF test with PaddleOCRProcessor")
    print(f"Summary: {summary_path}")
    print(f"Totals: {summary['totals']}")
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run PaddleOCRProcessor on a real PDF and export structured JSON outputs."
    )
    parser.add_argument("--pdf", type=str, default=None, help="Path to input PDF.")
    parser.add_argument(
        "--out-dir",
        type=str,
        default="results_paddle_ocr",
        help="Directory to save page JSON and summary.",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=1,
        help="Maximum number of pages to process.",
    )
    parser.add_argument("--cpu", action="store_true", help="Force CPU mode (disable GPU).")
    parser.add_argument(
        "--no-trt",
        action="store_true",
        help="Disable TensorRT.",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="fp16",
        choices=["fp16", "fp32", "int8"],
        help="Inference precision.",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    pdf_path = _resolve_pdf_path(args.pdf)
    out_dir = Path(args.out_dir).resolve()

    run_real_pdf_test(
        pdf_path=pdf_path,
        out_dir=out_dir,
        max_pages=max(1, int(args.max_pages)),
        use_gpu=not args.cpu,
        use_tensorrt=not args.no_trt,
        precision=args.precision,
    )


if __name__ == "__main__":
    main()
