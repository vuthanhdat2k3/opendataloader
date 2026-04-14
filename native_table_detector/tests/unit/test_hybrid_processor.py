import argparse
from pathlib import Path
from typing import Dict

import cv2
import numpy as np

from native_table_detector.src.processors.hybrid_processor import (
    HybridTableProcessor,
    export_results_to_json,
    generate_markdown_from_hybrid_results,
)


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


def _to_bgr(image_rgb: np.ndarray) -> np.ndarray:
    if image_rgb.ndim == 2:
        return cv2.cvtColor(image_rgb, cv2.COLOR_GRAY2BGR)
    if image_rgb.ndim == 3 and image_rgb.shape[2] == 3:
        return cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    if image_rgb.ndim == 3 and image_rgb.shape[2] == 4:
        return cv2.cvtColor(image_rgb, cv2.COLOR_RGBA2BGR)
    return image_rgb


def _save_rotated_table_images(results: Dict, images_dir: Path, save_before: bool, save_after: bool) -> int:
    images_dir.mkdir(parents=True, exist_ok=True)
    total = 0

    for page_result in results.get("pages", []):
        page_no = page_result.get("page_number", 0)
        for t_idx, table in enumerate(page_result.get("rotated_tables", []), start=1):
            angle = float(table.get("angle", 0.0))
            before_patch = table.get("patch_before_rotate")
            after_patch = table.get("patch_image")

            if save_before and isinstance(before_patch, np.ndarray) and before_patch.size > 0:
                before_name = f"page_{page_no}_table_{t_idx}_angle_{angle:+.1f}_before.png"
                cv2.imwrite(str(images_dir / before_name), _to_bgr(before_patch))

            if save_after and isinstance(after_patch, np.ndarray) and after_patch.size > 0:
                after_name = f"page_{page_no}_table_{t_idx}_angle_{angle:+.1f}_ocr.png"
                cv2.imwrite(str(images_dir / after_name), _to_bgr(after_patch))

            total += 1

    return total


def run_hybrid_real(
    pdf_path: Path,
    out_dir: Path,
    max_pages: int,
    angle_threshold: float,
    spatial_dist_threshold: float,
    ocr_lang: str,
    save_before: bool,
    save_after: bool,
) -> Dict:
    out_dir.mkdir(parents=True, exist_ok=True)

    processor = HybridTableProcessor(
        angle_threshold=angle_threshold,
        spatial_dist_threshold=spatial_dist_threshold,
        ocr_lang=ocr_lang,
    )

    print(f"Processing PDF with HybridTableProcessor: {pdf_path}")
    results = processor.process_pdf(pdf_path=pdf_path, max_pages=max_pages)

    json_path = out_dir / "hybrid_results.json"
    export_results_to_json(results, json_path)

    md_path = out_dir / "hybrid_output.md"
    md_text = generate_markdown_from_hybrid_results(results)
    md_path.write_text(md_text, encoding="utf-8")

    images_dir = out_dir / "rotated_tables"
    count = _save_rotated_table_images(results, images_dir, save_before, save_after)

    summary = {
        "pdf_path": str(pdf_path),
        "max_pages": int(max_pages),
        "total_pages": int(results.get("total_pages", 0)),
        "total_rotated_tables": int(results.get("total_rotated_tables", 0)),
        "saved_table_images": int(count),
        "json_output": str(json_path),
        "markdown_output": str(md_path),
        "images_dir": str(images_dir),
    }

    summary_path = out_dir / "hybrid_summary.txt"
    lines = [f"{k}: {v}" for k, v in summary.items()]
    summary_path.write_text("\n".join(lines), encoding="utf-8")

    print("Done hybrid real test")
    print(f"JSON: {json_path}")
    print(f"Markdown: {md_path}")
    print(f"Images: {images_dir}")
    print(f"Summary: {summary_path}")
    print(f"Total rotated tables: {summary['total_rotated_tables']}")
    return results


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run HybridTableProcessor on a real PDF and export JSON/Markdown/images."
    )
    parser.add_argument("--pdf", type=str, default=None, help="Path to input PDF.")
    parser.add_argument(
        "--out-dir",
        type=str,
        default="results_hybrid_real",
        help="Directory to save outputs.",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=1,
        help="Maximum number of pages to process. 0 means all pages.",
    )
    parser.add_argument(
        "--angle-threshold",
        type=float,
        default=2.0,
        help="Ignore rotations under this threshold.",
    )
    parser.add_argument(
        "--spatial-dist-threshold",
        type=float,
        default=100.0,
        help="Distance threshold for grouping rotated spans.",
    )
    parser.add_argument("--ocr-lang", type=str, default="vi", help="OCR language code.")
    parser.add_argument("--no-save-before", action="store_true", help="Do not save before images.")
    parser.add_argument("--no-save-after", action="store_true", help="Do not save after images.")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    pdf_path = _resolve_pdf_path(args.pdf)
    out_dir = Path(args.out_dir).resolve()

    run_hybrid_real(
        pdf_path=pdf_path,
        out_dir=out_dir,
        max_pages=int(args.max_pages),
        angle_threshold=float(args.angle_threshold),
        spatial_dist_threshold=float(args.spatial_dist_threshold),
        ocr_lang=str(args.ocr_lang),
        save_before=not args.no_save_before,
        save_after=not args.no_save_after,
    )


if __name__ == "__main__":
    main()