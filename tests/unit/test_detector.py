import argparse
import json
from pathlib import Path
from typing import Dict, List

import cv2
import fitz
import numpy as np

from src.core.detector import NativePDFTableDetector


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


def _make_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_make_serializable(x) for x in obj]
    return obj


def run_detector_real(
    pdf_path: Path,
    out_dir: Path,
    max_pages: int,
    angle_threshold: float,
    spatial_dist_threshold: float,
    save_before: bool,
    save_after: bool,
) -> Dict:
    out_dir.mkdir(parents=True, exist_ok=True)

    detector = NativePDFTableDetector(
        angle_threshold=angle_threshold,
        spatial_dist_threshold=spatial_dist_threshold,
    )

    summary: Dict = {
        "pdf_path": str(pdf_path),
        "max_pages": int(max_pages),
        "angle_threshold": float(angle_threshold),
        "spatial_dist_threshold": float(spatial_dist_threshold),
        "pages": [],
        "totals": {"rotated_tables": 0},
    }

    images_dir = out_dir / "rotated_tables"
    images_dir.mkdir(parents=True, exist_ok=True)

    doc = fitz.open(str(pdf_path))
    try:
        total_pages = len(doc)
        pages_to_run = min(total_pages, max_pages) if max_pages > 0 else total_pages

        print(f"Processing PDF: {pdf_path}")
        print(f"Total pages: {total_pages}, run pages: {pages_to_run}")

        for page_idx in range(pages_to_run):
            page_no = page_idx + 1
            page = doc[page_idx]

            results = detector.process_page(page)
            rotated_tables: List[Dict] = results.get("rotated_tables", [])

            page_tables: List[Dict] = []
            for t_idx, table in enumerate(rotated_tables, start=1):
                angle = float(table.get("angle", 0.0))
                obb = table.get("obb", {})
                spans = table.get("spans", [])

                patch_before = table.get("patch_before_rotate")
                patch_after = table.get("patch_image")

                before_path = None
                after_path = None

                if (
                    save_before
                    and isinstance(patch_before, np.ndarray)
                    and patch_before.size > 0
                ):
                    before_name = (
                        f"page_{page_no}_table_{t_idx}_angle_{angle:+.1f}_before.png"
                    )
                    before_path = images_dir / before_name
                    cv2.imwrite(str(before_path), _to_bgr(patch_before))

                if save_after and isinstance(patch_after, np.ndarray) and patch_after.size > 0:
                    after_name = (
                        f"page_{page_no}_table_{t_idx}_angle_{angle:+.1f}_after.png"
                    )
                    after_path = images_dir / after_name
                    cv2.imwrite(str(after_path), _to_bgr(patch_after))

                page_tables.append(
                    {
                        "table_index": t_idx,
                        "angle": angle,
                        "obb": _make_serializable(obb),
                        "span_count": len(spans),
                        "patch_before_shape": list(patch_before.shape)
                        if isinstance(patch_before, np.ndarray)
                        else None,
                        "patch_after_shape": list(patch_after.shape)
                        if isinstance(patch_after, np.ndarray)
                        else None,
                        "image_before": str(before_path) if before_path else None,
                        "image_after": str(after_path) if after_path else None,
                    }
                )

            page_result = {
                "page": page_no,
                "rotation": page.rotation,
                "rotated_table_count": len(page_tables),
                "tables": page_tables,
            }
            summary["pages"].append(page_result)
            summary["totals"]["rotated_tables"] += len(page_tables)

            page_json = out_dir / f"page_{page_no}_detector.json"
            page_json.write_text(
                json.dumps(page_result, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print(f" -> Page {page_no}: {len(page_tables)} rotated table(s)")
            print(f"    [+] Saved page result: {page_json}")

    finally:
        doc.close()

    summary_path = out_dir / "detector_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\nDone detector real test")
    print(f"Summary: {summary_path}")
    print(f"Totals: {summary['totals']}")
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run NativePDFTableDetector on a real PDF and export JSON/images."
    )
    parser.add_argument("--pdf", type=str, default=None, help="Path to input PDF.")
    parser.add_argument(
        "--out-dir",
        type=str,
        default="results_detector_real",
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
    parser.add_argument("--no-save-before", action="store_true", help="Do not save before images.")
    parser.add_argument("--no-save-after", action="store_true", help="Do not save after images.")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    pdf_path = _resolve_pdf_path(args.pdf)
    out_dir = Path(args.out_dir).resolve()

    run_detector_real(
        pdf_path=pdf_path,
        out_dir=out_dir,
        max_pages=int(args.max_pages),
        angle_threshold=float(args.angle_threshold),
        spatial_dist_threshold=float(args.spatial_dist_threshold),
        save_before=not args.no_save_before,
        save_after=not args.no_save_after,
    )


if __name__ == "__main__":
    main()