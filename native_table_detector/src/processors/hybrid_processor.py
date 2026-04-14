import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import fitz
import numpy as np

from native_table_detector.src.core.detector import NativePDFTableDetector
from native_table_detector.src.processors.paddle_ocr_table_processor import PaddleOCRTableProcessor


class HybridTableProcessor:
    """Hybrid processor combining NativePDFTableDetector + TesseractOCR for rotated tables."""

    def __init__(
        self,
        angle_threshold: float = 2.0,
        spatial_dist_threshold: float = 100,
        ocr_lang: str = "vi",
        normalize_orientation: bool = False,
        logger: Optional[logging.Logger] = None,
    ):
        self.detector = NativePDFTableDetector(
            angle_threshold=angle_threshold,
            spatial_dist_threshold=spatial_dist_threshold,
        )
        self.ocr = PaddleOCRTableProcessor(lang=ocr_lang)
        self.normalize_orientation = normalize_orientation
        self.logger = logger or logging.getLogger(self.__class__.__name__)

    def process_page(self, page: fitz.Page) -> Dict[str, Any]:
        """
        Process a PDF page and OCR rotated tables.

        Returns:
            Dict with:
                - page_rotation: page rotation angle
                - straight_tables: list of tables detected as straight (not processed by OCR)
                - rotated_tables: list of dicts with table data including OCR results
        """
        results = {
            "page_rotation": page.rotation,
            "straight_tables": [],
            "rotated_tables": [],
        }

        groups = self.detector._group_spans_by_angle_and_space(page)

        for group_id, group_data in groups.items():
            angle = group_data["angle"]
            spans = group_data["spans"]

            if abs(angle) < self.detector.angle_threshold:
                continue
            else:
                obb = self.detector._compute_obb(page, spans, angle, padding=30)
                patch_before, patch_after = self.detector._extract_cells_from_rotated(
                    page, obb, angle
                )

                if not isinstance(patch_after, np.ndarray) or patch_after.size == 0:
                    continue

                # patch_after is already deskewed by NativePDFTableDetector.
                # Re-normalizing orientation here can rotate the table a second time.
                patch_oriented = (
                    self._normalize_patch(patch_after)
                    if self.normalize_orientation
                    else patch_after
                )
                ocr_result = self.ocr.process_table_image(patch_oriented)

                table_data = {
                    "obb": obb,
                    "angle": angle,
                    "spans": spans,
                    "patch_image": patch_oriented,
                    "patch_deskewed": patch_after,
                    "patch_before_rotate": patch_before,
                    "ocr_html": ocr_result["html"],
                    "ocr_markdown": ocr_result["markdown"],
                    "ocr_text": ocr_result["text"],
                    "ocr_confidence": ocr_result["confidence"],
                }

                results["rotated_tables"].append(table_data)

        return results

    def _normalize_patch(self, patch_rgb: np.ndarray) -> np.ndarray:
        """Normalize patch orientation using doctr predictor."""
        try:
            from doctr.models import crop_orientation_predictor
        except Exception:
            crop_orientation_predictor = None

        if crop_orientation_predictor is None:
            if patch_rgb.shape[0] > patch_rgb.shape[1]:
                return cv2.rotate(patch_rgb, cv2.ROTATE_90_CLOCKWISE)
            return patch_rgb

        try:
            candidates = [
                (0, patch_rgb),
                (90, cv2.rotate(patch_rgb, cv2.ROTATE_90_CLOCKWISE)),
                (180, cv2.rotate(patch_rgb, cv2.ROTATE_180)),
                (270, cv2.rotate(patch_rgb, cv2.ROTATE_90_COUNTERCLOCKWISE)),
            ]

            best_patch = patch_rgb
            best_score = float("inf")

            for rot_deg, cand in candidates:
                _, angles, confidences = crop_orientation_predictor([cand])
                pred_angle = float(angles[0]) if len(angles) else 0.0
                conf = float(confidences[0]) if len(confidences) else 0.0
                quantized = int(round(pred_angle / 90.0) * 90) % 360

                mismatch = 0.0 if quantized == 0 else 1.0
                score = mismatch + (1.0 - conf) * 0.1
                if score < best_score:
                    best_score = score
                    best_patch = cand

            return best_patch
        except Exception:
            if patch_rgb.shape[0] > patch_rgb.shape[1]:
                return cv2.rotate(patch_rgb, cv2.ROTATE_90_CLOCKWISE)
            return patch_rgb

    def process_pdf(self, pdf_path: str | Path, max_pages: int = 0) -> Dict[str, Any]:
        """
        Process entire PDF and return results for all pages.

        Args:
            pdf_path: Path to PDF file
            max_pages: Maximum pages to process (0 = all)

        Returns:
            Dict with page results and summary
        """
        doc = fitz.open(str(pdf_path))
        total_rotated = 0

        summary: Dict[str, Any] = {
            "pdf_path": str(pdf_path),
            "total_pages": len(doc),
            "pages": [],
        }

        try:
            pages_to_process = min(len(doc), max_pages) if max_pages > 0 else len(doc)

            for page_idx in range(pages_to_process):
                page = doc[page_idx]
                page_result = self.process_page(page)
                page_result["page_number"] = page_idx + 1
                summary["pages"].append(page_result)
                total_rotated += len(page_result["rotated_tables"])

                self.logger.info(
                    f"Page {page_idx + 1}: found {len(page_result['rotated_tables'])} rotated tables"
                )

        finally:
            doc.close()

        summary["total_rotated_tables"] = total_rotated
        return summary


def generate_markdown_from_hybrid_results(results: Dict[str, Any]) -> str:
    """Generate markdown output containing only OCR table content."""
    md_parts = []

    for page_result in results.get("pages", []):
        for table in page_result.get("rotated_tables", []):
            markdown = (table.get("ocr_markdown", "") or "").strip()
            if markdown:
                md_parts.append(markdown)

    return "\n\n".join(md_parts)


def export_results_to_json(results: Dict[str, Any], output_path: Path) -> None:
    """Export hybrid results to JSON file."""
    import json

    serializable_results = _make_serializable(results)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(serializable_results, f, ensure_ascii=False, indent=2)


def _make_serializable(obj: Any) -> Any:
    """Convert numpy types to Python native types for JSON serialization."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_make_serializable(item) for item in obj]
    return obj
