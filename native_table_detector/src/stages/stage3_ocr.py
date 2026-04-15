from __future__ import annotations

from pathlib import Path

import fitz
import numpy as np
from PIL import Image

# Compat shim for imgaug/paddleocr on NumPy>=2.0.
if not hasattr(np, "sctypes"):
    np.sctypes = {
        "int": [np.int8, np.int16, np.int32, np.int64],
        "uint": [np.uint8, np.uint16, np.uint32, np.uint64],
        "float": [np.float16, np.float32, np.float64],
        "complex": [np.complex64, np.complex128],
        "others": [np.bool_, np.object_, np.str_],
    }

from paddleocr import PaddleOCR

try:
    from ..core.detector import NativePDFTableDetector
except ImportError:
    from src.core.detector import NativePDFTableDetector
from ..pipeline.contracts import OCRResult, Stage2Result, Stage3Result
from ..utils.io import save_json


class Stage3HybridOCR:
    def __init__(
        self,
        angle_threshold: float,
        save_debug_artifacts: bool = True,
        ocr_lang: str = "vi",
        ocr_use_gpu: bool = False,
        spatial_dist_threshold: float = 100.0,
        normalize_orientation: bool = False,
    ):
        self.save_debug_artifacts = save_debug_artifacts
        self.normalize_orientation = normalize_orientation
        try:
            self.ocr_engine = PaddleOCR(
                lang=ocr_lang,
                use_textline_orientation=True,
                use_gpu=ocr_use_gpu,
                # Avoid MKLDNN/oneDNN executor paths that can crash on some builds.
                enable_mkldnn=False,
            )
        except ValueError as exc:
            # Runtime can miss GPUs (driver/container mismatch). Fallback to CPU.
            if ocr_use_gpu and "GPU count is: 0" in str(exc):
                self.ocr_engine = PaddleOCR(
                    lang=ocr_lang,
                    use_textline_orientation=True,
                    use_gpu=False,
                    enable_mkldnn=False,
                )
            else:
                raise
        self.detector = NativePDFTableDetector(
            angle_threshold=angle_threshold,
            spatial_dist_threshold=spatial_dist_threshold,
        )

    @staticmethod
    def _call_ocr(ocr_engine: PaddleOCR, image_rgb: np.ndarray):
        """
        PaddleOCR v2 accepted `cls=...`; PaddleOCR v3 routes to `.predict()` which
        no longer supports `cls`. Try the old call first, then fallback.
        """
        try:
            return ocr_engine.ocr(image_rgb, cls=True)
        except TypeError:
            return ocr_engine.ocr(image_rgb)
        except ValueError:
            # PaddleOCR v3 raises ValueError("Unknown argument: cls")
            return ocr_engine.ocr(image_rgb)

    @staticmethod
    def _ocr_patch(patch_rgb: np.ndarray, ocr_engine: PaddleOCR) -> dict:
        result = Stage3HybridOCR._call_ocr(ocr_engine, patch_rgb)
        if not result:
            return {"markdown": "", "confidence": 0.0}

        items = []
        # PaddleOCR v2 format: result[0] = list[[bbox, (text, conf)], ...]
        # PaddleOCR v3 format: result = [ { "dt_polys": [np.ndarray(4,2),...],
        #                                 "rec_texts": [...], "rec_scores": [...] }, ...]
        first = result[0]
        if isinstance(first, dict) and "dt_polys" in first:
            polys = first.get("dt_polys") or []
            texts = first.get("rec_texts") or []
            scores = first.get("rec_scores") or []
            n = min(len(polys), len(texts), len(scores))
            for i in range(n):
                poly = polys[i]
                text = str(texts[i]).strip()
                try:
                    conf = float(scores[i])
                except Exception:
                    conf = 0.5
                try:
                    xs = [float(p[0]) for p in poly]
                    ys = [float(p[1]) for p in poly]
                except Exception:
                    continue
                if not text:
                    continue
                items.append({"text": text, "x": min(xs), "y": min(ys), "conf": conf})
        else:
            page0 = first
            if not isinstance(page0, list):
                return {"markdown": "", "confidence": 0.0}
            for line in page0:
                if line and len(line) >= 2:
                    bbox = line[0]
                    text_info = line[1]
                    if isinstance(text_info, (list, tuple)) and len(text_info) >= 2:
                        text = text_info[0]
                        conf = float(text_info[1])
                    else:
                        text = str(text_info)
                        conf = 0.5

                    try:
                        x_coords = [p[0] for p in bbox]
                        y_coords = [p[1] for p in bbox]
                    except Exception:
                        continue
                    items.append(
                        {
                            "text": str(text).strip(),
                            "x": min(x_coords),
                            "y": min(y_coords),
                            "conf": conf,
                        }
                    )

        if not items:
            return {"markdown": "", "confidence": 0.0}

        items.sort(key=lambda x: x["y"])
        rows = []
        current_row = [items[0]]
        current_y = items[0]["y"]
        row_threshold = 10
        for item in items[1:]:
            if abs(item["y"] - current_y) <= row_threshold:
                current_row.append(item)
            else:
                rows.append(sorted(current_row, key=lambda x: x["x"]))
                current_row = [item]
                current_y = item["y"]
        rows.append(sorted(current_row, key=lambda x: x["x"]))

        md_lines = []
        for row in rows:
            md_lines.append("| " + " | ".join([i["text"] for i in row]) + " |")
        if len(rows) > 1:
            md_lines.insert(1, "| " + " | ".join(["---"] * len(rows[0])) + " |")

        return {
            "markdown": "\n".join(md_lines),
            "confidence": sum(i["conf"] for i in items) / len(items),
        }

    def run(
        self,
        pdf_path: str,
        stage2: Stage2Result,
        output_dir: str,
        doc: fitz.Document | None = None,
    ) -> Stage3Result:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        local_doc = doc or fitz.open(pdf_path)
        hybrid_results: list[OCRResult] = []
        try:
            for det in stage2.detector_rotated_tables:
                selected_obb = det.detector_obb_tight or det.detector_obb
                if not selected_obb:
                    continue

                page = local_doc[det.page_number - 1]
                patch_before, patch_deskewed = self.detector._extract_cells_from_rotated(
                    page, selected_obb, det.angle
                )
                if self.normalize_orientation:
                    patch_deskewed = self.detector._normalize_patch_upright(patch_deskewed)
                patch_tight = self.detector._tight_crop_white_border(patch_deskewed)
                ocr_result = self._ocr_patch(patch_tight, self.ocr_engine)

                patch_dir = out / f"detection_{det.detection_id}"
                if self.save_debug_artifacts:
                    patch_dir.mkdir(exist_ok=True)
                    if isinstance(patch_before, np.ndarray) and patch_before.size > 0:
                        Image.fromarray(patch_before.astype("uint8"), mode="RGB").save(
                            patch_dir / "before.png"
                        )
                    if isinstance(patch_deskewed, np.ndarray) and patch_deskewed.size > 0:
                        Image.fromarray(patch_deskewed.astype("uint8"), mode="RGB").save(
                            patch_dir / "deskewed.png"
                        )
                    if isinstance(patch_tight, np.ndarray) and patch_tight.size > 0:
                        Image.fromarray(patch_tight.astype("uint8"), mode="RGB").save(
                            patch_dir / "tight.png"
                        )

                hybrid_results.append(
                    OCRResult(
                        detection_id=det.detection_id,
                        matched_table_id=det.matched_table_id,
                        page_number=det.page_number,
                        bbox=det.matched_bbox,
                        crop_mode="tight" if det.detector_obb_tight is not None else "loose",
                        detector_obb_tight=det.detector_obb_tight,
                        detector_obb=det.detector_obb,
                        angle=det.angle,
                        ocr_markdown=ocr_result["markdown"],
                        ocr_confidence=ocr_result["confidence"],
                        patch_before=str(patch_dir / "before.png"),
                        patch_deskewed=str(patch_dir / "deskewed.png"),
                        patch_tight=str(patch_dir / "tight.png"),
                    )
                )
        finally:
            if doc is None:
                local_doc.close()

        result = Stage3Result(total_rotated=len(hybrid_results), hybrid_results=hybrid_results)
        save_json(
            {
                "total_rotated": result.total_rotated,
                "hybrid_results": [
                    {
                        "detection_id": r.detection_id,
                        "matched_table_id": r.matched_table_id,
                        "page_number": r.page_number,
                        "bbox": r.bbox,
                        "crop_mode": r.crop_mode,
                        "detector_obb_tight": r.detector_obb_tight,
                        "detector_obb": r.detector_obb,
                        "angle": r.angle,
                        "ocr_markdown": r.ocr_markdown,
                        "ocr_confidence": r.ocr_confidence,
                        "patch_before": r.patch_before,
                        "patch_deskewed": r.patch_deskewed,
                        "patch_tight": r.patch_tight,
                    }
                    for r in hybrid_results
                ],
            },
            out / "stage3_hybrid_ocr.json",
        )
        return result

