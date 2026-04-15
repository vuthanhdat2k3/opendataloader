import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np


class PaddleOCRTableProcessor:
    """OCR processor using PaddleOCR for table text extraction."""

    def __init__(
        self,
        lang: str = "vi",
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.lang = lang
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        self._engine = None
        self._ensure_engine_initialized()

    def _ensure_engine_initialized(self):
        if self._engine is None:
            self.logger.info("Initializing PaddleOCR engine...")
            self._patch_numpy_sctypes()
            from paddleocr import PaddleOCR

            ocr_kwargs = {
                "lang": self.lang,
                "use_textline_orientation": True,
                "device": "cpu",
                "enable_mkldnn": False,
            }
            if self.lang.lower() in {"vi", "vietnamese"}:
                vi_cfg = self._resolve_vi_ocr_config()
                if vi_cfg:
                    ocr_kwargs.update(vi_cfg)
                else:
                    self.logger.warning(
                        "Vietnamese OCR assets not found. Falling back to Paddle default language resolution for '%s'. "
                        "Set PADDLEOCR_VI_REC_MODEL_DIR and PADDLEOCR_VI_DICT_PATH for best Vietnamese accuracy.",
                        self.lang,
                    )

            self._engine = PaddleOCR(**ocr_kwargs)
            self.logger.info("PaddleOCR engine initialized.")

    def _patch_numpy_sctypes(self) -> None:
        """
        Compat shim for deps that still access np.sctypes (removed in NumPy 2.0).
        """
        if hasattr(np, "sctypes"):
            return
        np.sctypes = {
            "int": [np.int8, np.int16, np.int32, np.int64],
            "uint": [np.uint8, np.uint16, np.uint32, np.uint64],
            "float": [np.float16, np.float32, np.float64],
            "complex": [np.complex64, np.complex128],
            "others": [np.bool_, np.object_, np.str_],
        }

    def _resolve_vi_ocr_config(self) -> Optional[Dict[str, Any]]:
        """
        Prefer explicit Vietnamese recognizer assets for correct diacritics.
        Users can provide:
          - PADDLEOCR_VI_REC_MODEL_DIR
          - PADDLEOCR_VI_DICT_PATH
        """
        model_dir_env = (os.getenv("PADDLEOCR_VI_REC_MODEL_DIR") or "").strip()
        dict_path_env = (os.getenv("PADDLEOCR_VI_DICT_PATH") or "").strip()

        model_dir = Path(model_dir_env) if model_dir_env else None
        dict_path = Path(dict_path_env) if dict_path_env else None

        if not model_dir or not dict_path:
            return None
        if not model_dir.exists() or not model_dir.is_dir():
            self.logger.warning(
                "PADDLEOCR_VI_REC_MODEL_DIR is set but invalid: %s", model_dir
            )
            return None
        if not dict_path.exists() or not dict_path.is_file():
            self.logger.warning(
                "PADDLEOCR_VI_DICT_PATH is set but invalid: %s", dict_path
            )
            return None

        self.logger.info(
            "Using Vietnamese OCR assets: rec_model_dir=%s, dict=%s",
            model_dir,
            dict_path,
        )
        # Keep a compatible language key while forcing recognizer assets.
        return {
            "lang": "en",
            "rec_model_dir": str(model_dir),
            "rec_char_dict_path": str(dict_path),
        }

    def _to_bgr(self, image_rgb: np.ndarray) -> np.ndarray:
        if image_rgb.ndim == 2:
            return cv2.cvtColor(image_rgb, cv2.COLOR_GRAY2BGR)
        if image_rgb.ndim == 3 and image_rgb.shape[2] == 3:
            return cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        if image_rgb.ndim == 3 and image_rgb.shape[2] == 4:
            return cv2.cvtColor(image_rgb, cv2.COLOR_RGBA2BGR)
        return image_rgb

    def process_table_image(self, image_rgb: np.ndarray) -> Dict[str, Any]:
        """
        Process a table image and extract structured text.

        Args:
            image_rgb: RGB image of a table (already deskewed and orientation-corrected)

        Returns:
            Dict with keys:
                - html: HTML table string
                - markdown: Markdown table string
                - text: Plain text with cell separators
                - confidence: Average OCR confidence
                - ocr_result: Raw PaddleOCR result for debugging
        """
        try:
            h, w = image_rgb.shape[:2]
            if h < 20 or w < 20:
                return self._empty_result()

            image_bgr = self._to_bgr(image_rgb)

            try:
                ocr_result = self._engine.ocr(image_bgr, cls=True)
            except TypeError:
                ocr_result = self._engine.ocr(image_bgr)
            except ValueError:
                ocr_result = self._engine.ocr(image_bgr)

            if not ocr_result or not ocr_result[0]:
                return self._empty_result()

            items = self._parse_ocr_result(ocr_result)
            rows = self._group_into_rows(items)
            html = self._rows_to_html(rows)
            markdown = self._rows_to_markdown(rows)
            text = self._rows_to_text(rows)
            avg_conf = self._calculate_avg_confidence(items)

            return {
                "html": html,
                "markdown": markdown,
                "text": text,
                "confidence": avg_conf,
                "ocr_result": ocr_result,
            }

        except Exception:
            self.logger.exception("Table OCR processing failed")
            return self._empty_result()

    def _parse_ocr_result(self, ocr_result: Any) -> List[Dict]:
        items = []
        for line in ocr_result[0]:
            if not line or len(line) < 2:
                continue
            bbox = line[0]
            text_info = line[1]
            if isinstance(text_info, (list, tuple)) and len(text_info) >= 2:
                text = text_info[0]
                conf = float(text_info[1])
            else:
                text = str(text_info)
                conf = 0.5

            x_coords = [p[0] for p in bbox]
            y_coords = [p[1] for p in bbox]
            x0, x1 = min(x_coords), max(x_coords)
            y0, y1 = min(y_coords), max(y_coords)

            items.append(
                {
                    "text": text.strip(),
                    "x": x0,
                    "y": y0,
                    "w": x1 - x0,
                    "h": y1 - y0,
                    "conf": conf,
                    "bbox": bbox,
                }
            )

        return items

    def _group_into_rows(
        self, items: List[Dict], row_gap_threshold: float = 1.5
    ) -> List[List[Dict]]:
        if not items:
            return []

        items_sorted = sorted(items, key=lambda x: (round(x["y"] / 30) * 30, x["x"]))

        cell_heights = [item["h"] for item in items_sorted if item["h"] > 0]
        avg_h = sum(cell_heights) / len(cell_heights) if cell_heights else 30
        gap_threshold = avg_h * row_gap_threshold

        rows: List[List[Dict]] = []
        current_row: List[Dict] = []
        current_y = None

        for item in items_sorted:
            y = item["y"]
            if current_y is None:
                current_y = y
                current_row.append(item)
            elif abs(y - current_y) <= gap_threshold:
                current_row.append(item)
            else:
                if current_row:
                    current_row.sort(key=lambda x: x["x"])
                    rows.append(current_row)
                current_row = [item]
                current_y = y

        if current_row:
            current_row.sort(key=lambda x: x["x"])
            rows.append(current_row)

        return rows

    def _calculate_avg_confidence(self, items: List[Dict]) -> float:
        if not items:
            return 0.0
        confs = [item["conf"] for item in items]
        return sum(confs) / len(confs)

    def _rows_to_html(self, rows: List[List[Dict]]) -> str:
        if not rows:
            return "<table><tr><td></td></tr></table>"

        html_parts = ["<table>"]
        for row in rows:
            html_parts.append("  <tr>")
            for cell in row:
                text = (
                    cell["text"]
                    .replace("&", "&amp;")
                    .replace("<", "&lt;")
                    .replace(">", "&gt;")
                )
                html_parts.append(f"    <td>{text}</td>")
            html_parts.append("  </tr>")
        html_parts.append("</table>")
        return "\n".join(html_parts)

    def _rows_to_markdown(self, rows: List[List[Dict]]) -> str:
        if not rows:
            return "| |\n| |"

        markdown_parts = []
        for row in rows:
            cells = [cell["text"] for cell in row]
            markdown_parts.append("| " + " | ".join(cells) + " |")

        if markdown_parts:
            col_count = len(rows[0]) if rows else 1
            markdown_parts.insert(1, "| " + " | ".join(["---"] * col_count) + " |")

        return "\n".join(markdown_parts)

    def _rows_to_text(self, rows: List[List[Dict]]) -> str:
        lines = []
        for row in rows:
            cells = [cell["text"] for cell in row]
            lines.append(" | ".join(cells))
        return "\n".join(lines)

    def _empty_result(self) -> Dict[str, Any]:
        return {
            "html": "<table><tr><td></td></tr></table>",
            "markdown": "| |\n| |",
            "text": "",
            "confidence": 0.0,
            "ocr_result": None,
        }

    def process_image_simple(self, image_rgb: np.ndarray) -> Tuple[str, float]:
        """
        Simple OCR on image without table structure detection.

        Returns:
            Tuple of (text, confidence)
        """
        try:
            image_bgr = self._to_bgr(image_rgb)
            try:
                ocr_result = self._engine.ocr(image_bgr, cls=True)
            except TypeError:
                ocr_result = self._engine.ocr(image_bgr)
            except ValueError:
                ocr_result = self._engine.ocr(image_bgr)

            if not ocr_result or not ocr_result[0]:
                return "", 0.0

            texts = []
            confs = []
            for line in ocr_result[0]:
                if line and len(line) >= 2:
                    text_info = line[1]
                    if isinstance(text_info, (list, tuple)) and len(text_info) >= 2:
                        texts.append(text_info[0])
                        confs.append(float(text_info[1]))
                    elif isinstance(text_info, str):
                        texts.append(text_info)
                        confs.append(0.5)

            text = "\n".join(texts)
            avg_conf = sum(confs) / len(confs) if confs else 0.0
            return text, avg_conf

        except Exception:
            self.logger.exception("Simple OCR failed")
            return "", 0.0
