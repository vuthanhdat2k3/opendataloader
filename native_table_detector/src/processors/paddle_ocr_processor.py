import logging
import os
import threading
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

try:
    from paddleocr import PPStructure as _PPStructure
except Exception:
    _PPStructure = None

try:
    from paddleocr import PPStructureV3 as _PPStructureV3
except Exception:
    _PPStructureV3 = None

from paddleocr import PaddleOCR


class PaddleOCRProcessor:
    """Production-grade OCR processor for Vietnamese documents with complex layouts."""

    _engine_lock = threading.Lock()
    _ocr_engine: Optional[PaddleOCR] = None
    _layout_engine: Optional[Any] = None

    def __init__(
        self,
        lang: str = "vi",
        use_gpu: bool = True,
        use_tensorrt: bool = True,
        precision: str = "fp16",
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.lang = lang
        self.use_gpu = use_gpu
        self.use_tensorrt = use_tensorrt
        self.precision = precision
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        self._ensure_engines_initialized()

    def _ensure_engines_initialized(self) -> None:
        if self.__class__._ocr_engine is not None and self.__class__._layout_engine is not None:
            return

        with self.__class__._engine_lock:
            if self.__class__._ocr_engine is not None and self.__class__._layout_engine is not None:
                return

            self.logger.info("Initializing PaddleOCR + PPStructure engines (singleton)...")
            self.__class__._ocr_engine = self._build_ocr_engine()
            self.__class__._layout_engine = self._build_layout_engine()
            self.logger.info("PaddleOCR engines initialized successfully.")

    def _build_ocr_engine(self) -> PaddleOCR:
        device = "gpu" if self.use_gpu else "cpu"
        base_kwargs = {
            "lang": self.lang,
            "device": device,
            "use_tensorrt": self.use_tensorrt,
            "precision": self.precision,
            "use_textline_orientation": True,
            # Avoid MKLDNN/oneDNN executor paths that can crash on some builds.
            "enable_mkldnn": False,
            "det": True,
            "rec": True,
            "cls": True,
            "show_log": False,
        }
        return self._create_with_fallback(PaddleOCR, base_kwargs, "PaddleOCR")

    def _build_layout_engine(self) -> Any:
        if _PPStructure is not None:
            device = "gpu" if self.use_gpu else "cpu"
            base_kwargs = {
                "lang": self.lang,
                "device": device,
                "use_tensorrt": self.use_tensorrt,
                "precision": self.precision,
                "enable_mkldnn": False,
                "layout": True,
                "table": True,
                "ocr": True,
                "show_log": False,
                # Orientation classifier for 0/90/180/270 at document-level in PP-Structure.
                "image_orientation": True,
            }
            return self._create_with_fallback(_PPStructure, base_kwargs, "PPStructure")

        if _PPStructureV3 is not None:
            # PaddleOCR>=3 exports PPStructureV3 instead of PPStructure.
            base_kwargs = {
                "lang": self.lang,
                "use_doc_orientation_classify": True,
                "use_textline_orientation": True,
                "use_table_recognition": True,
                "use_region_detection": True,
            }
            return self._create_with_fallback(_PPStructureV3, base_kwargs, "PPStructureV3")

        raise RuntimeError("Neither PPStructure nor PPStructureV3 is available in paddleocr.")

    def _create_with_fallback(self, cls_obj: Any, kwargs: Dict[str, Any], name: str) -> Any:
        try:
            return cls_obj(**kwargs)
        except Exception as first_error:
            self.logger.warning(
                "%s advanced config failed, fallback to safer config. Error: %s",
                name,
                first_error,
            )

        fallback_kwargs = dict(kwargs)
        fallback_kwargs["use_tensorrt"] = False
        fallback_kwargs.pop("precision", None)
        fallback_kwargs.pop("image_orientation", None)

        try:
            return cls_obj(**fallback_kwargs)
        except Exception as second_error:
            self.logger.exception("%s initialization failed after fallback.", name)
            raise RuntimeError(f"Cannot initialize {name}: {second_error}") from second_error

    def process_document(
        self, image_input: Union[str, np.ndarray]
    ) -> List[Dict[str, Any]]:
        """
        Process an image path or numpy image and return structured regions.

        Returns:
            List[Dict]: [{
                "type": "text|table|figure",
                "bbox": [[x1, y1], [x2, y2], [x3, y3], [x4, y4]],
                "content": text or html,
                "confidence_score": float
            }]
        """
        try:
            image_bgr = self._load_image(image_input)
            rotation = self._detect_best_rotation(image_bgr)
            if rotation != 0:
                self.logger.info("Rotate image by %d degrees to normalize orientation.", rotation)
                image_bgr = self._rotate_orthogonal(image_bgr, rotation)

            layout_engine = self.__class__._layout_engine
            if layout_engine is None:
                raise RuntimeError("PPStructure engine is not initialized.")

            layout_results = layout_engine(image_bgr)
            structured_output: List[Dict[str, Any]] = []

            for region in layout_results:
                parsed = self._parse_layout_region(region=region, image_bgr=image_bgr)
                if parsed is not None:
                    structured_output.append(parsed)

            return structured_output
        except Exception:
            self.logger.exception("Document processing failed.")
            return []

    def _load_image(self, image_input: Union[str, np.ndarray]) -> np.ndarray:
        if isinstance(image_input, str):
            if not os.path.exists(image_input):
                raise FileNotFoundError(f"Input image path does not exist: {image_input}")
            image = cv2.imread(image_input)
            if image is None:
                raise ValueError(f"Cannot read image from path: {image_input}")
            return image

        if isinstance(image_input, np.ndarray):
            if image_input.size == 0:
                raise ValueError("Input numpy image is empty.")
            if image_input.ndim == 2:
                return cv2.cvtColor(image_input, cv2.COLOR_GRAY2BGR)
            if image_input.ndim == 3 and image_input.shape[2] == 3:
                return image_input.copy()
            if image_input.ndim == 3 and image_input.shape[2] == 4:
                return cv2.cvtColor(image_input, cv2.COLOR_BGRA2BGR)
            raise ValueError(f"Unsupported numpy image shape: {image_input.shape}")

        raise TypeError("image_input must be a file path (str) or numpy array.")

    def _detect_best_rotation(self, image_bgr: np.ndarray) -> int:
        """Pick best orientation among 0/90/180/270 by OCR confidence."""
        ocr_engine = self.__class__._ocr_engine
        if ocr_engine is None:
            raise RuntimeError("PaddleOCR engine is not initialized.")

        h, w = image_bgr.shape[:2]
        scale = min(1.0, 1024.0 / max(h, w))
        if scale < 1.0:
            resized = cv2.resize(image_bgr, (int(w * scale), int(h * scale)))
        else:
            resized = image_bgr

        best_angle = 0
        best_score = -1.0

        for angle in (0, 90, 180, 270):
            rotated = self._rotate_orthogonal(resized, angle)
            try:
                # Full pipeline: DBNet detection + angle cls + SVTR-LCNet recognition.
                try:
                    ocr_result = ocr_engine.ocr(rotated, cls=True, det=True, rec=True)
                except TypeError:
                    ocr_result = ocr_engine.ocr(rotated)
                except ValueError:
                    ocr_result = ocr_engine.ocr(rotated)
                score = self._score_ocr_result(ocr_result)
            except Exception as e:
                self.logger.warning("Rotation %d failed during OCR scoring: %s", angle, e)
                score = -1.0

            if score > best_score:
                best_score = score
                best_angle = angle

        self.logger.info("Selected orientation angle=%d with score=%.4f", best_angle, best_score)
        return best_angle

    def _score_ocr_result(self, ocr_result: Any) -> float:
        total_conf = 0.0
        count = 0

        if not isinstance(ocr_result, list):
            return 0.0

        for page in ocr_result:
            if not isinstance(page, list):
                continue
            for line in page:
                if not isinstance(line, (list, tuple)) or len(line) < 2:
                    continue
                rec_info = line[1]
                if isinstance(rec_info, (list, tuple)) and len(rec_info) >= 2:
                    conf = rec_info[1]
                    try:
                        total_conf += float(conf)
                        count += 1
                    except Exception:
                        continue

        if count == 0:
            return 0.0
        return total_conf / count

    def _parse_layout_region(
        self, region: Dict[str, Any], image_bgr: np.ndarray
    ) -> Optional[Dict[str, Any]]:
        try:
            raw_type = str(region.get("type", "text")).lower()
            normalized_type = self._normalize_region_type(raw_type)
            bbox = self._normalize_bbox(region)

            if normalized_type == "table":
                content = self._extract_table_html(region)
                confidence = self._extract_region_confidence(region)
            elif normalized_type == "figure":
                content = ""
                confidence = self._extract_region_confidence(region)
            else:
                content, confidence = self._extract_text_content(region, image_bgr)

            return {
                "type": normalized_type,
                "bbox": bbox,
                "content": content,
                "confidence_score": float(confidence),
            }
        except Exception:
            self.logger.exception("Failed to parse layout region: %s", region)
            return None

    def _normalize_region_type(self, raw_type: str) -> str:
        if raw_type in {"table"}:
            return "table"
        if raw_type in {"figure", "image", "pic"}:
            return "figure"
        return "text"

    def _normalize_bbox(self, region: Dict[str, Any]) -> List[List[int]]:
        bbox = region.get("bbox")
        if isinstance(bbox, (list, tuple)) and len(bbox) == 4 and all(
            isinstance(v, (int, float)) for v in bbox
        ):
            x0, y0, x1, y1 = [int(round(v)) for v in bbox]
            return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]

        if isinstance(bbox, (list, tuple)) and len(bbox) == 4 and all(
            isinstance(v, (list, tuple)) and len(v) == 2 for v in bbox
        ):
            return [[int(round(pt[0])), int(round(pt[1]))] for pt in bbox]

        points = region.get("points")
        if isinstance(points, (list, tuple)) and len(points) == 4:
            return [[int(round(pt[0])), int(round(pt[1]))] for pt in points]

        return []

    def _extract_table_html(self, region: Dict[str, Any]) -> str:
        res = region.get("res")
        if isinstance(res, dict):
            html = res.get("html")
            if isinstance(html, str):
                return html

        if isinstance(res, list):
            for item in res:
                if isinstance(item, dict) and isinstance(item.get("html"), str):
                    return item["html"]

        self.logger.warning("Table region does not contain HTML, returning empty content.")
        return ""

    def _extract_text_content(
        self, region: Dict[str, Any], image_bgr: np.ndarray
    ) -> Tuple[str, float]:
        text_chunks: List[str] = []
        conf_list: List[float] = []

        res = region.get("res")
        if isinstance(res, list):
            for item in res:
                if not isinstance(item, dict):
                    continue
                text = item.get("text")
                score = item.get("confidence", item.get("score", item.get("conf", 0.0)))
                if isinstance(text, str) and text.strip():
                    text_chunks.append(text.strip())
                    try:
                        conf_list.append(float(score))
                    except Exception:
                        conf_list.append(0.0)

        if text_chunks:
            mean_conf = sum(conf_list) / max(1, len(conf_list))
            return "\n".join(text_chunks), mean_conf

        # Fallback OCR on cropped text area when PPStructure has empty OCR result.
        bbox = self._normalize_bbox(region)
        if len(bbox) != 4:
            return "", 0.0

        x_coords = [pt[0] for pt in bbox]
        y_coords = [pt[1] for pt in bbox]
        x0, x1 = max(0, min(x_coords)), min(image_bgr.shape[1], max(x_coords))
        y0, y1 = max(0, min(y_coords)), min(image_bgr.shape[0], max(y_coords))

        if x1 <= x0 or y1 <= y0:
            return "", 0.0

        crop = image_bgr[y0:y1, x0:x1]
        ocr_engine = self.__class__._ocr_engine
        if ocr_engine is None:
            return "", 0.0

        try:
            try:
                ocr_res = ocr_engine.ocr(crop, cls=True, det=True, rec=True)
            except TypeError:
                ocr_res = ocr_engine.ocr(crop)
            except ValueError:
                ocr_res = ocr_engine.ocr(crop)
            return self._flatten_ocr_text(ocr_res)
        except Exception as e:
            self.logger.warning("Fallback OCR failed on text region: %s", e)
            return "", 0.0

    def _flatten_ocr_text(self, ocr_result: Any) -> Tuple[str, float]:
        texts: List[str] = []
        confs: List[float] = []

        if isinstance(ocr_result, list):
            for page in ocr_result:
                if not isinstance(page, list):
                    continue
                for line in page:
                    if not isinstance(line, (list, tuple)) or len(line) < 2:
                        continue
                    rec_info = line[1]
                    if isinstance(rec_info, (list, tuple)) and len(rec_info) >= 2:
                        txt, conf = rec_info[0], rec_info[1]
                        if isinstance(txt, str) and txt.strip():
                            texts.append(txt.strip())
                            try:
                                confs.append(float(conf))
                            except Exception:
                                confs.append(0.0)

        if not texts:
            return "", 0.0

        return "\n".join(texts), sum(confs) / max(1, len(confs))

    def _extract_region_confidence(self, region: Dict[str, Any]) -> float:
        for key in ("confidence", "score", "conf"):
            val = region.get(key)
            if isinstance(val, (int, float)):
                return float(val)

        res = region.get("res")
        if isinstance(res, dict):
            for key in ("confidence", "score", "conf"):
                val = res.get(key)
                if isinstance(val, (int, float)):
                    return float(val)

        if isinstance(res, list):
            confs = []
            for item in res:
                if not isinstance(item, dict):
                    continue
                for key in ("confidence", "score", "conf"):
                    val = item.get(key)
                    if isinstance(val, (int, float)):
                        confs.append(float(val))
                        break
            if confs:
                return sum(confs) / len(confs)

        return 0.0

    def _rotate_orthogonal(self, image_bgr: np.ndarray, angle: int) -> np.ndarray:
        normalized = angle % 360
        if normalized == 90:
            return cv2.rotate(image_bgr, cv2.ROTATE_90_CLOCKWISE)
        if normalized == 180:
            return cv2.rotate(image_bgr, cv2.ROTATE_180)
        if normalized == 270:
            return cv2.rotate(image_bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return image_bgr
