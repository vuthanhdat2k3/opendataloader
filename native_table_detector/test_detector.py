import argparse
import json
from pathlib import Path

import cv2
import fitz
import numpy as np
import pytest

from detector import NativePDFTableDetector
from paddle_ocr_processor import PaddleOCRProcessor


class _FakeOCR:
	def __init__(self, score_by_angle=None):
		self.score_by_angle = score_by_angle or {0: 0.5}

	def ocr(self, image, cls=True, det=True, rec=True):
		angle = 0
		if isinstance(image, dict):
			angle = int(image.get("angle", 0))
		conf = float(self.score_by_angle.get(angle, 0.0))
		return [[[[0, 0], [1, 0], [1, 1], [0, 1]], ("text", conf)]]


class _FakeLayout:
	def __init__(self, regions):
		self.regions = regions

	def __call__(self, image):
		return self.regions


@pytest.fixture(autouse=True)
def _reset_paddle_ocr_singleton():
	PaddleOCRProcessor._ocr_engine = None
	PaddleOCRProcessor._layout_engine = None
	yield
	PaddleOCRProcessor._ocr_engine = None
	PaddleOCRProcessor._layout_engine = None


def _resolve_pdf_path(cli_pdf: str | None) -> Path:
	if cli_pdf:
		path = Path(cli_pdf)
		if path.exists():
			return path.resolve()
		raise FileNotFoundError(f"PDF not found: {path}")

	candidates = [
		Path("../data/tet_rotated.pdf"),
		Path("../data/rotated_tables_clean.pdf"),
		Path("../../data/complex_tables_rotated.pdf"),
	]
	for path in candidates:
		if path.exists():
			return path.resolve()

	raise FileNotFoundError(
		"Could not find a default PDF. Pass one with --pdf /path/to/file.pdf"
	)


def _to_bgr(image_rgb: np.ndarray) -> np.ndarray:
	if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
		return image_rgb
	return cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)


def run_detector(
	pdf_path: Path,
	out_dir: Path,
	angle_threshold: float,
	spatial_dist_threshold: float,
	save_before: bool,
	save_after: bool,
) -> int:
	out_dir.mkdir(parents=True, exist_ok=True)

	print(f"Processing PDF: {pdf_path}")
	detector = NativePDFTableDetector(
		angle_threshold=angle_threshold,
		spatial_dist_threshold=spatial_dist_threshold,
	)

	doc = fitz.open(str(pdf_path))
	total_rotated = 0
	manifest: list[dict] = []

	try:
		for page_idx in range(len(doc)):
			page = doc[page_idx]
			page_number = page_idx + 1
			results = detector.process_page(page)
			rotated_tables = results.get("rotated_tables", [])

			if not rotated_tables:
				continue

			print(f"\n--- Page {page_number} ---")
			print(f"Found {len(rotated_tables)} rotated table(s).")

			for table_idx, table in enumerate(rotated_tables, start=1):
				angle = float(table.get("angle", 0.0))
				patch_before = table.get("patch_before_rotate")
				patch_after = table.get("patch_image")

				if not isinstance(patch_after, np.ndarray) or patch_after.size == 0:
					print(f" -> Table {table_idx}: skipped (empty patch)")
					continue

				print(
					f" -> Table {table_idx}: rotated {angle:+.1f} degrees. "
					f"Size after deskew: {patch_after.shape}"
				)

				before_path = None
				after_path = None

				if save_before and isinstance(patch_before, np.ndarray) and patch_before.size > 0:
					before_name = (
						f"page_{page_number}_table_{table_idx}_"
						f"angle_{angle:+.1f}_before_rotate.png"
					)
					before_path = out_dir / before_name
					cv2.imwrite(str(before_path), _to_bgr(patch_before))
					print(f"    [+] Saved pre-rotate image to {before_path}")

				if save_after:
					after_name = (
						f"page_{page_number}_table_{table_idx}_"
						f"angle_{angle:+.1f}_after_rotate.png"
					)
					after_path = out_dir / after_name
					cv2.imwrite(str(after_path), _to_bgr(patch_after))
					print(f"    [+] Saved post-rotate image to {after_path}")

				manifest.append(
					{
						"page": page_number,
						"table_index": table_idx,
						"angle": angle,
						"before_image": str(before_path) if before_path else None,
						"after_image": str(after_path) if after_path else None,
						"patch_shape": list(patch_after.shape),
					}
				)
				total_rotated += 1
	finally:
		doc.close()

	manifest_path = out_dir / "manifest.json"
	manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

	print(f"\nDone! Extracted {total_rotated} rotated table(s) to {out_dir}")
	print(f"Manifest: {manifest_path}")
	return total_rotated


def build_arg_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description="Test NativePDFTableDetector on a PDF file.")
	parser.add_argument("--pdf", type=str, default=None, help="Path to input PDF.")
	parser.add_argument("--out-dir", type=str, default="results", help="Directory to save outputs.")
	parser.add_argument("--angle-threshold", type=float, default=2.0, help="Ignore rotations under this threshold.")
	parser.add_argument(
		"--spatial-dist-threshold",
		type=float,
		default=100.0,
		help="Distance threshold for grouping rotated spans.",
	)
	parser.add_argument("--no-save-before", action="store_true", help="Do not save pre-rotate crops.")
	parser.add_argument("--no-save-after", action="store_true", help="Do not save post-rotate crops.")
	return parser


def main() -> None:
	parser = build_arg_parser()
	args = parser.parse_args()

	pdf_path = _resolve_pdf_path(args.pdf)
	out_dir = Path(args.out_dir).resolve()

	run_detector(
		pdf_path=pdf_path,
		out_dir=out_dir,
		angle_threshold=float(args.angle_threshold),
		spatial_dist_threshold=float(args.spatial_dist_threshold),
		save_before=not args.no_save_before,
		save_after=not args.no_save_after,
	)


if __name__ == "__main__":
	main()


def test_paddle_ocr_processor_singleton_init(monkeypatch):
	calls = {"ocr": 0, "layout": 0}

	def _fake_build_ocr(self):
		calls["ocr"] += 1
		return _FakeOCR()

	def _fake_build_layout(self):
		calls["layout"] += 1
		return _FakeLayout([])

	monkeypatch.setattr(PaddleOCRProcessor, "_build_ocr_engine", _fake_build_ocr)
	monkeypatch.setattr(PaddleOCRProcessor, "_build_layout_engine", _fake_build_layout)

	p1 = PaddleOCRProcessor()
	p2 = PaddleOCRProcessor()

	assert calls["ocr"] == 1
	assert calls["layout"] == 1
	assert p1.__class__._ocr_engine is p2.__class__._ocr_engine
	assert p1.__class__._layout_engine is p2.__class__._layout_engine


def test_paddle_ocr_processor_process_document_layout_output(monkeypatch):
	regions = [
		{
			"type": "table",
			"bbox": [10, 20, 110, 220],
			"res": {"html": "<table><tr><td>A</td></tr></table>"},
			"score": 0.98,
		},
		{
			"type": "text",
			"bbox": [120, 20, 220, 80],
			"res": [{"text": "Xin chao", "confidence": 0.91}],
		},
		{
			"type": "figure",
			"bbox": [230, 20, 320, 160],
			"confidence": 0.77,
		},
	]

	monkeypatch.setattr(PaddleOCRProcessor, "_build_ocr_engine", lambda self: _FakeOCR())
	monkeypatch.setattr(PaddleOCRProcessor, "_build_layout_engine", lambda self: _FakeLayout(regions))

	processor = PaddleOCRProcessor()
	monkeypatch.setattr(processor, "_detect_best_rotation", lambda image: 0)

	image = np.zeros((256, 256, 3), dtype=np.uint8)
	output = processor.process_document(image)

	assert len(output) == 3
	assert output[0]["type"] == "table"
	assert "<table" in output[0]["content"]
	assert output[1]["type"] == "text"
	assert output[1]["content"] == "Xin chao"
	assert output[2]["type"] == "figure"


def test_paddle_ocr_processor_detect_best_rotation(monkeypatch):
	monkeypatch.setattr(
		PaddleOCRProcessor,
		"_build_ocr_engine",
		lambda self: _FakeOCR({0: 0.1, 90: 0.2, 180: 0.95, 270: 0.3}),
	)
	monkeypatch.setattr(PaddleOCRProcessor, "_build_layout_engine", lambda self: _FakeLayout([]))

	processor = PaddleOCRProcessor()
	monkeypatch.setattr(
		processor,
		"_rotate_orthogonal",
		lambda image, angle: {"angle": int(angle)},
	)

	angle = processor._detect_best_rotation(np.zeros((100, 200, 3), dtype=np.uint8))
	assert angle == 180


def test_paddle_ocr_processor_invalid_input_returns_empty(monkeypatch):
	monkeypatch.setattr(PaddleOCRProcessor, "_build_ocr_engine", lambda self: _FakeOCR())
	monkeypatch.setattr(PaddleOCRProcessor, "_build_layout_engine", lambda self: _FakeLayout([]))

	processor = PaddleOCRProcessor()
	output = processor.process_document("/not/exist/image.png")
	assert output == []
