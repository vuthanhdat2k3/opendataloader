from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import fitz  # pymupdf
import numpy as np

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
        return ocr_engine.ocr(image_rgb)


def _extract_items(ocr_result: Any) -> list[dict]:
    """
    Normalize PaddleOCR v2/v3 outputs into:
    { text, conf, poly: [(x,y),...4] } in image pixel space.
    """
    if not ocr_result:
        return []

    first = ocr_result[0]
    items: list[dict] = []

    # v3: list[dict(dt_polys, rec_texts, rec_scores)]
    if isinstance(first, dict) and "dt_polys" in first:
        polys = first.get("dt_polys") or []
        texts = first.get("rec_texts") or []
        scores = first.get("rec_scores") or []
        n = min(len(polys), len(texts), len(scores))
        for i in range(n):
            text = str(texts[i]).strip()
            if not text:
                continue
            try:
                conf = float(scores[i])
            except Exception:
                conf = 0.5
            poly = polys[i]
            try:
                pts = [(float(p[0]), float(p[1])) for p in poly]
            except Exception:
                continue
            if len(pts) >= 4:
                pts = pts[:4]
            elif len(pts) == 2:
                # Convert a line into a thin box
                (x0, y0), (x1, y1) = pts
                pts = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
            else:
                continue
            items.append({"text": text, "conf": conf, "poly": pts})
        return items

    # v2: result[0] = list[[bbox4pts, (text, conf)], ...]
    if not isinstance(first, list):
        return []
    for line in first:
        if not line or len(line) < 2:
            continue
        bbox = line[0]
        text_info = line[1]
        if isinstance(text_info, (list, tuple)) and len(text_info) >= 2:
            text = str(text_info[0]).strip()
            try:
                conf = float(text_info[1])
            except Exception:
                conf = 0.5
        else:
            text = str(text_info).strip()
            conf = 0.5
        if not text:
            continue
        try:
            pts = [(float(p[0]), float(p[1])) for p in bbox]
        except Exception:
            continue
        if len(pts) >= 4:
            pts = pts[:4]
        else:
            continue
        items.append({"text": text, "conf": conf, "poly": pts})
    return items


def _reading_order_key(item: dict) -> tuple[float, float]:
    pts = item.get("poly") or []
    xs = [p[0] for p in pts] if pts else [0.0]
    ys = [p[1] for p in pts] if pts else [0.0]
    return (min(ys), min(xs))


def _annotate_pdf_page(
    src_pdf: Path,
    page_number: int,
    items: list[dict],
    *,
    render_scale: int,
    output_pdf: Path,
) -> None:
    doc = fitz.open(str(src_pdf))
    try:
        page = doc[page_number - 1]
        shape = page.new_shape()
        for it in items:
            pts = it["poly"]
            # Map image pixel coords back to PDF coords
            pdf_pts = [(p[0] / render_scale, p[1] / render_scale) for p in pts]
            # Draw polygon
            shape.draw_polyline(pdf_pts + [pdf_pts[0]])
        shape.finish(color=(1, 0, 0), width=0.8)
        shape.commit()

        # Add light labels (first 60 chars) at each box top-left
        for it in items:
            pts = it["poly"]
            pdf_pts = [(p[0] / render_scale, p[1] / render_scale) for p in pts]
            x = min(p[0] for p in pdf_pts)
            y = min(p[1] for p in pdf_pts)
            txt = it["text"].replace("\n", " ").strip()
            if len(txt) > 60:
                txt = txt[:57] + "..."
            conf = float(it.get("conf") or 0.0)
            label = f"{txt} ({conf:.2f})"
            try:
                page.insert_text(
                    fitz.Point(x, max(0.0, y - 2)),
                    label,
                    fontsize=6,
                    color=(0.1, 0.1, 0.1),
                )
            except Exception:
                # Best-effort label; ignore font failures.
                pass

        doc.save(str(output_pdf))
    finally:
        doc.close()


def _items_to_markdown(items: list[dict]) -> str:
    if not items:
        return "No OCR text detected.\n"
    items_sorted = sorted(items, key=_reading_order_key)
    lines = ["## OCR result", ""]
    for it in items_sorted:
        pts = it["poly"]
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)
        txt = it["text"].replace("\n", " ").strip()
        conf = float(it.get("conf") or 0.0)
        lines.append(f"- **{txt}**  \n  conf={conf:.4f} bbox=[{x0:.1f},{y0:.1f},{x1:.1f},{y1:.1f}]")
    lines.append("")
    lines.append("## Plain text")
    lines.append("")
    lines.append("\n".join(it["text"] for it in items_sorted))
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Test PaddleOCR on a single PDF page")
    parser.add_argument("--pdf", type=str, required=True, help="Path to input PDF")
    parser.add_argument("--page", type=int, default=1, help="1-indexed page number")
    parser.add_argument("--out", type=str, default="output/ocr_page_test", help="Output directory")
    parser.add_argument("--lang", type=str, default="vi", help="PaddleOCR language")
    parser.add_argument("--use-gpu", action="store_true", help="Use GPU if available")
    parser.add_argument("--render-scale", type=int, default=2, help="Pixmap render scale (1 or 2)")
    args = parser.parse_args()

    pdf_path = Path(args.pdf).resolve()
    if not pdf_path.exists():
        raise SystemExit(f"PDF not found: {pdf_path}")

    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    page_number = int(args.page)
    if page_number < 1:
        raise SystemExit("--page must be >= 1")

    render_scale = int(args.render_scale)
    if render_scale not in (1, 2):
        raise SystemExit("--render-scale must be 1 or 2")

    # Render page to image
    doc = fitz.open(str(pdf_path))
    try:
        if page_number > len(doc):
            raise SystemExit(f"--page out of range. PDF pages={len(doc)}")
        page = doc[page_number - 1]
        pix = page.get_pixmap(matrix=fitz.Matrix(render_scale, render_scale))
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
        if pix.n == 4:
            # RGBA -> RGB
            import cv2

            img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
        elif pix.n == 3:
            pass
        else:
            img = img[:, :, :3]
    finally:
        doc.close()

    # OCR
    ocr = PaddleOCR(
        lang=args.lang,
        use_textline_orientation=True,
        use_gpu=bool(args.use_gpu),
        enable_mkldnn=False,
    )
    raw = _call_ocr(ocr, img)
    items = _extract_items(raw)

    # Save outputs
    stem = pdf_path.stem
    annotated_pdf = out_dir / f"{stem}_page{page_number:03d}_annotated.pdf"
    md_path = out_dir / f"{stem}_page{page_number:03d}.md"

    _annotate_pdf_page(
        src_pdf=pdf_path,
        page_number=page_number,
        items=items,
        render_scale=render_scale,
        output_pdf=annotated_pdf,
    )
    md_path.write_text(_items_to_markdown(items), encoding="utf-8")

    print(f"[ok] annotated_pdf={annotated_pdf}")
    print(f"[ok] markdown={md_path}")
    print(f"[ok] items={len(items)}")


if __name__ == "__main__":
    main()

