import argparse
import cv2
import json
import math
import re
import shutil
import time
from pathlib import Path
from typing import Any

import fitz
import numpy as np
import opendataloader_pdf
from PIL import Image

from native_table_detector.src.processors.merge_results import (
    merge_opendataloader_with_hybrid,
)
from native_table_detector.src.core.detector import NativePDFTableDetector
from native_table_detector.src.processors.paddle_ocr_table_processor import PaddleOCRTableProcessor


def _resolve_pdf_path(cli_pdf: str | None) -> Path:
    if cli_pdf:
        path = Path(cli_pdf)
        if path.exists() and path.is_file():
            return path.resolve()
        raise FileNotFoundError(f"PDF not found: {path}")

    candidates = [
        Path("data/table_rotated.pdf"),
        Path("data/rotated_tables_one_page.pdf"),
        Path("../data/table_rotated.pdf"),
    ]
    for path in candidates:
        if path.exists() and path.is_file():
            return path.resolve()

    raise FileNotFoundError(
        "Could not find a default PDF. Pass one with --pdf /path/to/file.pdf"
    )


def _default_output_dir() -> Path:
    return Path("output/merge_results_runs").resolve()


def _build_selected_pdf(input_pdf: Path, out_dir: Path, max_pages: int) -> tuple[Path, list[int]]:
    doc = fitz.open(str(input_pdf))
    try:
        if max_pages <= 0 or max_pages >= len(doc):
            return input_pdf, list(range(1, len(doc) + 1))

        selected_pages = list(range(1, max_pages + 1))
        selected_pdf = out_dir / f"{input_pdf.stem}_selected_pages.pdf"
        out_doc = fitz.open()
        try:
            for page_no in selected_pages:
                out_doc.insert_pdf(doc, from_page=page_no - 1, to_page=page_no - 1)
            out_doc.save(str(selected_pdf))
        finally:
            out_doc.close()
        return selected_pdf, selected_pages
    finally:
        doc.close()


def _run_detector_and_patch_pdf(
    input_pdf: Path,
    detector_dir: Path,
    angle_threshold: float,
    spatial_dist_threshold: float,
    original_page_numbers: list[int],
) -> tuple[Path, dict]:
    detector_dir.mkdir(parents=True, exist_ok=True)
    rotated_dir = detector_dir / "rotated_tables"
    rotated_dir.mkdir(parents=True, exist_ok=True)
    patched_pdf = detector_dir / f"{input_pdf.stem}_patched.pdf"

    detector = NativePDFTableDetector(
        angle_threshold=angle_threshold,
        spatial_dist_threshold=spatial_dist_threshold,
    )
    skipped_for_overlap = 0

    def _rect_iou(a: fitz.Rect, b: fitz.Rect) -> float:
        inter = a & b
        if inter.is_empty:
            return 0.0
        inter_area = inter.width * inter.height
        a_area = max(0.0, a.width * a.height)
        b_area = max(0.0, b.width * b.height)
        union = a_area + b_area - inter_area
        return inter_area / union if union > 0 else 0.0

    manifest: list[dict] = []
    total = 0
    doc = fitz.open(str(input_pdf))
    try:
        for local_idx in range(len(doc)):
            page = doc[local_idx]
            text_blocks = [
                fitz.Rect(b[:4])
                for b in page.get_text("blocks")
                if len(b) >= 5 and str(b[4]).strip()
            ]
            page_no = (
                original_page_numbers[local_idx]
                if local_idx < len(original_page_numbers)
                else local_idx + 1
            )
            results = detector.process_page(page)
            rotated_tables = results.get("rotated_tables", [])
            if not rotated_tables:
                continue

            for t_idx, table in enumerate(rotated_tables, start=1):
                patch_before = table.get("patch_before_rotate")
                patch_after = table.get("patch_image")
                angle = float(table.get("angle", 0.0))
                spans = table.get("spans", [])
                if not isinstance(patch_after, np.ndarray) or patch_after.size == 0:
                    continue

                after_name = f"page_{page_no}_table_{t_idx}_angle_{angle:+.1f}_after.png"
                after_path = rotated_dir / after_name
                Image.fromarray(patch_after.astype("uint8"), mode="RGB").save(after_path)

                before_path = None
                if isinstance(patch_before, np.ndarray) and patch_before.size > 0:
                    before_name = (
                        f"page_{page_no}_table_{t_idx}_angle_{angle:+.1f}_before.png"
                    )
                    before_path = rotated_dir / before_name
                    Image.fromarray(patch_before.astype("uint8"), mode="RGB").save(
                        before_path
                    )

                    tight_before_path = None
                    tight_after_path = None
                    tight_before = None
                    tight_after = None
                    if table.get("obb_tight"):
                        tight_before, tight_after = detector._extract_cells_from_rotated(
                            page, table["obb_tight"], angle
                        )
                        if (
                            isinstance(tight_before, np.ndarray)
                            and tight_before.size > 0
                        ):
                            tight_before_name = (
                                f"page_{page_no}_table_{t_idx}_angle_{angle:+.1f}_before_tight.png"
                            )
                            tight_before_path = rotated_dir / tight_before_name
                            Image.fromarray(
                                tight_before.astype("uint8"), mode="RGB"
                            ).save(tight_before_path)
                        if isinstance(tight_after, np.ndarray) and tight_after.size > 0:
                            tight_after_name = (
                                f"page_{page_no}_table_{t_idx}_angle_{angle:+.1f}_tight.png"
                            )
                            tight_after_path = rotated_dir / tight_after_name
                            Image.fromarray(
                                tight_after.astype("uint8"), mode="RGB"
                            ).save(tight_after_path)
                else:
                    tight_before_path = None
                    tight_after_path = None

                loose_obb = table.get("obb_loose") or table.get("obb")
                tight_obb = table.get("obb_tight") or table.get("obb")
                if not loose_obb or not tight_obb:
                    continue
                cx, cy = tight_obb["cx"], tight_obb["cy"]
                tw, th = tight_obb["w"], tight_obb["h"]
                if tw <= 1 or th <= 1:
                    continue
                span_rect = fitz.Rect(
                    min(s["bbox"][0] for s in spans),
                    min(s["bbox"][1] for s in spans),
                    max(s["bbox"][2] for s in spans),
                    max(s["bbox"][3] for s in spans),
                )
                a_rad = math.radians(-angle)
                cos_a, sin_a = math.cos(a_rad), math.sin(a_rad)

                ul_x = cx - tw / 2 * cos_a + th / 2 * sin_a
                ul_y = cy - tw / 2 * sin_a - th / 2 * cos_a
                ur_x = cx + tw / 2 * cos_a + th / 2 * sin_a
                ur_y = cy + tw / 2 * sin_a - th / 2 * cos_a
                lr_x = cx + tw / 2 * cos_a - th / 2 * sin_a
                lr_y = cy + tw / 2 * sin_a + th / 2 * cos_a
                ll_x = cx - tw / 2 * cos_a - th / 2 * sin_a
                ll_y = cy - tw / 2 * sin_a + th / 2 * cos_a

                quad = fitz.Quad(
                    fitz.Point(ul_x, ul_y),
                    fitz.Point(ur_x, ur_y),
                    fitz.Point(ll_x, ll_y),
                    fitz.Point(lr_x, lr_y),
                )
                upright_rect = fitz.Rect(
                    cx - tw / 2, cy - th / 2, cx + tw / 2, cy + th / 2
                )

                # Safety guard: do not patch if replacement area collides
                # with non-table text blocks.
                unsafe_overlap = False
                for block_rect in text_blocks:
                    if not block_rect.intersects(upright_rect):
                        continue
                    # Consider it table content if it overlaps table span area enough.
                    if _rect_iou(block_rect, span_rect) >= 0.15:
                        continue
                    unsafe_overlap = True
                    break

                if unsafe_overlap:
                    skipped_for_overlap += 1
                    white_name = (
                        f"page_{page_no}_table_{t_idx}_angle_{angle:+.1f}_white_mask.png"
                    )
                    white_path = rotated_dir / white_name
                    base_patch = (
                        tight_before
                        if isinstance(tight_before, np.ndarray) and tight_before.size > 0
                        else (
                            tight_after
                            if isinstance(tight_after, np.ndarray) and tight_after.size > 0
                            else (
                                patch_before
                                if isinstance(patch_before, np.ndarray)
                                and patch_before.size > 0
                                else patch_after
                            )
                        )
                    )
                    white_patch = np.full_like(base_patch, 255, dtype=np.uint8)
                    Image.fromarray(white_patch.astype("uint8"), mode="RGB").save(
                        white_path
                    )

                    white_rect = upright_rect

                    page.add_redact_annot(quad, fill=(1, 1, 1))
                    page.apply_redactions()
                    page.insert_image(white_rect, filename=str(white_path))

                    manifest.append(
                        {
                            "page": page_no,
                            "table_index": t_idx,
                            "angle": angle,
                            "obb_loose": loose_obb,
                            "replacement_obb": tight_obb,
                            "obb_area_ratio_tight_to_loose": (
                                float(tight_obb["w"] * tight_obb["h"])
                                / max(1e-6, float(loose_obb["w"] * loose_obb["h"]))
                            ),
                            "image_before_rotate": str(before_path)
                            if before_path
                            else None,
                            "image_before_tight": str(tight_before_path)
                            if tight_before_path
                            else None,
                            "image_tight": str(tight_after_path)
                            if tight_after_path
                            else str(after_path),
                            "patched_to_pdf": True,
                            "patched_with_white_mask": True,
                            "white_mask_image": str(white_path),
                            "white_mask_rect": list(white_rect),
                            "white_mask_based_on": (
                                "image_before_tight"
                                if isinstance(tight_before, np.ndarray)
                                and tight_before.size > 0
                                else "image_tight"
                            ),
                            "reason": "overlap_with_non_table_text",
                        }
                    )
                    total += 1
                    continue

                page.add_redact_annot(quad, fill=(1, 1, 1))
                page.apply_redactions()
                image_for_patch = (
                    str(tight_after_path) if tight_after_path else str(after_path)
                )
                page.insert_image(upright_rect, filename=image_for_patch)

                manifest.append(
                    {
                        "page": page_no,
                        "table_index": t_idx,
                        "angle": angle,
                        "obb_loose": loose_obb,
                        "replacement_obb": tight_obb,
                        "obb_area_ratio_tight_to_loose": (
                            float(tight_obb["w"] * tight_obb["h"])
                            / max(1e-6, float(loose_obb["w"] * loose_obb["h"]))
                        ),
                        "image_before_rotate": str(before_path) if before_path else None,
                        "image_before_tight": str(tight_before_path)
                        if tight_before_path
                        else None,
                        "image_tight": str(tight_after_path)
                        if tight_after_path
                        else str(after_path),
                        "patched_to_pdf": True,
                        "patched_with_white_mask": False,
                    }
                )
                total += 1

        doc.save(str(patched_pdf))
    finally:
        doc.close()

    manifest_path = rotated_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    if total == 0:
        shutil.copy2(input_pdf, patched_pdf)

    return patched_pdf, {
        "totals": {"rotated_tables": total},
        "skipped_for_overlap": skipped_for_overlap,
        "manifest_path": str(manifest_path),
        "manifest": manifest,
    }


def _run_hybrid_on_tight_images(
    detector_manifest: list[dict],
    hybrid_dir: Path,
    ocr_lang: str,
) -> dict:
    hybrid_dir.mkdir(parents=True, exist_ok=True)
    processor = PaddleOCRTableProcessor(lang=ocr_lang)

    pages_map: dict[int, dict] = {}
    for item in detector_manifest:
        page_no = int(item.get("page", 1))
        image_tight = item.get("image_tight")
        if not image_tight:
            continue
        image_path = Path(image_tight)
        if not image_path.exists():
            continue

        img_bgr = cv2.imread(str(image_path))
        if img_bgr is None:
            continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        ocr = processor.process_table_image(img_rgb)

        page_obj = pages_map.setdefault(
            page_no,
            {"page_number": page_no, "rotated_tables": []},
        )
        page_obj["rotated_tables"].append(
            {
                "obb": item.get("replacement_obb", {}),
                "angle": float(item.get("angle", 0.0)),
                "ocr_markdown": ocr.get("markdown", ""),
                "ocr_html": ocr.get("html", ""),
                "ocr_text": ocr.get("text", ""),
                "ocr_confidence": float(ocr.get("confidence", 0.0)),
                "image_tight": str(image_path),
            }
        )

    pages = [pages_map[k] for k in sorted(pages_map.keys())]
    results = {
        "total_pages": len(pages),
        "pages": pages,
        "total_rotated_tables": sum(len(p.get("rotated_tables", [])) for p in pages),
    }
    (hybrid_dir / "hybrid_results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # Raw markdown output only (no extra metadata headings).
    md_lines = []
    for page in pages:
        rotated_tables = page.get("rotated_tables", [])
        if not rotated_tables:
            continue
        for idx, table in enumerate(rotated_tables, start=1):
            md_block = str(table.get("ocr_markdown", "")).strip()
            if md_block:
                md_lines.append(md_block)
                md_lines.append("")
    (hybrid_dir / "hybrid_output.md").write_text(
        "\n".join(md_lines).strip() + "\n" if md_lines else "",
        encoding="utf-8",
    )
    return results


def _table_to_markdown(table_kid: dict) -> str:
    rows = table_kid.get("rows", [])
    if not rows:
        return ""

    matrix: list[list[str]] = []
    for row in rows:
        row_cells = []
        for cell in row.get("cells", []):
            paragraphs = cell.get("kids", [])
            text = paragraphs[0].get("content", "") if paragraphs else ""
            row_cells.append(str(text))
        matrix.append(row_cells)

    if not matrix:
        return ""

    col_count = max(len(r) for r in matrix)
    normalized = [r + [""] * (col_count - len(r)) for r in matrix]
    header = "| " + " | ".join([f"Col {i + 1}" for i in range(col_count)]) + " |"
    sep = "| " + " | ".join(["---"] * col_count) + " |"
    body = ["| " + " | ".join(r) + " |" for r in normalized]
    return "\n".join([header, sep, *body])


def _obb_to_aabb(obb: dict) -> list[float]:
    cx = float(obb.get("cx", 0.0))
    cy = float(obb.get("cy", 0.0))
    w = max(0.0, float(obb.get("w", 0.0)))
    h = max(0.0, float(obb.get("h", 0.0)))
    return [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]


def _bbox_iou(box1: list[float], box2: list[float]) -> float:
    x0_1, y0_1, x1_1, y1_1 = box1
    x0_2, y0_2, x1_2, y1_2 = box2
    xi0 = max(x0_1, x0_2)
    yi0 = max(y0_1, y0_2)
    xi1 = min(x1_1, x1_2)
    yi1 = min(y1_1, y1_2)
    if xi1 <= xi0 or yi1 <= yi0:
        return 0.0
    inter = (xi1 - xi0) * (yi1 - yi0)
    area1 = max(0.0, (x1_1 - x0_1) * (y1_1 - y0_1))
    area2 = max(0.0, (x1_2 - x0_2) * (y1_2 - y0_2))
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


def _bbox_intersection_area(box1: list[float], box2: list[float]) -> float:
    x0_1, y0_1, x1_1, y1_1 = box1
    x0_2, y0_2, x1_2, y1_2 = box2
    xi0 = max(x0_1, x0_2)
    yi0 = max(y0_1, y0_2)
    xi1 = min(x1_1, x1_2)
    yi1 = min(y1_1, y1_2)
    if xi1 <= xi0 or yi1 <= yi0:
        return 0.0
    return (xi1 - xi0) * (yi1 - yi0)


def _bbox_area(box: list[float]) -> float:
    return max(0.0, (box[2] - box[0]) * (box[3] - box[1]))


def _filter_odl_tables_by_detector_boxes(
    odl_json: dict,
    detector_manifest: list[dict],
    iou_threshold: float = 0.3,
) -> tuple[dict, dict]:
    detector_boxes = [
        _obb_to_aabb(item.get("replacement_obb", {}))
        for item in detector_manifest
        if isinstance(item.get("replacement_obb"), dict)
    ]
    filtered_kids = []
    removed = []
    total_tables = 0

    for kid in odl_json.get("kids", []):
        if kid.get("type") != "table":
            filtered_kids.append(kid)
            continue
        total_tables += 1
        bbox = kid.get("bounding box")
        if not (isinstance(bbox, list) and len(bbox) == 4 and detector_boxes):
            filtered_kids.append(kid)
            continue

        best_iou = 0.0
        best_detector_cover = 0.0
        best_table_cover = 0.0
        kid_area = _bbox_area(bbox)
        for dbox in detector_boxes:
            inter = _bbox_intersection_area(bbox, dbox)
            det_area = _bbox_area(dbox)
            iou = _bbox_iou(bbox, dbox)
            det_cover = inter / det_area if det_area > 0 else 0.0
            tbl_cover = inter / kid_area if kid_area > 0 else 0.0
            best_iou = max(best_iou, iou)
            best_detector_cover = max(best_detector_cover, det_cover)
            best_table_cover = max(best_table_cover, tbl_cover)

        # IoU catches similar-sized overlaps.
        # detector_cover catches the common case where detector box is fully inside
        # a larger ODL table box (IoU can be small but still same table region).
        should_remove = (
            best_iou >= iou_threshold
            or best_detector_cover >= 0.35
            or best_table_cover >= 0.4
        )
        if should_remove:
            removed.append(
                {
                    "table_id": kid.get("id"),
                    "page_number": kid.get("page number"),
                    "bbox": bbox,
                    "max_iou": round(float(best_iou), 4),
                    "max_detector_cover": round(float(best_detector_cover), 4),
                    "max_table_cover": round(float(best_table_cover), 4),
                }
            )
            continue
        filtered_kids.append(kid)

    filtered = {**odl_json, "kids": filtered_kids}
    report = {
        "total_tables": total_tables,
        "removed_tables": len(removed),
        "iou_threshold": iou_threshold,
        "removed_details": removed,
    }
    return filtered, report


def _render_odl_json_to_review_markdown(odl_json: dict) -> str:
    lines = []
    for kid in odl_json.get("kids", []):
        ktype = kid.get("type")
        if ktype == "table":
            md_table = _table_to_markdown(kid).strip()
            if md_table:
                lines.append(md_table)
                lines.append("")
        elif ktype == "paragraph":
            content = str(kid.get("content", "")).strip()
            if content:
                lines.append(content)
                lines.append("")
        elif ktype == "image":
            src = kid.get("source")
            if src:
                lines.append(f"![image]({src})")
                lines.append("")
    return ("\n".join(lines).strip() + "\n") if lines else ""


def _merge_into_original_markdown(original_md: str, merged_json: dict) -> str:
    """
    Patch OCR-merged table content back into the original markdown by table position.
    """
    # Match contiguous markdown table blocks.
    table_pattern = re.compile(r"(?:^\|.*\|\n)+", re.MULTILINE)
    table_matches = list(table_pattern.finditer(original_md))
    if not table_matches:
        return original_md

    merged_tables = [
        kid
        for kid in merged_json.get("kids", [])
        if kid.get("type") == "table" and kid.get("rotated_table_replaced")
    ]
    if not merged_tables:
        return original_md

    replacement_blocks = []
    for table_kid in merged_tables:
        md_block = _table_to_markdown(table_kid).strip()
        if md_block:
            replacement_blocks.append(md_block + "\n")

    if not replacement_blocks:
        return original_md

    max_replace = min(len(table_matches), len(replacement_blocks))
    chunks: list[str] = []
    cursor = 0
    for idx in range(max_replace):
        match = table_matches[idx]
        chunks.append(original_md[cursor : match.start()])
        chunks.append(replacement_blocks[idx])
        cursor = match.end()
    chunks.append(original_md[cursor:])
    return "".join(chunks)


def _hybrid_tables_to_markdown_blocks(hybrid_results: dict) -> list[str]:
    blocks: list[str] = []
    for page in hybrid_results.get("pages", []):
        for table in page.get("rotated_tables", []):
            md = str(table.get("ocr_markdown", "")).strip()
            if md:
                blocks.append(md + "\n")
    return blocks


def _merge_into_original_markdown_with_fallback(
    original_md: str, merged_json: dict, hybrid_results: dict
) -> str:
    """
    Prefer merged-json replacements; fallback to hybrid OCR table markdown blocks
    by markdown-table order when JSON merge cannot replace enough tables.
    """
    table_pattern = re.compile(r"(?:^\|.*\|\n)+", re.MULTILINE)
    table_matches = list(table_pattern.finditer(original_md))
    if not table_matches:
        return original_md

    replacement_blocks = []
    merged_tables = [
        kid
        for kid in merged_json.get("kids", [])
        if kid.get("type") == "table" and kid.get("rotated_table_replaced")
    ]
    for table_kid in merged_tables:
        md_block = _table_to_markdown(table_kid).strip()
        if md_block:
            replacement_blocks.append(md_block + "\n")

    hybrid_blocks = _hybrid_tables_to_markdown_blocks(hybrid_results)
    if len(replacement_blocks) < len(hybrid_blocks):
        replacement_blocks.extend(hybrid_blocks[len(replacement_blocks) :])

    if not replacement_blocks:
        return original_md

    max_replace = min(len(table_matches), len(replacement_blocks))
    chunks: list[str] = []
    cursor = 0
    for idx in range(max_replace):
        match = table_matches[idx]
        chunks.append(original_md[cursor : match.start()])
        chunks.append(replacement_blocks[idx])
        cursor = match.end()
    chunks.append(original_md[cursor:])
    return "".join(chunks)


def run_merge_pipeline(
    pdf_path: Path,
    out_dir: Path,
    max_pages: int,
    angle_threshold: float,
    spatial_dist_threshold: float,
    ocr_lang: str,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)

    detector_dir = out_dir / "detector"
    hybrid_dir = out_dir / "hybrid"
    merge_dir = out_dir / "merge"
    merge_dir.mkdir(parents=True, exist_ok=True)

    processing_pdf, selected_pages = _build_selected_pdf(pdf_path, merge_dir, max_pages)

    print(f"[1/4] detector + patch pdf -> {detector_dir}")
    patched_pdf, detector_summary = _run_detector_and_patch_pdf(
        input_pdf=processing_pdf,
        detector_dir=detector_dir,
        angle_threshold=angle_threshold,
        spatial_dist_threshold=spatial_dist_threshold,
        original_page_numbers=selected_pages,
    )

    print(f"[2/4] opendataloader convert on patched pdf -> {merge_dir}")
    start = time.perf_counter()
    convert_kwargs = {
        "input_path": str(patched_pdf),
        "output_dir": str(merge_dir),
        "format": "json,html,markdown,pdf",
    }
    opendataloader_pdf.convert(**convert_kwargs)
    elapsed = time.perf_counter() - start

    output_stem = patched_pdf.stem
    odl_json_path = merge_dir / f"{output_stem}.json"
    original_md_path = merge_dir / f"{output_stem}.md"
    if not odl_json_path.exists():
        raise FileNotFoundError(f"OpenDataLoader JSON was not generated: {odl_json_path}")
    if not original_md_path.exists():
        raise FileNotFoundError(
            f"OpenDataLoader markdown was not generated: {original_md_path}"
        )
    raw_odl_review_md = merge_dir / "opendataloader_raw_output.md"
    raw_odl_review_md.write_text(
        original_md_path.read_text(encoding="utf-8", errors="replace"),
        encoding="utf-8",
    )

    odl_json = json.loads(odl_json_path.read_text(encoding="utf-8", errors="replace"))
    filtered_odl_json, filter_report = _filter_odl_tables_by_detector_boxes(
        odl_json=odl_json,
        detector_manifest=detector_summary.get("manifest", []),
        iou_threshold=0.3,
    )
    filtered_odl_json_path = merge_dir / "opendataloader_filtered.json"
    filtered_odl_json_path.write_text(
        json.dumps(filtered_odl_json, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    filtered_odl_review_md = merge_dir / "opendataloader_filtered_output.md"
    filtered_odl_review_md.write_text(
        _render_odl_json_to_review_markdown(filtered_odl_json),
        encoding="utf-8",
    )
    (merge_dir / "opendataloader_filter_report.json").write_text(
        json.dumps(filter_report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"[3/4] hybrid on image_tight -> {hybrid_dir}")
    hybrid_results = _run_hybrid_on_tight_images(
        detector_manifest=detector_summary.get("manifest", []),
        hybrid_dir=hybrid_dir,
        ocr_lang=ocr_lang,
    )
    print(f"[4/4] merge OCR into original markdown/json -> {merge_dir}")

    merged_json_path = merge_dir / "merged_results.json"
    merged_md_path = merge_dir / "merged_results_patched.md"

    merged_json = merge_opendataloader_with_hybrid(
        odl_json_path=filtered_odl_json_path,
        hybrid_results=hybrid_results,
        output_json_path=merged_json_path,
    )

    original_md = filtered_odl_review_md.read_text(encoding="utf-8", errors="replace")
    patched_md = _merge_into_original_markdown_with_fallback(
        original_md, merged_json, hybrid_results
    )
    merged_md_path.write_text(patched_md, encoding="utf-8")

    summary = {
        "input_pdf": str(pdf_path),
        "processing_pdf": str(processing_pdf),
        "patched_pdf": str(patched_pdf),
        "odl_json": str(odl_json_path),
        "detector_dir": str(detector_dir),
        "hybrid_dir": str(hybrid_dir),
        "merge_dir": str(merge_dir),
        "opendataloader_elapsed_sec": round(elapsed, 2),
        "detector_rotated_tables": int(detector_summary.get("totals", {}).get("rotated_tables", 0)),
        "hybrid_rotated_tables": int(hybrid_results.get("total_rotated_tables", 0)),
        "merged_rotated_tables_replaced": int(merged_json.get("rotated_tables_replaced", 0)),
        "merged_json": str(merged_json_path),
        "merged_md": str(merged_md_path),
        "hybrid_md_review": str(hybrid_dir / "hybrid_output.md"),
        "odl_md_review": str(raw_odl_review_md),
        "odl_filtered_json": str(filtered_odl_json_path),
        "odl_filtered_md_review": str(filtered_odl_review_md),
        "odl_filtered_removed_tables": int(filter_report.get("removed_tables", 0)),
    }

    summary_path = merge_dir / "pipeline_summary.txt"
    summary_path.write_text(
        "\n".join(f"{key}: {value}" for key, value in summary.items()),
        encoding="utf-8",
    )

    print("Done merge pipeline")
    print(f"Detector summary: {detector_dir / 'detector_summary.json'}")
    print(f"Hybrid results: {hybrid_dir / 'hybrid_results.json'}")
    print(f"Merged JSON: {merged_json_path}")
    print(f"Merged Markdown: {merged_md_path}")
    print(f"Summary: {summary_path}")

    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run detector -> hybrid -> merge pipeline and save the outputs."
    )
    parser.add_argument("--pdf", type=str, default=None, help="Input PDF path.")
    parser.add_argument(
        "--out-dir",
        type=str,
        default=str(_default_output_dir()),
        help="Base directory for detector, hybrid, and merge outputs.",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=1,
        help="Maximum pages to process. 0 means all pages.",
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
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    pdf_path = _resolve_pdf_path(args.pdf)
    out_dir = Path(args.out_dir).resolve()

    run_merge_pipeline(
        pdf_path=pdf_path,
        out_dir=out_dir,
        max_pages=int(args.max_pages),
        angle_threshold=float(args.angle_threshold),
        spatial_dist_threshold=float(args.spatial_dist_threshold),
        ocr_lang=str(args.ocr_lang),
    )


if __name__ == "__main__":
    main()
