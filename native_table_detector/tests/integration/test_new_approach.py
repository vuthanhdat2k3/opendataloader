"""
Test script for new approach:
1. OpenDataLoader → Get all table boxes with content
2. For each box → Check rotation using text direction metadata
3. If rotated → Extract patch → Rotate + OCR → New content
4. If normal → Keep original content
5. Merge → Final JSON → Markdown

Stage results are saved for review.
"""

import json
import math
import shutil
import time
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import cv2
import fitz
import numpy as np
from PIL import Image

import opendataloader_pdf

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
    from native_table_detector.src.core.detector import NativePDFTableDetector
    from native_table_detector.src.pipeline.orchestrator import run_pipeline
except ModuleNotFoundError:
    from src.core.detector import NativePDFTableDetector
    from src.pipeline.orchestrator import run_pipeline


def save_json(data: Any, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def compute_text_angle_from_span(span: dict) -> float:
    """Compute angle from text span direction vector (like detector.py)."""
    origin = span.get("origin")
    if origin is None:
        return 0.0

    # Get the bbox to calculate direction
    bbox = span.get("bbox", [0, 0, 0, 0])
    if len(bbox) != 4:
        return 0.0

    # For horizontal text, direction is (1, 0), for vertical it's (0, 1)
    # We need to look at the line direction
    dx = bbox[2] - bbox[0]  # width
    dy = bbox[3] - bbox[1]  # height

    if dx == 0 and dy == 0:
        return 0.0

    angle = round(-math.degrees(math.atan2(dy, dx)) / 5) * 5
    return angle


def get_text_angle_from_block(block: dict) -> float:
    """Get dominant text angle from a block."""
    if block.get("type") != 0:  # Not text block
        return 0.0

    angles = []
    for line in block.get("lines", []):
        dir_vec = line.get("dir")
        if dir_vec:
            dx, dy = dir_vec
            if dx != 0 or dy != 0:
                angle = round(-math.degrees(math.atan2(dy, dx)) / 5) * 5
                angles.append(angle)

    if not angles:
        return 0.0

    # Return most common angle
    from collections import Counter

    angle_counts = Counter(angles)
    return angle_counts.most_common(1)[0][0]


def check_box_rotation(
    page: fitz.Page, bbox: list, angle_threshold: float = 2.0
) -> dict:
    """
    Check if text in a bounding box is rotated.
    Uses text direction metadata from PDF (same as detector.py).

    Returns:
        dict with: is_rotated, angle, text_content
    """
    x0, y0, x1, y1 = bbox

    # Get text blocks in this region
    blocks = page.get_text("rawdict", flags=0)["blocks"]

    box_angles = []
    text_parts = []

    for block in blocks:
        if block.get("type") != 0:
            continue

        block_bbox = block.get("bbox", [0, 0, 0, 0])

        # Check if block intersects with our bbox
        if not (
            block_bbox[2] > x0
            and block_bbox[0] < x1
            and block_bbox[3] > y0
            and block_bbox[1] < y1
        ):
            continue

        angle = get_text_angle_from_block(block)
        if abs(angle) >= angle_threshold:
            box_angles.append(angle)

        # Extract text
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                text = span.get("text", "")
                if text.strip():
                    text_parts.append(text.strip())

    if not box_angles:
        return {
            "is_rotated": False,
            "angle": 0.0,
            "text_content": " ".join(text_parts),
            "confidence": 0.0,
        }

    # Get dominant rotation angle
    from collections import Counter

    angle_counts = Counter(box_angles)
    dominant_angle = angle_counts.most_common(1)[0][0]

    return {
        "is_rotated": True,
        "angle": dominant_angle,
        "text_content": " ".join(text_parts),
        "confidence": angle_counts[dominant_angle] / len(box_angles),
    }


def _obb_to_aabb(obb: dict[str, float]) -> list[float]:
    cx, cy, w, h = obb["cx"], obb["cy"], obb["w"], obb["h"]
    return [cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0]


def _bbox_iou(box_a: list[float], box_b: list[float]) -> float:
    x0 = max(box_a[0], box_b[0])
    y0 = max(box_a[1], box_b[1])
    x1 = min(box_a[2], box_b[2])
    y1 = min(box_a[3], box_b[3])
    if x1 <= x0 or y1 <= y0:
        return 0.0
    inter = (x1 - x0) * (y1 - y0)
    area_a = max(0.0, (box_a[2] - box_a[0]) * (box_a[3] - box_a[1]))
    area_b = max(0.0, (box_b[2] - box_b[0]) * (box_b[3] - box_b[1]))
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _overlap_ratio_min_area(box_a: list[float], box_b: list[float]) -> float:
    x0 = max(box_a[0], box_b[0])
    y0 = max(box_a[1], box_b[1])
    x1 = min(box_a[2], box_b[2])
    y1 = min(box_a[3], box_b[3])
    if x1 <= x0 or y1 <= y0:
        return 0.0
    inter = (x1 - x0) * (y1 - y0)
    area_a = max(0.0, (box_a[2] - box_a[0]) * (box_a[3] - box_a[1]))
    area_b = max(0.0, (box_b[2] - box_b[0]) * (box_b[3] - box_b[1]))
    min_area = min(area_a, area_b)
    return inter / min_area if min_area > 0 else 0.0


def _match_odl_box_with_detector_tables(
    bbox: list[float], detector_tables: list[dict], overlap_threshold: float
) -> tuple[dict | None, float, str]:
    best_match = None
    best_overlap = 0.0
    best_mode = "none"
    for table in detector_tables:
        tight_obb = table.get("obb_tight")
        loose_obb = table.get("obb_loose") or table.get("obb")

        tight_overlap = (
            _overlap_ratio_min_area(bbox, _obb_to_aabb(tight_obb)) if tight_obb else 0.0
        )
        loose_overlap = (
            _overlap_ratio_min_area(bbox, _obb_to_aabb(loose_obb)) if loose_obb else 0.0
        )

        if tight_overlap >= loose_overlap:
            candidate_overlap = tight_overlap
            candidate_mode = "tight"
        else:
            candidate_overlap = loose_overlap
            candidate_mode = "loose"

        if candidate_overlap > best_overlap:
            best_overlap = candidate_overlap
            best_match = table
            best_mode = candidate_mode

    if best_overlap < overlap_threshold:
        return None, best_overlap, "none"
    return best_match, best_overlap, best_mode


def _match_detector_table_with_odl_boxes(
    detector_table: dict, odl_tables_on_page: list[dict], overlap_threshold: float
) -> tuple[dict | None, float, str]:
    best_table = None
    best_overlap = 0.0
    best_mode = "none"
    tight_obb = detector_table.get("obb_tight")
    loose_obb = detector_table.get("obb_loose") or detector_table.get("obb")

    for table in odl_tables_on_page:
        bbox = table["bbox"]
        tight_overlap = (
            _overlap_ratio_min_area(bbox, _obb_to_aabb(tight_obb)) if tight_obb else 0.0
        )
        loose_overlap = (
            _overlap_ratio_min_area(bbox, _obb_to_aabb(loose_obb)) if loose_obb else 0.0
        )

        if tight_overlap >= loose_overlap:
            candidate_overlap = tight_overlap
            candidate_mode = "tight"
        else:
            candidate_overlap = loose_overlap
            candidate_mode = "loose"

        if candidate_overlap > best_overlap:
            best_overlap = candidate_overlap
            best_table = table
            best_mode = candidate_mode

    if best_overlap < overlap_threshold:
        return None, best_overlap, "none"
    return best_table, best_overlap, best_mode


def extract_patch_from_bbox(page: fitz.Page, bbox: list, angle: float) -> tuple:
    """
    Extract and deskew image patch from bbox at given angle.
    Returns: (patch_before, patch_deskewed)

    Note: PDF coordinates have origin at bottom-left, image at top-left.
    Need to flip y coordinates when cropping.
    """
    x0, y0_pdf, x1, y1_pdf = bbox

    # Render page at 2x resolution
    mat = fitz.Matrix(2, 2)
    pix = page.get_pixmap(matrix=mat)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
        pix.height, pix.width, pix.n
    )

    page_height = pix.height

    if pix.n == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    elif pix.n == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # Scale bbox coordinates
    x0, y0_pdf, x1, y1_pdf = x0 * 2, y0_pdf * 2, x1 * 2, y1_pdf * 2

    # Convert PDF y coordinates to image y coordinates (flip vertically)
    y0_img = page_height - y1_pdf
    y1_img = page_height - y0_pdf

    # Ensure coordinates are within bounds
    y0_img = max(0, min(y0_img, page_height - 1))
    y1_img = max(0, min(y1_img, page_height))
    x0 = max(0, min(x0, pix.width - 1))
    x1 = max(0, min(x1, pix.width))

    # Crop the region
    crop = img[int(y0_img) : int(y1_img), int(x0) : int(x1)]

    # Rotate to deskew
    if abs(angle) < 2.0:
        return crop, crop

    # Rotate by -angle to deskew
    h, w = crop.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, -angle, 1.0)

    # Calculate new bounding box size
    cos_a = abs(M[0, 0])
    sin_a = abs(M[0, 1])
    new_w = int(w * cos_a + h * sin_a)
    new_h = int(w * sin_a + h * cos_a)

    M[0, 2] += (new_w - w) // 2
    M[1, 2] += (new_h - h) // 2

    deskewed = cv2.warpAffine(crop, M, (new_w, new_h), borderValue=(255, 255, 255))

    return crop, deskewed


def ocr_patch(patch: np.ndarray, ocr_engine) -> dict:
    """OCR a patch and return markdown table."""
    patch_rgb = patch

    try:
        result = ocr_engine.ocr(patch_rgb, cls=True)

        if not result or not result[0]:
            return {"markdown": "", "confidence": 0.0}

        # Parse OCR result
        items = []
        for line in result[0]:
            if line and len(line) >= 2:
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
                x0 = min(x_coords)
                y0 = min(y_coords)

                items.append({"text": text.strip(), "x": x0, "y": y0, "conf": conf})

        if not items:
            return {"markdown": "", "confidence": 0.0}

        # Group into rows
        items.sort(key=lambda x: x["y"])

        # Simple row grouping by y coordinate
        rows = []
        current_row = [items[0]]
        current_y = items[0]["y"]
        row_threshold = 10

        for item in items[1:]:
            if abs(item["y"] - current_y) <= row_threshold:
                current_row.append(item)
            else:
                rows.append(sorted(current_row, key=lambda x: x.get("x", 0)))
                current_row = [item]
                current_y = item["y"]
        rows.append(sorted(current_row, key=lambda x: x.get("x", 0)))

        # Generate markdown
        md_lines = []
        for row in rows:
            cells = [item["text"] for item in row]
            md_lines.append("| " + " | ".join(cells) + " |")

        if len(rows) > 1:
            col_count = len(rows[0])
            md_lines.insert(1, "| " + " | ".join(["---"] * col_count) + " |")

        return {
            "markdown": "\n".join(md_lines),
            "confidence": sum(item["conf"] for item in items) / len(items)
            if items
            else 0.0,
        }

    except Exception as e:
        print(f"OCR error: {e}")
        return {"markdown": "", "confidence": 0.0}


def run_stage1_opendataloader(pdf_path: Path, output_dir: Path) -> dict:
    """Stage 1: OpenDataLoader to get all table boxes."""
    print("\n" + "=" * 60)
    print("STAGE 1: OpenDataLoader")
    print("=" * 60)

    output_dir.mkdir(parents=True, exist_ok=True)

    start = time.perf_counter()
    opendataloader_pdf.convert(
        input_path=str(pdf_path),
        output_dir=str(output_dir),
        format="json,html,markdown,pdf",
    )
    elapsed = time.perf_counter() - start

    generated_annotated_pdf = output_dir / f"{pdf_path.stem}_annotated.pdf"
    annotated_pdf = output_dir / "annotated.pdf"
    if generated_annotated_pdf.exists():
        shutil.copy2(generated_annotated_pdf, annotated_pdf)

    # Load JSON result
    json_path = output_dir / f"{pdf_path.stem}.json"
    with open(json_path, "r", encoding="utf-8") as f:
        odl_data = json.load(f)

    # Extract table boxes
    tables = []
    for kid in odl_data.get("kids", []):
        if kid.get("type") == "table":
            tables.append(
                {
                    "id": kid.get("id"),
                    "page_number": kid.get("page number"),
                    "bbox": kid.get("bounding box"),
                    "num_rows": kid.get("number of rows", 0),
                    "num_cols": kid.get("number of columns", 0),
                    "content": kid.get("rows", []),
                }
            )

    result = {
        "input_pdf": str(pdf_path),
        "output_dir": str(output_dir),
        "json_path": str(json_path),
        "markdown_path": str(output_dir / f"{pdf_path.stem}.md"),
        "annotated_pdf_path": str(annotated_pdf)
        if annotated_pdf.exists()
        else str(generated_annotated_pdf),
        "elapsed_sec": round(elapsed, 2),
        "total_tables": len(tables),
        "tables": tables,
    }

    print(f"Found {len(tables)} tables")
    for t in tables:
        print(f"  Table {t['id']}: bbox={[round(x, 1) for x in t['bbox']]}")
    if generated_annotated_pdf.exists():
        print(f"Saved: {generated_annotated_pdf}")
    if annotated_pdf.exists():
        print(f"Saved: {annotated_pdf}")

    save_json(result, output_dir / "stage1_opendataloader.json")
    print(f"Saved: {output_dir / 'stage1_opendataloader.json'}")

    return result


def run_stage2_rotation_check(
    pdf_path: Path,
    stage1_result: dict,
    output_dir: Path,
    angle_threshold: float = 2.0,
    overlap_threshold: float = 0.08,
) -> dict:
    """Stage 2: Detector-first rotated table detection and ODL mapping."""
    print("\n" + "=" * 60)
    print("STAGE 2: Rotation Check")
    print("=" * 60)

    output_dir.mkdir(parents=True, exist_ok=True)

    doc = fitz.open(str(pdf_path))
    detector = NativePDFTableDetector(angle_threshold=angle_threshold)
    odl_tables_by_page: dict[int, list[dict]] = {}
    for table in stage1_result["tables"]:
        odl_tables_by_page.setdefault(table["page_number"], []).append(table)

    detector_rotated_tables = []
    detected_count = 0
    matched_count = 0
    unmatched_count = 0
    detection_counter = 1

    for page_idx in range(len(doc)):
        page_num = page_idx + 1
        page_result = detector.process_page(doc[page_idx])
        rotated_tables = page_result.get("rotated_tables", [])
        if not rotated_tables:
            continue

        for rot in rotated_tables:
            detected_count += 1
            matched_odl, overlap_score, match_mode = _match_detector_table_with_odl_boxes(
                detector_table=rot,
                odl_tables_on_page=odl_tables_by_page.get(page_num, []),
                overlap_threshold=overlap_threshold,
            )

            item = {
                "detection_id": detection_counter,
                "page_number": page_num,
                "angle": float(rot.get("angle", 0.0)),
                "match_score": overlap_score,
                "match_mode": match_mode,
                "matched_table_id": matched_odl.get("id") if matched_odl else None,
                "matched_bbox": matched_odl.get("bbox") if matched_odl else None,
                "detector_obb": rot.get("obb_loose") or rot.get("obb"),
                "detector_obb_tight": rot.get("obb_tight"),
            }
            detector_rotated_tables.append(item)

            if matched_odl:
                matched_count += 1
                print(
                    f"  Detect#{detection_counter} page {page_num}: ROTATED angle={item['angle']:.1f}° -> ODL table {item['matched_table_id']} (overlap={overlap_score:.2f}, mode={match_mode})"
                )
            else:
                unmatched_count += 1
                print(
                    f"  Detect#{detection_counter} page {page_num}: ROTATED angle={item['angle']:.1f}° -> UNMATCHED (overlap={overlap_score:.2f})"
                )
            detection_counter += 1

    doc.close()

    stage2_result = {
        "angle_threshold": angle_threshold,
        "overlap_threshold": overlap_threshold,
        "odl_total_tables": len(stage1_result["tables"]),
        "detector_rotated_total": detected_count,
        "detector_matched_to_odl": matched_count,
        "detector_unmatched": unmatched_count,
        "detector_rotated_tables": detector_rotated_tables,
    }

    save_json(stage2_result, output_dir / "stage2_rotation_check.json")
    print(
        f"\nSummary: detector_rotated={detected_count}, matched={matched_count}, unmatched={unmatched_count}"
    )
    print(f"Saved: {output_dir / 'stage2_rotation_check.json'}")

    return stage2_result


def run_stage3_hybrid_ocr(
    pdf_path: Path, stage2_result: dict, output_dir: Path
) -> dict:
    """Stage 3: Hybrid OCR for rotated tables using detector deskew."""
    print("\n" + "=" * 60)
    print("STAGE 3: Hybrid OCR")
    print("=" * 60)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize PaddleOCR (singleton)
    print("Initializing PaddleOCR...")
    ocr_engine = PaddleOCR(lang="vi", use_angle_cls=True, use_gpu=False)
    detector = NativePDFTableDetector(angle_threshold=stage2_result["angle_threshold"])

    doc = fitz.open(str(pdf_path))

    hybrid_results = []

    for rot_result in stage2_result.get("detector_rotated_tables", []):
        detection_id = rot_result["detection_id"]
        matched_table_id = rot_result.get("matched_table_id")
        page_num = rot_result["page_number"]
        angle = rot_result["angle"]
        detector_obb_tight = rot_result.get("detector_obb_tight")
        detector_obb = rot_result.get("detector_obb")
        bbox = rot_result.get("matched_bbox")
        selected_obb = detector_obb_tight or detector_obb
        crop_mode = "tight" if detector_obb_tight is not None else "loose"
        if not selected_obb:
            print(f"  Detect#{detection_id}: skipped (missing detector_obb)")
            continue

        page = doc[page_num - 1]

        # Extract patch with detector's own geometry/deskew logic.
        patch_before, patch_deskewed = detector._extract_cells_from_rotated(
            page, selected_obb, angle
        )
        patch_deskewed = detector._normalize_patch_upright(patch_deskewed)

        # Save patches for review
        patch_dir = output_dir / f"detection_{detection_id}"
        patch_dir.mkdir(exist_ok=True)

        if patch_before.size > 0:
            Image.fromarray(patch_before.astype("uint8"), mode="RGB").save(
                patch_dir / "before.png"
            )
        if patch_deskewed.size > 0:
            Image.fromarray(patch_deskewed.astype("uint8"), mode="RGB").save(
                patch_dir / "deskewed.png"
            )

        # OCR
        ocr_result = ocr_patch(patch_deskewed, ocr_engine)

        result = {
            "detection_id": detection_id,
            "matched_table_id": matched_table_id,
            "page_number": page_num,
            "bbox": bbox,
            "crop_mode": crop_mode,
            "detector_obb_tight": detector_obb_tight,
            "detector_obb": detector_obb,
            "angle": angle,
            "ocr_markdown": ocr_result["markdown"],
            "ocr_confidence": ocr_result["confidence"],
            "patch_before": str(patch_dir / "before.png"),
            "patch_deskewed": str(patch_dir / "deskewed.png"),
        }
        hybrid_results.append(result)

        print(
            f"  Detect#{detection_id}: angle={angle:.1f}°, conf={ocr_result['confidence']:.2f}, crop={crop_mode}, matched_table_id={matched_table_id}"
        )
        print(f"    Markdown: {ocr_result['markdown'][:100]}...")

    doc.close()

    stage3_result = {
        "total_rotated": len(hybrid_results),
        "hybrid_results": hybrid_results,
    }

    save_json(stage3_result, output_dir / "stage3_hybrid_ocr.json")
    print(f"\nOCR'd {len(hybrid_results)} rotated tables")
    print(f"Saved: {output_dir / 'stage3_hybrid_ocr.json'}")

    return stage3_result


def run_stage4_merge(
    stage1_result: dict, stage2_result: dict, stage3_result: dict, output_dir: Path
) -> dict:
    """Stage 4: Merge hybrid OCR results with opendataloader JSON."""
    print("\n" + "=" * 60)
    print("STAGE 4: Merge Results")
    print("=" * 60)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load original opendataloader JSON
    odl_json_path = Path(stage1_result["json_path"])
    with open(odl_json_path, "r", encoding="utf-8") as f:
        odl_json = json.load(f)

    # Create lookup for matched hybrid results (ODL table id -> best OCR result).
    hybrid_lookup: dict[Any, dict] = {}
    unmatched_hybrid = []
    for item in stage3_result["hybrid_results"]:
        matched_table_id = item.get("matched_table_id")
        if matched_table_id is None:
            unmatched_hybrid.append(item)
            continue
        current = hybrid_lookup.get(matched_table_id)
        if current is None or item.get("ocr_confidence", 0.0) > current.get(
            "ocr_confidence", 0.0
        ):
            hybrid_lookup[matched_table_id] = item

    # Merge
    merged_kids = []
    replaced_count = 0

    for kid in odl_json.get("kids", []):
        if kid.get("type") == "table":
            table_id = kid.get("id")

            if table_id in hybrid_lookup:
                # Replace content with OCR result
                hybrid = hybrid_lookup[table_id]

                # Parse markdown to get rows
                md = hybrid["ocr_markdown"]
                rows_data = parse_markdown_to_rows(md)

                # Update kid with new content
                kid["number of rows"] = len(rows_data)
                kid["rotated_table_replaced"] = True
                kid["ocr_confidence"] = hybrid["ocr_confidence"]
                # Preserve original OCR markdown verbatim for markdown patch stage.
                kid["ocr_markdown_raw"] = md
                kid["rows"] = rows_data

                replaced_count += 1
                print(f"  Table {table_id}: REPLACED")
            else:
                print(f"  Table {table_id}: KEPT (original)")

        merged_kids.append(kid)

    # Save merged JSON
    merged_json = {
        **odl_json,
        "kids": merged_kids,
        "merged_with_hybrid": True,
        "rotated_tables_replaced": replaced_count,
        "detector_rotated_total": stage2_result.get("detector_rotated_total", 0),
        "detector_unmatched_count": len(unmatched_hybrid),
        "unmatched_rotated_tables": [
            {
                "detection_id": item.get("detection_id"),
                "page_number": item.get("page_number"),
                "angle": item.get("angle"),
                "ocr_confidence": item.get("ocr_confidence"),
                "ocr_markdown": item.get("ocr_markdown"),
            }
            for item in unmatched_hybrid
        ],
    }

    merged_json_path = output_dir / "merged_results.json"
    save_json(merged_json, merged_json_path)

    # Generate markdown by patching Stage1 markdown in-place.
    merged_md = merge_into_original_markdown_with_replacements(
        original_markdown_path=Path(stage1_result["markdown_path"]),
        original_json=odl_json,
        merged_json=merged_json,
    )
    merged_md_path = output_dir / "merged_results.md"
    with open(merged_md_path, "w", encoding="utf-8") as f:
        f.write(merged_md)

    print(f"\nMerged {replaced_count} tables")
    print(f"Saved JSON: {merged_json_path}")
    print(f"Saved Markdown: {merged_md_path}")

    return {
        "merged_json_path": str(merged_json_path),
        "merged_md_path": str(merged_md_path),
        "tables_replaced": replaced_count,
    }


def parse_markdown_to_rows(markdown: str) -> list:
    """Parse markdown table to rows structure."""
    lines = markdown.strip().split("\n")
    if len(lines) < 2:
        return []

    data_rows = lines[2:] if "---" in lines[1] else lines[1:]

    rows = []
    for row_idx, line in enumerate(data_rows, start=1):
        line = line.strip()
        if not line.startswith("|") or not line.endswith("|"):
            continue

        cells = [c.strip() for c in line.split("|")[1:-1]]

        row = {"type": "table row", "row number": row_idx, "cells": []}

        for col_idx, cell_text in enumerate(cells, start=1):
            cell = {
                "type": "table cell",
                "page number": 1,
                "bounding box": [0, 0, 0, 0],
                "row number": row_idx,
                "column number": col_idx,
                "row span": 1,
                "column span": 1,
                "kids": [
                    {
                        "type": "paragraph",
                        "id": (row_idx - 1) * len(cells) + col_idx,
                        "page number": 1,
                        "bounding box": [0, 0, 0, 0],
                        "content": cell_text,
                    }
                ],
            }
            row["cells"].append(cell)

        rows.append(row)

    return rows


def _rows_to_cell_matrix(rows: list[dict]) -> list[list[str]]:
    matrix = []
    for row in rows:
        cells = []
        for cell in row.get("cells", []):
            paragraphs = cell.get("kids", [])
            text = paragraphs[0].get("content", "") if paragraphs else ""
            cells.append(text)
        if cells:
            matrix.append(cells)
    return matrix


def _render_table_markdown_from_rows(rows: list[dict]) -> str:
    matrix = _rows_to_cell_matrix(rows)
    if not matrix:
        return ""

    max_cols = max(len(r) for r in matrix)
    matrix = [r + [""] * (max_cols - len(r)) for r in matrix]
    header = matrix[0]
    body = matrix[1:]

    lines = [
        "|" + "|".join(header) + "|",
        "|" + "|".join(["---"] * max_cols) + "|",
    ]
    for row in body:
        lines.append("|" + "|".join(row) + "|")
    return "\n".join(lines)


def _extract_markdown_table_blocks(markdown_text: str) -> list[dict]:
    lines = markdown_text.splitlines()
    blocks = []
    start = None
    for idx, line in enumerate(lines):
        is_table_line = line.strip().startswith("|") and line.strip().endswith("|")
        if is_table_line and start is None:
            start = idx
        elif not is_table_line and start is not None:
            blocks.append({"start": start, "end": idx - 1})
            start = None
    if start is not None:
        blocks.append({"start": start, "end": len(lines) - 1})

    for block in blocks:
        block_lines = lines[block["start"] : block["end"] + 1]
        block["text"] = "\n".join(block_lines)
    return blocks


def _normalize_table_markdown(table_md: str) -> str:
    parts = []
    for line in table_md.splitlines():
        line = line.strip()
        if not line.startswith("|") or not line.endswith("|"):
            continue
        cells = [c.strip() for c in line.split("|")[1:-1]]
        parts.append("|" + "|".join(cells) + "|")
    return "\n".join(parts)


def merge_into_original_markdown_with_replacements(
    original_markdown_path: Path, original_json: dict, merged_json: dict
) -> str:
    if not original_markdown_path.exists():
        return generate_markdown_from_json(merged_json)

    original_md = original_markdown_path.read_text(encoding="utf-8")
    blocks = _extract_markdown_table_blocks(original_md)
    if not blocks:
        return original_md

    original_tables = [k for k in original_json.get("kids", []) if k.get("type") == "table"]
    merged_tables_by_id = {
        k.get("id"): k for k in merged_json.get("kids", []) if k.get("type") == "table"
    }

    used_block_idxs: set[int] = set()
    table_id_to_block_idx: dict[Any, int] = {}

    for table in original_tables:
        table_id = table.get("id")
        rendered = _normalize_table_markdown(
            _render_table_markdown_from_rows(table.get("rows", []))
        )
        if not rendered:
            continue

        best_idx = -1
        best_score = 0.0
        for idx, block in enumerate(blocks):
            if idx in used_block_idxs:
                continue
            block_norm = _normalize_table_markdown(block["text"])
            score = SequenceMatcher(None, rendered, block_norm).ratio()
            if score > best_score:
                best_score = score
                best_idx = idx

        if best_idx != -1:
            used_block_idxs.add(best_idx)
            table_id_to_block_idx[table_id] = best_idx

    lines = original_md.splitlines()
    replacement_ranges = []
    for table in merged_json.get("kids", []):
        if table.get("type") != "table":
            continue
        if not table.get("rotated_table_replaced"):
            continue

        table_id = table.get("id")
        block_idx = table_id_to_block_idx.get(table_id)
        if block_idx is None:
            continue

        raw_ocr_md = (table.get("ocr_markdown_raw") or "").strip()
        if raw_ocr_md:
            new_table_md = _normalize_table_markdown(raw_ocr_md)
        else:
            new_table_md = _render_table_markdown_from_rows(table.get("rows", []))
        if not new_table_md:
            continue
        block = blocks[block_idx]
        replacement_ranges.append(
            (block["start"], block["end"], new_table_md.splitlines())
        )

    if not replacement_ranges:
        return original_md

    replacement_ranges.sort(key=lambda x: x[0], reverse=True)
    for start, end, new_lines in replacement_ranges:
        lines[start : end + 1] = new_lines

    return "\n".join(lines) + ("\n" if original_md.endswith("\n") else "")


def generate_markdown_from_json(json_data: dict) -> str:
    """Generate markdown from merged JSON."""
    md_parts = ["# Document", ""]

    for kid in json_data.get("kids", []):
        if kid.get("type") == "table":
            md_parts.append(f"## Table {kid.get('id')}")
            md_parts.append("")

            rows = kid.get("rows", [])
            if rows:
                all_cells = []
                for row in rows:
                    cells = []
                    for cell in row.get("cells", []):
                        paragraphs = cell.get("kids", [])
                        text = paragraphs[0].get("content", "") if paragraphs else ""
                        cells.append(text)
                    all_cells.append(cells)

                if all_cells:
                    col_count = len(all_cells[0])
                    md_parts.append(
                        "| "
                        + " | ".join([f"Col {i + 1}" for i in range(col_count)])
                        + " |"
                    )
                    md_parts.append("| " + " | ".join(["---"] * col_count) + " |")

                    for row_cells in all_cells:
                        md_parts.append("| " + " | ".join(row_cells) + " |")

            md_parts.append("")
            md_parts.append("---")
            md_parts.append("")

    unmatched = json_data.get("unmatched_rotated_tables", [])
    if unmatched:
        md_parts.append("## Unmatched Detector Rotated Tables")
        md_parts.append("")
        for idx, item in enumerate(unmatched, start=1):
            md_parts.append(
                f"### Detector #{item.get('detection_id', idx)} (page {item.get('page_number')}, angle={item.get('angle')})"
            )
            md_parts.append(f"Confidence: {item.get('ocr_confidence', 0):.2f}")
            md_parts.append("")
            md_parts.append(item.get("ocr_markdown", ""))
            md_parts.append("")
            md_parts.append("---")
            md_parts.append("")

    return "\n".join(md_parts)


def run_full_pipeline(
    pdf_path: Path,
    output_dir: Path,
    angle_threshold: float = 2.0,
    overlap_threshold: float = 0.08,
) -> dict:
    """Run the full production pipeline."""
    result = run_pipeline(
        pdf_path=pdf_path,
        output_dir=output_dir,
        angle_threshold=angle_threshold,
        overlap_threshold=overlap_threshold,
        save_debug_artifacts=True,
    )
    stage1_result = {
        "total_tables": result.stage1.total_tables,
    }
    stage2_result = {
        "detector_rotated_total": result.stage2.detector_rotated_total,
    }
    stage4_result = {
        "tables_replaced": result.stage4.tables_replaced,
    }

    # Summary
    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)
    print(f"Input: {pdf_path}")
    print(f"Output: {output_dir}")
    print(f"Tables found: {stage1_result['total_tables']}")
    print(f"Detector rotated tables: {stage2_result['detector_rotated_total']}")
    print(f"Tables replaced with OCR: {stage4_result['tables_replaced']}")

    return {
        "stage1": result.stage1,
        "stage2": result.stage2,
        "stage3": result.stage3,
        "stage4": result.stage4,
        "metrics": result.metrics,
    }


if __name__ == "__main__":
    import argparse
    import time

    parser = argparse.ArgumentParser(description="Test new approach pipeline")
    parser.add_argument("--pdf", type=str, default="data/rotated_tables_one_page.pdf")
    parser.add_argument("--out", type=str, default="output/new_approach_test")
    parser.add_argument("--threshold", type=float, default=2.0)
    parser.add_argument("--overlap-threshold", type=float, default=0.08)

    args = parser.parse_args()

    pdf_path = Path(args.pdf)
    output_dir = Path(args.out)

    run_full_pipeline(pdf_path, output_dir, args.threshold, args.overlap_threshold)
