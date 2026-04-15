import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def load_json(json_path: Path) -> Dict[str, Any]:
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Dict[str, Any], json_path: Path) -> None:
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def compute_iou(box1: List[float], box2: List[float]) -> float:
    """Compute Intersection over Union for two bounding boxes [x0, y0, x1, y1]."""
    x0_1, y0_1, x1_1, y1_1 = box1
    x0_2, y0_2, x1_2, y1_2 = box2

    xi = max(x0_1, x0_2)
    yi = max(y0_1, y0_2)
    x1i = min(x1_1, x1_2)
    y1i = min(y1_1, y1_2)

    if xi >= x1i or yi >= y1i:
        return 0.0

    inter_area = (x1i - xi) * (y1i - yi)
    box1_area = (x1_1 - x0_1) * (y1_1 - y0_1)
    box2_area = (x1_2 - x0_2) * (y1_2 - y0_2)
    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area if union_area > 0 else 0.0


def parse_opendataloader_tables(odl_json: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract table information from opendataloader JSON."""
    tables = []
    kids = odl_json.get("kids", [])

    for kid in kids:
        if kid.get("type") == "table":
            table_info = {
                "id": kid.get("id"),
                "page_number": kid.get("page number"),
                "bounding_box": kid.get("bounding box"),
                "num_rows": kid.get("number of rows", 0),
                "num_cols": kid.get("number of columns", 0),
                "rows": kid.get("rows", []),
            }
            tables.append(table_info)

    return tables


def parse_merged_tables(merged_results: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract rotated tables from hybrid processor results."""
    tables = []

    for page_result in merged_results.get("pages", []):
        page_num = page_result.get("page_number", 1)
        rotated_tables = page_result.get("rotated_tables", [])

        for table_idx, table in enumerate(rotated_tables, start=1):
            obb = table.get("obb", {})
            cx = obb.get("cx", 0)
            cy = obb.get("cy", 0)
            w = obb.get("w", 0)
            h = obb.get("h", 0)

            table_info = {
                "page_number": page_num,
                "bounding_box": [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2],
                "angle": table.get("angle", 0),
                "ocr_markdown": table.get("ocr_markdown", ""),
                "ocr_html": table.get("ocr_html", ""),
                "ocr_confidence": table.get("ocr_confidence", 0),
                "table_index": table_idx,
            }
            tables.append(table_info)

    return tables


def match_tables_by_position(
    odl_tables: List[Dict[str, Any]],
    merged_tables: List[Dict[str, Any]],
    iou_threshold: float = 0.3,
) -> List:
    """Match tables from opendataloader with rotated tables from hybrid processor."""
    matches = []
    used_merged_indices: set[int] = set()

    for odl_table in odl_tables:
        odl_box = odl_table.get("bounding_box", [0, 0, 0, 0])
        if not isinstance(odl_box, list) or len(odl_box) != 4:
            matches.append((odl_table, None))
            continue

        odl_page = odl_table.get("page_number")
        best_match = None
        best_iou = 0.0
        best_idx = -1

        for idx, merged_table in enumerate(merged_tables):
            if idx in used_merged_indices:
                continue
            if merged_table.get("page_number") != odl_page:
                continue
            merged_box = merged_table.get("bounding_box", [0, 0, 0, 0])
            if not isinstance(merged_box, list) or len(merged_box) != 4:
                continue
            iou = compute_iou(odl_box, merged_box)

            if iou > best_iou and iou >= iou_threshold:
                best_iou = iou
                best_match = merged_table
                best_idx = idx

        if best_idx >= 0:
            used_merged_indices.add(best_idx)

        matches.append((odl_table, best_match))

    return matches


def parse_markdown_table(markdown_table: str) -> List[List[str]]:
    """Parse markdown table to 2D array of cells."""
    lines = [line.strip() for line in markdown_table.strip().split("\n") if line.strip()]
    if len(lines) < 2:
        return []

    def is_separator_row(line: str) -> bool:
        if not (line.startswith("|") and line.endswith("|")):
            return False
        cells = [cell.strip() for cell in line.split("|")[1:-1]]
        if not cells:
            return False
        return all(cell.replace("-", "").replace(":", "") == "" for cell in cells)

    data_rows = lines[1:] if is_separator_row(lines[1]) else lines[1:]
    if is_separator_row(lines[1]):
        data_rows = lines[2:]

    rows = []
    for line in data_rows:
        line = line.strip()
        if line.startswith("|") and line.endswith("|"):
            cells = [cell.strip() for cell in line.split("|")[1:-1]]
            rows.append(cells)

    return rows


def replace_table_content(
    odl_table: Dict[str, Any],
    merged_table: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Replace table content with PaddleOCR results."""
    if merged_table is None:
        return odl_table

    confidence = float(merged_table.get("ocr_confidence", 0.0) or 0.0)
    if confidence < 0.95:
        return odl_table

    markdown = merged_table.get("ocr_markdown", "")
    if not markdown:
        return odl_table

    new_rows = parse_markdown_table(markdown)
    if not new_rows:
        return odl_table

    page_num = odl_table.get("page number", odl_table.get("page_number", 1))
    new_num_rows = len(new_rows)
    new_num_cols = len(new_rows[0]) if new_rows else 0
    old_num_rows = int(
        odl_table.get("number of rows", odl_table.get("num_rows", new_num_rows))
    )
    old_num_cols = int(
        odl_table.get("number of columns", odl_table.get("num_cols", new_num_cols))
    )

    # Guardrail: avoid replacing a dense/complete table with a clearly weaker OCR table.
    if old_num_rows >= 3 and new_num_rows < max(2, int(old_num_rows * 0.7)):
        return odl_table
    if old_num_cols >= 3 and new_num_cols < max(2, int(old_num_cols * 0.7)):
        return odl_table

    new_kid = dict(odl_table)
    new_rows = create_rows_from_cells(new_rows, page_num)

    new_kid.update(
        {
            "type": "table",
            "page number": page_num,
            "number of rows": new_num_rows,
            "number of columns": new_num_cols,
            "rotated_table_replaced": True,
            "ocr_confidence": merged_table.get("ocr_confidence", 0),
            "rows": new_rows,
            # Keep only OCR-derived rows. Old table-level kids/content may contain
            # misaligned text blocks from the expanded detector box.
            "kids": [],
            "content": "",
            "hybrid_ocr_metadata": {
                "page_number": merged_table.get("page_number"),
                "table_index": merged_table.get("table_index"),
                "angle": merged_table.get("angle"),
                "confidence": merged_table.get("ocr_confidence", 0),
            },
        }
    )

    if "bounding box" not in new_kid:
        new_kid["bounding box"] = odl_table.get("bounding_box")

    if new_kid.get("bounding box") is None and isinstance(
        odl_table.get("bounding_box"), list
    ):
        new_kid["bounding box"] = odl_table.get("bounding_box")

    if "bounding_box" in new_kid:
        new_kid.pop("bounding_box", None)

    return new_kid


def create_rows_from_cells(
    cells: List[List[str]], page_num: int
) -> List[Dict[str, Any]]:
    """Convert 2D cell array to opendataloader row structure."""
    rows = []
    for row_idx, row_cells in enumerate(cells, start=1):
        row_dict = {
            "type": "table row",
            "row number": row_idx,
            "cells": [],
        }

        for col_idx, cell_text in enumerate(row_cells, start=1):
            cell_dict = {
                "type": "table cell",
                "page number": page_num,
                "bounding box": [0, 0, 0, 0],
                "row number": row_idx,
                "column number": col_idx,
                "row span": 1,
                "column span": 1,
                "kids": [
                    {
                        "type": "paragraph",
                        "id": (row_idx - 1) * len(row_cells) + col_idx,
                        "page number": page_num,
                        "bounding box": [0, 0, 0, 0],
                        "content": cell_text,
                    }
                ],
            }
            row_dict["cells"].append(cell_dict)

        rows.append(row_dict)

    return rows


def merge_opendataloader_with_hybrid(
    odl_json_path: Path,
    hybrid_results: Dict[str, Any],
    output_json_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Merge opendataloader JSON with PaddleOCR results from hybrid processor."""
    odl_json = load_json(odl_json_path)

    odl_tables = parse_opendataloader_tables(odl_json)
    merged_tables = parse_merged_tables(hybrid_results)

    matches = match_tables_by_position(odl_tables, merged_tables)

    new_kids = []
    replacements_count = 0

    for kid in odl_json.get("kids", []):
        if kid.get("type") == "table":
            odl_table = kid
            matched = False
            for orig, replacement in matches:
                if orig.get("id") == odl_table.get("id"):
                    new_kid = replace_table_content(odl_table, replacement)
                    if new_kid.get("rotated_table_replaced"):
                        replacements_count += 1
                    new_kids.append(new_kid)
                    matched = True
                    break
            if not matched:
                new_kids.append(kid)
        else:
            new_kids.append(kid)

    merged_json = {
        **odl_json,
        "kids": new_kids,
        "merged_with_hybrid": True,
        "rotated_tables_replaced": replacements_count,
    }

    if output_json_path:
        save_json(merged_json, output_json_path)
        print(f"Merged JSON saved to: {output_json_path}")

    return merged_json


def generate_merged_markdown(merged_json: Dict[str, Any]) -> str:
    """Generate markdown from merged JSON using content only."""
    md_parts = []

    for kid in merged_json.get("kids", []):
        if kid.get("type") != "table":
            continue

        rows = kid.get("rows", [])
        if not rows:
            continue

        all_cells = []
        for row in rows:
            cells = row.get("cells", [])
            row_cells = []
            for cell in cells:
                paragraphs = cell.get("kids", [])
                text = paragraphs[0].get("content", "") if paragraphs else ""
                row_cells.append(text)
            all_cells.append(row_cells)

        if not all_cells:
            continue

        col_count = len(all_cells[0])
        md_parts.append("| " + " | ".join([f"Col {i + 1}" for i in range(col_count)]) + " |")
        md_parts.append("| " + " | ".join(["---"] * col_count) + " |")

        for row_cells in all_cells:
            md_parts.append("| " + " | ".join(row_cells) + " |")

        md_parts.append("")

    return "\n".join(md_parts).strip()


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print(
            "Usage: python merge_results.py <opendataloader.json> <hybrid_results.json>"
        )
        sys.exit(1)

    odl_json_path = Path(sys.argv[1])
    hybrid_json_path = Path(sys.argv[2])

    hybrid_results = load_json(hybrid_json_path)
    merged_json = merge_opendataloader_with_hybrid(odl_json_path, hybrid_results)

    output_md_path = odl_json_path.with_suffix(".merged.md")
    md_content = generate_merged_markdown(merged_json)
    output_md_path.write_text(md_content, encoding="utf-8")
    print(f"Merged markdown saved to: {output_md_path}")
    print(md_content)
