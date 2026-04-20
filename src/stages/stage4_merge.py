from __future__ import annotations

import re
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from ..pipeline.contracts import Stage1Result, Stage2Result, Stage3Result, Stage4Result
from ..utils.io import read_json, save_json


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _split_bbox_grid(
    table_bbox: list[float], num_rows: int, num_cols: int
) -> list[list[list[float]]]:
    """
    Create a simple grid of [x0,y0,x1,y1] cells covering the table bbox.
    This preserves stage1 schema expectations (page number + bbox present)
    even when OCR doesn't provide per-cell geometry.
    """
    if not (isinstance(table_bbox, list) and len(table_bbox) == 4):
        table_bbox = [0.0, 0.0, 0.0, 0.0]
    x0, y0, x1, y1 = [float(v) for v in table_bbox]
    if num_rows <= 0 or num_cols <= 0:
        return []
    w = max(1.0, x1 - x0)
    h = max(1.0, y1 - y0)
    cell_w = w / num_cols
    cell_h = h / num_rows
    grid: list[list[list[float]]] = []
    for r in range(num_rows):
        row_boxes = []
        for c in range(num_cols):
            cx0 = x0 + c * cell_w
            cy0 = y0 + r * cell_h
            cx1 = x0 + (c + 1) * cell_w
            cy1 = y0 + (r + 1) * cell_h
            # Keep strictly inside table bbox
            row_boxes.append(
                [
                    _clamp(cx0, x0, x1),
                    _clamp(cy0, y0, y1),
                    _clamp(cx1, x0, x1),
                    _clamp(cy1, y0, y1),
                ]
            )
        grid.append(row_boxes)
    return grid


def parse_markdown_to_rows(markdown: str, *, page_number: int, table_bbox: list[float]) -> list:
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
        if not cells:
            continue
        row = {"type": "table row", "row number": row_idx, "cells": []}
        # Build grid once we know final shape; for now collect cell texts.
        for col_idx, cell_text in enumerate(cells, start=1):
            row["cells"].append(
                {
                    "type": "table cell",
                    "page number": int(page_number),
                    "bounding box": [0.0, 0.0, 0.0, 0.0],
                    "row number": row_idx,
                    "column number": col_idx,
                    "row span": 1,
                    "column span": 1,
                    "kids": [
                        {
                            "type": "paragraph",
                            "id": (row_idx - 1) * len(cells) + col_idx,
                            "page number": int(page_number),
                            "bounding box": [0.0, 0.0, 0.0, 0.0],
                            "content": cell_text,
                        }
                    ],
                }
            )
        rows.append(row)

    # Populate approximate geometry so schema matches stage1 expectations.
    if not rows:
        return []
    num_rows = len(rows)
    num_cols = max((len(r.get("cells", [])) for r in rows), default=0)
    if num_cols <= 0:
        return rows
    grid = _split_bbox_grid(table_bbox, num_rows, num_cols)
    for r_idx, row in enumerate(rows):
        for c_idx, cell in enumerate(row.get("cells", [])):
            if r_idx < len(grid) and c_idx < len(grid[r_idx]):
                bbox = grid[r_idx][c_idx]
                cell["bounding box"] = bbox
                kids = cell.get("kids") or []
                if kids:
                    kids[0]["bounding box"] = bbox
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
        block["text"] = "\n".join(lines[block["start"] : block["end"] + 1])
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


def _table_vocab_from_markdown(table_md: str) -> set[str]:
    return {
        tok.lower()
        for tok in re.findall(r"[A-Za-z0-9]+", table_md)
        if tok.strip()
    }


def _token_matches_table_vocab(token: str, table_vocab: set[str]) -> bool:
    if token in table_vocab:
        return True

    # Accept compacted tokens like "a1b1c1d1" or "col1col2col3col4"
    # if they can be segmented into >=2 known table tokens.
    n = len(token)
    best_parts = [-1] * (n + 1)
    best_parts[0] = 0
    for i in range(n):
        if best_parts[i] < 0:
            continue
        for j in range(i + 1, n + 1):
            if token[i:j] in table_vocab:
                best_parts[j] = max(best_parts[j], best_parts[i] + 1)
    return best_parts[n] >= 2


def _is_table_noise_line(line: str, table_vocab: set[str]) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    if stripped.startswith("|") and stripped.endswith("|"):
        return False

    candidate = stripped.lstrip("-* ").strip()
    tokens = [tok.lower() for tok in re.findall(r"[A-Za-z0-9]+", candidate)]
    if not tokens:
        return False

    matched = sum(1 for tok in tokens if _token_matches_table_vocab(tok, table_vocab))
    return matched / len(tokens) >= 0.8


def merge_into_original_markdown_with_replacements(
    original_markdown_path: Path, original_json: dict, merged_json: dict
) -> str:
    if not original_markdown_path.exists():
        return ""

    original_md = original_markdown_path.read_text(encoding="utf-8")
    blocks = _extract_markdown_table_blocks(original_md)
    if not blocks:
        return original_md

    original_tables = [k for k in original_json.get("kids", []) if k.get("type") == "table"]
    used_block_idxs: set[int] = set()
    table_id_to_block_idx: dict[Any, int] = {}

    for table in original_tables:
        table_id = table.get("id")
        rendered = _normalize_table_markdown(
            _render_table_markdown_from_rows(table.get("rows", []))
        )
        if not rendered:
            continue
        best_idx, best_score = -1, 0.0
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
    replacements = []
    for table in merged_json.get("kids", []):
        if table.get("type") != "table" or not table.get("rotated_table_replaced"):
            continue
        block_idx = table_id_to_block_idx.get(table.get("id"))
        if block_idx is None:
            continue
        raw_ocr_md = (table.get("ocr_markdown_raw") or "").strip()
        new_table_md = (
            _normalize_table_markdown(raw_ocr_md)
            if raw_ocr_md
            else _render_table_markdown_from_rows(table.get("rows", []))
        )
        if not new_table_md:
            continue
        block = blocks[block_idx]
        replace_end = block["end"]

        # Some ODL markdown outputs include duplicate/noisy table text immediately
        # after a table block (e.g., flattened bullets). Remove that tail as well.
        next_table_start = len(lines)
        for other in blocks:
            if other["start"] > block["end"]:
                next_table_start = min(next_table_start, other["start"])

        tail_lines = lines[block["end"] + 1 : next_table_start]
        non_empty_tail = [ln for ln in tail_lines if ln.strip()]
        table_vocab = _table_vocab_from_markdown(new_table_md)
        if non_empty_tail and all(
            _is_table_noise_line(ln, table_vocab) for ln in non_empty_tail
        ):
            replace_end = next_table_start - 1

        replacement_lines = new_table_md.splitlines()
        if block["start"] > 0 and lines[block["start"] - 1].strip():
            replacement_lines = [""] + replacement_lines
        if replace_end + 1 < len(lines) and lines[replace_end + 1].strip():
            replacement_lines = replacement_lines + [""]

        replacements.append((block["start"], replace_end, replacement_lines))

    replacements.sort(key=lambda x: x[0], reverse=True)
    for start, end, new_lines in replacements:
        lines[start : end + 1] = new_lines
    return "\n".join(lines) + ("\n" if original_md.endswith("\n") else "")


class Stage4Merger:
    def run(
        self,
        stage1: Stage1Result,
        stage2: Stage2Result,
        stage3: Stage3Result,
        output_dir: str,
    ) -> Stage4Result:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        odl_json = read_json(Path(stage1.json_path))

        # Build redaction boxes (tight AABB) from stage2 detections.
        redaction_boxes: list[dict[str, Any]] = []
        for det in stage2.detector_rotated_tables:
            box = det.detector_aabb_tight or det.detector_aabb
            if not box or len(box) != 4:
                continue
            redaction_boxes.append(
                {"page_number": int(det.page_number), "bbox": [float(v) for v in box]}
            )

        def _intersects(a: list[float], b: list[float]) -> bool:
            x0 = max(a[0], b[0])
            y0 = max(a[1], b[1])
            x1 = min(a[2], b[2])
            y1 = min(a[3], b[3])
            return x1 > x0 and y1 > y0

        def _kid_bbox(k: dict) -> list[float] | None:
            bb = k.get("bounding box")
            if isinstance(bb, list) and len(bb) == 4:
                return [float(v) for v in bb]
            return None

        def _kid_page(k: dict) -> int | None:
            pn = k.get("page number")
            try:
                return int(pn)
            except Exception:
                return None

        def _should_redact(k: dict) -> bool:
            bb = _kid_bbox(k)
            pn = _kid_page(k)
            if bb is None or pn is None:
                return False
            for rb in redaction_boxes:
                if rb["page_number"] != pn:
                    continue
                if _intersects(bb, rb["bbox"]):
                    return True
            return False

        kids_redacted = 0

        def _redact_tree(node: dict) -> dict | None:
            # If this node itself intersects a redaction box, drop it.
            nonlocal kids_redacted
            # IMPORTANT: do not drop table nodes here. Tables are replaced later
            # (matched by id) or inserted (unmatched detections). Dropping them
            # prevents replacement and leaves original noisy markdown untouched.
            if node.get("type") != "table" and _should_redact(node):
                kids_redacted += 1
                return None
            # Otherwise recursively filter kids (if present).
            kids = node.get("kids")
            if isinstance(kids, list) and kids:
                new_kids = []
                for ch in kids:
                    if not isinstance(ch, dict):
                        continue
                    kept = _redact_tree(ch)
                    if kept is not None:
                        new_kids.append(kept)
                node = dict(node)
                node["kids"] = new_kids
            return node
        hybrid_lookup: dict[Any, dict] = {}
        unmatched_hybrid = []
        for item in stage3.hybrid_results:
            if item.matched_table_id is None:
                unmatched_hybrid.append(item)
                continue
            current = hybrid_lookup.get(item.matched_table_id)
            if current is None or item.ocr_confidence > current["ocr_confidence"]:
                hybrid_lookup[item.matched_table_id] = {
                    "ocr_markdown": item.ocr_markdown,
                    "ocr_confidence": item.ocr_confidence,
                    "detection_id": item.detection_id,
                }

        replaced_count = 0
        merged_kids: list[dict] = []
        for kid in odl_json.get("kids", []):
            if not isinstance(kid, dict):
                continue
            # Redact all intersecting content first.
            redacted = _redact_tree(kid)
            if redacted is None:
                continue
            kid = redacted

            # Replace matched ODL tables by id.
            if kid.get("type") == "table" and kid.get("id") in hybrid_lookup:
                hybrid = hybrid_lookup[kid.get("id")]
                page_number = int(kid.get("page number") or 1)
                table_bbox = kid.get("bounding box") or [0.0, 0.0, 0.0, 0.0]
                rows_data = parse_markdown_to_rows(
                    hybrid["ocr_markdown"],
                    page_number=page_number,
                    table_bbox=table_bbox,
                )
                kid["number of rows"] = len(rows_data)
                kid["rotated_table_replaced"] = True
                kid["ocr_confidence"] = hybrid["ocr_confidence"]
                kid["ocr_markdown_raw"] = hybrid["ocr_markdown"]
                kid["rows"] = rows_data
                kid["kids"] = []
                kid["content"] = ""
                replaced_count += 1
            merged_kids.append(kid)

        # Insert OCR-only tables for unmatched detections (when ODL didn't have an id to match).
        det_by_id: dict[int, Any] = {d.detection_id: d for d in stage2.detector_rotated_tables}
        next_table_id = (
            max(
                [int(k.get("id")) for k in odl_json.get("kids", []) if isinstance(k, dict) and str(k.get("id", "")).isdigit()],
                default=0,
            )
            + 1
        )
        for item in unmatched_hybrid:
            det = det_by_id.get(int(item.detection_id))
            if det is None:
                continue
            page_number = int(det.page_number)
            table_bbox = det.detector_aabb_tight or det.detector_aabb
            if not table_bbox:
                continue
            rows_data = parse_markdown_to_rows(
                item.ocr_markdown,
                page_number=page_number,
                table_bbox=table_bbox,
            )
            if not rows_data:
                continue
            merged_kids.append(
                {
                    "type": "table",
                    "id": f"hybrid_{next_table_id}",
                    "page number": page_number,
                    "bounding box": [float(v) for v in table_bbox],
                    "number of rows": len(rows_data),
                    "number of columns": max(
                        (len(r.get("cells", [])) for r in rows_data if isinstance(r, dict)),
                        default=0,
                    ),
                    "rotated_table_replaced": True,
                    "ocr_confidence": float(item.ocr_confidence),
                    "ocr_markdown_raw": item.ocr_markdown,
                    "rows": rows_data,
                    "kids": [],
                    "content": "",
                }
            )
            next_table_id += 1

        merged_json = {
            **odl_json,
            "kids": merged_kids,
            "merged_with_hybrid": True,
            "rotated_tables_replaced": replaced_count,
            "detector_rotated_total": stage3.total_rotated,
            "detector_unmatched_count": len(unmatched_hybrid),
            "redaction_boxes_total": len(redaction_boxes),
            "unmatched_rotated_tables": [
                {
                    "detection_id": i.detection_id,
                    "page_number": i.page_number,
                    "angle": i.angle,
                    "ocr_confidence": i.ocr_confidence,
                    "ocr_markdown": i.ocr_markdown,
                }
                for i in unmatched_hybrid
            ],
        }

        merged_json_path = out / "merged_results.json"
        save_json(merged_json, merged_json_path)
        merged_md = merge_into_original_markdown_with_replacements(
            Path(stage1.markdown_path), odl_json, merged_json
        )
        merged_md_path = out / "merged_results.md"
        merged_md_path.write_text(merged_md, encoding="utf-8")
        return Stage4Result(
            merged_json_path=str(merged_json_path),
            merged_md_path=str(merged_md_path),
            tables_replaced=replaced_count,
            kids_redacted=kids_redacted,
        )

