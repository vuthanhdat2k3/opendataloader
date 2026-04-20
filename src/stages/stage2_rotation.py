from __future__ import annotations

import fitz

try:
    from ..core.detector import NativePDFTableDetector
except ImportError:
    from src.core.detector import NativePDFTableDetector
from ..pipeline.contracts import RotationDetection, Stage1Result, Stage2Result
from ..utils.io import save_json


def _obb_to_aabb(obb: dict[str, float]) -> list[float]:
    cx, cy, w, h = obb["cx"], obb["cy"], obb["w"], obb["h"]
    return [cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0]


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


class Stage2RotationDetector:
    def __init__(
        self, angle_threshold: float, overlap_threshold: float, spatial_dist_threshold: float = 100.0
    ):
        self.angle_threshold = angle_threshold
        self.overlap_threshold = overlap_threshold
        self.detector = NativePDFTableDetector(
            angle_threshold=angle_threshold,
            spatial_dist_threshold=spatial_dist_threshold,
            extract_debug_patches=False,
        )

    def run(
        self,
        pdf_path: str,
        stage1: Stage1Result,
        output_dir: str,
        doc: fitz.Document | None = None,
    ) -> Stage2Result:
        import pathlib

        output_path = pathlib.Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        local_doc = doc or fitz.open(pdf_path)
        try:
            odl_tables_by_page: dict[int, list[dict]] = {}
            for table in stage1.tables:
                odl_tables_by_page.setdefault(table.page_number, []).append(
                    {
                        "id": table.table_id,
                        "page_number": table.page_number,
                        "bbox": table.bbox,
                    }
                )

            detection_counter = 1
            detections: list[RotationDetection] = []
            matched_count = 0
            unmatched_count = 0

            for page_idx in range(len(local_doc)):
                page_num = page_idx + 1
                page_result = self.detector.process_page(local_doc[page_idx])
                for rot in page_result.get("rotated_tables", []):
                    loose_obb = rot.get("obb_loose") or rot.get("obb")
                    tight_obb = rot.get("obb_tight")
                    matched_odl, overlap_score, match_mode = _match_detector_table_with_odl_boxes(
                        rot,
                        odl_tables_by_page.get(page_num, []),
                        self.overlap_threshold,
                    )
                    if matched_odl:
                        matched_count += 1
                    else:
                        unmatched_count += 1

                    detections.append(
                        RotationDetection(
                            detection_id=detection_counter,
                            page_number=page_num,
                            angle=float(rot.get("angle", 0.0)),
                            match_score=overlap_score,
                            match_mode=match_mode,
                            matched_table_id=matched_odl.get("id") if matched_odl else None,
                            matched_bbox=matched_odl.get("bbox") if matched_odl else None,
                            detector_obb=loose_obb,
                            detector_obb_tight=tight_obb,
                            detector_aabb=_obb_to_aabb(loose_obb) if loose_obb else None,
                            detector_aabb_tight=_obb_to_aabb(tight_obb) if tight_obb else None,
                        )
                    )
                    detection_counter += 1

            result = Stage2Result(
                angle_threshold=self.angle_threshold,
                overlap_threshold=self.overlap_threshold,
                odl_total_tables=stage1.total_tables,
                detector_rotated_total=len(detections),
                detector_matched_to_odl=matched_count,
                detector_unmatched=unmatched_count,
                detector_rotated_tables=detections,
            )

            save_json(
                {
                    "angle_threshold": result.angle_threshold,
                    "overlap_threshold": result.overlap_threshold,
                    "odl_total_tables": result.odl_total_tables,
                    "detector_rotated_total": result.detector_rotated_total,
                    "detector_matched_to_odl": result.detector_matched_to_odl,
                    "detector_unmatched": result.detector_unmatched,
                    "detector_rotated_tables": [
                        {
                            "detection_id": d.detection_id,
                            "page_number": d.page_number,
                            "angle": d.angle,
                            "match_score": d.match_score,
                            "match_mode": d.match_mode,
                            "matched_table_id": d.matched_table_id,
                            "matched_bbox": d.matched_bbox,
                            "detector_obb": d.detector_obb,
                            "detector_obb_tight": d.detector_obb_tight,
                            "detector_aabb": d.detector_aabb,
                            "detector_aabb_tight": d.detector_aabb_tight,
                        }
                        for d in detections
                    ],
                },
                output_path / "stage2_rotation_check.json",
            )
            return result
        finally:
            if doc is None:
                local_doc.close()

