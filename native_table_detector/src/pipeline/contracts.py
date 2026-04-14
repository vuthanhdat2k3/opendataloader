from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class PipelineRequest:
    pdf_path: Path
    output_dir: Path
    angle_threshold: float = 2.0
    overlap_threshold: float = 0.08
    save_debug_artifacts: bool = True


@dataclass
class Stage1Table:
    table_id: Any
    page_number: int
    bbox: list[float]
    num_rows: int
    num_cols: int
    content: list[dict]


@dataclass
class Stage1Result:
    input_pdf: str
    output_dir: str
    json_path: str
    markdown_path: str
    annotated_pdf_path: str
    elapsed_sec: float
    total_tables: int
    tables: list[Stage1Table] = field(default_factory=list)


@dataclass
class RotationDetection:
    detection_id: int
    page_number: int
    angle: float
    match_score: float
    match_mode: str
    matched_table_id: Any | None
    matched_bbox: list[float] | None
    detector_obb: dict[str, float] | None
    detector_obb_tight: dict[str, float] | None


@dataclass
class Stage2Result:
    angle_threshold: float
    overlap_threshold: float
    odl_total_tables: int
    detector_rotated_total: int
    detector_matched_to_odl: int
    detector_unmatched: int
    detector_rotated_tables: list[RotationDetection] = field(default_factory=list)


@dataclass
class OCRResult:
    detection_id: int
    matched_table_id: Any | None
    page_number: int
    bbox: list[float] | None
    crop_mode: str
    detector_obb_tight: dict[str, float] | None
    detector_obb: dict[str, float] | None
    angle: float
    ocr_markdown: str
    ocr_confidence: float
    patch_before: str
    patch_deskewed: str


@dataclass
class Stage3Result:
    total_rotated: int
    hybrid_results: list[OCRResult] = field(default_factory=list)


@dataclass
class Stage4Result:
    merged_json_path: str
    merged_md_path: str
    tables_replaced: int


@dataclass
class PipelineResult:
    stage1: Stage1Result
    stage2: Stage2Result
    stage3: Stage3Result
    stage4: Stage4Result
    metrics: dict[str, float] = field(default_factory=dict)

