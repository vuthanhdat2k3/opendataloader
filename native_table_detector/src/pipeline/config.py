from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PipelineConfig:
    angle_threshold: float = 2.0
    overlap_threshold: float = 0.08
    save_debug_artifacts: bool = True
    ocr_lang: str = "vi"
    ocr_use_gpu: bool = False
    detector_spatial_dist_threshold: float = 100.0
    ocr_workers: int = 1

