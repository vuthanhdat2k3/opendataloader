from __future__ import annotations

import time
from pathlib import Path

import fitz

from ..stages.stage1_odl import Stage1ODLExtractor
from ..stages.stage2_rotation import Stage2RotationDetector
from ..stages.stage3_ocr import Stage3HybridOCR
from ..stages.stage4_merge import Stage4Merger
from .config import PipelineConfig
from .contracts import PipelineRequest, PipelineResult


class ProductionPipeline:
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.stage1 = Stage1ODLExtractor()
        self.stage2 = Stage2RotationDetector(
            angle_threshold=config.angle_threshold,
            overlap_threshold=config.overlap_threshold,
            spatial_dist_threshold=config.detector_spatial_dist_threshold,
        )
        self.stage3 = Stage3HybridOCR(
            angle_threshold=config.angle_threshold,
            save_debug_artifacts=config.save_debug_artifacts,
            ocr_lang=config.ocr_lang,
            ocr_use_gpu=config.ocr_use_gpu,
            spatial_dist_threshold=config.detector_spatial_dist_threshold,
        )
        self.stage4 = Stage4Merger()

    def run(self, request: PipelineRequest) -> PipelineResult:
        request.output_dir.mkdir(parents=True, exist_ok=True)
        metrics: dict[str, float] = {}

        t0 = time.perf_counter()
        stage1 = self.stage1.run(
            pdf_path=request.pdf_path,
            output_dir=request.output_dir / "stage1_opendataloader",
        )
        metrics["stage1_sec"] = round(time.perf_counter() - t0, 3)

        t1 = time.perf_counter()
        shared_doc = fitz.open(str(request.pdf_path))
        try:
            stage2 = self.stage2.run(
                pdf_path=str(request.pdf_path),
                stage1=stage1,
                output_dir=str(request.output_dir / "stage2_rotation_check"),
                doc=shared_doc,
            )
            metrics["stage2_sec"] = round(time.perf_counter() - t1, 3)

            t2 = time.perf_counter()
            stage3 = self.stage3.run(
                pdf_path=str(request.pdf_path),
                stage2=stage2,
                output_dir=str(request.output_dir / "stage3_hybrid_ocr"),
                doc=shared_doc,
            )
            metrics["stage3_sec"] = round(time.perf_counter() - t2, 3)
        finally:
            shared_doc.close()

        t3 = time.perf_counter()
        stage4 = self.stage4.run(
            stage1=stage1,
            stage3=stage3,
            output_dir=str(request.output_dir / "stage4_merge"),
        )
        metrics["stage4_sec"] = round(time.perf_counter() - t3, 3)
        metrics["total_sec"] = round(sum(metrics.values()), 3)

        return PipelineResult(
            stage1=stage1,
            stage2=stage2,
            stage3=stage3,
            stage4=stage4,
            metrics=metrics,
        )


def run_pipeline(
    pdf_path: Path,
    output_dir: Path,
    angle_threshold: float = 2.0,
    overlap_threshold: float = 0.08,
    save_debug_artifacts: bool = True,
) -> PipelineResult:
    config = PipelineConfig(
        angle_threshold=angle_threshold,
        overlap_threshold=overlap_threshold,
        save_debug_artifacts=save_debug_artifacts,
    )
    pipeline = ProductionPipeline(config)
    return pipeline.run(PipelineRequest(pdf_path=pdf_path, output_dir=output_dir))

