from __future__ import annotations

from .config import PipelineConfig
from .orchestrator import ProductionPipeline


class PipelineFactory:
    @staticmethod
    def build(config: PipelineConfig | None = None) -> ProductionPipeline:
        return ProductionPipeline(config or PipelineConfig())

