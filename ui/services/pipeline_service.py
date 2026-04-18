from __future__ import annotations

import time
from pathlib import Path

from native_table_detector.src.pipeline.config import PipelineConfig
from native_table_detector.src.pipeline.contracts import PipelineRequest, PipelineResult
from native_table_detector.src.pipeline.factory import PipelineFactory


_GLOBAL_PIPELINE = None


def get_pipeline():
    global _GLOBAL_PIPELINE
    if _GLOBAL_PIPELINE is None:
        _GLOBAL_PIPELINE = PipelineFactory.build(
            PipelineConfig(
                angle_threshold=2.0,
                overlap_threshold=0.08,
                save_debug_artifacts=True,
                ocr_use_gpu=True,
            )
        )
    return _GLOBAL_PIPELINE


def run_pdf_pipeline(
    *,
    input_pdf: Path,
    output_dir: Path,
    page_selection: str,
    hybrid_mode: str,
    hybrid_url: str,
) -> tuple[PipelineResult, float, list[int], Path]:
    """
    Runs the native pipeline on a PDF and returns:
    - PipelineResult
    - elapsed seconds
    - selected page numbers
    - the PDF path actually processed (may be a subset PDF)
    """
    import fitz

    def parse_page_selection(spec: str, total_pages: int) -> list[int]:
        spec = (spec or "all").strip().lower()
        if spec in ("", "all"):
            return list(range(1, total_pages + 1))
        pages: set[int] = set()
        for part in spec.split(","):
            token = part.strip()
            if not token:
                continue
            if "-" in token:
                a, b = token.split("-", 1)
                start, end = int(a), int(b)
                if start > end:
                    start, end = end, start
                pages.update(range(start, end + 1))
            else:
                pages.add(int(token))
        valid = sorted(p for p in pages if 1 <= p <= total_pages)
        if not valid:
            raise ValueError(f"No valid pages found in '{spec}'. Total pages: {total_pages}.")
        return valid

    def extract_pages(src_pdf: Path, pages_spec: str, out_dir: Path, stem: str) -> tuple[Path, list[int]]:
        src_doc = fitz.open(str(src_pdf))
        try:
            selected = parse_page_selection(pages_spec, src_doc.page_count)
            if len(selected) == src_doc.page_count:
                return src_pdf, selected
            selected_pdf = out_dir / f"{stem}_selected_pages.pdf"
            out_doc = fitz.open()
            try:
                for p in selected:
                    out_doc.insert_pdf(src_doc, from_page=p - 1, to_page=p - 1)
                out_doc.save(str(selected_pdf))
            finally:
                out_doc.close()
            return selected_pdf.resolve(), selected
        finally:
            src_doc.close()

    stem = input_pdf.stem
    processing_pdf, selected_pages = extract_pages(input_pdf, page_selection, output_dir, stem)

    start = time.perf_counter()
    pipeline = get_pipeline()
    result = pipeline.run(
        PipelineRequest(
            pdf_path=processing_pdf,
            output_dir=output_dir,
            angle_threshold=2.0,
            overlap_threshold=0.08,
            save_debug_artifacts=True,
            hybrid_mode=hybrid_mode,
            hybrid_url=hybrid_url,
        )
    )
    elapsed = time.perf_counter() - start
    return result, elapsed, selected_pages, processing_pdf

