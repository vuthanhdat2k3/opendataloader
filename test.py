from __future__ import annotations

import base64
import json
import os
import shutil
import time
from pathlib import Path
from urllib.parse import quote, urlparse
from urllib.request import urlopen

import gradio as gr
import fitz
import numpy as np
import opendataloader_pdf
from PIL import Image

from native_table_detector.src.core.detector import NativePDFTableDetector
from native_table_detector.src.pipeline.config import PipelineConfig
from native_table_detector.src.pipeline.contracts import PipelineRequest
from native_table_detector.src.pipeline.factory import PipelineFactory


RUNS_DIR = Path("output/gradio_runs")
RUNS_DIR.mkdir(parents=True, exist_ok=True)
RUNS_DIR_ABS = RUNS_DIR.resolve()


def _safe_stem(name: str) -> str:
    stem = Path(name).stem
    safe = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in stem)
    return safe or "document"


def _to_file_url(path: Path) -> str:
    return f"/gradio_api/file={quote(str(path.resolve()))}"


def _pdf_iframe(path: Path | None, empty_text: str) -> str:
    if not path or not path.exists():
        return f"<div style='padding:8px'>{empty_text}</div>"

    pdf_bytes = path.read_bytes()
    pdf_base64 = base64.b64encode(pdf_bytes).decode("ascii")

    return (
        "<embed "
        f'src="data:application/pdf;base64,{pdf_base64}" '
        "type='application/pdf' width='100%' height='640px' "
        "style='border:1px solid #ddd;border-radius:8px;'></embed>"
    )


def _preview_pdf(file_obj: str | os.PathLike | None) -> str:
    path = Path(str(file_obj)) if file_obj else None
    return _pdf_iframe(path, "Please upload a PDF to preview.")


def _read_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def _create_run_dir(name: str) -> tuple[Path, str]:
    safe_stem = _safe_stem(name)
    run_id = time.strftime("%Y%m%d_%H%M%S")
    run_dir = RUNS_DIR / f"{run_id}_{safe_stem}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir.resolve(), safe_stem


def _prepare_upload(file_obj: str | os.PathLike) -> tuple[Path, Path, str]:
    src = Path(str(file_obj))
    run_dir, safe_stem = _create_run_dir(src.name)
    return src.resolve(), run_dir.resolve(), safe_stem


def _download_pdf_from_url(url: str) -> tuple[Path, Path, str]:
    parsed = urlparse(url)
    url_name = Path(parsed.path).name or "url_document.pdf"
    if not url_name.lower().endswith(".pdf"):
        url_name = f"{url_name}.pdf"

    run_dir, safe_stem = _create_run_dir(url_name)
    saved_pdf = run_dir / f"{safe_stem}.pdf"
    with urlopen(url.strip(), timeout=30) as response:
        content = response.read()
    saved_pdf.write_bytes(content)
    return saved_pdf.resolve(), run_dir.resolve(), safe_stem


def _parse_page_selection(page_selection: str, total_pages: int) -> list[int]:
    spec = (page_selection or "all").strip().lower()
    if spec in ("", "all"):
        return list(range(1, total_pages + 1))

    pages: set[int] = set()
    for part in spec.split(","):
        token = part.strip()
        if not token:
            continue
        if "-" in token:
            start_str, end_str = token.split("-", 1)
            start_page = int(start_str)
            end_page = int(end_str)
            if start_page > end_page:
                start_page, end_page = end_page, start_page
            pages.update(range(start_page, end_page + 1))
        else:
            pages.add(int(token))

    valid_pages = sorted(page for page in pages if 1 <= page <= total_pages)
    if not valid_pages:
        raise ValueError(
            f"No valid pages found in '{page_selection}'. Total pages: {total_pages}."
        )
    return valid_pages


def _extract_pages(
    input_pdf: Path, pages_spec: str, run_dir: Path, stem: str
) -> tuple[Path, list[int]]:
    src_doc = fitz.open(str(input_pdf))
    try:
        selected_pages = _parse_page_selection(pages_spec, src_doc.page_count)
        if len(selected_pages) == src_doc.page_count:
            return input_pdf, selected_pages

        selected_pdf = run_dir / f"{stem}_selected_pages.pdf"
        out_doc = fitz.open()
        try:
            for page_number in selected_pages:
                out_doc.insert_pdf(
                    src_doc, from_page=page_number - 1, to_page=page_number - 1
                )
            out_doc.save(str(selected_pdf))
        finally:
            out_doc.close()
        return selected_pdf.resolve(), selected_pages
    finally:
        src_doc.close()


def _run_rotated_table_deskew(
    input_pdf: Path,
    run_dir: Path,
    original_page_numbers: list[int],
    enabled: bool,
) -> tuple[str, str, int, Path]:
    patched_pdf = run_dir / f"{input_pdf.stem}_patched.pdf"
    if not enabled:
        import shutil

        shutil.copy2(input_pdf, patched_pdf)
        return "Rotated-table deskew: off", "", 0, patched_pdf

    detector = NativePDFTableDetector(spatial_dist_threshold=100)
    rotated_dir = run_dir / "rotated_tables"
    rotated_dir.mkdir(parents=True, exist_ok=True)

    doc = fitz.open(str(input_pdf))
    summary_rows: list[str] = []
    html_blocks: list[str] = []
    manifest: list[dict] = []
    total = 0

    try:
        for local_idx in range(len(doc)):
            page = doc[local_idx]
            page_number = (
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
                patch = table.get("patch_image")
                angle = float(table.get("angle", 0.0))
                if not isinstance(patch, np.ndarray) or patch.size == 0:
                    continue

                out_after_name = f"page_{page_number}_table_{t_idx}_angle_{angle:.1f}_after_rotate.png"
                out_after_path = rotated_dir / out_after_name
                Image.fromarray(patch.astype("uint8"), mode="RGB").save(out_after_path)

                out_before_path = None
                if isinstance(patch_before, np.ndarray) and patch_before.size > 0:
                    out_before_name = f"page_{page_number}_table_{t_idx}_angle_{angle:.1f}_before_rotate.png"
                    out_before_path = rotated_dir / out_before_name
                    Image.fromarray(patch_before.astype("uint8"), mode="RGB").save(
                        out_before_path
                    )

                # >>> PATHING THE PDF <<<
                # Tính toạ độ các góc gốc của bảng để xoá (Redact)
                cx, cy, tw, th = (
                    table["obb"]["cx"],
                    table["obb"]["cy"],
                    table["obb"]["w"],
                    table["obb"]["h"],
                )
                import math

                a_rad = math.radians(-angle)
                cos_a, sin_a = math.cos(a_rad), math.sin(a_rad)

                # 4 góc xoay của cái bảng nghiêng ban đầu
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

                # Bôi sạch nội dung bảng rác
                page.add_redact_annot(quad, fill=(1, 1, 1))
                page.apply_redactions()

                # Chèn ảnh bảng thẳng tắp vào
                upright_rect = fitz.Rect(
                    cx - tw / 2, cy - th / 2, cx + tw / 2, cy + th / 2
                )
                page.insert_image(upright_rect, filename=str(out_after_path))

                # Mở rộng bounds trang nếu cái bảng thẳng nó phình ra ngoài (nằm ngang)
                new_box = page.rect | upright_rect
                page.set_mediabox(new_box)
                page.set_cropbox(new_box)

                manifest.append(
                    {
                        "page": page_number,
                        "table_index": t_idx,
                        "angle": angle,
                        "image_before_rotate": str(out_before_path)
                        if out_before_path
                        else None,
                        "image_after_rotate": str(out_after_path),
                    }
                )
                summary_rows.append(
                    f"Page {page_number} | Table {t_idx} | angle={angle:.1f}°"
                )
                before_html = (
                    (
                        "<div><div style='font-size:12px;color:#666;margin-bottom:4px'>Before rotate</div>"
                        f"<img src='{_to_file_url(out_before_path)}' style='max-width:100%;border:1px solid #ddd;border-radius:6px'/></div>"
                    )
                    if out_before_path
                    else ""
                )
                after_html = (
                    "<div><div style='font-size:12px;color:#666;margin-bottom:4px'>After rotate</div>"
                    f"<img src='{_to_file_url(out_after_path)}' style='max-width:100%;border:1px solid #ddd;border-radius:6px'/></div>"
                )
                html_blocks.append(
                    "<div style='margin-bottom:12px'>"
                    f"<div><b>Page {page_number} - Table {t_idx}</b> | angle={angle:.1f}°</div>"
                    "<div style='display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:8px'>"
                    f"{before_html}{after_html}</div></div>"
                )
                total += 1
        doc.save(str(patched_pdf))
    finally:
        doc.close()

    manifest_path = rotated_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    if total == 0:
        import shutil

        shutil.copy2(input_pdf, patched_pdf)
        return (
            "Rotated-table deskew: enabled, no rotated table detected",
            "",
            0,
            patched_pdf,
        )

    summary = (
        f"Rotated-table deskew: detected {total} table(s). "
        f"Saved to {rotated_dir}.\n" + "\n".join(summary_rows)
    )
    return summary, "\n".join(html_blocks), total, patched_pdf


def _run_hybrid_rotated_table_ocr(
    input_pdf: Path,
    run_dir: Path,
    original_page_numbers: list[int],
    enabled: bool,
) -> tuple[str, str, int, dict]:
    if not enabled:
        return "Hybrid OCR: off", "", 0, {}

    rotated_dir = run_dir / "rotated_tables_hybrid"
    rotated_dir.mkdir(parents=True, exist_ok=True)

    processor = HybridTableProcessor(spatial_dist_threshold=100)

    doc = fitz.open(str(input_pdf))
    summary_rows: list[str] = []
    html_blocks: list[str] = []
    total = 0
    all_results = {"pages": []}

    try:
        for local_idx in range(len(doc)):
            page = doc[local_idx]
            page_number = (
                original_page_numbers[local_idx]
                if local_idx < len(original_page_numbers)
                else local_idx + 1
            )

            page_result = processor.process_page(page)
            page_result["page_number"] = page_number
            all_results["pages"].append(page_result)

            rotated_tables = page_result.get("rotated_tables", [])
            if not rotated_tables:
                continue

            for t_idx, table in enumerate(rotated_tables, start=1):
                patch = table.get("patch_image")
                angle = float(table.get("angle", 0.0))
                ocr_md = table.get("ocr_markdown", "")
                ocr_conf = float(table.get("ocr_confidence", 0.0))

                if not isinstance(patch, np.ndarray) or patch.size == 0:
                    continue

                out_after_name = (
                    f"page_{page_number}_table_{t_idx}_angle_{angle:.1f}_ocr.png"
                )
                out_after_path = rotated_dir / out_after_name
                Image.fromarray(patch.astype("uint8"), mode="RGB").save(out_after_path)

                summary_rows.append(
                    f"Page {page_number} | Table {t_idx} | angle={angle:.1f}° | conf={ocr_conf:.2f}"
                )

                before_patch = table.get("patch_before_rotate")
                before_html = ""
                if isinstance(before_patch, np.ndarray) and before_patch.size > 0:
                    out_before_name = (
                        f"page_{page_number}_table_{t_idx}_before_rotate.png"
                    )
                    out_before_path = rotated_dir / out_before_name
                    Image.fromarray(before_patch.astype("uint8"), mode="RGB").save(
                        out_before_path
                    )
                    before_html = (
                        "<div><div style='font-size:12px;color:#666;margin-bottom:4px'>Before rotate</div>"
                        f"<img src='{_to_file_url(out_before_path)}' style='max-width:100%;border:1px solid #ddd;border-radius:6px'/></div>"
                    )

                after_html = (
                    "<div><div style='font-size:12px;color:#666;margin-bottom:4px'>After OCR</div>"
                    f"<img src='{_to_file_url(out_after_path)}' style='max-width:100%;border:1px solid #ddd;border-radius:6px'/></div>"
                )

                table_html = (
                    "<div style='margin-bottom:12px'>"
                    f"<div><b>Page {page_number} - Table {t_idx}</b> | angle={angle:.1f}° | conf={ocr_conf:.2f}</div>"
                    f"<pre style='background:#f5f5f5;padding:8px;border-radius:4px;overflow-x:auto'>{ocr_md}</pre>"
                    "<div style='display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:8px;margin-top:8px'>"
                    f"{before_html}{after_html}</div></div>"
                )
                html_blocks.append(table_html)
                total += 1

    finally:
        doc.close()

    export_results_to_json(all_results, rotated_dir / "hybrid_results.json")

    markdown_output = generate_markdown_from_hybrid_results(all_results)
    markdown_path = rotated_dir / "output.md"
    markdown_path.write_text(markdown_output, encoding="utf-8")

    if total == 0:
        return "Hybrid OCR: enabled, no rotated table detected", "", 0, all_results

    summary = (
        f"Hybrid OCR: detected {total} table(s). Saved to {rotated_dir}.\n"
        + "\n".join(summary_rows)
    )
    return summary, "\n".join(html_blocks), total, all_results


def process_pdf(
    file_obj: str | os.PathLike | None,
    pdf_url: str,
    page_selection: str,
    use_hybrid_docling_fast: bool,
    hybrid_url: str,
    enable_rotated_table_deskew: bool,
    enable_hybrid_ocr: bool,
):
    file_obj = str(file_obj) if file_obj else None
    pdf_url = (pdf_url or "").strip()
    hybrid_url = (hybrid_url or "").strip()

    if not file_obj and not pdf_url:
        return (
            None,
            None,
            _pdf_iframe(None, "Annotated PDF preview will appear here."),
            "",
            "",
            "",
            "",
            "Please upload a PDF or provide a PDF URL.",
        )

    try:
        if file_obj:
            input_pdf = Path(file_obj)
            if input_pdf.suffix.lower() != ".pdf":
                return (
                    None,
                    None,
                    _pdf_iframe(None, "Annotated PDF preview will appear here."),
                    "",
                    "",
                    "",
                    "",
                    "Only PDF files are supported.",
                )
            saved_input_pdf, run_dir, stem = _prepare_upload(file_obj)
            source_name = "upload"
        else:
            saved_input_pdf, run_dir, stem = _download_pdf_from_url(pdf_url)
            source_name = "url"

        processing_input_pdf, selected_pages = _extract_pages(
            saved_input_pdf,
            page_selection,
            run_dir,
            stem,
        )
    except Exception as exc:
        return (
            None,
            None,
            _pdf_iframe(None, "Annotated PDF preview will appear here."),
            "",
            "",
            "",
            "",
            f"Input preparation failed: {exc}",
        )

    try:
        start = time.perf_counter()
        pipeline = PipelineFactory.build(
            PipelineConfig(
                angle_threshold=2.0,
                overlap_threshold=0.08,
                save_debug_artifacts=True,
            )
        )
        pipeline_result = pipeline.run(
            PipelineRequest(
                pdf_path=processing_input_pdf,
                output_dir=run_dir,
                angle_threshold=2.0,
                overlap_threshold=0.08,
                save_debug_artifacts=True,
            )
        )
        elapsed = time.perf_counter() - start
    except Exception as exc:
        return (
            str(saved_input_pdf),
            None,
            _pdf_iframe(None, "Annotated PDF preview will appear here."),
            "",
            "",
            "",
            "",
            f"Pipeline failed: {exc}",
        )

    annotated_pdf = Path(pipeline_result.stage1.annotated_pdf_path)
    merged_md_path = Path(pipeline_result.stage4.merged_md_path)
    stage1_html_path = (
        Path(pipeline_result.stage1.output_dir)
        / f"{Path(pipeline_result.stage1.input_pdf).stem}.html"
    )
    md_text = _read_text(merged_md_path)
    html_text = _read_text(stage1_html_path)
    annotated_preview = _pdf_iframe(annotated_pdf, "Annotated PDF was not generated.")

    rotated_blocks = []
    for item in pipeline_result.stage3.hybrid_results:
        patch = Path(item.patch_deskewed)
        if patch.exists():
            rotated_blocks.append(
                "<div style='margin-bottom:8px'>"
                f"<div><b>Detect#{item.detection_id}</b> page={item.page_number}, "
                f"angle={item.angle:.1f}°, conf={item.ocr_confidence:.2f}</div>"
                f"<img src='{_to_file_url(patch)}' style='max-width:100%;border:1px solid #ddd;border-radius:6px'/>"
                "</div>"
            )
    rotated_tables_combined_html = "\n".join(rotated_blocks)

    status = (
        f"Processed in {elapsed:.2f} seconds. Source={source_name}. "
        f"Pages={','.join(map(str, selected_pages))}. "
        f"Saved run: {run_dir}. "
        f"detected={pipeline_result.stage2.detector_rotated_total}, "
        f"matched={pipeline_result.stage2.detector_matched_to_odl}, "
        f"replaced={pipeline_result.stage4.tables_replaced}. "
        f"Metrics={pipeline_result.metrics}"
    )

    return (
        str(saved_input_pdf),
        str(annotated_pdf) if annotated_pdf.exists() else None,
        annotated_preview,
        md_text,
        md_text,
        html_text,
        rotated_tables_combined_html,
        status,
    )


def clear_ui_cache():
    return (
        None,
        "",
        "all",
        True,
        "http://0.0.0.0:5004",
        True,
        False,
        None,
        None,
        _pdf_iframe(None, "Please upload a PDF to preview."),
        _pdf_iframe(None, "Annotated PDF preview will appear here."),
        "",
        "",
        "",
        "",
        "UI cache cleared.",
    )


def clear_saved_outputs():
    removed = 0
    if RUNS_DIR.exists():
        for child in RUNS_DIR.iterdir():
            if child.is_dir():
                shutil.rmtree(child, ignore_errors=True)
            else:
                child.unlink(missing_ok=True)
            removed += 1

    return f"Deleted {removed} saved run(s) in {RUNS_DIR_ABS}."


with gr.Blocks(title="OpenDataLoader PDF Tester") as demo:
    gr.Markdown("## OpenDataLoader PDF Test UI")
    gr.Markdown(
        "Upload PDF, run pipeline, then review annotated PDF + Markdown + HTML output."
    )

    with gr.Row(equal_height=True):
        with gr.Column(scale=1):
            input_pdf = gr.File(label="Input PDF", file_types=[".pdf"], type="filepath")
            pdf_url = gr.Textbox(
                label="Or PDF URL",
                placeholder="https://example.com/file.pdf",
            )
            page_selection = gr.Textbox(
                label="Pages to Process",
                value="all",
                placeholder="all or 1,3-5",
            )
            use_hybrid_docling_fast = gr.Checkbox(
                label="Use Hybrid docling-fast (hybrid_mode=full)",
                value=True,
            )
            hybrid_url = gr.Textbox(
                label="Hybrid Server URL",
                value="http://0.0.0.0:5004",
                placeholder="http://0.0.0.0:5004",
            )
            enable_rotated_table_deskew = gr.Checkbox(
                label="Detect & Deskew Rotated Tables (native)",
                value=True,
            )
            enable_hybrid_ocr = gr.Checkbox(
                label="Hybrid OCR for Rotated Tables (Tesseract)",
                value=False,
            )
            with gr.Row():
                run_btn = gr.Button("Run Processing", variant="primary", size="lg")
                clear_cache_btn = gr.Button("Clear UI Cache")
                clear_output_btn = gr.Button("Delete Saved Outputs", variant="stop")
            processing_time = gr.Textbox(label="Processing Status", lines=2)
            saved_input = gr.File(label="Saved Input PDF")
            output_pdf = gr.File(label="Output Annotated PDF")

        with gr.Column(scale=2):
            with gr.Tabs():
                with gr.Tab("Input PDF Preview"):
                    pdf_preview = gr.HTML(
                        "<div style='padding:8px'>Please upload a PDF to preview.</div>"
                    )
                with gr.Tab("Annotated PDF Preview"):
                    annotated_pdf_preview = gr.HTML(
                        "<div style='padding:8px'>Annotated PDF preview will appear here.</div>"
                    )
                with gr.Tab("Markdown"):
                    preview_md = gr.Markdown(label="Preview Markdown")
                    output_md = gr.Textbox(label="Markdown Output", lines=14)
                with gr.Tab("HTML"):
                    output_html = gr.Textbox(label="HTML Output", lines=18)
                with gr.Tab("Rotated Tables"):
                    rotated_tables_preview = gr.HTML(
                        "<div style='padding:8px'>Rotated table patches will appear here.</div>"
                    )

    input_pdf.change(fn=_preview_pdf, inputs=input_pdf, outputs=pdf_preview)
    run_btn.click(
        fn=process_pdf,
        inputs=[
            input_pdf,
            pdf_url,
            page_selection,
            use_hybrid_docling_fast,
            hybrid_url,
            enable_rotated_table_deskew,
            enable_hybrid_ocr,
        ],
        outputs=[
            saved_input,
            output_pdf,
            annotated_pdf_preview,
            output_md,
            preview_md,
            output_html,
            rotated_tables_preview,
            processing_time,
        ],
    )
    clear_cache_btn.click(
        fn=clear_ui_cache,
        inputs=[],
        outputs=[
            input_pdf,
            pdf_url,
            page_selection,
            use_hybrid_docling_fast,
            hybrid_url,
            enable_rotated_table_deskew,
            enable_hybrid_ocr,
            saved_input,
            output_pdf,
            pdf_preview,
            annotated_pdf_preview,
            output_md,
            preview_md,
            output_html,
            rotated_tables_preview,
            processing_time,
        ],
    )
    clear_output_btn.click(
        fn=clear_saved_outputs,
        inputs=[],
        outputs=[processing_time],
    )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", allowed_paths=[str(RUNS_DIR_ABS)])
