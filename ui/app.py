from __future__ import annotations

import os
import shutil
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import gradio as gr

from .common import (
    RUNS_DIR,
    RUNS_DIR_ABS,
    create_run_dir,
    download_to_tempfile,
    normalize_input_files,
    pdf_iframe,
    safe_stem,
    split_lines,
    to_file_url,
)
from .services.gateway_service import call_gateway, coerce_annotated_pdf_bytes
from .services.pipeline_service import run_pdf_pipeline


def _read_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def _is_pdf(path: Path) -> bool:
    return path.suffix.lower() == ".pdf"


def _format_triage(triage_data: Any, summary: Any | None = None) -> str:
    if not triage_data and not summary:
        return ""
    lines: list[str] = []
    if summary:
        lines.append("### Summary")
        if isinstance(summary, dict):
            for k in ("totalPages", "javaPages", "backendPages", "doclingPages"):
                if k in summary:
                    lines.append(f"- **{k}**: {summary.get(k)}")
        else:
            lines.append(f"- {summary}")
        lines.append("")

    if isinstance(triage_data, list) and triage_data:
        java_pages = []
        backend_pages = []
        other_pages = []
        for item in triage_data:
            if not isinstance(item, dict):
                continue
            page = item.get("page")
            decision = str(item.get("decision") or "").upper()
            if decision == "JAVA":
                java_pages.append(page)
            elif decision in ("BACKEND", "DOCLING", "DOCLING_SERVER", "DOC_SERVER"):
                backend_pages.append(page)
            else:
                other_pages.append((page, decision))

        lines.append("### Pages")
        if java_pages:
            lines.append(f"- **JAVA**: {', '.join(map(str, java_pages))}")
        if backend_pages:
            lines.append(f"- **DOCLING/BACKEND**: {', '.join(map(str, backend_pages))}")
        if other_pages:
            lines.append(
                "- **Other**: "
                + ", ".join(f"{p}({d})" for p, d in other_pages if p is not None)
            )
        lines.append("")
        lines.append("### Raw triage")
        import json

        lines.append("```json")
        lines.append(json.dumps(triage_data, ensure_ascii=False, indent=2))
        lines.append("```")
    else:
        # Fallback: dump as JSON
        import json

        lines.append("```json")
        lines.append(json.dumps(triage_data, ensure_ascii=False, indent=2))
        lines.append("```")

    return "\n".join(lines).strip()


def _save_gateway_bundle(run_dir: Path, src_name: str, payload: dict) -> tuple[Path, Path | None]:
    import json

    safe_name = safe_stem(src_name)
    doc_dir = run_dir / safe_name
    doc_dir.mkdir(parents=True, exist_ok=True)

    document = payload.get("document", {})
    (doc_dir / "response_raw.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    triage = payload.get("triage")
    if triage is not None:
        (doc_dir / "triage.json").write_text(
            json.dumps(triage, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    triage_summary = payload.get("summary") or payload.get("triage_summary")
    if triage_summary is not None:
        (doc_dir / "triage_summary.json").write_text(
            json.dumps(triage_summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    json_content = document.get("json_content")
    if json_content is not None:
        if isinstance(json_content, str):
            (doc_dir / f"{safe_name}.json").write_text(json_content, encoding="utf-8")
        else:
            (doc_dir / f"{safe_name}.json").write_text(
                json.dumps(json_content, ensure_ascii=False, indent=2), encoding="utf-8"
            )

    markdown = document.get("markdown", "")
    (doc_dir / f"{safe_name}.md").write_text(markdown or "", encoding="utf-8")

    html = document.get("html", "")
    if html:
        (doc_dir / f"{safe_name}.html").write_text(html or "", encoding="utf-8")

    annotated_pdf_path = None
    annotated_pdf_bytes = coerce_annotated_pdf_bytes(document.get("annotated_pdf"))
    if annotated_pdf_bytes:
        annotated_pdf_path = doc_dir / f"{safe_name}_annotated.pdf"
        annotated_pdf_path.write_bytes(annotated_pdf_bytes)

    return doc_dir, annotated_pdf_path


def _download_url_to_run_dir(url: str, run_dir: Path) -> tuple[str, Path]:
    parsed = urlparse(url)
    name = Path(parsed.path).name or "url_document"
    downloaded = download_to_tempfile(url, run_dir)
    return name, downloaded


def process_documents(
    input_docs: list[object] | None,
    docs_url: str,
    api_url: str,
    page_selection: str,
    page_ranges: str,
    hybrid_mode: str,
    hybrid_url: str,
):
    # Inputs
    local_files = normalize_input_files(input_docs)
    input_urls = split_lines(docs_url)
    if not local_files and not input_urls:
        return (
            None,
            None,
            pdf_iframe(None, "Input preview will appear here."),
            pdf_iframe(None, "Annotated PDF preview will appear here."),
            "",
            "",
            "",
            "",
            "",
            "Please upload documents or provide document URLs.",
        )

    run_dir = create_run_dir("unified_ui")
    work_items: list[tuple[str, Path]] = []
    for f in local_files:
        work_items.append((f.name, f))
    for url in input_urls:
        try:
            name, downloaded = _download_url_to_run_dir(url, run_dir)
            work_items.append((name, downloaded))
        except Exception as exc:
            work_items.append((url, Path()))

    status_rows: list[str] = []
    combined_md_parts: list[str] = []
    combined_html_parts: list[str] = []
    rotated_blocks: list[str] = []
    triage_parts: list[str] = []

    first_input_pdf: Path | None = None
    first_annotated_pdf: Path | None = None

    for src_name, path in work_items:
        if not path or not isinstance(path, Path) or not path.exists():
            status_rows.append(f"[ERROR] {src_name}: download failed or file missing")
            continue

        try:
            if _is_pdf(path):
                if first_input_pdf is None:
                    first_input_pdf = path
                pdf_run_dir = run_dir / safe_stem(src_name)
                pdf_run_dir.mkdir(parents=True, exist_ok=True)

                result, elapsed, selected_pages, processing_pdf = run_pdf_pipeline(
                    input_pdf=path,
                    output_dir=pdf_run_dir,
                    page_selection=page_selection,
                    hybrid_mode=hybrid_mode,
                    hybrid_url=hybrid_url,
                )

                annotated_pdf = Path(result.stage1.annotated_pdf_path)
                merged_md_path = Path(result.stage4.merged_md_path)
                stage1_html_path = (
                    Path(result.stage1.output_dir) / f"{Path(result.stage1.input_pdf).stem}.html"
                )
                md_text = _read_text(merged_md_path)
                html_text = _read_text(stage1_html_path)

                # Stage1 triage files (saved by stage1 gateway)
                triage_path = Path(result.stage1.output_dir) / f"{Path(result.stage1.input_pdf).stem}_triage.json"
                triage_summary_path = (
                    Path(result.stage1.output_dir)
                    / f"{Path(result.stage1.input_pdf).stem}_triage_summary.json"
                )
                triage_text = ""
                if triage_path.exists():
                    import json

                    triage_data = json.loads(triage_path.read_text(encoding="utf-8"))
                    summary_data = (
                        json.loads(triage_summary_path.read_text(encoding="utf-8"))
                        if triage_summary_path.exists()
                        else None
                    )
                    triage_text = _format_triage(triage_data, summary_data)
                if triage_text:
                    triage_parts.append(f"## {src_name}\n\n{triage_text}\n")

                combined_md_parts.append(f"## {src_name}\n\n{md_text.strip()}\n")
                if html_text.strip():
                    combined_html_parts.append(f"<!-- {src_name} -->\n{html_text.strip()}\n")

                if first_annotated_pdf is None and annotated_pdf.exists():
                    first_annotated_pdf = annotated_pdf

                # Rotated tables preview
                for item in result.stage3.hybrid_results:
                    patch_before = Path(item.patch_before) if getattr(item, "patch_before", None) else None
                    patch_deskewed = Path(item.patch_deskewed) if getattr(item, "patch_deskewed", None) else None
                    patch_tight = Path(item.patch_tight) if getattr(item, "patch_tight", None) else None

                    images_html = ""
                    if patch_before and patch_before.exists():
                        images_html += (
                            "<div><div style='font-size:12px;color:#666'>Before Rotate</div>"
                            f"<img src='{to_file_url(patch_before)}' style='max-width:100%;border:1px solid #ddd;border-radius:6px'/></div>"
                        )
                    if patch_deskewed and patch_deskewed.exists():
                        images_html += (
                            "<div><div style='font-size:12px;color:#666'>Rotated</div>"
                            f"<img src='{to_file_url(patch_deskewed)}' style='max-width:100%;border:1px solid #ddd;border-radius:6px'/></div>"
                        )
                    if patch_tight and patch_tight.exists():
                        images_html += (
                            "<div><div style='font-size:12px;color:#666'>Tight (OCR Input)</div>"
                            f"<img src='{to_file_url(patch_tight)}' style='max-width:100%;border:1px solid #ddd;border-radius:6px'/></div>"
                        )

                    rotated_blocks.append(
                        "<div style='margin-bottom:12px;border-bottom:1px solid #eee;padding-bottom:12px'>"
                        f"<div><b>{src_name} / Detect#{item.detection_id}</b> page={item.page_number}, angle={item.angle:.1f}°, conf={item.ocr_confidence:.2f}</div>"
                        f"<pre style='background:#f5f5f5;padding:8px;border-radius:4px;overflow-x:auto;max-height:200px'>{item.ocr_markdown}</pre>"
                        "<div style='display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:8px'>"
                        f"{images_html}</div></div>"
                    )

                status_rows.append(
                    f"[OK][PDF] {src_name}: {elapsed:.2f}s pages={','.join(map(str, selected_pages))} "
                    f"detected={result.stage2.detector_rotated_total} matched={result.stage2.detector_matched_to_odl} "
                    f"replaced={result.stage4.tables_replaced} redacted={result.stage4.kids_redacted}"
                )
            else:
                # Non-PDF: gateway only
                payload, debug_request = call_gateway(
                    api_url=api_url,
                    file_path=path,
                    page_ranges=page_ranges,
                    hybrid="docling-fast",
                    hybrid_mode=hybrid_mode,
                    hybrid_url=hybrid_url,
                )
                doc_dir, annotated_pdf = _save_gateway_bundle(run_dir, src_name, payload)
                doc = payload.get("document", {})
                md = (doc.get("markdown") or "").strip()
                js = doc.get("json_content")
                if js is not None:
                    # Already saved; no need to display raw JSON here.
                    pass
                combined_md_parts.append(f"## {src_name}\n\n{md}\n" if md else f"## {src_name}\n\n")
                # For non-pdf, html may be missing; if present, still show.
                html = (doc.get("html") or "").strip()
                if html:
                    combined_html_parts.append(f"<!-- {src_name} -->\n{html}\n")
                triage_text = _format_triage(payload.get("triage"), payload.get("summary"))
                if triage_text:
                    triage_parts.append(f"## {src_name}\n\n{triage_text}\n")
                if first_annotated_pdf is None and annotated_pdf and annotated_pdf.exists():
                    first_annotated_pdf = annotated_pdf
                status_rows.append(f"[OK][GATEWAY] {src_name}: keys={', '.join(sorted(doc.keys()))} | {debug_request}")
        except Exception as exc:
            status_rows.append(f"[ERROR] {src_name}: {exc}")

    combined_md = "\n\n".join(p for p in combined_md_parts if p.strip())
    combined_html = "\n\n".join(p for p in combined_html_parts if p.strip())
    rotated_html = "\n".join(rotated_blocks) if rotated_blocks else "<div style='padding:8px'>No rotated tables detected.</div>"
    combined_triage = "\n\n".join(p for p in triage_parts if p.strip())

    input_preview = pdf_iframe(first_input_pdf, "Input preview will appear here.") if first_input_pdf else pdf_iframe(None, "Input preview will appear here.")
    annotated_preview = pdf_iframe(first_annotated_pdf, "Annotated PDF preview will appear here.") if first_annotated_pdf else pdf_iframe(None, "Annotated PDF preview will appear here.")

    status = "\n".join(status_rows) + f"\n\nSaved run: {run_dir}"

    return (
        str(first_input_pdf) if first_input_pdf and first_input_pdf.exists() else None,
        str(first_annotated_pdf) if first_annotated_pdf and first_annotated_pdf.exists() else None,
        input_preview,
        annotated_preview,
        combined_md,
        combined_md,
        combined_html,
        rotated_html,
        combined_triage,
        str(run_dir),
        status,
    )


def clear_ui_cache():
    return (
        None,
        "",
        "http://localhost:8000/v1/convert/file",
        "all",
        "",
        "full",
        "http://hybrid-server:5002",
        None,
        None,
        pdf_iframe(None, "Input preview will appear here."),
        pdf_iframe(None, "Annotated PDF preview will appear here."),
        "",
        "",
        "",
        "<div style='padding:8px'>No rotated tables detected.</div>",
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


def build_demo() -> gr.Blocks:
    with gr.Blocks(title="OpenDataLoader Unified UI") as demo:
        gr.Markdown("## OpenDataLoader Unified UI")
        gr.Markdown(
            "Upload documents or provide URLs. PDF files run the native pipeline; other files are sent to the API gateway."
        )

        with gr.Row(equal_height=True):
            with gr.Column(scale=1):
                input_docs = gr.File(
                    label="Input Documents (Multi)",
                    file_count="multiple",
                    file_types=[
                        ".pdf",
                        ".png",
                        ".jpg",
                        ".jpeg",
                        ".webp",
                        ".doc",
                        ".docx",
                        ".xls",
                        ".xlsx",
                        ".ppt",
                        ".pptx",
                    ],
                    type="filepath",
                )
                docs_url = gr.Textbox(
                    label="Document URLs (one per line)",
                    placeholder="https://example.com/file1.pdf\nhttps://example.com/file2.docx",
                    lines=4,
                )
                api_url = gr.Textbox(
                    label="API URL (gateway)",
                    value="http://localhost:8000/v1/convert/file",
                )

                with gr.Accordion("PDF Pipeline Options", open=False):
                    page_selection = gr.Textbox(
                        label="Pages to Process (PDF only)",
                        value="all",
                        placeholder="all or 1,3-5",
                    )
                    hybrid_mode = gr.Radio(
                        choices=["off", "full", "auto"],
                        label="Hybrid Mode (docling-fast)",
                        value="full",
                    )
                    hybrid_url = gr.Textbox(
                        label="Hybrid Server URL",
                        value="http://hybrid-server:5002",
                    )

                with gr.Accordion("Gateway Options (non-PDF)", open=False):
                    page_ranges = gr.Textbox(
                        label="page_ranges (optional)",
                        placeholder="1,3,5-7",
                    )

                with gr.Row():
                    run_btn = gr.Button("Run Processing", variant="primary", size="lg")
                    clear_cache_btn = gr.Button("Clear UI Cache")
                    clear_output_btn = gr.Button("Delete Saved Outputs", variant="stop")
                processing_status = gr.Textbox(label="Processing Status", lines=10)
                saved_run_dir = gr.Textbox(label="Saved run directory", lines=1)
                saved_input_pdf = gr.File(label="First Input PDF (if any)")
                output_pdf = gr.File(label="First Annotated PDF (if any)")

            with gr.Column(scale=2):
                with gr.Tabs():
                    with gr.Tab("Input PDF Preview"):
                        pdf_preview = gr.HTML("<div style='padding:8px'>Input preview will appear here.</div>")
                    with gr.Tab("Annotated PDF Preview"):
                        annotated_pdf_preview = gr.HTML(
                            "<div style='padding:8px'>Annotated PDF preview will appear here.</div>"
                        )
                    with gr.Tab("Markdown"):
                        preview_md = gr.Markdown(label="Preview Markdown")
                        output_md = gr.Textbox(label="Markdown Output", lines=16)
                    with gr.Tab("HTML (if available)"):
                        output_html = gr.Textbox(label="HTML Output", lines=18)
                    with gr.Tab("Rotated Tables (PDF only)"):
                        rotated_tables_preview = gr.HTML(
                            "<div style='padding:8px'>No rotated tables detected.</div>"
                        )
                with gr.Tab("Triage (JAVA vs DOCLING)"):
                    triage_md = gr.Markdown(
                        value="",
                        label="Triage",
                    )

        run_btn.click(
            fn=process_documents,
            inputs=[
                input_docs,
                docs_url,
                api_url,
                page_selection,
                page_ranges,
                hybrid_mode,
                hybrid_url,
            ],
            outputs=[
                saved_input_pdf,
                output_pdf,
                pdf_preview,
                annotated_pdf_preview,
                output_md,
                preview_md,
                output_html,
                rotated_tables_preview,
                triage_md,
                saved_run_dir,
                processing_status,
            ],
        )
        clear_cache_btn.click(
            fn=clear_ui_cache,
            inputs=[],
            outputs=[
                input_docs,
                docs_url,
                api_url,
                page_selection,
                page_ranges,
                hybrid_mode,
                hybrid_url,
                saved_input_pdf,
                output_pdf,
                pdf_preview,
                annotated_pdf_preview,
                output_md,
                preview_md,
                output_html,
                rotated_tables_preview,
                triage_md,
                saved_run_dir,
                processing_status,
            ],
        )
        clear_output_btn.click(
            fn=clear_saved_outputs,
            inputs=[],
            outputs=[processing_status],
        )

    return demo


demo = build_demo()


def main() -> None:
    demo.launch(
        server_name=os.getenv("GRADIO_SERVER_NAME", "127.0.0.1"),
        share=os.getenv("GRADIO_SHARE", "0") == "1",
        allowed_paths=[str(RUNS_DIR_ABS)],
    )


if __name__ == "__main__":
    main()

