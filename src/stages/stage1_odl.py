from __future__ import annotations

import base64
import mimetypes
import os
import time
from pathlib import Path
import json

import requests

from ..pipeline.contracts import Stage1Result, Stage1Table
from ..utils.io import read_json, save_json


class Stage1ODLExtractor:
    @staticmethod
    def _decode_annotated_pdf(value: str | bytes | None) -> bytes | None:
        if value is None:
            return None
        if isinstance(value, bytes):
            return value
        raw = value.strip()
        if not raw:
            return None
        if raw.startswith("data:application/pdf;base64,"):
            raw = raw.split(",", 1)[1]
        try:
            return base64.b64decode(raw, validate=True)
        except Exception:
            return raw.encode("utf-8", errors="ignore")

    @staticmethod
    def _convert_via_gateway(
        pdf_path: Path,
        output_dir: Path,
        hybrid_mode: str,
        hybrid_url: str,
    ) -> None:
        api_url = os.getenv("ODL_API_URL", "http://localhost:8000/v1/convert/file").strip()
        timeout_sec = float(os.getenv("ODL_API_TIMEOUT", "300"))
        data_payload: dict[str, str] = {}
        if hybrid_mode in ("full", "auto"):
            data_payload["hybrid"] = "docling-fast"
            data_payload["hybrid_mode"] = hybrid_mode
            if hybrid_url:
                data_payload["hybrid_url"] = hybrid_url

        mime_type = mimetypes.guess_type(pdf_path.name)[0] or "application/pdf"
        with pdf_path.open("rb") as f:
            response = requests.post(
                api_url,
                files={"files": (pdf_path.name, f, mime_type)},
                data=data_payload,
                timeout=timeout_sec,
            )
        if not response.ok:
            raise RuntimeError(
                f"Stage1 gateway failed with HTTP {response.status_code}: {response.text}"
            )
        payload = response.json()
        document = payload.get("document", {})
        triage = payload.get("triage")
        if triage is not None:
            (output_dir / f"{pdf_path.stem}_triage.json").write_text(
                json.dumps(triage, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        triage_summary = payload.get("summary") or payload.get("triage_summary")
        if triage_summary is not None:
            (output_dir / f"{pdf_path.stem}_triage_summary.json").write_text(
                json.dumps(triage_summary, ensure_ascii=False, indent=2), encoding="utf-8"
            )

        json_content = document.get("json_content")
        json_path = output_dir / f"{pdf_path.stem}.json"
        if isinstance(json_content, str):
            json_path.write_text(json_content, encoding="utf-8")
        else:
            json_path.write_text(
                json.dumps(json_content or {}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

        markdown = document.get("markdown", "")
        (output_dir / f"{pdf_path.stem}.md").write_text(markdown or "", encoding="utf-8")

        html = document.get("html", "")
        (output_dir / f"{pdf_path.stem}.html").write_text(html or "", encoding="utf-8")

        annotated_pdf_bytes = Stage1ODLExtractor._decode_annotated_pdf(
            document.get("annotated_pdf")
        )
        if annotated_pdf_bytes:
            (output_dir / f"{pdf_path.stem}_annotated.pdf").write_bytes(annotated_pdf_bytes)

    def run(
        self,
        pdf_path: Path,
        output_dir: Path,
        hybrid_mode: str = "off",
        hybrid_url: str = "",
    ) -> Stage1Result:
        output_dir.mkdir(parents=True, exist_ok=True)

        start = time.perf_counter()
        self._convert_via_gateway(
            pdf_path=pdf_path,
            output_dir=output_dir,
            hybrid_mode=hybrid_mode,
            hybrid_url=hybrid_url,
        )
        elapsed = time.perf_counter() - start

        json_path = output_dir / f"{pdf_path.stem}.json"
        markdown_path = output_dir / f"{pdf_path.stem}.md"
        annotated_pdf = output_dir / f"{pdf_path.stem}_annotated.pdf"
        fallback_annotated = output_dir / "annotated.pdf"
        if fallback_annotated.exists():
            annotated_pdf = fallback_annotated

        odl_data = read_json(json_path)
        tables: list[Stage1Table] = []
        for kid in odl_data.get("kids", []):
            if kid.get("type") != "table":
                continue
            tables.append(
                Stage1Table(
                    table_id=kid.get("id"),
                    page_number=int(kid.get("page number")),
                    bbox=list(kid.get("bounding box", [])),
                    num_rows=int(kid.get("number of rows", 0)),
                    num_cols=int(kid.get("number of columns", 0)),
                    content=list(kid.get("rows", [])),
                )
            )

        result = Stage1Result(
            input_pdf=str(pdf_path),
            output_dir=str(output_dir),
            json_path=str(json_path),
            markdown_path=str(markdown_path),
            annotated_pdf_path=str(annotated_pdf),
            elapsed_sec=round(elapsed, 2),
            total_tables=len(tables),
            tables=tables,
        )

        save_json(
            {
                "input_pdf": result.input_pdf,
                "output_dir": result.output_dir,
                "json_path": result.json_path,
                "markdown_path": result.markdown_path,
                "annotated_pdf_path": result.annotated_pdf_path,
                "elapsed_sec": result.elapsed_sec,
                "total_tables": result.total_tables,
                "tables": [
                    {
                        "id": t.table_id,
                        "page_number": t.page_number,
                        "bbox": t.bbox,
                        "num_rows": t.num_rows,
                        "num_cols": t.num_cols,
                        "content": t.content,
                    }
                    for t in tables
                ],
            },
            output_dir / "stage1_opendataloader.json",
        )
        return result

