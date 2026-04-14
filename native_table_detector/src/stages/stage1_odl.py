from __future__ import annotations

import time
from pathlib import Path

import opendataloader_pdf

from ..pipeline.contracts import Stage1Result, Stage1Table
from ..utils.io import read_json, save_json


class Stage1ODLExtractor:
    def run(self, pdf_path: Path, output_dir: Path) -> Stage1Result:
        output_dir.mkdir(parents=True, exist_ok=True)

        start = time.perf_counter()
        opendataloader_pdf.convert(
            input_path=str(pdf_path),
            output_dir=str(output_dir),
            format="json,html,markdown,pdf",
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

