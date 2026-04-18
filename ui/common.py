from __future__ import annotations

import base64
import time
from pathlib import Path
from typing import Iterable
from urllib.parse import quote
from urllib.request import urlopen


RUNS_DIR = Path("output/gradio_runs")
RUNS_DIR.mkdir(parents=True, exist_ok=True)
RUNS_DIR_ABS = RUNS_DIR.resolve()


def safe_stem(name: str) -> str:
    stem = Path(name).stem
    safe = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in stem)
    return safe or "document"


def to_file_url(path: Path) -> str:
    return f"/file={quote(str(path.resolve()))}"


def pdf_iframe(path: Path | None, empty_text: str) -> str:
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


def split_lines(text: str) -> list[str]:
    return [line.strip() for line in (text or "").splitlines() if line.strip()]


def normalize_input_files(file_objs: list[object] | None) -> list[Path]:
    if not file_objs:
        return []
    normalized: list[Path] = []
    for file_obj in file_objs:
        file_path: str | None = None
        if isinstance(file_obj, str):
            file_path = file_obj
        elif isinstance(file_obj, dict):
            file_path = file_obj.get("path")
        else:
            file_path = getattr(file_obj, "path", None) or getattr(file_obj, "name", None)
        if file_path:
            normalized.append(Path(file_path).resolve())
    return normalized


def download_to_tempfile(url: str, run_dir: Path) -> Path:
    parsed_name = Path(url.split("?")[0].rstrip("/")).name or "url_document"
    with urlopen(url, timeout=30) as response:
        content = response.read()
    suffix = Path(parsed_name).suffix or ".bin"
    tmp_path = run_dir / f"url_{int(time.time() * 1000)}{suffix}"
    tmp_path.write_bytes(content)
    return tmp_path.resolve()


def create_run_dir(prefix: str) -> Path:
    run_id = time.strftime("%Y%m%d_%H%M%S")
    run_dir = RUNS_DIR / f"{run_id}_{prefix}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir.resolve()


def first_existing(paths: Iterable[Path | None]) -> Path | None:
    for p in paths:
        if p and p.exists():
            return p
    return None

