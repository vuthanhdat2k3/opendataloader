from __future__ import annotations

import base64
import json
import mimetypes
from pathlib import Path

import requests


def coerce_annotated_pdf_bytes(value: str | bytes | None) -> bytes | None:
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


def call_gateway(
    *,
    api_url: str,
    file_path: Path,
    page_ranges: str,
    hybrid: str,
    hybrid_mode: str,
    hybrid_url: str,
    timeout_sec: float = 180,
) -> tuple[dict, str]:
    data_payload: dict[str, str] = {}
    if page_ranges.strip():
        data_payload["page_ranges"] = page_ranges.strip()
    if hybrid.strip():
        data_payload["hybrid"] = hybrid.strip()
    if hybrid_mode.strip():
        data_payload["hybrid_mode"] = hybrid_mode.strip()
    if hybrid_url.strip():
        data_payload["hybrid_url"] = hybrid_url.strip()

    with file_path.open("rb") as f:
        mime_type = mimetypes.guess_type(file_path.name)[0] or "application/octet-stream"
        resp = requests.post(
            api_url.strip(),
            files={"files": (file_path.name, f, mime_type)},
            data=data_payload,
            timeout=timeout_sec,
        )

    debug_request = (
        f"api={api_url.strip()} | file={file_path.name} | mime={mime_type} | "
        f"data={json.dumps(data_payload, ensure_ascii=False)}"
    )
    if not resp.ok:
        raise RuntimeError(
            f"HTTP {resp.status_code}. request=({debug_request}) response={resp.text}"
        )
    try:
        return resp.json(), debug_request
    except Exception as exc:
        raise RuntimeError(
            f"Invalid JSON response. request=({debug_request}) response={resp.text}"
        ) from exc

