from __future__ import annotations

import os
from pathlib import Path


def _env(key: str, default: str) -> str:
    v = (os.getenv(key) or "").strip()
    return v if v else default


def main() -> None:
    """
    Pre-download model assets into a shared directory so runtime can be offline.

    This script is intentionally conservative: it only initializes PaddleOCR once.
    """
    # Avoid oneDNN/PIR conversion path that can crash on some builds.
    os.environ.setdefault("FLAGS_use_onednn", "0")
    os.environ.setdefault("FLAGS_use_mkldnn", "0")

    models_root = Path(_env("MODELS_ROOT", "/models")).resolve()
    paddleocr_home = Path(_env("PADDLEOCR_HOME", str(models_root / "paddleocr"))).resolve()
    paddleocr_home.mkdir(parents=True, exist_ok=True)

    # PaddleOCR respects this env in most versions; also helps keep caches contained.
    os.environ.setdefault("PADDLEOCR_HOME", str(paddleocr_home))

    # Reduce noisy/slow online hoster checks; we want local cache behavior.
    os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

    lang = _env("PADDLEOCR_LANG", "vi")
    use_gpu = _env("PADDLEOCR_USE_GPU", "False").lower() in {"1", "true", "yes"}
    device = "gpu" if use_gpu else "cpu"

    print(f"[preload] MODELS_ROOT={models_root}")
    print(f"[preload] PADDLEOCR_HOME={paddleocr_home}")
    print(f"[preload] Initializing PaddleOCR(lang={lang}, device={device}) to download assets...")

    # Import here so env vars above take effect.
    from paddleocr import PaddleOCR

    _ = PaddleOCR(
        lang=lang,
        use_textline_orientation=True,
        device=device,
        enable_mkldnn=False,
    )
    print("[preload] PaddleOCR assets ready.")


if __name__ == "__main__":
    main()

