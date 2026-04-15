FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    # Avoid oneDNN/PIR conversion crashes in some Paddle builds.
    FLAGS_use_onednn=0 \
    FLAGS_use_mkldnn=0 \
    # Keep model caches in a mounted volume.
    MODELS_ROOT=/models \
    PADDLEOCR_HOME=/models/paddleocr \
    # Skip model hoster connectivity probe (we want local/offline behavior).
    PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True

WORKDIR /app

# System deps for opencv/pymupdf and general runtime.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    curl \
  && rm -rf /var/lib/apt/lists/*

# Install uv (fast, lockfile-aware).
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

# Install Python deps from lock.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen

# Copy application source.
COPY . .

# Preload OCR models into /models (can be persisted via volume).
RUN uv run python scripts/preload_models.py

EXPOSE 7860

# Run Gradio UI.
CMD ["uv", "run", "python", "test.py"]

