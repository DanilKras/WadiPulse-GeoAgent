FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_SYSTEM_PYTHON=1 \
    CUDA_VISIBLE_DEVICES=""

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    libexpat1 \
    libgdal-dev \
    libgeos-dev \
    libproj-dev \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
COPY app ./app
COPY scripts ./scripts
COPY tests ./tests
COPY data/few_shot_tiles ./data/few_shot_tiles
COPY data/artifacts/ablation_results.json ./data/artifacts/ablation_results.json
COPY data/artifacts/prithvi_decoder_20shot.pt ./data/artifacts/prithvi_decoder_20shot.pt

RUN uv pip install --no-cache .

EXPOSE 8000 8501

CMD ["streamlit", "run", "app/dashboard.py", "--server.address=0.0.0.0", "--server.port=8501"]