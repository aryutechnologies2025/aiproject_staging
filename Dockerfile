# ---------- STAGE 1: BUILDER ----------
FROM python:3.10-slim AS builder

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY req.txt .

# Enabled Pip Caching with BuildKit
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip setuptools wheel \
    && pip install --prefix=/install -r req.txt


# ---------- STAGE 2: FINAL ----------
FROM python:3.10-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /install /usr/local

# Copy code
COPY app ./app
COPY alembic ./alembic
COPY alembic.ini .

ENV OLLAMA_HOST=http://172.17.0.1:11434
ENV OLLAMA_MODEL=gemma4:31b-cloud
ENV OLLAMA_TIMEOUT=120

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -sf "${OLLAMA_HOST}/api/tags" > /dev/null || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
