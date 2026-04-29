# ---------- STAGE 1: BUILDER ----------
FROM python:3.10-slim AS builder

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY req.txt .

RUN pip install --upgrade pip setuptools wheel \
    && pip install --prefix=/install --no-cache-dir -r req.txt


# ---------- STAGE 2: FINAL ----------
FROM python:3.10-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# httpx is used for async Ollama calls; curl for healthcheck probe
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages
COPY --from=builder /install /usr/local

# Copy app
COPY app ./app
COPY adapters ./adapters
COPY inference ./inference
COPY training ./training
COPY dataset ./dataset
COPY hrms_ai ./hrms_ai
COPY alembic ./alembic
COPY alembic.ini .

# Ollama lives on the HOST (or a sidecar); we just need the URL reachable.
# OLLAMA_HOST is injected via .env / docker-compose; default shown below.
ENV OLLAMA_HOST=http://host.docker.internal:11434
ENV OLLAMA_MODEL=gemma4:31b-cloud
ENV OLLAMA_TIMEOUT=120

EXPOSE 8000

# Healthcheck: verify Ollama is reachable before reporting healthy
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -sf "${OLLAMA_HOST}/api/tags" > /dev/null || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]