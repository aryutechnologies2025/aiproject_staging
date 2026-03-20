# -------- BASE --------
FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies first (cache layer)
COPY req.txt .
RUN pip install --upgrade pip setuptools wheel \
    && pip install --no-cache-dir --prefer-binary -r req.txt

# Copy ONLY required folders (your case)
COPY app ./app
COPY adapters ./adapters
COPY inference ./inference
COPY training ./training
COPY dataset ./dataset
COPY hmns_ai ./hmns_ai

# Copy config files if needed
COPY alembic ./alembic
COPY alembic.ini .

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]