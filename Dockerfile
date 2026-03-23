# ---------- STAGE 1: BUILDER ----------
FROM python:3.10-slim as builder

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

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]