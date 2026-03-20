# -------- BASE --------
FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install ONLY runtime deps (remove build tools)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libssl-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY req.txt .

# Install dependencies (optimized)
RUN pip install --upgrade pip \
    && pip install --no-cache-dir --prefer-binary -r req.txt

# Copy only required code (avoid full copy)
COPY app/ ./app/
COPY *.py ./

EXPOSE 8000

# Use better production server
CMD ["gunicorn", "-k", "uvicorn.workers.UvicornWorker", "app.main:app", "--bind", "0.0.0.0:8000", "--workers", "2"]