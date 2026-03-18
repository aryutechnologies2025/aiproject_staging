FROM python:3.10-slim-bookworm

# Prevent Python cache + improve logs
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

# System deps (minimal)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libssl-dev \
    pkg-config \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install dependencies separately (better caching)
COPY req.txt .

# Upgrade pip and install CPU-only torch + deps
RUN pip install --upgrade pip setuptools wheel \
    && pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu -r req.txt \
    && pip uninstall -y nvidia-* triton || true

# Copy only required app files
COPY . .

# Reduce size further
RUN find /usr/local -type d -name "__pycache__" -exec rm -rf {} + \
    && find /usr/local -type f -name "*.pyc" -delete

EXPOSE 8000

# Use production workers
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
