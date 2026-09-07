FROM python:3.12-slim

WORKDIR /app

# Install system utilities
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    gcc \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code, models, data, and web assets
COPY . .

# Ensure ports are exposed (8000 for web portal & API, 8001 for metrics exporter)
EXPOSE 8000 8001

ENV PORT=8000
ENV PYTHONUNBUFFERED=1

HEALTHCHECK --interval=15s --timeout=5s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "module3_workload_prediction.api:app", "--host", "0.0.0.0", "--port", "8000"]
