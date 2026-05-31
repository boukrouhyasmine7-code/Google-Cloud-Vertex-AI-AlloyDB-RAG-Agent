# ============================================================
# Dockerfile — Vertex AI Agent Platform
# Deploys to: Google Cloud Run
# ============================================================

FROM python:3.11-slim

# Metadata
LABEL org.opencontainers.image.title="Vertex AI Agent Platform — AlloyDB RAG"
LABEL org.opencontainers.image.description="Gemini Pro + AlloyDB RAG chat agent"
LABEL com.google.cloud.service="cloud-run"

# Security: run as non-root
RUN groupadd --gid 1001 appgroup && \
    useradd --uid 1001 --gid appgroup --shell /bin/bash --create-home appuser

WORKDIR /app

# Install dependencies (layer cached separately from app code)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/ ./app/
COPY main.py .

# Cloud Run uses PORT env var (default 8080)
ENV API_PORT=8080
ENV API_HOST=0.0.0.0
ENV API_RELOAD=false

USER appuser

EXPOSE 8080

# Gunicorn with uvicorn workers for Cloud Run production
CMD ["python", "-m", "uvicorn", "main:app", \
     "--host", "0.0.0.0", \
     "--port", "8080", \
     "--workers", "2", \
     "--log-level", "info"]
