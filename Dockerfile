# Heart Disease Risk Predictor API — production image
# BITS Pilani AIMLCZG523 · Assignment 01
FROM python:3.10-slim

WORKDIR /app

# System deps: curl is needed for the container HEALTHCHECK below.
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl && rm -rf /var/lib/apt/lists/*

# Install Python deps first so this layer is cached across code changes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Application code + the trained model artifact.
COPY src/ ./src/
COPY models/ ./models/

# Run as a non-root user for security.
RUN adduser --disabled-password --gecos '' appuser
USER appuser

EXPOSE 8000

# Container-level health check hitting the FastAPI /health endpoint.
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
