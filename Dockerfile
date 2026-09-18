FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt scipy>=1.12.0

# Copy application source
COPY verigraph/ ./verigraph/
COPY frontend/ ./frontend/
COPY sample_data/ ./sample_data/
COPY evaluation/ ./evaluation/
COPY pyproject.toml .
COPY README.md .

# Create data directories
RUN mkdir -p data/storage data/uploads data/benchmarks

# Environment configuration
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app
ENV PORT=8080
ENV HOST=0.0.0.0

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

CMD ["uvicorn", "verigraph.api.app:app", "--host", "0.0.0.0", "--port", "8080"]
