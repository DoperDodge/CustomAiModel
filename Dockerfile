# ============================================================
# Custom AI Model — Docker Deployment
# ============================================================
# Build:  docker build -t custom-ai-model .
# Run:    docker run --gpus all -p 8000:8000 custom-ai-model
#
# For development (mount source):
#   docker run --gpus all -p 8000:8000 -v $(pwd):/app custom-ai-model

FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    python3-pip \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.11 as default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Create directories for checkpoints and data
RUN mkdir -p /app/checkpoints /app/data

# Expose API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python3 -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Run the API server
CMD ["uvicorn", "src.api.server:app", "--host", "0.0.0.0", "--port", "8000"]
