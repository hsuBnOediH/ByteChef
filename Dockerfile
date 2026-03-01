# ── Dockerfile ───────────────────────────────────────────────────────────────
# Builds a lean Python 3.11 image for the ByteChef Streamlit app.
# state.md is stored in a Docker volume so data persists across container restarts.

FROM python:3.11-slim

# Install only curl for health check (ffmpeg removed — format detected via magic bytes in Python)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app.py agent_logic.py ./
COPY .streamlit/ .streamlit/

# state.md will live in a mounted volume (/app/state.md)
# so we don't COPY it into the image.

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "app.py"]
