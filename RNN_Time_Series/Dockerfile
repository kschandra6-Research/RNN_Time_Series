# ==============================================================================
# NeuroCast: Production Dockerfile for Web Hosting
# Concept & Design by K.S.Chandra 2026
#
# Deployable on:
# - Google Cloud Run
# - AWS App Runner / ECS
# - Render / Railway / Fly.io / Heroku
# - Any Docker Container Registry / VPS
# ==============================================================================

FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=5000 \
    HOST=0.0.0.0

WORKDIR /app

# Copy dependency specifications
COPY requirements.txt .

# Install optional acceleration dependencies if present (gracefully ignores if offline)
RUN pip install --no-cache-dir -r requirements.txt || true

# Copy application source code
COPY . /app

# Cloud providers automatically assign $PORT (default 5000)
EXPOSE 5000

# Start production multi-threaded web server
CMD ["python", "server.py", "--no-browser"]
