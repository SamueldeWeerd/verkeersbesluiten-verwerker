# Use Python 3.11 slim image as base
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies for OpenCV, geospatial libraries, and ROMA
RUN apt-get update && apt-get install -y \
    # OpenCV dependencies
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    # GDAL dependencies
    libgdal-dev \
    gdal-bin \
    libproj-dev \
    proj-data \
    proj-bin \
    # Build dependencies
    libffi-dev \
    build-essential \
    python3-dev \
    # Tools
    curl \
    git \
    procps \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables for GDAL
ENV CPLUS_INCLUDE_PATH=/usr/include/gdal
ENV C_INCLUDE_PATH=/usr/include/gdal

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip
# Install NumPy first and pin it
RUN pip install --no-cache-dir "numpy==1.24.4"
# Install requirements
RUN pip install --no-cache-dir -r requirements.txt
# Force reinstall NumPy to ensure correct version
RUN pip install --no-cache-dir --force-reinstall "numpy==1.24.4"
RUN pip install --no-cache-dir gunicorn
# Verify final NumPy version
RUN python -c "import numpy; print(f'Final NumPy version: {numpy.__version__}')"

# Copy application code (excluding large model files first for better caching)
COPY app.py .
COPY memory_config.py .

# Copy refactored architecture components
COPY services/ ./services/
COPY utils/ ./utils/
COPY models/ ./models/

# Copy third party components
COPY third_party/ ./third_party/

# Setup ROMA model (AFTER copying the files)
RUN python third_party/setup_roma.py && \
    echo "ROMA model setup complete"

# Create necessary directories
RUN mkdir -p uploads outputs ROMA_checkpoints/roma

# Expose the port
EXPOSE 8000

# Health check with increased timeout
HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Create a startup script with memory optimization
RUN echo '#!/bin/bash\n\
echo "🚀 Starting Refactored Feature Matching API with ROMA support..."\n\
echo "📂 ROMA model status:"\n\
ls -la /app/third_party/RoMa || echo "ROMA not found"\n\
echo "💾 Available memory: $(free -h)"\n\
echo "🔧 Starting server as single process (no workers)..."\n\
exec python -m uvicorn app:app \
    --host 0.0.0.0 \
    --port 8000 \
    --timeout-keep-alive 120 \
    --log-level info' > /app/start.sh && \
    chmod +x /app/start.sh

# Set the command to run the application
CMD ["/app/start.sh"] 