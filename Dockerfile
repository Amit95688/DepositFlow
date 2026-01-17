FROM python:3.9-slim

# Set Python configuration environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MLFLOW_TRACKING_URI=file:/app/mlruns

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /tmp/requirements.txt

COPY . .

# Install WSGI server (gunicorn)
RUN pip install gunicorn

# Create directory for MLflow artifacts
RUN mkdir -p /app/mlruns

EXPOSE 5000 5001

# Use gunicorn as WSGI server (equivalent to EB's WSGIPath: application:application)
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "4", "--access-logfile", "-", "--error-logfile", "-", "application:application"]