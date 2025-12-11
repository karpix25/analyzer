FROM python:3.11-slim

WORKDIR /app

# Без буфера, чтобы логи print сразу уходили в stdout/stderr (easypanel их забирает)
ENV PYTHONUNBUFFERED=1

# EasyOCR model directory - сохраняем модели в образе
ENV EASYOCR_MODULE_PATH=/app/.EasyOCR

# Системные зависимости для OpenCV
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Python зависимости
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download EasyOCR models during build
RUN python -c "import easyocr; reader = easyocr.Reader(['en', 'ru'], gpu=False, model_storage_directory='/app/.EasyOCR', download_enabled=True)"

COPY . .

# Uvicorn with explicit logging to stdout for Easypanel
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--log-level", "info", "--access-log"]
