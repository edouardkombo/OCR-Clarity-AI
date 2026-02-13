FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Minimal runtime deps for PyMuPDF + OpenCV headless
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt requirements-extras.txt /app/
ARG INSTALL_EXTRAS=0
RUN pip install --no-cache-dir -r requirements.txt
RUN if [ "$INSTALL_EXTRAS" = "1" ]; then pip install --no-cache-dir -r requirements-extras.txt; fi

COPY clarity_ocr.py clarity_ocr_hard.py clarity_worker.py enqueue_job.py api_service.py load_test.py /app/

# Data mount points (compose will mount)
RUN mkdir -p /data/in /data/out

EXPOSE 8000
CMD ["python", "clarity_worker.py"]
