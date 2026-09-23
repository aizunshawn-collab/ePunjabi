# Container image for ai_backend_server.py, built for GPU-backed cloud hosting
# (e.g. Fly.io GPU machines). Uses an official CUDA-enabled PyTorch base image
# instead of a plain python:slim image, since the local dev venv's CPU-only
# torch build won't use a cloud GPU even if one is attached.
FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime

# ffmpeg is required by Whisper to decode uploaded audio.
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
# torch is already provided by the base image (CUDA build) - skip reinstalling
# it from requirements.txt so pip doesn't silently swap in a CPU-only wheel.
RUN grep -v '^torch' requirements.txt > requirements.docker.txt \
    && pip install --no-cache-dir -r requirements.docker.txt gunicorn

COPY ai_backend_server.py indictrans_processor.py ./

EXPOSE 8080

# Single worker: the model is loaded once into GPU memory per worker, so
# more workers would multiply GPU memory usage rather than add capacity.
# Threads give some request concurrency without a second model copy.
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--workers", "1", "--threads", "4", \
     "--timeout", "180", "ai_backend_server:app"]
