# Container image for ai_backend_server.py, built for GPU-backed cloud hosting
# (e.g. Fly.io GPU machines). Uses an official CUDA-enabled PyTorch base image
# instead of a plain python:slim image, since the local dev venv's CPU-only
# torch build won't use a cloud GPU even if one is attached.
# Pinned to 2.5+ because requirements.txt installs an unpinned (latest)
# transformers, which refuses to enable its PyTorch backend below 2.5.
FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime

# Unbuffered stdout so startup/model-load prints reach container logs in
# real time instead of sitting in a buffer until the process exits.
ENV PYTHONUNBUFFERED=1

# ffmpeg is required by Whisper to decode uploaded audio. curl is used by the
# in-container watchdog to poll /health.
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
# torch is already provided by the base image (CUDA build) - skip reinstalling
# it from requirements.txt so pip doesn't silently swap in a CPU-only wheel.
# supervisor runs gunicorn as a supervised process so a full process death
# (not just a worker crash, which gunicorn already handles on its own) gets
# restarted automatically - this matters most on a persistent Pod, which has
# no equivalent of Serverless's own health-check-based worker restarts.
RUN grep -v '^torch' requirements.txt > requirements.docker.txt \
    && pip install --no-cache-dir -r requirements.docker.txt gunicorn supervisor

COPY ai_backend_server.py indictrans_processor.py supervisord.conf watchdog.sh ./
# Fine-tuned Punjabi Whisper checkpoint (see finetune_whisper_punjabi.py) -
# without this, the app silently falls back to only the generic Whisper
# model for Punjabi audio too. Pulled from HF Hub (not COPYed from the repo)
# since the checkpoint is 922MB - over GitHub's 100MB push limit.
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='Pro-Developer/whisper-punjabi-finetuned', local_dir='models/whisper-punjabi-final')"
RUN chmod +x watchdog.sh

# Pre-download/load every model at build time by importing the app module
# (its model-loading code runs unconditionally at import time, the same way
# gunicorn will import it later). This bakes ~8-10GB of weights into the
# image layer so cold-start workers load them from local disk instead of
# downloading from Hugging Face/OpenAI over the network on every cold start -
# that network download, not raw model-load time, is the dominant cold-start
# cost for this app. No GPU is available at build time, so this just runs on
# CPU long enough to populate the on-disk caches; it does not run inference.
RUN python -c "import ai_backend_server"

EXPOSE 8080

# supervisord runs both gunicorn and watchdog.sh (see supervisord.conf):
# gunicorn's own --timeout 900 tolerates the long cold-start model load, and
# the watchdog force-restarts gunicorn if /health ever stops responding after
# a successful start (a hang, not just a crash gunicorn would already retry).
CMD ["supervisord", "-c", "/app/supervisord.conf"]
