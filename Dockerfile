# syntax=docker/dockerfile:1
# Container image for ai_backend_server.py. No Whisper anywhere (removed
# 2026-10, see repo memory): translation is IndicTrans2 (CTranslate2),
# English ASR is Parakeet-unified (ONNX, torch-free), Punjabi ASR is
# IndicConformer (native NeMo RNNT decoder, torch + AI4Bharat's NeMo fork -
# see indicconformer_nemo_asr.py docstring + repo memory "OPTION 1
# IMPLEMENTED"), and spoken-language auto-detection is SpeechBrain's
# VoxLingua107 model (torch + torchaudio, reusing the torch install already
# needed for NeMo RNNT).
#
# Multi-stage build (added 2026-10-03): the single-stage version pushed the
# final image past Fly's 8GB uncompressed-unpack limit (confirmed via
# `flyctl logs`: "Not enough space to unpack image, possibly exceeds maximum
# of 8GB uncompressed" - this failure is SILENT in `flyctl deploy`'s own
# output, which still reports "complete"/shows the DNS-verified message even
# though the machine never actually updated). Splitting into a `builder`
# stage (apt build-essential/git + pip/NeMo source builds) and a slim
# `final` stage that only COPYs the installed venv removes build-essential,
# git, pip wheel caches, and NeMo's cloned repo source tree from the shipped
# image - none of that is needed at runtime.
#
# MODEL WEIGHTS (2026-10-19): no longer downloaded at build time / baked into
# the image at all - see the NOTE further down, right before the `final`
# stage, for why (the gated NeMo RNNT checkpoint alone was enough to push
# the image back over the 8GB limit above). They're downloaded at container
# STARTUP instead, onto a persistent Fly Volume mounted at /app/models (see
# download_models.py, entrypoint.sh, and fly.toml's [[mounts]]).
FROM python:3.11-slim AS builder

# git is needed to clone AI4Bharat's NeMo fork below. build-essential is
# needed because some of NeMo's ASR-extra deps (e.g. youtokentome) have C
# extensions without prebuilt wheels for this Python version/platform.
# Neither is needed in the final runtime image.
RUN apt-get update && apt-get install -y --no-install-recommends \
    git build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt gunicorn supervisor
# NOTE: requirements.txt's spaCy line + en_core_web_sm wheel URL are already
# installed by the -r above - no separate `spacy download` step is needed
# here, since the model is a direct wheel (not spaCy's own downloader CLI).

# --- Punjabi ASR accuracy fix: native NeMo RNNT decoder (torch) ---
# CPU-only torch/torchaudio wheels - never the CUDA build (no GPU on Fly
# machines, and CUDA wheels are multiple GB bigger for nothing we'd use).
# torchaudio is pulled in here (not left to speechbrain's own dependency
# resolution below) so it's guaranteed to match this exact torch version.
RUN pip install --no-cache-dir torch torchaudio --index-url https://download.pytorch.org/whl/cpu

# SpeechBrain (spoken-language-ID, see ai_backend_server.py's
# get_language_id_model()). Installed here, BEFORE the NeMo install and the
# pytorch-lightning/numpy pins below, so those two pins (which NeMo needs to
# actually work) always win if speechbrain's own dependency resolution would
# otherwise pull a different numpy/torch version.
RUN pip install --no-cache-dir speechbrain

# Public nemo_toolkit cannot load this checkpoint's "multilingual" tokenizer
# type - only AI4Bharat's own fork does (discovered via
# test_indicconformer_colab.ipynb). Installing only the "asr" extra (not
# "[all]"/reinstall.sh) avoids NLP/TTS/multimodal deps that don't have
# prebuilt wheels for this Python version and would fail to build from source.
RUN git clone --depth 1 --branch nemo-v2 https://github.com/AI4Bharat/NeMo.git /tmp/nemo-ai4bharat \
    && pip install --no-cache-dir "/tmp/nemo-ai4bharat[asr]" \
    && rm -rf /tmp/nemo-ai4bharat

# Pin pytorch-lightning AFTER the nemo install above, which has no upper
# bound (just >=2.2.1) and resolves to whatever the latest release is at
# build time. Latest releases (e.g. 2.6.6) removed NeptuneLogger from
# pytorch_lightning.loggers' public API, which NeMo's exp_manager.py
# unconditionally imports - breaks `import nemo.collections.asr` entirely
# with "ImportError: cannot import name 'NeptuneLogger'". 2.4.0 is confirmed
# (via its own loggers/__init__.py) to still export NeptuneLogger and still
# satisfies NeMo's own >=2.2.1 constraint.
RUN pip install --no-cache-dir "pytorch-lightning==2.4.0"

# Pin numpy to the 1.x line AFTER the nemo install above. The nemo install
# (or one of its deps) pulls in numpy 2.x, but AI4Bharat's NeMo fork's own
# audio preprocessing code (nemo/collections/asr/parts/preprocessing/
# segment.py: `samples.dtype in np.sctypes['int']`) uses the `np.sctypes`
# attribute, which was REMOVED in NumPy 2.0. This doesn't break import or
# model loading - it only fails at actual transcribe() call time, with:
# "AttributeError: `np.sctypes` was removed in the NumPy 2.0 release."
# Confirmed via a live 500 error/traceback on /speech-to-text after NeMo
# RNNT loaded successfully. numpy<2 restores `np.sctypes` and is otherwise
# fully compatible with torch/NeMo's >=1.22 style constraints.
RUN pip install --no-cache-dir "numpy<2"

# Pin pyarrow to a pre-2.0-NumPy-era release, also AFTER the numpy<2 pin
# above. `nemo.collections.asr` transitively imports HF `datasets` (only
# used by NeMo's own training-data loaders, never invoked by this inference-
# only backend) -> `datasets` imports `pyarrow` -> newer pyarrow releases
# (19.0+) hard-require NumPy>=2.0 at import time ("ImportError: pyarrow
# requires NumPy 2.0 or newer, found 1.26.4"), which directly conflicts with
# the numpy<2 pin above and breaks `import nemo.collections.asr` itself -
# confirmed via a live traceback while diagnosing why NeMo RNNT silently
# fell back to the ONNX CTC decoder in production. pyarrow<19 has no such
# requirement and is otherwise unused by this backend at runtime.
RUN pip install --no-cache-dir "pyarrow<19"

# Pin pandas to an exact version verified (in this project's own local dev
# venv) to import cleanly against numpy 1.26.4, also AFTER the numpy<2 pin
# above. requirements.txt's indic-nlp-library (installed earlier, before the
# NeMo/numpy<2 sequence) transitively pulls in pandas (indicnlp/script/
# indic_scripts.py does `import pandas as pd`), unpinned - a fresh build
# resolves whatever the latest pandas release is, built against whatever
# numpy is present AT THAT POINT (unconstrained, likely numpy 2.x) - the
# numpy<2 pin further below then downgrades numpy afterward WITHOUT
# reinstalling pandas, breaking pandas' compiled C-extension ABI against the
# now-older numpy and crashing every gunicorn worker on boot with
# "ImportError: Please upgrade numpy to >= 2.0.2 to use this pandas version"
# (confirmed via a live traceback - /health never passed, every worker
# crashed on `from indictrans_processor import IndicProcessor`). A first
# attempt at pinning just "pandas<2.2.2" (open-ended) instead produced a
# DIFFERENT crash ("ImportError: C extension: None not built" - pip
# resolved a version/build without a usable prebuilt wheel here and fell
# back to an incomplete source build) - also confirmed via a live
# traceback. pandas==3.0.5 is confirmed (via `import numpy, pandas` in this
# project's own local venv, numpy 1.26.4) to import without error.
# `--only-binary=:all:` makes pip fail loudly at `docker build` time instead
# of silently shipping another broken install if a wheel ever goes missing.
RUN pip install --no-cache-dir --only-binary=:all: "pandas==3.0.5"

# NOTE 2026-10-19: model downloads used to happen HERE, at build time, baked
# directly into the image. Re-adding the gated ~523MB NeMo RNNT checkpoint
# pushed the final image's UNCOMPRESSED size back over Fly's 8GB-per-machine
# unpack limit - a failure mode that is SILENT in `flyctl deploy`'s own
# output (reports "complete"/healthy) while the machine keeps serving its old
# image forever; only `flyctl logs`' separate runner-level entries show the
# real "Not enough space to unpack image, possibly exceeds maximum of 8GiB
# uncompressed" error. Model downloads now happen at CONTAINER STARTUP
# instead (see download_models.py + entrypoint.sh), writing into /app/models
# which is a persistent Fly Volume mount (see fly.toml's [[mounts]]) kept
# entirely OFF the image. This keeps the image itself (just the venv/code) a
# small, stable size regardless of how many/how large the models get.

# --- Final runtime stage: only the installed venv, no model weights ---
FROM python:3.11-slim AS final

# curl is needed by watchdog.sh's health polling. decode_audio() (from the
# faster-whisper package, kept only for this utility - see
# ai_backend_server.py's import comment) decodes audio itself via PyAV
# (bundled ffmpeg decode libs) - no system ffmpeg needed. tesseract-ocr
# (+ eng/pan trained data) is needed by /ocr - pytesseract just shells out to
# the tesseract binary, it isn't a pip package. libglib2.0-0 is needed by
# opencv-python-headless (document_translator.py's cv2.inpaint) - the
# "-headless" build avoids the much heavier libGL/X11 GUI deps, but still
# dynamically links libgthread-2.0.so.0 from glib at import time. Neither
# git nor build-essential is needed at runtime - they stay in the builder
# stage only.
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl tesseract-ocr tesseract-ocr-eng tesseract-ocr-pan libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY ai_backend_server.py indictrans_processor.py indicconformer_asr.py indicconformer_nemo_asr.py parakeet_unified_asr.py punjabi_numerals.py english_numerals.py punjabi_name_corrections.py parakeet_name_corrections.py punjabi_alphanumeric_fix.py word_sense_disambiguation.py beam_sensitive_patterns.py google_translate_fallback.py google_tts_fallback.py ocr_processor.py document_translator.py download_models.py supervisord.conf watchdog.sh entrypoint.sh ./
COPY fonts ./fonts
RUN chmod +x watchdog.sh entrypoint.sh

EXPOSE 8080

CMD ["/app/entrypoint.sh"]
