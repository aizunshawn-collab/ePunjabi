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
# stage (apt build-essential/git + pip/NeMo source builds + model downloads)
# and a slim `final` stage that only COPYs the installed venv + downloaded
# models removes build-essential, git, pip wheel caches, and NeMo's cloned
# repo source tree from the shipped image - none of that is needed at
# runtime.
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

# Download the gated IndicConformer .nemo checkpoint (contains the native
# RNNT decoder head - the public ONNX export below is CTC-only). Needs a HF
# token with access to the gated ai4bharat repo (the account's own token
# already has it, from accepting the repo's gate in an earlier session).
# Passed as a BuildKit secret (`flyctl deploy --build-secret HF_TOKEN=...`)
# so it is never baked into an image layer/history.
RUN --mount=type=secret,id=HF_TOKEN \
    python -c "\
import os; \
from huggingface_hub import hf_hub_download; \
token = open('/run/secrets/HF_TOKEN').read().strip(); \
hf_hub_download(repo_id='ai4bharat/indicconformer_stt_pa_hybrid_ctc_rnnt_large', filename='indicconformer_stt_pa_hybrid_rnnt_large.nemo', local_dir='models/indicconformer-nemo', token=token)"

# Fetch all CT2/ONNX model directories from Hugging Face Hub at build time so
# the running container never needs to download anything at cold start.
# All source repos are public/ungated - no HF_TOKEN needed.
# NOTE: all Whisper models (whisper-small-ct2, whisper-punjabi-ct2,
# distil-whisper-large-v3-ct2) and the opt-in Parakeet-CTC-0.6b
# (AI-pro11/parakeet-ctc-0.6b-onnx) are intentionally NOT downloaded here -
# removed entirely 2026-10 in favor of Parakeet-unified (English) +
# IndicConformer (Punjabi) + SpeechBrain (language-ID), see repo memory.
RUN python -c "from huggingface_hub import snapshot_download; \
snapshot_download(repo_id='AI-pro11/indictrans2-en-indic-ct2-int8', local_dir='models/indictrans2-en-indic-ct2-int8'); \
snapshot_download(repo_id='AI-pro11/indictrans2-indic-en-ct2-int8', local_dir='models/indictrans2-indic-en-ct2-int8'); \
snapshot_download(repo_id='AI-pro11/indicconformer-punjabi-ctc-onnx', local_dir='models/indicconformer-punjabi-ctc-onnx')"

# Parakeet-unified-en-0.6b ONNX, INT8 only (~660MB; the fp32 variant is 2.4GB and
# would risk Fly's 8GB uncompressed image limit). The only English ASR model -
# see parakeet_unified_asr.py.  Lands in models/parakeet-unified-en-onnx/onnx_int8/.
RUN python -c "from huggingface_hub import snapshot_download; \
snapshot_download(repo_id='eschmidbauer/parakeet-unified-en-0.6b-onnx', allow_patterns=['onnx_int8/*'], local_dir='models/parakeet-unified-en-onnx')"

# SpeechBrain's VoxLingua107 ECAPA-TDNN spoken-language-ID model (~86MB) -
# replaces Whisper's detect_language() for 'auto'-mode /speech-to-text
# requests. Public/ungated, no HF_TOKEN needed.
RUN python -c "from huggingface_hub import snapshot_download; \
snapshot_download(repo_id='speechbrain/lang-id-voxlingua107-ecapa', local_dir='models/lang-id-voxlingua107-ecapa')"

# --- Final runtime stage: only the installed venv + downloaded models ---
FROM python:3.11-slim AS final

# curl is needed by watchdog.sh's health polling. decode_audio() (from the
# faster-whisper package, kept only for this utility - see
# ai_backend_server.py's import comment) decodes audio itself via PyAV
# (bundled ffmpeg decode libs) - no system ffmpeg needed. tesseract-ocr
# (+ eng/pan trained data) is needed by /ocr - pytesseract just shells out to
# the tesseract binary, it isn't a pip package. Neither git nor
# build-essential is needed at runtime - they stay in the builder stage only.
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl tesseract-ocr tesseract-ocr-eng tesseract-ocr-pan \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /app/models ./models
ENV PATH="/opt/venv/bin:$PATH"

COPY ai_backend_server.py indictrans_processor.py indicconformer_asr.py indicconformer_nemo_asr.py parakeet_unified_asr.py punjabi_numerals.py english_numerals.py punjabi_name_corrections.py parakeet_name_corrections.py punjabi_alphanumeric_fix.py ocr_processor.py supervisord.conf watchdog.sh ./
RUN chmod +x watchdog.sh

EXPOSE 8080

CMD ["supervisord", "-c", "supervisord.conf"]
