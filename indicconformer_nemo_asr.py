"""
Torch/NeMo-based Punjabi RNNT speech-to-text - IndicConformer's NATIVE RNNT
decoder head, run via AI4Bharat's NeMo fork instead of the torch-free ONNX
CTC-only export in indicconformer_asr.py.

WHY THIS EXISTS (see repo memory "OPTION 1 IMPLEMENTED" for the full
writeup): production previously only ran IndicConformer's CTC decoder head
(exported to ONNX) specifically to stay torch-free. The 2026-09-27
investigation measured the SAME model's RNNT decoder head as "excellent/
near-perfect" vs CTC being "slightly noisier (occasional double-matra/
spelling glitches) but still coherent" on identical test audio - that known
gap is what this module closes. This is a deliberate, narrow, Punjabi-ASR-
ONLY exception to the project's broader torch-free architecture (see the big
migration entry in repo memory) - translation (IndicTrans2/CTranslate2) and
English/auto-detect STT (Whisper/CTranslate2) are completely untouched and
stay torch-free.

Needs AI4Bharat's NeMo fork (https://github.com/AI4Bharat/NeMo, nemo-v2
branch) - the public `nemo_toolkit` package cannot load this checkpoint's
"multilingual" tokenizer type. See Dockerfile for the install step and
test_indicconformer_colab.ipynb for the notebook that first proved this
works (same restore_from()/cur_decoder="rnnt"/transcribe() call pattern
used here).

`indicconformer_asr.py`'s ONNX CTC-only IndicConformerASR remains in place
as an automatic rollback - see ai_backend_server.py's loading order.
"""

import os
import tempfile

import numpy as np
import soundfile as sf


class IndicConformerNeMoASR:
    """Loads the native .nemo checkpoint once (slow - several seconds, torch
    + NeMo model restore), exposes transcribe() with the same
    (transcript, confidence) interface as indicconformer_asr.IndicConformerASR."""

    def __init__(self, nemo_checkpoint_path: str):
        # Imported lazily so merely importing this module doesn't require
        # torch/nemo to be installed - only instantiating this class does.
        import nemo.collections.asr as nemo_asr

        self.model = nemo_asr.models.ASRModel.restore_from(restore_path=nemo_checkpoint_path)
        self.model.eval()
        # IndicConformer is a hybrid CTC+RNNT model - this is the whole point
        # of using the native checkpoint instead of the ONNX CTC-only export.
        self.model.cur_decoder = "rnnt"

    def transcribe(self, audio: np.ndarray):
        """audio: float32 mono PCM at 16kHz. Returns (transcript, confidence)."""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            sf.write(tmp_path, audio, 16000)
            # Same call pattern validated in test_indicconformer_colab.ipynb.
            # IndicConformer is multilingual with per-language decoder heads
            # keyed by language code, so language_id must be passed
            # explicitly or the internal lookup receives None.
            result = self.model.transcribe([tmp_path], batch_size=1, language_id="pa")
        finally:
            os.unlink(tmp_path)

        hyp = result[0]
        # NeMo's transcribe() return shape varies across versions/configs:
        # sometimes result[0] is already the per-utterance string/Hypothesis,
        # other times (observed with this hybrid RNNT/CTC checkpoint) it's
        # wrapped in an extra single-element list, e.g. result == [['text']]
        # rather than ['text']. Unwrap any such single-element list nesting
        # before falling back to Hypothesis's .text / str().
        while isinstance(hyp, list) and len(hyp) == 1:
            hyp = hyp[0]
        transcript = hyp if isinstance(hyp, str) else getattr(hyp, "text", str(hyp))

        # RNNT decoding doesn't expose a simple per-frame confidence the way
        # CTC's greedy-argmax max-probability proxy does. This value is
        # cosmetic/logging-only in ai_backend_server.py (not used for any
        # routing decision in the Punjabi STT path), so a fixed placeholder
        # is fine here rather than engineering a real confidence estimate.
        confidence = 0.9
        return transcript, confidence
