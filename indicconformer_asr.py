"""
Torch-free IndicConformer (AI4Bharat) Punjabi CTC speech-to-text.

Replaces the fine-tuned Whisper-CT2 Punjabi model for /speech-to-text when
available - IndicConformer's accuracy is meaningfully better on Punjabi audio
(see repo memory for the full investigation/validation).

This module has ZERO torch/NeMo dependency at runtime. It only needs:
  - onnxruntime (already a transitive dep of faster-whisper)
  - librosa + numpy (feature extraction, matching NeMo's exact
    FilterbankFeatures algorithm bit-for-bit - validated against the real
    torch preprocessor's output)
  - sentencepiece (already a project dependency, used for the plain,
    NeMo-independent tokenizer model file)

Model artifacts (encoder+CTC-decoder ONNX graph + tokenizer) are hosted at
https://huggingface.co/AI-pro11/indicconformer-punjabi-ctc-onnx and fetched at
Docker build time (see Dockerfile), same pattern as the other AI-pro11/* models.
"""

import numpy as np
import librosa
import onnxruntime as ort
import sentencepiece as spm

# --- Feature extraction config (from the NeMo checkpoint's preprocessor cfg) ---
_SAMPLE_RATE = 16000
_WIN_LENGTH = 400  # 25ms
_HOP_LENGTH = 160  # 10ms
_N_FFT = 512
_N_MELS = 80
_PREEMPH = 0.97
_LOG_ZERO_GUARD_VALUE = 2 ** -24
_MEL_NORM = "slaney"
_NORMALIZE_EPS = 1e-5

_HANN_WINDOW = np.hanning(_WIN_LENGTH)
_MEL_FILTERBANK = librosa.filters.mel(
    sr=_SAMPLE_RATE, n_fft=_N_FFT, n_mels=_N_MELS, fmin=0, fmax=_SAMPLE_RATE / 2, norm=_MEL_NORM
)


def extract_features(audio: np.ndarray):
    """Reimplementation of NeMo's FilterbankFeatures.forward() (eval mode, no
    dithering) in pure numpy/librosa. `audio` must be float32 mono PCM at
    16kHz. Returns (features [80, T] float32, seq_len)."""
    x = audio.astype(np.float64)

    x_pre = np.empty_like(x)
    x_pre[0] = x[0]
    x_pre[1:] = x[1:] - _PREEMPH * x[:-1]

    stft = librosa.stft(
        x_pre,
        n_fft=_N_FFT,
        hop_length=_HOP_LENGTH,
        win_length=_WIN_LENGTH,
        window=_HANN_WINDOW,
        center=True,
        pad_mode="reflect",
    )
    power = np.abs(stft) ** 2
    mel_spec = _MEL_FILTERBANK @ power
    log_mel = np.log(mel_spec + _LOG_ZERO_GUARD_VALUE)

    seq_len = int(audio.shape[0] // _HOP_LENGTH) + 1
    seq_len = min(seq_len, log_mel.shape[1])

    valid = log_mel[:, :seq_len]
    mean = valid.mean(axis=1, keepdims=True)
    std = valid.std(axis=1, keepdims=True) + _NORMALIZE_EPS
    normalized = (log_mel - mean) / std

    return normalized[:, :seq_len].astype(np.float32), seq_len


def ctc_greedy_decode(log_probs: np.ndarray):
    """log_probs: [T, vocab_size] (blank = last id). Returns (token_ids, mean_confidence)."""
    blank_id = log_probs.shape[-1] - 1
    ids = log_probs.argmax(axis=-1)
    frame_probs = np.exp(log_probs.max(axis=-1))  # confidence proxy per frame

    collapsed_ids = []
    collapsed_probs = []
    prev = None
    for idx, i in enumerate(ids.tolist()):
        if i != prev:
            collapsed_ids.append(i)
            collapsed_probs.append(frame_probs[idx])
        prev = i

    final_ids = [i for i, p in zip(collapsed_ids, collapsed_probs) if i != blank_id]
    kept_probs = [p for i, p in zip(collapsed_ids, collapsed_probs) if i != blank_id]
    confidence = float(np.mean(kept_probs)) if kept_probs else 0.0
    return final_ids, confidence


class IndicConformerASR:
    """Loads the ONNX encoder+CTC-decoder graph and tokenizer once, exposes transcribe()."""

    def __init__(self, model_path: str, tokenizer_path: str, cpu_threads: int = 4):
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = cpu_threads
        self.session = ort.InferenceSession(
            model_path, sess_options=sess_options, providers=["CPUExecutionProvider"]
        )
        self.tokenizer = spm.SentencePieceProcessor()
        self.tokenizer.Load(tokenizer_path)

    def transcribe(self, audio: np.ndarray):
        """audio: float32 mono PCM at 16kHz. Returns (transcript, confidence)."""
        features, seq_len = extract_features(audio)
        log_probs, _encoded_lengths = self.session.run(
            None,
            {
                "processed_signal": features[None, :, :],
                "processed_signal_length": np.array([seq_len], dtype=np.int64),
            },
        )
        token_ids, confidence = ctc_greedy_decode(log_probs[0])
        transcript = self.tokenizer.decode(token_ids)
        return transcript, confidence
