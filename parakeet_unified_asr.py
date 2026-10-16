"""NVIDIA Parakeet-unified-en-0.6b (RNNT, punctuation + casing) running on ONNX Runtime.

Torch-free English ASR. Uses the community ONNX export
`eschmidbauer/parakeet-unified-en-0.6b-onnx` (int8 variant: ~660MB, ~1GB RAM).

Measured 2026-10 on 200 random clips of Svarah (Indian-accented English), CPU,
4 threads, vs the production Whisper small (beam 1, VAD, vocab prompt):
    Whisper small      WER 13.94%   RTF 0.49
    Parakeet-unified   WER  7.58%   RTF 0.28   (int8)
See repo memory for details. Output already carries punctuation and casing, which
also helps the downstream en->pa translation step.

Architecture: FastConformer encoder (128 log-mel bins) + RNNT predictor/joint, greedy
decoding (max 10 symbols per frame).
"""

import numpy as np
import librosa
import onnxruntime as ort
import sentencepiece as spm

_SAMPLE_RATE = 16000
_WIN_LENGTH = 400  # 25ms
_HOP_LENGTH = 160  # 10ms
_N_FFT = 512
_N_MELS = 128
_PREEMPH = 0.97
_LOG_ZERO_GUARD_VALUE = 2 ** -24
_NORMALIZE_EPS = 1e-5

_BLANK_ID = 1024
_MAX_SYMBOLS_PER_STEP = 10
_PREDICTOR_LAYERS = 2
_PREDICTOR_HIDDEN = 640

_HANN_WINDOW = np.hanning(_WIN_LENGTH)
_MEL_FILTERBANK = librosa.filters.mel(
    sr=_SAMPLE_RATE, n_fft=_N_FFT, n_mels=_N_MELS, fmin=0, fmax=_SAMPLE_RATE / 2, norm="slaney"
)


def extract_features(audio: np.ndarray):
    """NeMo FilterbankFeatures (eval, no dither) in numpy/librosa.
    `audio`: float32 mono 16kHz. Returns (features [1, 128, T] float32, seq_len)."""
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
    log_mel = np.log(_MEL_FILTERBANK @ (np.abs(stft) ** 2) + _LOG_ZERO_GUARD_VALUE)

    seq_len = min(int(audio.shape[0] // _HOP_LENGTH) + 1, log_mel.shape[1])
    valid = log_mel[:, :seq_len]
    normalized = (valid - valid.mean(axis=1, keepdims=True)) / (valid.std(axis=1, keepdims=True) + _NORMALIZE_EPS)
    return normalized[None].astype(np.float32), seq_len


class ParakeetUnifiedASR:
    """Loads encoder + decoder/joint ONNX graphs and the tokenizer once; exposes transcribe()."""

    def __init__(self, encoder_path: str, decoder_path: str, tokenizer_path: str, cpu_threads: int = 4):
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = cpu_threads
        providers = ["CPUExecutionProvider"]
        self.encoder = ort.InferenceSession(encoder_path, sess_options=sess_options, providers=providers)
        self.decoder = ort.InferenceSession(decoder_path, sess_options=sess_options, providers=providers)
        self.tokenizer = spm.SentencePieceProcessor()
        self.tokenizer.Load(tokenizer_path)

    def transcribe(self, audio: np.ndarray):
        """audio: float32 mono PCM at 16kHz. Returns (transcript, confidence)."""
        if audio.shape[0] < _WIN_LENGTH:
            return "", 0.0

        features, seq_len = extract_features(audio)
        enc_out, enc_len = self.encoder.run(
            None,
            {"audio_signal": features, "length": np.array([seq_len], dtype=np.int64)},
        )

        state1 = np.zeros((_PREDICTOR_LAYERS, 1, _PREDICTOR_HIDDEN), dtype=np.float32)
        state2 = np.zeros((_PREDICTOR_LAYERS, 1, _PREDICTOR_HIDDEN), dtype=np.float32)
        last_token = _BLANK_ID
        token_ids = []
        token_probs = []
        target_length = np.array([1], dtype=np.int32)

        for t in range(int(enc_len[0])):
            frame = enc_out[:, :, t:t + 1]
            for _ in range(_MAX_SYMBOLS_PER_STEP):
                logits, _, new_state1, new_state2 = self.decoder.run(
                    None,
                    {
                        "encoder_outputs": frame,
                        "targets": np.array([[last_token]], dtype=np.int32),
                        "target_length": target_length,
                        "input_states_1": state1,
                        "input_states_2": state2,
                    },
                )
                scores = logits[0, 0, 0]
                idx = int(np.argmax(scores))
                if idx == _BLANK_ID:
                    break
                exp = np.exp(scores - scores[idx])
                token_probs.append(1.0 / float(exp.sum()))
                token_ids.append(idx)
                last_token = idx
                state1, state2 = new_state1, new_state2

        transcript = self.tokenizer.decode(token_ids).strip()
        confidence = float(np.mean(token_probs)) if token_probs else 0.0
        return transcript, confidence
