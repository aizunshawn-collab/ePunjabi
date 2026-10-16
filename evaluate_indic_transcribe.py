"""
Benchmarks the two new "Indic-Transcribe" ASR variants (bodhan-ai/indic-transcribe-core,
bodhan-ai/indic-transcribe-flex) against the current production Punjabi ASR
(IndicConformer, ONNX CTC) on the SAME held-out test split used elsewhere in this repo
(aaparajit02/punjabi-asr, train_test_split(test_size=0.02, seed=42)).

Measures both Word Error Rate and per-clip CPU latency, since the production backend
(Fly.io) is CPU-only and latency is the primary risk flagged for Indic-Transcribe's
autoregressive Transformer-decoder architecture (see repo memory: Whisper large-v3 had
the same architecture family and cost a fixed ~17-25s/request regardless of audio length).

Usage: python evaluate_indic_transcribe.py [--num-examples N] [--skip-indicconformer]
"""
import argparse
import io
import os
import sys
import time

# Gurmukhi print() output breaks with the default cp1252 console encoding when stdout
# is redirected to a file (no console attached) - force UTF-8 regardless of destination.
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")


class _Tee:
    """Writes to both the real stdout and a UTF-8 file, bypassing PowerShell's `>`
    redirection (which reinterprets a child process's raw UTF-8 bytes as UTF-16LE and
    mangles Gurmukhi text - this writes the log file directly instead)."""

    def __init__(self, stream, log_path):
        self._stream = stream
        self._log = open(log_path, "w", encoding="utf-8")

    def write(self, data):
        self._stream.write(data)
        self._log.write(data)

    def flush(self):
        self._stream.flush()
        self._log.flush()

import evaluate
import librosa
import numpy as np
import soundfile as sf
from datasets import Audio, load_dataset
from huggingface_hub import snapshot_download
from indicnlp.normalize.indic_normalize import IndicNormalizerFactory

SAMPLING_RATE = 16000
INDICCONFORMER_DIR = os.path.join(os.path.dirname(__file__), "models", "indicconformer-punjabi-ctc-onnx")

# Canonicalizes common Gurmukhi Unicode-level spelling variants (nukta forms, matras,
# etc.) - same normalizer production's IndicTrans2 pipeline uses (indictrans_processor.py).
# Lets us separate "genuine content errors" from "legitimate spelling variant, counted
# as wrong by strict surface-form WER".
_PA_NORMALIZER = IndicNormalizerFactory().get_normalizer("pa")


def indic_normalize(text):
    return _PA_NORMALIZER.normalize(text.strip())


def decode_audio(audio_column):
    """Decode raw audio bytes ourselves (soundfile/librosa) instead of via
    datasets' Audio() feature, which requires torchcodec + working FFmpeg
    shared libraries that aren't set up in this Windows environment."""
    data, sr = sf.read(io.BytesIO(audio_column["bytes"]), dtype="float32")
    if data.ndim > 1:
        data = data.mean(axis=1)  # downmix to mono
    if sr != SAMPLING_RATE:
        data = librosa.resample(data, orig_sr=sr, target_sr=SAMPLING_RATE)
    return data


def load_test_split(num_examples):
    print("Loading aaparajit02/punjabi-asr dataset (same held-out split as prior evals)...")
    raw = load_dataset("aaparajit02/punjabi-asr", split="train")
    split_raw = raw.train_test_split(test_size=0.02, seed=42)
    test_set = split_raw["test"].select_columns(["audio", "transcript"])
    test_set = test_set.cast_column("audio", Audio(decode=False))
    if num_examples is not None:
        test_set = test_set.select(range(min(num_examples, len(test_set))))
    print(f"Using {len(test_set)} held-out test examples.\n")
    return test_set


def summarize(label, predictions, references, latencies):
    wer_metric = evaluate.load("wer")
    wer = 100 * wer_metric.compute(predictions=predictions, references=references)
    norm_wer = 100 * wer_metric.compute(
        predictions=[indic_normalize(p) for p in predictions],
        references=[indic_normalize(r) for r in references],
    )
    avg_latency = sum(latencies) / len(latencies)
    print(f"  {label}: WER={wer:.1f}%  normalized_WER={norm_wer:.1f}%  avg_latency={avg_latency:.2f}s  "
          f"min={min(latencies):.2f}s  max={max(latencies):.2f}s\n")
    return wer, norm_wer, avg_latency


def eval_indicconformer(test_set):
    from indicconformer_asr import IndicConformerASR

    label = "IndicConformer (production baseline, ONNX CTC)"
    print(f"--- Evaluating: {label} ---")
    t0 = time.time()
    model = IndicConformerASR(
        os.path.join(INDICCONFORMER_DIR, "model.onnx"),
        os.path.join(INDICCONFORMER_DIR, "tokenizer_pa.model"),
    )
    print(f"  Loaded in {time.time() - t0:.1f}s")

    predictions, references, latencies = [], [], []
    for i, example in enumerate(test_set):
        audio_array = decode_audio(example["audio"])
        t_start = time.time()
        text, _confidence = model.transcribe(audio_array)
        elapsed = time.time() - t_start
        predictions.append(text)
        references.append(example["transcript"])
        latencies.append(elapsed)
        print(f"  [{i + 1}/{len(test_set)}] ({elapsed:.2f}s) ref: {example['transcript'][:60]}")
        print(f"                 hyp: {text[:60]}")

    return summarize(label, predictions, references, latencies)


def eval_indic_transcribe(repo_id, label, test_set):
    print(f"--- Evaluating: {label} ---")
    t0 = time.time()
    model_dir = snapshot_download(repo_id, ignore_patterns=["*.nemo", "nemo/*"])
    sys.path.insert(0, model_dir)
    from indic_transcribe import IndicTranscribe
    asr = IndicTranscribe.from_pretrained(model_dir, device="cpu")
    print(f"  Loaded in {time.time() - t0:.1f}s")

    predictions, references, latencies = [], [], []
    for i, example in enumerate(test_set):
        audio_array = decode_audio(example["audio"])  # transcribe() accepts numpy arrays directly
        try:
            t_start = time.time()
            text = asr.transcribe(audio_array, lang="pa", mode="native")
            if isinstance(text, tuple):
                text = text[0]
            elapsed = time.time() - t_start
        except ValueError as e:  # clip longer than the model's max_seconds (45s)
            print(f"  [{i + 1}/{len(test_set)}] SKIPPED: {e}")
            continue
        predictions.append(text)
        references.append(example["transcript"])
        latencies.append(elapsed)
        print(f"  [{i + 1}/{len(test_set)}] ({elapsed:.2f}s) ref: {example['transcript'][:60]}")
        print(f"                 hyp: {text[:60]}")

    # Remove this model's modules from sys.path/sys.modules before loading the next
    # variant, since both repos ship same-named modules (indic_transcribe, lid, etc.)
    sys.path.remove(model_dir)
    for mod_name in list(sys.modules):
        mod = sys.modules[mod_name]
        if getattr(mod, "__file__", None) and str(mod.__file__).startswith(model_dir):
            del sys.modules[mod_name]

    return summarize(label, predictions, references, latencies)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-examples", type=int, default=15,
                         help="Number of held-out test clips to evaluate (kept small since this runs on CPU).")
    parser.add_argument("--skip-indicconformer", action="store_true")
    parser.add_argument("--skip-core", action="store_true")
    parser.add_argument("--skip-flex", action="store_true")
    parser.add_argument("--log-file", type=str, default=None,
                         help="Also write clean UTF-8 output directly to this file.")
    args = parser.parse_args()

    if args.log_file:
        sys.stdout = _Tee(sys.stdout, args.log_file)

    test_set = load_test_split(args.num_examples)
    results = {}

    if not args.skip_indicconformer:
        results["IndicConformer (production)"] = eval_indicconformer(test_set)
    if not args.skip_core:
        results["Indic-Transcribe-core"] = eval_indic_transcribe(
            "bodhan-ai/indic-transcribe-core", "Indic-Transcribe-core", test_set
        )
    if not args.skip_flex:
        results["Indic-Transcribe-flex"] = eval_indic_transcribe(
            "bodhan-ai/indic-transcribe-flex", "Indic-Transcribe-flex", test_set
        )

    print("=" * 70)
    print(f"SUMMARY (on {len(test_set)} held-out clips, CPU)")
    print("=" * 70)
    for label, (wer, norm_wer, avg_latency) in results.items():
        print(f"  {label}: WER={wer:.1f}%  normalized_WER={norm_wer:.1f}%  avg_latency={avg_latency:.2f}s/clip")


if __name__ == "__main__":
    main()
