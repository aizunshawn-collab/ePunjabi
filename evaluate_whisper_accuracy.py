"""
Measures real Punjabi speech-to-text accuracy (Word Error Rate) on a genuinely
held-out test split of the same dataset used for fine-tuning
(aaparajit02/punjabi-asr), comparing:
  1. Base openai/whisper-small (no fine-tuning) - what we'd have without any
     of this fine-tuning effort.
  2. Our fine-tuned model (models/whisper-punjabi-final).
  3. openai-whisper large-v3 - what the app used/uses as a fallback for
     Punjabi when the fine-tuned model isn't available.

Uses the exact same train_test_split(test_size=0.02, seed=42) as
finetune_whisper_punjabi.py, so this test split was never seen during
training - a genuine, unbiased accuracy measurement.

Usage: python evaluate_whisper_accuracy.py [--num-examples N]
"""
import argparse
import io
import os
import time

import evaluate
import librosa
import numpy as np
import soundfile as sf
from datasets import Audio, load_dataset

SAMPLING_RATE = 16000
FINE_TUNED_MODEL_DIR = os.path.join(os.path.dirname(__file__), "models", "whisper-punjabi-final")


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
    print("Loading aaparajit02/punjabi-asr dataset (same as training)...")
    raw = load_dataset("aaparajit02/punjabi-asr", split="train")
    # Same split call/seed as finetune_whisper_punjabi.py - this "test" slice
    # was never used for training, regardless of how many epochs/examples
    # the actual training run used.
    split_raw = raw.train_test_split(test_size=0.02, seed=42)
    test_set = split_raw["test"].select_columns(["audio", "transcript"])
    # decode=False: keep raw bytes instead of auto-decoding via torchcodec,
    # which needs FFmpeg shared libraries not set up in this environment.
    test_set = test_set.cast_column("audio", Audio(decode=False))
    if num_examples is not None:
        test_set = test_set.select(range(min(num_examples, len(test_set))))
    print(f"Using {len(test_set)} held-out test examples.\n")
    return test_set


def eval_transformers_model(model_dir_or_name, label, test_set, base_processor_name="openai/whisper-small"):
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    print(f"--- Evaluating: {label} ---")
    t0 = time.time()
    # Always load tokenizer/feature-extractor from the base model - fine-tuning
    # doesn't change the vocabulary, and this avoids tokenizer.json version
    # mismatches between the training environment and this one.
    processor = WhisperProcessor.from_pretrained(base_processor_name, language="Punjabi", task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained(model_dir_or_name)
    model.eval()
    print(f"  Loaded in {time.time() - t0:.1f}s")

    predictions, references = [], []
    for i, example in enumerate(test_set):
        audio_array = decode_audio(example["audio"])
        inputs = processor.feature_extractor(audio_array, sampling_rate=SAMPLING_RATE, return_tensors="pt")
        forced_decoder_ids = processor.get_decoder_prompt_ids(language="pa", task="transcribe")
        t_start = time.time()
        predicted_ids = model.generate(inputs["input_features"], forced_decoder_ids=forced_decoder_ids)
        elapsed = time.time() - t_start
        text = processor.tokenizer.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        predictions.append(text)
        references.append(example["transcript"])
        print(f"  [{i + 1}/{len(test_set)}] ({elapsed:.1f}s) ref: {example['transcript'][:60]}")
        print(f"                 hyp: {text[:60]}")

    wer_metric = evaluate.load("wer")
    wer = 100 * wer_metric.compute(predictions=predictions, references=references)
    print(f"  {label} WER: {wer:.1f}%\n")
    return wer


def eval_openai_whisper_large_v3(test_set):
    import whisper

    label = "openai-whisper large-v3 (general-purpose baseline)"
    print(f"--- Evaluating: {label} ---")
    t0 = time.time()
    model = whisper.load_model("large-v3")
    print(f"  Loaded in {time.time() - t0:.1f}s")

    predictions, references = [], []
    for i, example in enumerate(test_set):
        audio = decode_audio(example["audio"]).astype(np.float32)
        t_start = time.time()
        result = model.transcribe(audio, language="pa", fp16=False, temperature=0.0)
        elapsed = time.time() - t_start
        text = result["text"].strip()
        predictions.append(text)
        references.append(example["transcript"])
        print(f"  [{i + 1}/{len(test_set)}] ({elapsed:.1f}s) ref: {example['transcript'][:60]}")
        print(f"                 hyp: {text[:60]}")

    wer_metric = evaluate.load("wer")
    wer = 100 * wer_metric.compute(predictions=predictions, references=references)
    print(f"  {label} WER: {wer:.1f}%\n")
    return wer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-examples", type=int, default=15, help="Number of held-out test clips to evaluate (kept small since this runs on CPU).")
    parser.add_argument("--skip-large-v3", action="store_true", help="Skip the large-v3 baseline (slowest part on CPU).")
    parser.add_argument("--skip-base-small", action="store_true", help="Skip the un-fine-tuned whisper-small baseline.")
    args = parser.parse_args()

    test_set = load_test_split(args.num_examples)

    results = {}
    if not args.skip_base_small:
        results["Base whisper-small (no fine-tuning)"] = eval_transformers_model(
            "openai/whisper-small", "Base whisper-small (no fine-tuning)", test_set
        )
    results["Fine-tuned whisper-small (ours)"] = eval_transformers_model(
        FINE_TUNED_MODEL_DIR, "Fine-tuned whisper-small (ours)", test_set
    )
    if not args.skip_large_v3:
        results["large-v3 (general-purpose)"] = eval_openai_whisper_large_v3(test_set)

    print("=" * 60)
    print(f"SUMMARY (Word Error Rate, lower = better, on {len(test_set)} held-out clips)")
    print("=" * 60)
    for label, wer in results.items():
        print(f"  {label}: {wer:.1f}%")


if __name__ == "__main__":
    main()
