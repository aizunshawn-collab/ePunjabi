"""
Fine-tune OpenAI Whisper specifically for Punjabi speech recognition, using
the AI4Bharat Shrutilipi corpus (All India Radio news broadcasts,
aaparajit02/punjabi-asr on Hugging Face).

WHY: Whisper's out-of-the-box Punjabi ASR accuracy is weaker than
higher-resource languages. Fine-tuning on real Punjabi speech data closes
that gap significantly - this follows the standard, widely-used Hugging
Face "Fine-Tune Whisper for Multilingual ASR" recipe.

NOTE: This originally used Mozilla Common Voice, but Mozilla discontinued
distributing Common Voice through Hugging Face as of October 2025 (moved to
their own "Mozilla Data Collective" platform), so this was switched to the
Shrutilipi corpus above, which is still freely available and ungated.

REQUIRES A GPU. No GPU was detected on this project's dev machine, so this
is meant to be run on a hosted GPU notebook (e.g. Kaggle Notebooks, which
gives ~30 free GPU-hours/week - Settings > Accelerator > GPU) or any other
CUDA machine, including rented cloud GPUs (Lambda Labs, RunPod, Vast.ai, etc.).

One-time setup (e.g. in a Kaggle notebook cell):
    !pip install -q transformers datasets accelerate evaluate jiwer librosa soundfile huggingface_hub

You must also:
  1. Create a (free) Hugging Face account.
  2. Log in (optional - this dataset isn't gated, but avoids anonymous rate limits):
       from huggingface_hub import notebook_login; notebook_login()
     (or set the HF_TOKEN environment variable / a Kaggle secret).

Run (on Kaggle, point --output-dir at the persisted /kaggle/working folder
so the saved model survives after the session ends):
    python finetune_whisper_punjabi.py --base-model openai/whisper-small --output-dir /kaggle/working/whisper-punjabi
"""

import argparse
import os

# Force single-GPU: on 2-GPU Kaggle sessions (e.g. "T4 x2"), Trainer auto-wraps the
# model in torch DataParallel across both GPUs, which reliably hangs mid-training on
# Kaggle due to NCCL/P2P communication issues between the two virtualized T4s. Must be
# set before torch is imported/initializes CUDA.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import evaluate
import torch
from datasets import Audio, load_dataset
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)

LANGUAGE = "Punjabi"
TASK = "transcribe"
SAMPLING_RATE = 16000


class DataCollatorSpeechSeq2SeqWithPadding:
    """Pads audio input features and label token sequences to the batch max,
    independently, since they have very different lengths/shapes."""

    def __init__(self, processor):
        self.processor = processor

    def __call__(self, features):
        input_features = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )
        # Strip the BOS token if it was already appended by an earlier step,
        # since the model adds it automatically during training.
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-model",
        default="openai/whisper-small",
        help="Use a smaller checkpoint (small/medium) for feasible fine-tuning on a single GPU.",
    )
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--max-train-examples",
        type=int,
        default=None,
        help="Use only the first N examples for a quick end-to-end test run.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Where to save checkpoints/final model. Defaults to ./<model>-punjabi. "
        "On Kaggle, pass /kaggle/working/<name> so it survives the session.",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit(
            "No GPU detected. This script needs a CUDA GPU (e.g. Google Colab's "
            "free T4) - fine-tuning Whisper on CPU is impractically slow."
        )

    # Common Voice was discontinued on Hugging Face (Oct 2025, moved to Mozilla Data
    # Collective) - using the AI4Bharat Shrutilipi corpus instead (All India Radio
    # news broadcasts), which is still freely available and ungated.
    print("Loading Punjabi ASR data (aaparajit02/punjabi-asr)...")
    raw = load_dataset("aaparajit02/punjabi-asr", split="train")
    if args.max_train_examples is not None:
        raw = raw.select(range(min(args.max_train_examples, len(raw))))
    split_raw = raw.train_test_split(test_size=0.02, seed=42)
    common_voice = {"train": split_raw["train"], "test": split_raw["test"]}

    # Keep only audio + transcript, resample to the 16kHz Whisper expects.
    for split in common_voice:
        common_voice[split] = common_voice[split].select_columns(["audio", "transcript"])
        common_voice[split] = common_voice[split].cast_column(
            "audio", Audio(sampling_rate=SAMPLING_RATE)
        )

    processor = WhisperProcessor.from_pretrained(
        args.base_model, language=LANGUAGE, task=TASK
    )

    # Whisper's decoder caps labels at 448 tokens - truncate here as a hard guarantee
    # training can never crash on an oversized label, regardless of dataset caching quirks.
    MAX_LABEL_LENGTH = 448

    def prepare_dataset(batch):
        audio = batch["audio"]
        batch["input_features"] = processor.feature_extractor(
            audio["array"], sampling_rate=audio["sampling_rate"]
        ).input_features[0]
        batch["labels"] = processor.tokenizer(
            batch["transcript"], truncation=True, max_length=MAX_LABEL_LENGTH
        ).input_ids
        return batch

    # num_proc=1 (no multiprocessing): CUDA is already initialized in this process by
    # the availability check above, and forking workers after CUDA init reliably hangs.
    common_voice = {
        split: ds.map(prepare_dataset, remove_columns=ds.column_names, num_proc=1)
        for split, ds in common_voice.items()
    }

    # Drop any example that got truncated above - keeps only examples whose full
    # transcript fit, so we never train on an incomplete/mismatched transcript.
    common_voice = {
        split: ds.filter(lambda x: len(x["labels"]) < MAX_LABEL_LENGTH)
        for split, ds in common_voice.items()
    }

    model = WhisperForConditionalGeneration.from_pretrained(args.base_model)
    model.generation_config.language = LANGUAGE.lower()
    model.generation_config.task = TASK
    model.generation_config.forced_decoder_ids = None

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor)
    wer_metric = evaluate.load("wer")

    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

        pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        wer = 100 * wer_metric.compute(predictions=pred_str, references=label_str)
        return {"wer": wer}

    output_dir = args.output_dir or f"./{args.base_model.split('/')[-1]}-punjabi"
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=2,
        learning_rate=1e-5,
        warmup_steps=500,
        num_train_epochs=args.epochs,
        gradient_checkpointing=True,
        fp16=True,
        eval_strategy="steps",
        per_device_eval_batch_size=8,
        predict_with_generate=True,
        generation_max_length=225,
        save_steps=1000,
        eval_steps=1000,
        logging_steps=50,
        save_total_limit=2,
        report_to="none",
    )

    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=common_voice["train"],
        eval_dataset=common_voice["test"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        processing_class=processor.feature_extractor,  # newer transformers renamed tokenizer= to this
    )

    trainer.train()
    trainer.save_model(output_dir + "-final")
    processor.save_pretrained(output_dir + "-final")
    print(f"Done. Fine-tuned model saved to {output_dir}-final")


if __name__ == "__main__":
    main()
