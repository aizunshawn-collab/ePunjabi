"""
LoRA fine-tune AI4Bharat's IndicTrans2 specifically on the English<->Punjabi
pair, using the Samanantar parallel corpus.

WHY: The stock indictrans2-en-indic-1B / indictrans2-indic-en-1B checkpoints
split their capacity across 22 Indic languages. LoRA-tuning just on the
Punjabi pair can sharpen accuracy specifically for en<->pa, without the cost
of a full retrain and without touching the base model's weights (LoRA only
trains a small adapter, so the result is a few MB, not a multi-GB model).

REQUIRES A GPU. No GPU was detected on this project's dev machine, so this
is meant to be run on Google Colab (free tier includes a T4 GPU) or any
other CUDA machine - NOT locally on a CPU-only PC (would be impractically
slow for a 1B-parameter model).

One-time setup (e.g. in a Colab cell):
    !pip install -q transformers peft accelerate datasets sentencepiece IndicTransToolkit huggingface_hub

You must also:
  1. Create a (free) Hugging Face account.
  2. Accept the license/access request on the model page(s) you intend to
     train, e.g. https://huggingface.co/ai4bharat/indictrans2-en-indic-1B
     (and .../indictrans2-indic-en-1B for the other direction).
  3. Log in so the gated model can be downloaded:
       from huggingface_hub import notebook_login; notebook_login()
     (or set the HF_TOKEN environment variable).

Run:
    python finetune_indictrans2_punjabi.py --direction en-pa
    python finetune_indictrans2_punjabi.py --direction pa-en
"""

import argparse

import torch
from datasets import load_dataset
from IndicTransToolkit.processor import IndicProcessor
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

# --- Configuration -----------------------------------------------------------
# Samanantar has ~700K English<->Punjabi pairs total; these caps keep a single
# training run within a reasonable time/VRAM budget on a free Colab GPU.
# Raise them if you have more time/compute available.
NUM_TRAIN_EXAMPLES = 200_000
NUM_EVAL_EXAMPLES = 2_000
MAX_LENGTH = 256
OUTPUT_DIR_TEMPLATE = "./indictrans2-{direction}-punjabi-lora"

LANG_CODES = {"en": "eng_Latn", "pa": "pan_Guru"}
BASE_MODEL = {
    "en-pa": "ai4bharat/indictrans2-en-indic-1B",
    "pa-en": "ai4bharat/indictrans2-indic-en-1B",
}


def build_dataset(direction: str, tokenizer, ip: IndicProcessor):
    src_lang_code = LANG_CODES["en" if direction == "en-pa" else "pa"]
    tgt_lang_code = LANG_CODES["pa" if direction == "en-pa" else "en"]

    # Samanantar's Punjabi config always has 'src' = English, 'tgt' = Punjabi,
    # regardless of which direction we're training - we swap below as needed.
    raw = load_dataset("ai4bharat/samanantar", "pa", split="train")
    raw = raw.shuffle(seed=42)
    total = min(NUM_TRAIN_EXAMPLES + NUM_EVAL_EXAMPLES, len(raw))
    raw = raw.select(range(total))
    split = raw.train_test_split(test_size=NUM_EVAL_EXAMPLES, seed=42)

    def preprocess(batch):
        sources = batch["src"] if direction == "en-pa" else batch["tgt"]
        targets = batch["tgt"] if direction == "en-pa" else batch["src"]

        processed_sources = ip.preprocess_batch(
            sources, src_lang=src_lang_code, tgt_lang=tgt_lang_code
        )
        model_inputs = tokenizer(processed_sources, truncation=True, max_length=MAX_LENGTH)

        with tokenizer.as_target_tokenizer():
            labels = tokenizer(targets, truncation=True, max_length=MAX_LENGTH)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    return split.map(preprocess, batched=True, remove_columns=split["train"].column_names)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--direction", choices=["en-pa", "pa-en"], required=True)
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit(
            "No GPU detected. This script needs a CUDA GPU (e.g. Google Colab's "
            "free T4) - training a 1B-parameter model on CPU is impractically slow."
        )

    model_name = BASE_MODEL[args.direction]
    print(f"Loading base model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, trust_remote_code=True)

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules="all-linear",  # robust to IndicTrans2's custom module names
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    ip = IndicProcessor(inference=True)
    dataset = build_dataset(args.direction, tokenizer, ip)

    collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True)

    output_dir = OUTPUT_DIR_TEMPLATE.format(direction=args.direction)
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        num_train_epochs=args.epochs,
        eval_strategy="steps",
        eval_steps=500,
        save_steps=500,
        save_total_limit=2,
        logging_steps=50,
        fp16=True,
        predict_with_generate=True,
        report_to="none",
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        data_collator=collator,
        tokenizer=tokenizer,
    )

    trainer.train()
    final_dir = output_dir + "-final"
    trainer.save_model(final_dir)
    print(f"Done. LoRA adapter saved to {final_dir}.")
    print(
        "Load it later with: "
        "PeftModel.from_pretrained(AutoModelForSeq2SeqLM.from_pretrained("
        f"'{model_name}', trust_remote_code=True), '{final_dir}')"
    )


if __name__ == "__main__":
    main()
