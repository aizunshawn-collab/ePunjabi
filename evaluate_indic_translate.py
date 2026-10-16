"""
Benchmarks bodhan-ai/indic-translate (8B Gemma-4 LLM) against production's IndicTrans2
on accuracy (chrF, same metric/library as evaluate_translation_accuracy.py) and CPU latency.

No GPU on this machine and only ~16GB total RAM, so the model is loaded 4-bit quantized
(bitsandbytes NF4) - this is a real deviation from the model card's own bf16-only
benchmarks, so don't expect its self-reported chrF numbers to reproduce exactly here.

Run in the disposable venv: C:\\pyenvs\\indic-translate-test
Usage:  python evaluate_indic_translate.py [--smoke-test]
"""
import os
import sys
import json
import time
import argparse

sys.stdout.reconfigure(encoding="utf-8")

# Avoid the ChunkedEncodingError/symlink issues hit during the earlier Indic-Transcribe
# investigation (see /memories/repo notes) - same fixes apply here.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")

import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig
from sacrebleu.metrics import CHRF

MODEL = "bodhan-ai/indic-translate"
LANG_NAME = {"en": "English", "pa": "Punjabi"}


def download_with_retries(repo_id, max_retries=40, backoff_seconds=5):
    """This connection drops/times out constantly on large files - HF's own internal
    retries aren't enough for a 15.9GB repo. snapshot_download resumes partial files
    on each call, so just keep retrying the whole thing (same fix as the earlier
    Indic-Transcribe investigation)."""
    for attempt in range(1, max_retries + 1):
        try:
            return snapshot_download(repo_id=repo_id)
        except Exception as e:
            print(f"[retry {attempt}/{max_retries}] download error: {e!r}")
            if attempt == max_retries:
                raise
            time.sleep(backoff_seconds)

# Copied verbatim from evaluate_translation_accuracy.py's TEST_SET.
TEST_SET = [
    ("en", "pa", "Hello", "ਸਤ ਸ੍ਰੀ ਅਕਾਲ"),
    ("en", "pa", "Thank you very much", "ਬਹੁਤ ਧੰਨਵਾਦ"),
    ("pa", "en", "ਬਾਥਰੂਮ ਕਿੱਥੇ ਹੈ?", "Where is the bathroom?"),
    ("pa", "en", "ਮੈਨੂੰ ਸਮਝ ਨਹੀਂ ਆਈ", "I don't understand"),
    ("en", "pa", "I am going to the market to buy vegetables.",
     "ਮੈਂ ਸਬਜ਼ੀਆਂ ਖਰੀਦਣ ਲਈ ਬਾਜ਼ਾਰ ਜਾ ਰਿਹਾ ਹਾਂ।"),
    ("en", "pa", "The weather is very nice today, so let's go for a walk.",
     ["ਅੱਜ ਮੌਸਮ ਬਹੁਤ ਵਧੀਆ ਹੈ, ਇਸ ਲਈ ਚਲੋ ਸੈਰ ਲਈ ਚੱਲੀਏ।",
      "ਅੱਜ ਮੌਸਮ ਬਹੁਤ ਚੰਗਾ ਹੈ, ਇਸ ਲਈ ਆਓ ਸੈਰ ਕਰਨ ਚੱਲੀਏ।"]),
    ("en", "pa", "Can you please help me find my way to the train station?",
     ["ਕੀ ਤੁਸੀਂ ਮੈਨੂੰ ਰੇਲਵੇ ਸਟੇਸ਼ਨ ਦਾ ਰਸਤਾ ਲੱਭਣ ਵਿੱਚ ਮਦਦ ਕਰ ਸਕਦੇ ਹੋ?",
      "ਕੀ ਤੁਸੀਂ ਰੇਲਵੇ ਸਟੇਸ਼ਨ ਤੱਕ ਪਹੁੰਚਣ ਦਾ ਰਸਤਾ ਲੱਭਣ ਵਿੱਚ ਮੇਰੀ ਮਦਦ ਕਰ ਸਕਦੇ ਹੋ?"]),
    ("pa", "en", "ਮੈਂ ਸਕੂਲ ਜਾ ਰਿਹਾ ਹਾਂ।", ["I am going to school.", "I'm going to school."]),
    ("pa", "en", "ਕੱਲ੍ਹ ਮੇਰੇ ਦੋਸਤ ਦਾ ਜਨਮਦਿਨ ਹੈ।", "Tomorrow is my friend's birthday."),
    ("pa", "en", "ਕਿਰਪਾ ਕਰਕੇ ਦਰਵਾਜ਼ਾ ਬੰਦ ਕਰ ਦਿਓ।", "Please close the door."),
    ("en", "pa", "Good morning", "ਸ਼ੁਭ ਸਵੇਰ"),
    ("en", "pa", "Nice to meet you", "ਤੁਹਾਨੂੰ ਮਿਲ ਕੇ ਖੁਸ਼ੀ ਹੋਈ"),
    ("en", "pa", "How much is the ticket?", "ਟਿਕਟ ਕਿੰਨੇ ਦੀ ਹੈ?"),
    ("en", "pa", "Turn left", "ਖੱਬੇ ਮੁੜੋ"),
    ("pa", "en", "ਸ਼ੁਭ ਸਵੇਰ", "Good morning"),
    ("pa", "en", "ਅਲਵਿਦਾ", "Goodbye"),
    ("pa", "en", "ਤੁਸੀਂ ਕਿਵੇਂ ਹੋ?", "How are you?"),
    ("pa", "en", "ਹਸਪਤਾਲ ਕਿੱਥੇ ਹੈ?", "Where is the hospital?"),
    ("en", "pa", "Where is the train station?", "ਰੇਲਵੇ ਸਟੇਸ਼ਨ ਕਿੱਥੇ ਹੈ?"),
    ("en", "pa", "I need a doctor", "ਮੈਨੂੰ ਡਾਕਟਰ ਚਾਹੀਦਾ ਹੈ"),
    ("en", "pa", "Call an ambulance", "ਐਂਬੂਲੈਂਸ ਬੁਲਾਓ"),
    ("en", "pa", "This is an emergency", "ਇਹ ਇੱਕ ਐਮਰਜੈਂਸੀ ਹੈ"),
    ("pa", "en", "ਮੈਨੂੰ ਡਾਕਟਰ ਚਾਹੀਦਾ ਹੈ", "I need a doctor"),
    ("pa", "en", "ਐਂਬੂਲੈਂਸ ਬੁਲਾਓ", "Call an ambulance"),
    ("pa", "en", "ਪੁਲਿਸ ਨੂੰ ਬੁਲਾਓ", "Call the police"),
    ("pa", "en", "ਮੈਨੂੰ ਬੁਖਾਰ ਹੈ", "I have a fever"),
    ("en", "pa", "Monday", "ਸੋਮਵਾਰ"),
    ("en", "pa", "Seven", "ਸੱਤ"),
    ("en", "pa", "Water", "ਪਾਣੀ"),
    ("pa", "en", "ਸ਼ੁੱਕਰਵਾਰ", "Friday"),
    ("pa", "en", "ਦਸ", "Ten"),
    ("pa", "en", "ਖਾਣਾ", "Food"),
    ("en", "pa", "My phone battery is almost dead.",
     ["ਮੇਰੇ ਫ਼ੋਨ ਦੀ ਬੈਟਰੀ ਲਗਭਗ ਖਤਮ ਹੋ ਗਈ ਹੈ।", "ਮੇਰੇ ਫ਼ੋਨ ਦੀ ਬੈਟਰੀ ਲਗਭਗ ਖ਼ਤਮ ਹੋ ਚੁੱਕੀ ਹੈ।"]),
    ("en", "pa", "She works at a hospital in the city.",
     "ਉਹ ਸ਼ਹਿਰ ਦੇ ਇੱਕ ਹਸਪਤਾਲ ਵਿੱਚ ਕੰਮ ਕਰਦੀ ਹੈ।"),
    ("en", "pa", "We will meet at the restaurant at seven o'clock.",
     "ਅਸੀਂ ਸੱਤ ਵਜੇ ਰੈਸਟੋਰੈਂਟ ਵਿੱਚ ਮਿਲਾਂਗੇ।"),
    ("en", "pa", "Please send me the documents by email.",
     ["ਕਿਰਪਾ ਕਰਕੇ ਮੈਨੂੰ ਦਸਤਾਵੇਜ਼ ਈਮੇਲ ਰਾਹੀਂ ਭੇਜੋ।", "ਕਿਰਪਾ ਕਰਕੇ ਮੈਨੂੰ ਈਮੇਲ ਰਾਹੀਂ ਦਸਤਾਵੇਜ਼ ਭੇਜੋ।"]),
    ("en", "pa", "I would like to book a room for two nights.",
     ["ਮੈਂ ਦੋ ਰਾਤਾਂ ਲਈ ਇੱਕ ਕਮਰਾ ਬੁੱਕ ਕਰਨਾ ਚਾਹੁੰਦਾ ਹਾਂ।", "ਮੈਂ ਦੋ ਰਾਤਾਂ ਲਈ ਕਮਰਾ ਬੁੱਕ ਕਰਨਾ ਚਾਹੁੰਦਾ ਹਾਂ।"]),
    ("en", "pa", "The train is running thirty minutes late.",
     "ਰੇਲਗੱਡੀ ਤੀਹ ਮਿੰਟ ਦੇਰੀ ਨਾਲ ਚੱਲ ਰਹੀ ਹੈ।"),
    ("en", "pa", "Can I have the bill, please?", "ਕੀ ਮੈਨੂੰ ਬਿੱਲ ਮਿਲ ਸਕਦਾ ਹੈ?"),
    ("en", "pa", "I lost my passport at the airport.",
     ["ਮੈਂ ਹਵਾਈ ਅੱਡੇ 'ਤੇ ਆਪਣਾ ਪਾਸਪੋਰਟ ਗੁਆ ਦਿੱਤਾ।", "ਮੇਰਾ ਪਾਸਪੋਰਟ ਹਵਾਈ ਅੱਡੇ ਉੱਤੇ ਗੁੰਮ ਹੋ ਗਿਆ।"]),
    ("en", "pa", "Please keep the change.", "ਕਿਰਪਾ ਕਰਕੇ ਬਾਕੀ ਪੈਸੇ ਰੱਖੋ।"),
    ("pa", "en", "ਮੈਂ ਹਰ ਰੋਜ਼ ਸਵੇਰੇ ਸੈਰ ਕਰਨ ਜਾਂਦਾ ਹਾਂ।", "I go for a walk every morning."),
    ("pa", "en", "ਉਸਨੇ ਬਹੁਤ ਮਿਹਨਤ ਨਾਲ ਪ੍ਰੀਖਿਆ ਪਾਸ ਕੀਤੀ।",
     ["He passed the exam with great effort.", "He passed the exam with a lot of hard work."]),
    ("pa", "en", "ਕੀ ਤੁਸੀਂ ਮੈਨੂੰ ਇੱਕ ਗਲਾਸ ਪਾਣੀ ਦੇ ਸਕਦੇ ਹੋ?", "Can you give me a glass of water?"),
    ("pa", "en", "ਬੱਚੇ ਬਾਗ਼ ਵਿੱਚ ਖੇਡ ਰਹੇ ਹਨ।", "The children are playing in the garden."),
    ("pa", "en", "ਮੌਸਮ ਵਿਭਾਗ ਨੇ ਕੱਲ੍ਹ ਮੀਂਹ ਦੀ ਭਵਿੱਖਬਾਣੀ ਕੀਤੀ ਹੈ।",
     ["The weather department has forecast rain for tomorrow.",
      "The Meteorological Department has predicted rain tomorrow."]),
    ("pa", "en", "ਮੈਨੂੰ ਇਹ ਕਿਤਾਬ ਬਹੁਤ ਪਸੰਦ ਆਈ।",
     ["I really liked this book.", "Absolutely loved this book."]),
    ("pa", "en", "ਦੁਕਾਨ ਸ਼ਾਮ ਨੂੰ ਅੱਠ ਵਜੇ ਬੰਦ ਹੋ ਜਾਂਦੀ ਹੈ।",
     ["The shop closes at eight in the evening.", "The shop closes at eight o'clock in the evening."]),
    ("pa", "en", "ਮੈਨੂੰ ਇੱਕ ਟੈਕਸੀ ਦੀ ਲੋੜ ਹੈ।", "I need a taxi."),
    ("pa", "en", "ਸਟੇਸ਼ਨ ਤੋਂ ਬਾਜ਼ਾਰ ਕਿੰਨਾ ਦੂਰ ਹੈ?", "How far is the market from the station?"),
]


def load_model():
    print("Downloading/resuming model snapshot (retrying through connection drops)...")
    local_dir = download_with_retries(MODEL)
    print(f"Snapshot ready at {local_dir}\n")

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float32,
    )
    processor = AutoProcessor.from_pretrained(local_dir)
    model = AutoModelForImageTextToText.from_pretrained(
        local_dir,
        quantization_config=quant_config,
        device_map="cpu",
        attn_implementation="sdpa",
    ).eval()
    return model, processor


def translate(model, processor, text, target_language, max_new_tokens=128):
    messages = [{
        "role": "user",
        "content": f"Translate the following text into {target_language}:\n\n{text}",
    }]
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    start = time.perf_counter()
    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, use_cache=True)
    elapsed = time.perf_counter() - start

    completion = output[0, inputs["input_ids"].shape[-1]:]
    return processor.decode(completion, skip_special_tokens=True).strip(), elapsed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-test", action="store_true", help="Run only 2 sentences to validate feasibility")
    parser.add_argument("--dump-json", help="Write per-case outputs/latencies here for compare_translation_models.py")
    args = parser.parse_args()

    print("Loading model (4-bit NF4, CPU)... this downloads ~15.9GB on first run.")
    t0 = time.perf_counter()
    model, processor = load_model()
    print(f"Model loaded in {time.perf_counter() - t0:.1f}s.\n")

    test_set = TEST_SET[:2] if args.smoke_test else TEST_SET
    chrf = CHRF()
    results = []

    for i, (source_lang, target_lang, text, reference) in enumerate(test_set, start=1):
        references = [reference] if isinstance(reference, str) else reference
        output, latency = translate(model, processor, text, LANG_NAME[target_lang])
        score = max(chrf.sentence_score(output, [ref]).score for ref in references)
        results.append({"source_lang": source_lang, "target_lang": target_lang, "text": text,
                         "references": references, "output": output, "latency": latency, "chrf": score})
        print(f"[{i}/{len(test_set)}] ({source_lang}->{target_lang}, {latency:.2f}s, chrF={score:.1f}) "
              f"{text!r} -> {output!r}")
        if args.dump_json:
            with open(args.dump_json, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=1)

    if not args.smoke_test:
        avg_chrf = sum(r["chrf"] for r in results) / len(results)
        avg_latency = sum(r["latency"] for r in results) / len(results)
        en_pa = [r for r in results if r["target_lang"] == "pa"]
        pa_en = [r for r in results if r["target_lang"] == "en"]
        print(f"\nOverall avg chrF:  {avg_chrf:.1f}/100  ({len(results)} cases)")
        print(f"  en->pa avg chrF: {sum(r['chrf'] for r in en_pa) / len(en_pa):.1f}/100  ({len(en_pa)} cases)")
        print(f"  pa->en avg chrF: {sum(r['chrf'] for r in pa_en) / len(pa_en):.1f}/100  ({len(pa_en)} cases)")
        print(f"Average latency/call: {avg_latency:.2f}s (4-bit NF4, CPU)")


if __name__ == "__main__":
    main()
