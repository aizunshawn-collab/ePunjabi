"""
Three-way translation comparison on evaluate_translation_accuracy.py's TEST_SET:
  1. raw IndicTrans2 (CT2 int8, no glossary)
  2. full production pipeline (IndicTrans2 + glossary overrides)
  3. Indic-Translate (outputs read from a JSON file), raw and with the glossary applied on top

Indic-Translate needs a separate venv (torch/transformers>=5), so generate its outputs first:
    C:\\pyenvs\\indic-translate-test\\Scripts\\python.exe evaluate_indic_translate.py --dump-json indic_translate_results.json
then run this in the production venv:
    C:\\pyenvs\\ep-voice-backend\\Scripts\\python.exe compare_translation_models.py [indic_translate_results.json]
"""
import json
import sys
import time
import unicodedata

from sacrebleu.metrics import CHRF

import ai_backend_server as backend
from evaluate_translation_accuracy import TEST_SET

sys.stdout.reconfigure(encoding="utf-8")


def nfc(text):
    return unicodedata.normalize("NFC", text)


def score(chrf, hypothesis, references):
    return chrf.sentence_score(nfc(hypothesis), [nfc(r) for r in references]).score


def indictrans2(text, src, tgt):
    start = time.perf_counter()
    norm = backend.normalize_text(text)
    chunks = backend.chunk_text(norm) or [norm]
    raw_chunks = backend.translate_chunks(chunks, src, tgt)
    raw_elapsed = time.perf_counter() - start
    raw = " ".join(raw_chunks)
    piped = " ".join(backend.apply_glossary(chunks, raw_chunks, src, tgt))
    return raw, raw_elapsed, piped


def with_glossary(text, output, src, tgt):
    chunks = [backend.normalize_text(text)]
    return " ".join(backend.apply_glossary(chunks, [output], src, tgt))


def summarize(rows, key):
    def avg(subset):
        return sum(r[key] for r in subset) / len(subset) if subset else float("nan")
    return avg(rows), avg([r for r in rows if r["tgt"] == "pa"]), avg([r for r in rows if r["tgt"] == "en"])


def main():
    json_path = sys.argv[1] if len(sys.argv) > 1 else "indic_translate_results.json"
    with open(json_path, encoding="utf-8") as f:
        it_results = json.load(f)
    if len(it_results) != len(TEST_SET):
        print(f"WARNING: {json_path} has {len(it_results)} cases, TEST_SET has {len(TEST_SET)}; comparing the overlap only.\n")

    chrf = CHRF()
    rows = []
    for (src, tgt, text, reference), it in zip(TEST_SET, it_results):
        refs = [reference] if isinstance(reference, str) else list(reference)
        raw, raw_latency, piped = indictrans2(text, src, tgt)
        it_out = it["output"]
        rows.append({
            "src": src, "tgt": tgt, "text": text,
            "raw": score(chrf, raw, refs),
            "pipeline": score(chrf, piped, refs),
            "it": score(chrf, it_out, refs),
            "it_gloss": score(chrf, with_glossary(text, it_out, src, tgt), refs),
            "raw_latency": raw_latency,
            "it_latency": it["latency"],
            "raw_out": raw, "pipeline_out": piped, "it_out": it_out,
        })

    print(f"{'Model':<42}{'overall':>9}{'en->pa':>9}{'pa->en':>9}{'latency':>10}")
    for label, key, latency_key in [
        ("IndicTrans2 raw (no glossary)", "raw", "raw_latency"),
        ("IndicTrans2 + glossary (production)", "pipeline", "raw_latency"),
        ("Indic-Translate raw", "it", "it_latency"),
        ("Indic-Translate + glossary", "it_gloss", "it_latency"),
    ]:
        overall, en_pa, pa_en = summarize(rows, key)
        latency = sum(r[latency_key] for r in rows) / len(rows)
        print(f"{label:<42}{overall:>9.1f}{en_pa:>9.1f}{pa_en:>9.1f}{latency:>9.2f}s")

    wins = {"raw": 0, "it": 0, "tie": 0}
    for r in rows:
        if r["it"] > r["raw"] + 0.5:
            wins["it"] += 1
        elif r["raw"] > r["it"] + 0.5:
            wins["raw"] += 1
        else:
            wins["tie"] += 1
    print(f"\nRaw-vs-raw per-case: Indic-Translate won {wins['it']}, IndicTrans2 won {wins['raw']}, tied {wins['tie']}")

    print("\nLargest per-case gaps between the two raw models (positive = Indic-Translate better):")
    for r in sorted(rows, key=lambda r: abs(r["it"] - r["raw"]), reverse=True)[:8]:
        print(f"  {r['it'] - r['raw']:+6.1f}  {r['src']}->{r['tgt']}  {r['text']!r}")
        print(f"          IndicTrans2:     {r['raw_out']!r}")
        print(f"          Indic-Translate: {r['it_out']!r}")


if __name__ == "__main__":
    main()
