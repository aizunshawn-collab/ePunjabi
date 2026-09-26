"""
Isolates how much of our translation accuracy comes from the raw IndicTrans2
model itself vs. the hand-built glossary override layer, by calling the same
chunk_text/translate_chunks functions ai_backend_server.py's own
/_debug_raw_translate route uses (glossary bypassed) - done in-process
(no HTTP server needed) so this also loads the models itself.

Usage:  python evaluate_raw_model_only.py
"""
import unicodedata

from sacrebleu.metrics import CHRF

import ai_backend_server as backend
from evaluate_translation_accuracy import TEST_SET


def raw_translate(text, source_lang, target_lang):
    text = backend.normalize_text(text)
    chunks = backend.chunk_text(text) or [text]
    translated = backend.translate_chunks(chunks, source_lang, target_lang)
    return ' '.join(translated)


def main():
    chrf = CHRF()
    scores = []

    for source_lang, target_lang, text, reference in TEST_SET:
        references = [reference] if isinstance(reference, str) else reference
        hypothesis = raw_translate(text, source_lang, target_lang)
        score = chrf.sentence_score(
            unicodedata.normalize('NFC', hypothesis),
            [unicodedata.normalize('NFC', r) for r in references],
        ).score
        scores.append(score)
        print(f"[{source_lang}->{target_lang}] {text}")
        print(f"  expected: {' | '.join(references)}")
        print(f"  raw model: {hypothesis}  (chrF {score:.1f})")
        print()

    average = sum(scores) / len(scores)
    print(f"Average chrF, RAW model (no glossary) across {len(scores)} cases: {average:.1f}/100")


if __name__ == "__main__":
    main()
