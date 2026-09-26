"""
Side-by-side accuracy comparison: our self-hosted IndicTrans2 backend vs.
Sarvam AI's translation API (Mayura v1), scored with chrF against the same
references used in evaluate_translation_accuracy.py's TEST_SET.

Sarvam's API requires a free account + API subscription key (unlike Google's
keyless "gtx" endpoint) - sign up at https://dashboard.sarvam.ai, create a key,
then set it as an environment variable before running this script:

    $env:SARVAM_API_KEY = "your-key-here"
    python compare_with_sarvam.py

New accounts get 100 INR in free credits, which comfortably covers this test
(Sarvam Translate is billed per 10K characters, this test set is a few
thousand characters total).

Requires ai_backend_server.py to be running locally (default http://localhost:5000).
"""
import os
import time
import unicodedata

import requests
from sacrebleu.metrics import CHRF

from evaluate_translation_accuracy import BACKEND_URL, TEST_SET

SARVAM_URL = "https://api.sarvam.ai/translate"
SARVAM_API_KEY = os.environ.get("SARVAM_API_KEY", "")

# evaluate_translation_accuracy.py uses plain 'en'/'pa' codes; Sarvam wants BCP-47 style.
_LANG_CODE_MAP = {"en": "en-IN", "pa": "pa-IN"}


def our_translate(text, source_lang, target_lang):
    response = requests.post(
        f"{BACKEND_URL}/translate",
        json={"text": text, "source_lang": source_lang, "target_lang": target_lang, "verify": False},
        timeout=90,
    )
    response.raise_for_status()
    return response.json().get("translation", "")


def sarvam_translate(text, source_lang, target_lang):
    response = requests.post(
        SARVAM_URL,
        headers={"api-subscription-key": SARVAM_API_KEY, "Content-Type": "application/json"},
        json={
            "input": text,
            "source_language_code": _LANG_CODE_MAP[source_lang],
            "target_language_code": _LANG_CODE_MAP[target_lang],
        },
        timeout=30,
    )
    response.raise_for_status()
    return response.json().get("translated_text", "")


def main():
    if not SARVAM_API_KEY:
        raise SystemExit(
            "SARVAM_API_KEY environment variable is not set. Get a free API key at "
            "https://dashboard.sarvam.ai and set it before running this script."
        )

    chrf = CHRF()
    ours_scores = []
    sarvam_scores = []
    ours_wins = 0
    sarvam_wins = 0
    ties = 0

    print(f"Comparing {len(TEST_SET)} cases: our IndicTrans2 backend vs. Sarvam AI (Mayura v1)\n")
    for source_lang, target_lang, text, reference in TEST_SET:
        references = [reference] if isinstance(reference, str) else reference
        normalized_references = [unicodedata.normalize('NFC', r) for r in references]

        ours_hypothesis = our_translate(text, source_lang, target_lang)
        try:
            sarvam_hypothesis = sarvam_translate(text, source_lang, target_lang)
        except Exception as error:
            print(f"  Sarvam request failed for '{text}': {error}")
            continue
        # Be polite to the rate limit (60 req/min on the Starter plan).
        time.sleep(1.1)

        ours_score = chrf.sentence_score(
            unicodedata.normalize('NFC', ours_hypothesis), normalized_references
        ).score
        sarvam_score = chrf.sentence_score(
            unicodedata.normalize('NFC', sarvam_hypothesis), normalized_references
        ).score
        ours_scores.append(ours_score)
        sarvam_scores.append(sarvam_score)

        if ours_score > sarvam_score + 0.5:
            ours_wins += 1
            verdict = "OURS wins"
        elif sarvam_score > ours_score + 0.5:
            sarvam_wins += 1
            verdict = "SARVAM wins"
        else:
            ties += 1
            verdict = "tie"

        print(f"[{source_lang}->{target_lang}] {text}")
        print(f"  expected: {' | '.join(references)}")
        print(f"  ours:     {ours_hypothesis}  (chrF {ours_score:.1f})")
        print(f"  sarvam:   {sarvam_hypothesis}  (chrF {sarvam_score:.1f})")
        print(f"  -> {verdict}")
        print()

    ours_average = sum(ours_scores) / len(ours_scores) if ours_scores else 0.0
    sarvam_average = sum(sarvam_scores) / len(sarvam_scores) if sarvam_scores else 0.0

    print("=" * 60)
    print(f"Our backend (IndicTrans2)  average chrF: {ours_average:.1f}/100")
    print(f"Sarvam AI (Mayura v1)      average chrF: {sarvam_average:.1f}/100")
    print(f"Case-by-case: ours won {ours_wins}, Sarvam won {sarvam_wins}, tied {ties} (of {len(ours_scores)})")


if __name__ == "__main__":
    main()
