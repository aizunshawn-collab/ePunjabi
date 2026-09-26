"""
Side-by-side accuracy comparison: our self-hosted IndicTrans2 backend vs. the
real Google Translate service, scored with chrF against the same references.

Uses Google's public "gtx" web-client endpoint (no API key/billing needed -
this is the same endpoint translate.google.com itself uses, just called
directly for evaluation purposes here).

Requires ai_backend_server.py to be running locally (default http://localhost:5000).
Usage:  python compare_with_google_translate.py
"""
import time
import unicodedata

import requests
from sacrebleu.metrics import CHRF

from evaluate_translation_accuracy import BACKEND_URL, TEST_SET

GOOGLE_URL = "https://translate.googleapis.com/translate_a/single"


def our_translate(text, source_lang, target_lang):
    response = requests.post(
        f"{BACKEND_URL}/translate",
        json={"text": text, "source_lang": source_lang, "target_lang": target_lang, "verify": False},
        timeout=90,
    )
    response.raise_for_status()
    return response.json().get("translation", "")


def google_translate(text, source_lang, target_lang):
    response = requests.get(
        GOOGLE_URL,
        params={"client": "gtx", "sl": source_lang, "tl": target_lang, "dt": "t", "q": text},
        timeout=30,
    )
    response.raise_for_status()
    data = response.json()
    return "".join(segment[0] for segment in data[0] if segment[0])


def main():
    chrf = CHRF()
    ours_scores = []
    google_scores = []
    ours_wins = 0
    google_wins = 0
    ties = 0

    print(f"Comparing {len(TEST_SET)} cases: our IndicTrans2 backend vs. Google Translate\n")
    for source_lang, target_lang, text, reference in TEST_SET:
        references = [reference] if isinstance(reference, str) else reference
        normalized_references = [unicodedata.normalize('NFC', r) for r in references]

        ours_hypothesis = our_translate(text, source_lang, target_lang)
        try:
            google_hypothesis = google_translate(text, source_lang, target_lang)
        except Exception as error:
            print(f"  Google Translate request failed for '{text}': {error}")
            continue
        # Google's free endpoint is rate-limited - avoid tripping it.
        time.sleep(0.5)

        ours_score = chrf.sentence_score(
            unicodedata.normalize('NFC', ours_hypothesis), normalized_references
        ).score
        google_score = chrf.sentence_score(
            unicodedata.normalize('NFC', google_hypothesis), normalized_references
        ).score
        ours_scores.append(ours_score)
        google_scores.append(google_score)

        if ours_score > google_score + 0.5:
            ours_wins += 1
            verdict = "OURS wins"
        elif google_score > ours_score + 0.5:
            google_wins += 1
            verdict = "GOOGLE wins"
        else:
            ties += 1
            verdict = "tie"

        print(f"[{source_lang}->{target_lang}] {text}")
        print(f"  expected: {' | '.join(references)}")
        print(f"  ours:     {ours_hypothesis}  (chrF {ours_score:.1f})")
        print(f"  google:   {google_hypothesis}  (chrF {google_score:.1f})")
        print(f"  -> {verdict}")
        print()

    ours_average = sum(ours_scores) / len(ours_scores) if ours_scores else 0.0
    google_average = sum(google_scores) / len(google_scores) if google_scores else 0.0

    print("=" * 60)
    print(f"Our backend (IndicTrans2)  average chrF: {ours_average:.1f}/100")
    print(f"Google Translate           average chrF: {google_average:.1f}/100")
    print(f"Case-by-case: ours won {ours_wins}, Google won {google_wins}, tied {ties} (of {len(ours_scores)})")


if __name__ == "__main__":
    main()
