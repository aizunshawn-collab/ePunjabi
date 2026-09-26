"""
Cross-checks the new travel/medical/emergency GLOSSARY entries (written by
hand, not yet reviewed by a native Punjabi speaker) against IndicTrans2's own
independent translation, using the /_debug_raw_translate endpoint which
bypasses the glossary. High agreement (chrF) is corroborating evidence the
glossary entry is reasonable; low agreement flags it for manual review.

Requires ai_backend_server.py to be running locally with the temporary
/_debug_raw_translate route.
Usage: python verify_glossary_translations.py
"""
import requests
from sacrebleu.metrics import CHRF

BACKEND_URL = "http://localhost:5000"

# (english, glossary_punjabi) pairs added in the travel/medical/emergency round.
ENTRIES = [
    ("Where is the train station?", "ਰੇਲਵੇ ਸਟੇਸ਼ਨ ਕਿੱਥੇ ਹੈ?"),
    ("How much is the ticket?", "ਟਿਕਟ ਕਿੰਨੇ ਦੀ ਹੈ?"),
    ("I need a taxi", "ਮੈਨੂੰ ਟੈਕਸੀ ਦੀ ਲੋੜ ਹੈ"),
    ("Turn left", "ਖੱਬੇ ਮੁੜੋ"),
    ("Turn right", "ਸੱਜੇ ਮੁੜੋ"),
    ("Go straight", "ਸਿੱਧਾ ਜਾਓ"),
    ("Stop here", "ਇੱਥੇ ਰੁਕੋ"),
    ("I need a doctor", "ਮੈਨੂੰ ਡਾਕਟਰ ਚਾਹੀਦਾ ਹੈ"),
    ("Call an ambulance", "ਐਂਬੂਲੈਂਸ ਬੁਲਾਓ"),
    ("Where is the hospital?", "ਹਸਪਤਾਲ ਕਿੱਥੇ ਹੈ?"),
    ("I am not feeling well", "ਮੇਰੀ ਤਬੀਅਤ ਠੀਕ ਨਹੀਂ ਹੈ"),
    ("I have a headache", "ਮੇਰੇ ਸਿਰ ਵਿੱਚ ਦਰਦ ਹੈ"),
    ("I have a fever", "ਮੈਨੂੰ ਬੁਖਾਰ ਹੈ"),
    ("Help me, please", "ਕਿਰਪਾ ਕਰਕੇ ਮੇਰੀ ਮਦਦ ਕਰੋ"),
    ("Call the police", "ਪੁਲਿਸ ਨੂੰ ਬੁਲਾਓ"),
    ("This is an emergency", "ਇਹ ਇੱਕ ਐਮਰਜੈਂਸੀ ਹੈ"),
    ("I need help", "ਮੈਨੂੰ ਮਦਦ ਚਾਹੀਦੀ ਹੈ"),
    ("I am lost", "ਮੈਂ ਗੁਆਚ ਗਿਆ ਹਾਂ"),
]

REVIEW_THRESHOLD = 60.0  # chrF below this in EITHER direction gets flagged


def raw_translate(text, source_lang, target_lang):
    response = requests.post(
        f"{BACKEND_URL}/_debug_raw_translate",
        json={"text": text, "source_lang": source_lang, "target_lang": target_lang},
        timeout=90,
    )
    response.raise_for_status()
    return response.json()["translation"]


def main():
    chrf = CHRF()
    flagged = []

    print(f"Cross-checking {len(ENTRIES)} glossary entries against the raw model...\n")
    for english, punjabi in ENTRIES:
        model_pa = raw_translate(english, "en", "pa")
        model_en = raw_translate(punjabi, "pa", "en")

        fwd_score = chrf.sentence_score(model_pa, [punjabi]).score
        back_score = chrf.sentence_score(model_en, [english]).score

        status = "OK"
        if fwd_score < REVIEW_THRESHOLD or back_score < REVIEW_THRESHOLD:
            status = "REVIEW"
            flagged.append((english, punjabi, model_pa, model_en, fwd_score, back_score))

        print(f"[{status}] \"{english}\" <-> \"{punjabi}\"")
        print(f"  model en->pa: {model_pa}   (chrF vs glossary: {fwd_score:.1f})")
        print(f"  model pa->en: {model_en}   (chrF vs original: {back_score:.1f})")
        print()

    print("=" * 60)
    if not flagged:
        print("All entries corroborated by the model - no review needed.")
    else:
        print(f"{len(flagged)} entries flagged for manual/native-speaker review:")
        for english, punjabi, model_pa, model_en, fwd_score, back_score in flagged:
            print(f"  - \"{english}\" / \"{punjabi}\" (fwd chrF {fwd_score:.1f}, back chrF {back_score:.1f})")


if __name__ == "__main__":
    main()
