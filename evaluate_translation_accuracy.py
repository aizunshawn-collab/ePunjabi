"""
Measures real translation accuracy against a small curated reference set,
instead of relying on impression/confidence scores alone.

Requires ai_backend_server.py to be running locally (default http://localhost:5000).
Usage:  python evaluate_translation_accuracy.py
"""
import requests
import unicodedata
from sacrebleu.metrics import CHRF

BACKEND_URL = "http://localhost:5000"

# (source_lang, target_lang, input_text, reference_translation_or_list_of_acceptable_translations)
TEST_SET = [
    # Glossary phrases - should score at/near 100 (exact overrides).
    ("en", "pa", "Hello", "ਸਤ ਸ੍ਰੀ ਅਕਾਲ"),
    ("en", "pa", "Thank you very much", "ਬਹੁਤ ਧੰਨਵਾਦ"),
    ("pa", "en", "ਬਾਥਰੂਮ ਕਿੱਥੇ ਹੈ?", "Where is the bathroom?"),
    ("pa", "en", "ਮੈਨੂੰ ਸਮਝ ਨਹੀਂ ਆਈ", "I don't understand"),
    # General sentences - not in the glossary, exercise the model itself.
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
    # Everyday conversational glossary phrases (en->pa).
    ("en", "pa", "Good morning", "ਸ਼ੁਭ ਸਵੇਰ"),
    ("en", "pa", "Nice to meet you", "ਤੁਹਾਨੂੰ ਮਿਲ ਕੇ ਖੁਸ਼ੀ ਹੋਈ"),
    ("en", "pa", "How much is the ticket?", "ਟਿਕਟ ਕਿੰਨੇ ਦੀ ਹੈ?"),
    ("en", "pa", "Turn left", "ਖੱਬੇ ਮੁੜੋ"),
    # Everyday conversational glossary phrases (pa->en).
    ("pa", "en", "ਸ਼ੁਭ ਸਵੇਰ", "Good morning"),
    ("pa", "en", "ਅਲਵਿਦਾ", "Goodbye"),
    ("pa", "en", "ਤੁਸੀਂ ਕਿਵੇਂ ਹੋ?", "How are you?"),
    ("pa", "en", "ਹਸਪਤਾਲ ਕਿੱਥੇ ਹੈ?", "Where is the hospital?"),
    # Travel/medical/emergency glossary phrases (en->pa).
    ("en", "pa", "Where is the train station?", "ਰੇਲਵੇ ਸਟੇਸ਼ਨ ਕਿੱਥੇ ਹੈ?"),
    ("en", "pa", "I need a doctor", "ਮੈਨੂੰ ਡਾਕਟਰ ਚਾਹੀਦਾ ਹੈ"),
    ("en", "pa", "Call an ambulance", "ਐਂਬੂਲੈਂਸ ਬੁਲਾਓ"),
    ("en", "pa", "This is an emergency", "ਇਹ ਇੱਕ ਐਮਰਜੈਂਸੀ ਹੈ"),
    # Travel/medical/emergency glossary phrases (pa->en).
    ("pa", "en", "ਮੈਨੂੰ ਡਾਕਟਰ ਚਾਹੀਦਾ ਹੈ", "I need a doctor"),
    ("pa", "en", "ਐਂਬੂਲੈਂਸ ਬੁਲਾਓ", "Call an ambulance"),
    ("pa", "en", "ਪੁਲਿਸ ਨੂੰ ਬੁਲਾਓ", "Call the police"),
    ("pa", "en", "ਮੈਨੂੰ ਬੁਖਾਰ ਹੈ", "I have a fever"),
    # Numbers / days glossary phrases (both directions).
    ("en", "pa", "Monday", "ਸੋਮਵਾਰ"),
    ("en", "pa", "Seven", "ਸੱਤ"),
    ("en", "pa", "Water", "ਪਾਣੀ"),
    ("pa", "en", "ਸ਼ੁੱਕਰਵਾਰ", "Friday"),
    ("pa", "en", "ਦਸ", "Ten"),
    ("pa", "en", "ਖਾਣਾ", "Food"),
    # More general sentences - exercise model fluency, not glossary hits.
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

# HELD-OUT set written independently of TEST_SET above and of the glossary -
# none of these sentences (or close paraphrases) were used while building
# GLOSSARY/GENDERED_GLOSSARY, so a good score here is a genuine signal about
# the model's general fluency rather than the glossary matching itself.
# Covers scenarios TEST_SET doesn't: banking, phone calls, small talk,
# complaints, multi-clause sentences, questions with negation.
HELD_OUT_SET = [
    ("en", "pa", "I would like to open a savings account at this bank.", None),
    ("en", "pa", "Could you please speak a little slower? I don't understand fully.", None),
    ("en", "pa", "My internet connection has not been working since yesterday.", None),
    ("en", "pa", "Is there a pharmacy nearby that's open right now?", None),
    ("en", "pa", "I have been waiting for the bus for almost an hour.", None),
    ("en", "pa", "Please let me know if the meeting time changes.", None),
    ("en", "pa", "He didn't come to work today because he wasn't feeling well.", None),
    ("en", "pa", "What time does the next flight to Delhi leave?", None),
    ("en", "pa", "I think there is a mistake in this bill, can you check again?", None),
    ("en", "pa", "My grandmother tells the best stories about her village.", None),
    ("en", "pa", "We should leave early tomorrow to avoid the traffic.", None),
    ("en", "pa", "Can you recommend a good restaurant near the city center?", None),
    ("en", "pa", "I am sorry for the delay, it won't happen again.", None),
    ("en", "pa", "The landlord increased the rent again this year.", None),
    ("en", "pa", "She is studying to become a nurse.", None),
    ("pa", "en", "ਮੈਨੂੰ ਕੱਲ੍ਹ ਸਵੇਰੇ ਜਲਦੀ ਉੱਠਣਾ ਪਏਗਾ।", None),
    ("pa", "en", "ਕੀ ਤੁਸੀਂ ਇਹ ਫਾਰਮ ਭਰਨ ਵਿੱਚ ਮੇਰੀ ਮਦਦ ਕਰ ਸਕਦੇ ਹੋ?", None),
    ("pa", "en", "ਮੇਰਾ ਭਰਾ ਵਿਦੇਸ਼ ਵਿੱਚ ਕੰਮ ਕਰਦਾ ਹੈ।", None),
    ("pa", "en", "ਇਹ ਜੁੱਤੀ ਮੈਨੂੰ ਥੋੜੀ ਛੋਟੀ ਹੈ, ਕੀ ਵੱਡਾ ਸਾਈਜ਼ ਹੈ?", None),
    ("pa", "en", "ਬਿਜਲੀ ਕਦੋਂ ਦੀ ਗਈ ਹੋਈ ਹੈ?", None),
    # Workplace / immigration-legal / landlord-tenant / school - daily-life
    # categories not covered above.
    ("en", "pa", "I need to request a day off next week for a family event.", None),
    ("en", "pa", "My manager asked me to finish the report by Friday.", None),
    ("en", "pa", "I need to renew my visa before it expires next month.", None),
    ("en", "pa", "Can you help me fill out this immigration form?", None),
    ("en", "pa", "The landlord has not fixed the heating in my apartment yet.", None),
    ("en", "pa", "My daughter's teacher wants to meet us about her homework.", None),
    ("en", "pa", "I was let go from my job last week and I'm looking for new work.", None),
    ("en", "pa", "The bank froze my account because of a suspicious transaction.", None),
    ("pa", "en", "ਮੈਨੂੰ ਆਪਣੀ ਤਨਖਾਹ ਬਾਰੇ ਗੱਲ ਕਰਨੀ ਹੈ।", None),
    ("pa", "en", "ਮੇਰਾ ਵੀਜ਼ਾ ਅਗਲੇ ਮਹੀਨੇ ਖ਼ਤਮ ਹੋ ਰਿਹਾ ਹੈ।", None),
    ("pa", "en", "ਮਕਾਨ ਮਾਲਕ ਨੇ ਅਜੇ ਤੱਕ ਕਿਰਾਏ ਦੀ ਰਸੀਦ ਨਹੀਂ ਦਿੱਤੀ।", None),
    ("pa", "en", "ਮੇਰੇ ਬੇਟੇ ਦਾ ਸਕੂਲ ਵਿੱਚ ਦਾਖਲਾ ਅਗਲੇ ਹਫ਼ਤੇ ਹੋਵੇਗਾ।", None),
    # Punjabi-English code-switching - very common in real diaspora speech,
    # untested until now.
    ("pa", "en", "ਮੈਨੂੰ ਅੱਜ office ਵਿੱਚ ਬਹੁਤ ਕੰਮ ਸੀ।", None),
    ("pa", "en", "ਮੇਰਾ meeting ਕੱਲ੍ਹ ਸਵੇਰੇ ਹੈ।", None),
    ("pa", "en", "ਉਹ ਮੇਰਾ best friend ਹੈ।", None),
    ("en", "pa", "Yaar, this traffic is so annoying today.", None),
]


def translate(text, source_lang, target_lang):
    response = requests.post(
        f"{BACKEND_URL}/translate",
        json={"text": text, "source_lang": source_lang, "target_lang": target_lang, "verify": True},
        timeout=90,
    )
    response.raise_for_status()
    return response.json()


def run_scored_set():
    chrf = CHRF()
    scores = []

    print(f"Running {len(TEST_SET)} scored test cases against {BACKEND_URL} ...\n")
    for source_lang, target_lang, text, reference in TEST_SET:
        references = [reference] if isinstance(reference, str) else reference
        data = translate(text, source_lang, target_lang)
        hypothesis = data.get("translation", "")
        confidence = data.get("confidence")

        score = chrf.sentence_score(
            unicodedata.normalize('NFC', hypothesis),
            [unicodedata.normalize('NFC', r) for r in references],
        ).score
        scores.append(score)

        print(f"[{source_lang}->{target_lang}] {text}")
        print(f"  expected: {' | '.join(references)}")
        print(f"  got:      {hypothesis}")
        print(f"  chrF: {score:.1f}/100" + (f"   backend confidence: {confidence}" if confidence is not None else ""))
        print()

    average = sum(scores) / len(scores) if scores else 0.0
    print(f"Average chrF across {len(scores)} scored cases: {average:.1f}/100")
    print("(chrF is a character n-gram similarity score - 100 means an exact match;")
    print(" scores above ~50-60 generally indicate a fluent, meaning-preserving translation")
    print(" even when wording differs from the reference.)")
    return average


def run_held_out_set():
    # No hand-written reference translations here (see HELD_OUT_SET comment) -
    # chrF needs a reference, so this reports the backend's own round-trip
    # confidence instead and prints the back-translation for a human
    # (ideally a Punjabi speaker) to sanity-check meaning by eye.
    print(f"\nRunning {len(HELD_OUT_SET)} held-out (reference-free) cases against {BACKEND_URL} ...\n")
    confidences = []
    for source_lang, target_lang, text, _ in HELD_OUT_SET:
        data = translate(text, source_lang, target_lang)
        hypothesis = data.get("translation", "")
        confidence = data.get("confidence")
        back_translation = data.get("back_translation")
        if confidence is not None:
            confidences.append(confidence)

        flag = "  <-- LOW CONFIDENCE, review this one" if confidence is not None and confidence < 0.5 else ""
        print(f"[{source_lang}->{target_lang}] {text}")
        print(f"  got:            {hypothesis}")
        print(f"  back-translated: {back_translation}")
        print(f"  confidence: {confidence}{flag}")
        print()

    if confidences:
        avg_confidence = sum(confidences) / len(confidences)
        print(f"Average round-trip confidence across {len(confidences)} held-out cases: {avg_confidence:.2f}")


def main():
    run_scored_set()
    run_held_out_set()


if __name__ == "__main__":
    main()
