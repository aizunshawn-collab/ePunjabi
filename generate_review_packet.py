"""
Generates NATIVE_SPEAKER_REVIEW_PACKET.md by calling the live backend for a
curated set of phrases: everyday conversational coverage plus every known
trouble spot (see repo memory), so a native Punjabi speaker's review session
is focused instead of open-ended.

Requires ai_backend_server.py running locally (default http://localhost:5000).
Usage:  python generate_review_packet.py
"""
import requests

BACKEND_URL = "http://localhost:5000"

# (section_title, [(source_lang, target_lang, text, note_for_reviewer), ...])
SECTIONS = [
    ("A. Everyday conversation (baseline sanity check)", [
        ("en", "pa", "Good morning, how did you sleep?", None),
        ("en", "pa", "Can you please help me find my way to the train station?", None),
        ("en", "pa", "I would like to book a room for two nights.", None),
        ("pa", "en", "ਅੱਜ ਮੌਸਮ ਬਹੁਤ ਵਧੀਆ ਹੈ, ਇਸ ਲਈ ਚਲੋ ਸੈਰ ਲਈ ਚੱਲੀਏ।", None),
        ("pa", "en", "ਮੌਸਮ ਵਿਭਾਗ ਨੇ ਕੱਲ੍ਹ ਮੀਂਹ ਦੀ ਭਵਿੱਖਬਾਣੀ ਕੀਤੀ ਹੈ।", None),
    ]),
    ("B. Travel / medical / emergency (should be exact - glossary-locked)", [
        ("en", "pa", "I need a doctor, call an ambulance please.", None),
        ("en", "pa", "I lost my passport at the airport.", None),
        ("pa", "en", "ਮੈਨੂੰ ਬੁਖਾਰ ਹੈ ਅਤੇ ਸਿਰ ਬਹੁਤ ਦਰਦ ਕਰਦਾ ਹੈ।", None),
    ]),
    ("C. KNOWN BUG - alphanumeric codes (expected WRONG, please describe how bad)", [
        ("en", "pa", "Your flight number is AI 202, and the booking reference is ABX123.",
         "Expected bug: codes get phonetically spelled out in Gurmukhi instead of kept as-is."),
        ("en", "pa", "Please quote invoice number INV-4567 when you call.",
         "Expected bug: same as above."),
    ]),
    ("D. KNOWN BUG - proper noun handling (mixed: one wrong, one right)", [
        ("en", "pa", "My name is Harpreet Singh and I live in Amritsar.",
         "Expected bug: model has previously substituted a different name (Gurpreet) here."),
        ("en", "pa", "My name is Jaspreet Kaur and I live in Ludhiana.",
         "Expected to be correct - included as a working comparison case."),
    ]),
    ("E. KNOWN EDGE CASE - narrow admission/ਦਾਖਲਾ drop", [
        ("pa", "en", "ਮੇਰੇ ਬੇਟੇ ਦਾ ਸਕੂਲ ਵਿੱਚ ਦਾਖਲਾ ਅਗਲੇ ਹਫ਼ਤੇ ਹੋਵੇਗਾ।",
         "Expected bug: 'admission/enrollment' meaning has been dropped in this exact phrasing before."),
        ("pa", "en", "ਕਾਲਜ ਵਿੱਚ ਦਾਖਲੇ ਦੀ ਆਖਰੀ ਤਾਰੀਖ ਕੀ ਹੈ?",
         "Expected to be correct - included as a working comparison case."),
    ]),
    ("F. Long/complex sentence (recently fixed chunking bug - please confirm still holds)", [
        ("en", "pa",
         "In order to apply for the small business loan, you will need to submit proof of "
         "residence, a valid business license, and a detailed plan describing how you intend "
         "to use the funds over the next twelve months.",
         "Previously this sentence dropped the first two conditions entirely when split into chunks."),
        ("pa", "en",
         "ਮੇਰੀ ਮਾਂ ਹਰ ਰੋਜ਼ ਸਵੇਰੇ ਜਲਦੀ ਉੱਠਦੀ ਹੈ, ਨਾਸ਼ਤਾ ਬਣਾਉਂਦੀ ਹੈ, ਅਤੇ ਫਿਰ ਸ਼ਾਮ ਨੂੰ ਕੰਮ ਤੋਂ ਬਾਅਦ "
         "ਰਾਤ ਦੀਆਂ ਕਲਾਸਾਂ ਵਿੱਚ ਪੜ੍ਹਨ ਜਾਂਦੀ ਹੈ ਕਿਉਂਕਿ ਉਹ ਨਰਸ ਬਣਨਾ ਚਾਹੁੰਦੀ ਹੈ।",
         "Previously this sentence lost the subject's gender ('she' became 'he') partway through."),
    ]),
    ("G. Code-switching (Punjabi-English mixed - confirmed working, spot-check)", [
        ("pa", "en", "ਮੈਨੂੰ ਅੱਜ office ਵਿੱਚ ਦੇਰ ਹੋ ਜਾਵੇਗੀ।", None),
        ("pa", "en", "ਉਹ ਮੇਰਾ best friend ਹੈ।", None),
    ]),
]

REVIEWER_QUESTIONS = """
## What to check for each pair
For every row below, please mark:
1. **Correct meaning?** (yes / no / partially - describe what's wrong)
2. **Natural phrasing?** (would a native speaker actually say it this way, or does it sound
   stiff/translated/robotic?)
3. **Tone/formality** appropriate for the context (casual vs. respectful/formal)?
4. **Gender agreement** correct throughout (especially section F)?

Sections C, D, and E are KNOWN issues already found by automated testing - your read on
*how bad* they are (embarrassing vs. minor vs. actually fine in context) is the most
valuable feedback, since automated tests can't judge that.
"""


def main():
    lines = [
        "# Native Speaker Review Packet",
        "",
        "Generated automatically from the live backend. For each row, English/Punjabi source",
        "is on top, the app's actual current translation is below it.",
        "",
        REVIEWER_QUESTIONS,
    ]

    for title, cases in SECTIONS:
        lines.append(f"## {title}")
        lines.append("")
        for i, (src, tgt, text, note) in enumerate(cases, 1):
            resp = requests.post(
                f"{BACKEND_URL}/translate",
                json={"text": text, "source_lang": src, "target_lang": tgt, "verify": True},
                timeout=60,
            )
            resp.raise_for_status()
            data = resp.json()
            translation = data.get("translation", "<no translation returned>")
            confidence = data.get("confidence")

            lines.append(f"**{i}. ({src} -> {tgt})**")
            lines.append(f"- Source: {text}")
            lines.append(f"- Translation: {translation}")
            if confidence is not None:
                lines.append(f"- App's own confidence score: {confidence:.2f}")
            if note:
                lines.append(f"- ⚠️ Note: {note}")
            lines.append("")
        lines.append("")

    with open("NATIVE_SPEAKER_REVIEW_PACKET.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print("Wrote NATIVE_SPEAKER_REVIEW_PACKET.md")


if __name__ == "__main__":
    main()
