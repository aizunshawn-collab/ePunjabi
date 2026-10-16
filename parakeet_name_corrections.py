"""
Narrow, curated post-decode fix for a verified Parakeet-unified-en-0.6b
(RNNT, ONNX) ASR mis-transcription: the name "Kaur" is sometimes rendered as
"Kor" / "Kore" / "Core" specifically when it is the LAST word of the
utterance.

Evidence (real local verification, since the production int8 model can't run
on this dev machine - see repo memory): using the non-quantized fp32 variant
of the same model (eschmidbauer/parakeet-unified-en-0.6b-onnx), "Kaur" was
transcribed CORRECTLY in every mid-sentence position tested (e.g. "Jaspreet
Kaur are going...", "Kaur called Manpreet...", "Kaur Gill lives..."), but was
misheard in several sentence-final cases:
    "My name is Kaur."            -> "My name is Kor"      (reproduced twice)
    "What is your name, Kaur?"    -> "What is your name, Kor?"
    "Please ask for Harpreet Kaur." -> "Please ask for Harpreet Core."
...while "Her surname is Kaur." was transcribed correctly, so this is not a
100%-reproducible rule, just a known failure mode worth correcting when it
does occur. False-positive check: ordinary sentence-final uses of the real
word "core" (e.g. "I ate the whole apple core.", "He felt bored after his
chore.", "The ship reached the shore.") were all transcribed correctly and
are NOT affected by this fix, because the "Core" correction below only
triggers when "Core" is immediately preceded by a known Indian/Punjabi given
name (never a bare/common-noun "core").

Same pattern as punjabi_name_corrections.py: plain string replacement on
already-decoded ASR output only, silent no-op if the exact pattern isn't
found - never touches model input, never introduces a new failure mode.
"""
import re

# Not real English words (aside from "core", handled separately below), so
# matching these at the very end of the transcript regardless of what
# precedes them is low-risk.
_BARE_WRONG_ENDINGS = {"kor", "kore"}

# "core" IS a common English word (e.g. "apple core"), so it's only
# corrected when immediately preceded by one of these known Indian/Punjabi
# given names - narrowing the fix to the exact "Firstname Core" ->
# "Firstname Kaur" pattern actually observed, never a bare/common-noun use.
_KNOWN_FIRST_NAMES = {
    "harpreet", "gurpreet", "jaspreet", "manpreet", "rajinder", "simran",
    "amandeep", "navjot", "baljeet", "hardeep", "harpal", "jagdeep",
    "sukhdeep", "ravinder", "parminder", "satinder",
}

_LAST_WORD_RE = re.compile(r"([A-Za-z]+)([.?!]*)$")


def correct_kaur_mishearing(transcript):
    """Fix a sentence-final 'Kaur' misheard as 'Kor' / 'Kore' / 'Core'."""
    if not transcript:
        return transcript

    match = _LAST_WORD_RE.search(transcript)
    if not match:
        return transcript

    last_word, trailing_punct = match.group(1), match.group(2)
    lower_last = last_word.lower()

    should_fix = lower_last in _BARE_WRONG_ENDINGS
    if not should_fix and lower_last == "core":
        words_before = transcript[:match.start()].split()
        if words_before and words_before[-1].strip(".,!?").lower() in _KNOWN_FIRST_NAMES:
            should_fix = True

    if not should_fix:
        return transcript

    return transcript[:match.start(1)] + "Kaur" + trailing_punct
