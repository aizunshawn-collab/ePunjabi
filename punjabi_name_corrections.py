"""Narrow, curated post-translation correction for a specific IndicTrans2
English->Punjabi proper-noun hallucination: a small number of Punjabi/Sikh
first names get silently SUBSTITUTED with a different, more common real name
when a surname follows them (e.g. "Harpreet Singh" -> "Gurpreet Singh"),
even though the exact same name translates correctly on its own.

Verified this session (see repo memory for full reproduction + methodology):
  - "Harpreet"            -> "ਹਰਪ੍ਰੀਤ" (correct, round-trips cleanly)
  - "Harpreet Singh"      -> "ਗੁਰਪ੍ਰੀਤ ਸਿੰਘ" (WRONG - substituted with Gurpreet)
  - "Hardeep"             -> "ਹਾਰਦੀਪ" (correct, round-trips cleanly)
  - "Hardeep Singh"       -> "ਗੁਰਦੀਪ ਸਿੰਘ" (WRONG - substituted with Gurdeep)
  - "Harpal"              -> "ਹਰਪਾਲ" (correct, round-trips cleanly)
  - "Harpal Singh"        -> "ਸਤਪਾਲ ਸਿੰਘ" (WRONG - substituted with Satpal)

The correct Gurmukhi spelling for each name below was NOT hand-typed/guessed
- it was obtained from IndicTrans2's own output when translating the bare
  name in isolation (a context where it reproducibly translates correctly),
  then confirmed via a pa->en back-translation round-trip matching the
  original English spelling exactly.

DELIBERATELY NARROW BY DESIGN: this project previously attempted a more
general placeholder-substitution mechanism to protect proper nouns (and
alphanumeric codes) during translation, and it was FULLY REVERTED after the
placeholder tokens themselves got corrupted/not preserved by the model (see
repo memory). To avoid repeating that failure, this module does NOT feed
anything into the model - it only does plain string replacement on the
ALREADY-DECODED Gurmukhi output, and only ever touches a wrong substitution
that exactly matches one of the curated entries below. If the real wrong
output doesn't exactly match (e.g. a different beam-search path produces some
other wrong spelling), this is a silent no-op - same as if the fix didn't
exist, never a new failure mode.

This is an intentionally small, incomplete list covering only the specific
cases reproduced and verified this session - not a general proper-noun
hallucination fix (no safe, general fix was found to be derivable; see repo
memory for the full investigation, including other failure modes - such as
a name being dropped entirely - that were investigated but left unaddressed
because no safe correction could be designed for them).
"""

import re

# source English first name (lowercase) -> (correct Gurmukhi spelling,
# [known wrong Gurmukhi substitutions it gets replaced with])
_NAME_SUBSTITUTION_FIXES = {
    'harpreet': ('ਹਰਪ੍ਰੀਤ', ['ਗੁਰਪ੍ਰੀਤ']),
    'hardeep': ('ਹਾਰਦੀਪ', ['ਗੁਰਦੀਪ']),
    'harpal': ('ਹਰਪਾਲ', ['ਸਤਪਾਲ']),
}

_ENGLISH_WORD_RE = re.compile(r"[A-Za-z]+")


def fix_name_substitutions(source_text, translated_text):
    """Correct known English->Punjabi name substitutions.

    If `source_text` (English) contains one of the curated names above, and
    `translated_text` (Gurmukhi) contains that specific name's known wrong
    substitution (and NOT already the correct spelling), replace the wrong
    spelling with the verified correct one. Otherwise returns the translation
    unchanged.
    """
    if not source_text or not translated_text:
        return translated_text

    source_names = {w.lower() for w in _ENGLISH_WORD_RE.findall(source_text)}
    result = translated_text
    for name in source_names:
        fix = _NAME_SUBSTITUTION_FIXES.get(name)
        if not fix:
            continue
        correct, wrong_variants = fix
        if correct in result:
            continue  # already correct, nothing to do
        for wrong in wrong_variants:
            if wrong in result:
                result = result.replace(wrong, correct)
    return result
