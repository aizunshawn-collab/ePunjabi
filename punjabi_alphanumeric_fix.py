"""Narrow, curated post-translation correction for IndicTrans2's English->Punjabi
mangling of short alphanumeric codes (flight/gate/seat codes, invoice/order
numbers, etc. - e.g. "Gate A12", "My seat is 14H.", "The code is B 99.").

Verified this session (see repo memory for the full reproduction + methodology,
collected by testing every letter A-Z glued/spaced against digits in several
sentence templates): a short (1-3 letter) run of UPPERCASE Latin letters
directly adjacent to a digit run is, depending on surrounding sentence
context, sometimes:
  - passed through unchanged (correct, e.g. "Gate B12" -> "ਗੇਟ B12"),
  - transliterated into Gurmukhi script (e.g. "Gate A12" -> "ਗੇਟ ਏ12", "W"
    consistently -> "ਡਬਲਯੂ" in every context tested), or
  - mistranslated as an entirely different WORD matching a plausible unit
    abbreviation (e.g. "My seat is 14H." -> "...14 ਘੰਟੇ..." [hours],
    "14L." -> "...14 ਲੀਟਰ..." [liters], "14M." -> "...14 ਮੀਟਰ..." [meters]).
Which of these happens is NOT a stable per-letter rule - the same letter
behaves differently depending on the sentence it appears in - so no static
letter -> Gurmukhi lookup table can reliably predict/reverse it up front.

DELIBERATELY NARROW BY DESIGN, same discipline as punjabi_name_corrections.py:
this project previously attempted a general placeholder-substitution
mechanism to protect codes during translation, and it was FULLY REVERTED
after the placeholder tokens themselves got corrupted by the model (see repo
memory). This module does NOT feed anything into the model. Instead it
detects the mangle AFTER translation by anchoring on the digit run, which was
never observed to be altered by the model in any test this session:
  1. Find short letter+digit (or digit+letter) codes in the ENGLISH source.
  2. Confirm that exact digit run appears exactly once in the Punjabi output
     (if it's missing or duplicated, the match is ambiguous - skip, no-op).
  3. Check whether the original Latin letters already sit directly next to
     that digit run in the output - if so, nothing is wrong, no-op.
  4. Otherwise, splice the verbatim original source token back in, replacing
     whatever token (Gurmukhi transliteration or mistranslated unit word)
     the model produced next to the digits.
Every branch that isn't an unambiguous match falls back to returning the
translation unchanged - never a new failure mode beyond "didn't fix it".

MITIGATED TRADEOFF: a handful of short uppercase letter runs directly
glued/spaced to digits are also legitimate unit abbreviations in English
("14H" hours, "14L" liters, "14M" meters, "5V" volts, "10KG" kilograms,
"500MB", etc.). Blindly reverting a correct unit translation back to the
literal code would be a regression, so before replacing, this module checks
two independent things: (1) whether the mangled fragment it's about to
overwrite IS one of a broad curated table of known Gurmukhi unit words
(_LETTER_UNIT_WORDS, covering time/length/mass/volume/electrical/
digital-storage/temperature units), and (2) whether a TRUE SEMANTIC PARSE of
the ENGLISH source (spaCy's statistical dependency parser - POS tags,
dependency relations, lemmas - not keyword matching) classifies the token's
grammatical role as a measurement rather than an identifier. See
`_semantic_intent()`: it looks at the actual sentence structure the token
plays in - is it the object of a measurement verb ("weighs 10KG"), an
npadvmod of an adjective ("14M tall"), part of a quantity-of-noun
construction ("5T of cargo", "2L bottle of milk"), vs. being/compounding with
an identifier noun ("Gate A12", "seat is 14H", "locker code is V12"). Only
when BOTH the output-side word match and the source-side semantic
classification agree it's a unit does the fix get skipped for that token.
Only H/L/M were directly reproduced as model mistranslation targets this
session; the rest of the Gurmukhi-word table is standard dictionary
translations included proactively. A token with no strong grammatical signal
either way still falls back to the original code-restoration behavior, so
this never introduces a new failure mode - only ever narrows how often the
restoration fires.
"""

import re

_CODE_TOKEN_RE = re.compile(
    r'\b([A-Z]{1,3})([ -]?)(\d{1,6})\b'
    r'|'
    r'\b(\d{1,6})([ -]?)([A-Z]{1,3})\b'
)

# Punctuation (Latin + Gurmukhi/Urdu danda marks) treated as a hard boundary
# when extending a replacement span outward from the matched digit run. Also
# treated as a separator to skip over (e.g. the model sometimes renders a
# multi-letter code as several dotted/spaced tokens, one per source letter -
# "AI" -> "ਏ. ਆਈ" - so all of them need to be swept up, not just the last).
_BOUNDARY_CHARS = ',.;:!?()[]{}"\'،۔।'


def _extend_left(text, idx, max_tokens):
    """Walk left from idx, optionally skipping a separator then capturing the
    word before it, up to `max_tokens` times - stopping early if a captured
    word contains a digit (never eat into another code) or there's nothing
    left to capture. The separator skip is a no-op when the mangled word sits
    directly glued to idx with no space/punctuation in between."""
    i = idx
    for _ in range(max_tokens):
        j = i
        while j > 0 and (text[j - 1].isspace() or text[j - 1] in _BOUNDARY_CHARS):
            j -= 1
        k = j
        while k > 0 and not text[k - 1].isspace() and text[k - 1] not in _BOUNDARY_CHARS:
            k -= 1
        if k == j or any(ch.isdigit() for ch in text[k:j]):
            break
        i = k
    return i


def _extend_right(text, idx, max_tokens):
    """Mirror of `_extend_left`, walking right from idx instead."""
    i = idx
    for _ in range(max_tokens):
        j = i
        while j < len(text) and (text[j].isspace() or text[j] in _BOUNDARY_CHARS):
            j += 1
        k = j
        while k < len(text) and not text[k].isspace() and text[k] not in _BOUNDARY_CHARS:
            k += 1
        if k == j or any(ch.isdigit() for ch in text[j:k]):
            break
        i = k
    return i


# Comprehensive curated table of standard SI/imperial/common abbreviations
# (1-3 uppercase letters, matching _CODE_TOKEN_RE) mapped to their known
# Gurmukhi unit-word translation(s). H/L/M confirmed directly this session as
# model mistranslation targets; the rest are standard dictionary Gurmukhi
# words for the same abbreviations, included proactively so the same guard
# covers them too if the model ever produces them in a measurement context.
# A letter can map to more than one plausible unit (e.g. "M" = meter or
# minute or month) - harmless, since the guard below also requires the
# source sentence to contain a supporting context clue.
_LETTER_UNIT_WORDS = {
    'H': ('ਘੰਟੇ', 'ਘੰਟਾ', 'ਘੰਟਿਆਂ'),                         # hour(s)
    'L': ('ਲੀਟਰ',),                                          # liter
    'M': ('ਮੀਟਰ', 'ਮਿੰਟ', 'ਮਹੀਨਾ', 'ਮਹੀਨੇ'),                  # meter / minute / month
    'S': ('ਸਕਿੰਟ',),                                         # second
    'D': ('ਦਿਨ',),                                           # day
    'Y': ('ਸਾਲ',),                                           # year
    'W': ('ਵਾਟ',),                                           # watt
    'V': ('ਵੋਲਟ',),                                          # volt
    'A': ('ਐਂਪੀਅਰ', 'ਐਂਪ'),                                   # amp(ere)
    'N': ('ਨਿਊਟਨ',),                                         # newton
    'J': ('ਜੂਲ',),                                           # joule
    'C': ('ਸੈਲਸੀਅਸ',),                                       # celsius
    'F': ('ਫਾਰਨਹੀਟ',),                                       # fahrenheit
    'K': ('ਕੈਲਵਿਨ', 'ਕਿਲੋਗ੍ਰਾਮ', 'ਕਿਲੋ'),                      # kelvin / kilogram
    'G': ('ਗ੍ਰਾਮ',),                                         # gram
    'T': ('ਟਨ',),                                            # ton
    'P': ('ਪਾਸਕਲ',),                                         # pascal
    'KM': ('ਕਿਲੋਮੀਟਰ',),                                     # kilometer
    'CM': ('ਸੈਂਟੀਮੀਟਰ',),                                    # centimeter
    'MM': ('ਮਿਲੀਮੀਟਰ',),                                     # millimeter
    'KG': ('ਕਿਲੋਗ੍ਰਾਮ', 'ਕਿਲੋ'),                              # kilogram
    'MG': ('ਮਿਲੀਗ੍ਰਾਮ',),                                    # milligram
    'ML': ('ਮਿਲੀਲੀਟਰ',),                                     # milliliter
    'LB': ('ਪੌਂਡ',),                                         # pound
    'OZ': ('ਔਂਸ',),                                          # ounce
    'FT': ('ਫੁੱਟ',),                                         # feet
    'IN': ('ਇੰਚ',),                                          # inch
    'YD': ('ਗਜ਼',),                                          # yard
    'MI': ('ਮੀਲ',),                                          # mile
    'HZ': ('ਹਰਟਜ਼',),                                        # hertz
    'KW': ('ਕਿਲੋਵਾਟ',),                                      # kilowatt
    'MA': ('ਮਿਲੀਐਂਪੀਅਰ',),                                   # milliamp
    'KB': ('ਕਿਲੋਬਾਈਟ',),                                     # kilobyte
    'MB': ('ਮੈਗਾਬਾਈਟ',),                                     # megabyte
    'GB': ('ਗੀਗਾਬਾਈਟ',),                                     # gigabyte
    'TB': ('ਟੈਰਾਬਾਈਟ',),                                     # terabyte
    'HP': ('ਹਾਰਸਪਾਵਰ',),                                     # horsepower
}

# Lazily-loaded spaCy English pipeline (POS tagger + dependency parser) -
# used only to analyze the ENGLISH SOURCE sentence's grammar, never fed
# anything related to the translation model itself.
_NLP = None


def _get_nlp():
    global _NLP
    if _NLP is None:
        import spacy
        _NLP = spacy.load('en_core_web_sm')
    return _NLP


# Verbs whose direct object is characteristically a measurement, not an
# identifier ("weighs 10KG", "takes 14H", "outputs 5V").
_MEASURE_VERBS = {
    'weigh', 'cost', 'take', 'last', 'hold', 'contain', 'measure', 'output',
    'consume', 'span', 'stretch', 'reach', 'travel', 'run', 'drive', 'charge',
    'pour', 'fill', 'wait',
}

# Nouns that characteristically hold/carry a quantity rather than name an
# identifier ("2L bottle", "5T of cargo", "10MB file").
_QUANTITY_NOUNS = {
    'bottle', 'tank', 'bag', 'box', 'cup', 'glass', 'can', 'jar', 'package',
    'bucket', 'barrel', 'container', 'cargo', 'load', 'amount', 'weight',
    'distance', 'height', 'length', 'speed', 'size', 'capacity', 'file',
    'wall', 'road', 'fuel', 'water', 'milk',
}

# Nouns that characteristically name an identifier/code, not a measurement
# ("Gate A12", "seat is 14H", "locker code is V12").
_IDENTIFIER_NOUNS = {
    'gate', 'seat', 'room', 'flight', 'code', 'number', 'order', 'invoice',
    'reference', 'platform', 'locker', 'apartment', 'flat', 'ticket', 'track',
    'bus', 'train', 'model', 'account', 'license', 'plate', 'door', 'unit',
    'suite', 'floor', 'lane', 'aisle', 'counter',
}


def _semantic_intent(source_text, token_start, token_end):
    """Classify a digit+letter token's intent as 'unit', 'code', or None (no
    strong signal) by looking at the grammatical role it plays in a real
    dependency parse of the English source - not by scanning for keywords
    anywhere in the sentence."""
    doc = _get_nlp()(source_text)
    span = doc.char_span(token_start, token_end, alignment_mode='expand')
    if span is None:
        return None

    def _is_identifier(tok):
        return tok.lemma_.lower() in _IDENTIFIER_NOUNS

    def _is_quantity_noun(tok):
        return tok.lemma_.lower() in _QUANTITY_NOUNS

    def _classify(anchor):
        head = anchor.head

        # "14M tall" - number acts as an adverbial modifier of an adjective
        if anchor.dep_ == 'npadvmod' and head.pos_ == 'ADJ':
            return 'unit'

        # "weighs 10KG" / "takes 14H" - token is the object of a measurement verb
        if head.pos_ == 'VERB' and head.lemma_.lower() in _MEASURE_VERBS:
            return 'unit'

        # "wait for 14H" - duration object of a measurement verb via a
        # preposition ("for"/"of") rather than a direct object
        if anchor.dep_ == 'pobj' and head.dep_ == 'prep' and head.lemma_.lower() in ('for', 'of'):
            grandhead = head.head
            if grandhead.pos_ == 'VERB' and grandhead.lemma_.lower() in _MEASURE_VERBS:
                return 'unit'

        # "5T of cargo" - quantity-of-noun construction
        for child in anchor.children:
            if child.dep_ == 'prep' and child.lemma_.lower() == 'of':
                if any(gc.dep_ == 'pobj' for gc in child.children):
                    return 'unit'

        # "2L bottle" / "10MB file" - compounds with a container/quantity noun
        if anchor.dep_ == 'compound' and _is_quantity_noun(head):
            return 'unit'
        if _is_quantity_noun(anchor):
            return 'unit'

        # "Gate A12" / "locker code is V12" / "seat is 14H" / "room B12" -
        # token is, or is modified by, or is the object/complement of, an
        # identifier noun
        if any(_is_identifier(child) for child in anchor.children):
            return 'code'
        if _is_identifier(head):
            return 'code'
        if anchor.dep_ in ('attr', 'nsubj') and head.pos_ == 'AUX':
            subj = next((c for c in head.children if c.dep_ == 'nsubj'), None)
            if subj is not None and _is_identifier(subj):
                return 'code'

        return None

    # A multi-word token (e.g. "10MB" split into separate "10"/"MB" tokens by
    # spaCy) may carry its clearest grammatical signal on either part, so
    # every token spanned by the match is considered a candidate anchor - a
    # 'unit' signal from any of them is trusted first, falling back to a
    # 'code' signal from any of them.
    results = [_classify(tok) for tok in span]
    if 'unit' in results:
        return 'unit'
    if 'code' in results:
        return 'code'
    return None


def _looks_like_unit_use(letters, mangled_fragment, source_text, token_start, token_end):
    """True if the fragment we'd overwrite is a known unit word for `letters`
    AND a semantic parse of the source sentence classifies this token's
    grammatical role as a measurement - in which case trust the model's unit
    translation instead of reverting it."""
    unit_words = _LETTER_UNIT_WORDS.get(letters)
    if not unit_words or not any(w in mangled_fragment for w in unit_words):
        return False
    return _semantic_intent(source_text, token_start, token_end) == 'unit'


def fix_alphanumeric_codes(source_text, translated_text):
    """Restore short letter+digit codes the model mangled, anchored on digits.

    If `source_text` (English) contains a short uppercase-letter + digit code
    and `translated_text` (Gurmukhi) still has the exact digits but not the
    original letters sitting next to them, replace whatever is there with the
    verbatim original code. Otherwise returns the translation unchanged.
    """
    if not source_text or not translated_text:
        return translated_text

    result = translated_text
    for m in _CODE_TOKEN_RE.finditer(source_text):
        if m.group(1):
            letters, digits, original_token, letters_first = m.group(1), m.group(3), m.group(0), True
        else:
            letters, digits, original_token, letters_first = m.group(6), m.group(4), m.group(0), False

        digit_matches = list(re.finditer(r'(?<!\d)' + re.escape(digits) + r'(?!\d)', result))
        if len(digit_matches) != 1:
            continue  # missing or duplicated - ambiguous, do not guess

        d_start, d_end = digit_matches[0].span()

        if letters_first:
            before = result[max(0, d_start - 12):d_start].rstrip(' -')
            if before.endswith(letters):
                continue  # already correct
            replace_start, replace_end = _extend_left(result, d_start, len(letters)), d_end
        else:
            after = result[d_end:d_end + 12].lstrip(' -')
            if after.startswith(letters):
                continue  # already correct
            replace_start, replace_end = d_start, _extend_right(result, d_end, len(letters))

        if _looks_like_unit_use(letters, result[replace_start:replace_end], source_text, m.start(), m.end()):
            continue  # looks like a genuine unit-of-measurement, not a mangled code

        result = result[:replace_start] + original_token + result[replace_end:]

    return result
