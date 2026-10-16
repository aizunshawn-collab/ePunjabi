"""
Converts spelled-out Punjabi number words (as heard in real ASR transcripts,
which never contain digits) into plain ASCII digit strings, so IndicTrans2
only has to copy a digit sequence through rather than translate a number
lexically - something it does far more reliably (see repo memory: production
pa->en voice translation was found to mangle spoken numbers, e.g. "ਪੰਦਰਾਂ ਸੌ"
(fifteen hundred/1500) -> "100 jobs", "ਪੰਜਤਾਲੀ" (45) -> "one year").

Word forms below were generated via the `num_to_words` PyPI package
(https://pypi.org/project/indic-num2words/, lang='pa') for n in range(100) -
NOT hand-typed - specifically to avoid introducing incorrect Gurmukhi
spellings into production. They are only used here for PARSING (recognizing
these exact forms in text), never for generating output, so a speaker/ASR
using a slightly different spelling variant just falls back to normal
translation (a miss, not a correctness bug).
"""

import unicodedata

# 0-99, generated via num_to_words.num_to_word(n, lang='pa') for n in range(100).
_PA_NUMBER_WORDS = {
    0: 'ਸਿਫਰ', 1: 'ਇੱਕ', 2: 'ਦੋ', 3: 'ਤਿੰਨ', 4: 'ਚਾਰ', 5: 'ਪੰਜ', 6: 'ਛੇ',
    7: 'ਸੱਤ', 8: 'ਅੱਠ', 9: 'ਨੌਂ', 10: 'ਦੱਸ',
    11: 'ਗਿਆਰਾਂ', 12: 'ਬਾਰਾਂ', 13: 'ਤੇਰਾਂ', 14: 'ਚੌਦਾਂ', 15: 'ਪੰਦਰਾਂ',
    16: 'ਸੋਲ਼ਾਂ', 17: 'ਸਤਾਰਾਂ', 18: 'ਅਠਾਰਾਂ', 19: 'ਉਨੀ', 20: 'ਵੀਹ',
    21: 'ਇੱਕੀ', 22: 'ਬਾਈ', 23: 'ਤੇਈ', 24: 'ਚੌਵੀ', 25: 'ਪੰਝੀ',
    26: 'ਛੱਬੀ', 27: 'ਸਤਾਈ', 28: 'ਅਠਾਈ', 29: 'ਉਨੱਤੀ', 30: 'ਤੀਹ',
    31: 'ਇਕੱਤੀ', 32: 'ਬੱਤੀ', 33: 'ਤੇਤੀ', 34: 'ਚੌਂਤੀ', 35: 'ਪੈਂਤੀ',
    36: 'ਛੱਤੀ', 37: 'ਸੈਂਤੀ', 38: 'ਅਠੱਤੀ', 39: 'ਉਨਤਾਲੀ', 40: 'ਚਾਲੀ',
    41: 'ਇਕਤਾਲੀ', 42: 'ਬਤਾਲੀ', 43: 'ਤਰਤਾਲੀ', 44: 'ਚੌਤਾਲੀ', 45: 'ਪੰਜਤਾਲੀ',
    46: 'ਛਿਆਲੀ', 47: 'ਸੰਤਾਲੀ', 48: 'ਅੱਠਤਾਲੀ', 49: 'ਉਣਿੰਜਾ', 50: 'ਪੰਜਾਹ',
    51: 'ਇਕਵਿੰਜਾ', 52: 'ਬਵਿੰਜਾ', 53: 'ਤਰਵਿੰਜਾ', 54: 'ਚਰਿੰਜਾ', 55: 'ਪਚਵਿੰਜਾ',
    56: 'ਛਪਿੰਜਾ', 57: 'ਸਤਵਿੰਜਾ', 58: 'ਅੱਠਵਿੰਜਾ', 59: 'ਉਣਾਠ', 60: 'ਸੱਠ',
    61: 'ਇਕਾਠ', 62: 'ਬਾਠ੍ਹ', 63: 'ਤਰੇਠ੍ਹ', 64: 'ਚੌਠ੍ਹ', 65: 'ਪੈਂਠ',
    66: 'ਛਿਆਠ', 67: 'ਸਤਾਹਠ', 68: 'ਅੱਠਾਠ', 69: 'ਉਣੱਤਰ', 70: 'ਸੱਤਰ',
    71: 'ਇਕ੍ਹੱਤਰ', 72: 'ਬਹੱਤਰ', 73: 'ਤਹੱਤਰ', 74: 'ਚੌਹੱਤਰ', 75: 'ਪੰਜੱਤਰ',
    76: 'ਛਿਹੱਤਰ', 77: 'ਸਤੱਤਰ', 78: 'ਅਠੱਤਰ', 79: 'ਉਣਾਸੀ', 80: 'ਅੱਸੀ',
    81: 'ਇਕਾਸੀ', 82: 'ਬਿਆਸੀ', 83: 'ਤਰਾਸੀ', 84: 'ਚਰਾਸੀ', 85: 'ਪੰਜਾਸੀ',
    86: 'ਛਿਆਸੀ', 87: 'ਸਤਾਸੀ', 88: 'ਅਠਾਸੀ', 89: 'ਉਣਾਨਵੇਂ', 90: 'ਨੱਬੇ',
    91: 'ਇਕਾਨਵੇਂ', 92: 'ਬਿਆਨਵੇਂ', 93: 'ਤਰਾਨਵੇਂ', 94: 'ਚਰਾਨਵੇਂ', 95: 'ਪਚਾਨਵੇਂ',
    96: 'ਛਿਆਨਵੇਂ', 97: 'ਸਤਾਨਵੇਂ', 98: 'ਅਠਾਨਵੇਂ', 99: 'ਨਿੜਾਨਵੇਂ',
}

# Reverse lookup, NFC-normalized - guards against the same Gurmukhi conjunct
# being encoded with different underlying codepoint sequences depending on
# source (same convention used elsewhere in this project, e.g. the glossary).
_WORD_TO_VALUE = {
    unicodedata.normalize('NFC', word): value
    for value, word in _PA_NUMBER_WORDS.items()
}
# Words 11+ are unambiguous standalone numerals (never doubling as an
# article/quantifier the way 0-10 commonly do, e.g. "ਇੱਕ ਆਦਮੀ" = "a man") -
# safe to convert even with no adjacent multiplier word.
_UNAMBIGUOUS_STANDALONE_MIN = 11

_MULTIPLIERS = {
    unicodedata.normalize('NFC', k): v for k, v in {
        'ਸੌ': 100,
        'ਹਜਾਰ': 1000,
        'ਹਜ਼ਾਰ': 1000,  # common alternate spelling (with nukta)
        'ਲੱਖ': 100000,
        'ਕਰੋੜ': 10000000,
    }.items()
}

_TRAILING_PUNCT = '.,!?;:।॥'


def convert_number_words_to_digits(text):
    """Replace spelled-out Punjabi number phrases (e.g. "ਪੰਦਰਾਂ ਸੌ" -> "1500",
    "ਦੋ ਹਜਾਰ ਬਾਈ" -> "2022", standalone "ਪੰਜਤਾਲੀ" -> "45") with plain digit
    strings. Conservative by design: a bare 0-10 number word with no
    multiplier is left untouched - these double as common indefinite-
    article-like quantifiers in natural Punjabi (e.g. "ਇੱਕ ਆਦਮੀ" = "a man"),
    and converting them unconditionally would create a worse, confidently-
    wrong regression than the bug this is meant to fix."""
    words = text.split()
    out = []
    i = 0
    n = len(words)
    while i < n:
        key = unicodedata.normalize('NFC', words[i].rstrip(_TRAILING_PUNCT))
        if key not in _WORD_TO_VALUE and key not in _MULTIPLIERS:
            out.append(words[i])
            i += 1
            continue

        # Parse a maximal number-phrase run starting at i.
        j = i
        total = 0
        current = None
        used_multiplier = False
        while j < n:
            w = unicodedata.normalize('NFC', words[j].rstrip(_TRAILING_PUNCT))
            if w in _MULTIPLIERS:
                total += (current if current is not None else 1) * _MULTIPLIERS[w]
                current = None
                used_multiplier = True
                j += 1
            elif w in _WORD_TO_VALUE and current is None:
                current = _WORD_TO_VALUE[w]
                j += 1
            else:
                break
        if current is not None:
            total += current

        run_len = j - i
        accept = used_multiplier or (run_len == 1 and current is not None and current >= _UNAMBIGUOUS_STANDALONE_MIN)
        if accept:
            last_word = words[j - 1]
            trailing = last_word[len(last_word.rstrip(_TRAILING_PUNCT)):]
            out.append(str(total) + trailing)
            i = j
        else:
            out.append(words[i])
            i += 1
    return ' '.join(out)
