"""
Converts spelled-out English cardinal number words (as sometimes heard in
Parakeet-unified-en-0.6b ASR transcripts, e.g. "two hundred" instead of
"200") into plain digit strings, matching the digit-based output convention
the old Whisper-based pipeline reliably produced (see repo memory: flagged
as a follow-up when Parakeet replaced Whisper for English ASR).

Mirrors `punjabi_numerals.py`'s safety design: a bare standalone 0-10 number
word ("one", "two", ..., "ten") is left UNTOUCHED, since these double as
common pronouns/quantifiers in natural English ("I need one", "no one",
"give me two minutes", "the one I like") - converting them unconditionally
would be a confidently-wrong regression, not an improvement. Words 11-99 are
unambiguous standalone numerals and are always converted, as is any
unit/teen/tens word adjacent to a hundred/thousand/million/billion
multiplier (grammatically validated - see _VALID_ADJACENT below).

Grammar validation also guards against a real failure mode: digit-by-digit
speech (e.g. reading out a code as "zero two one five") must NOT be merged
into a single accumulated number - only grammatically valid English number
compounds (e.g. "twenty two", "two hundred", "two thousand twenty two") are
merged into one run; adjacent bare unit/teen words with no multiplier or
tens-pairing between them are treated as separate, independent words.

Year-reading special case (see _merge_year_runs): spoken years are commonly
read as two independent two-digit groups rather than one arithmetic
compound (e.g. "nineteen ninety nine" = 1999, NOT 19*100+99). Without this,
each group converts correctly on its own but is left as two separate digit
strings ("19 99") instead of being concatenated into one year ("1999"). Only
applies to adjacent non-multiplier runs that each read as 10-99, joined by a
single plain space.
"""

import re

_UNITS = {
    'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
    'six': 6, 'seven': 7, 'eight': 8, 'nine': 9,
}
_TEENS = {
    'ten': 10, 'eleven': 11, 'twelve': 12, 'thirteen': 13, 'fourteen': 14,
    'fifteen': 15, 'sixteen': 16, 'seventeen': 17, 'eighteen': 18, 'nineteen': 19,
}
_TENS = {
    'twenty': 20, 'thirty': 30, 'forty': 40, 'fifty': 50, 'sixty': 60,
    'seventy': 70, 'eighty': 80, 'ninety': 90,
}
_MULTIPLIERS = {'hundred': 100, 'thousand': 1000, 'million': 1_000_000, 'billion': 1_000_000_000}

# (scale, increment) per word, used by the classic text2num-style accumulator
# in _words_to_number().
_NUMWORDS = {}
for _w, _v in _UNITS.items():
    _NUMWORDS[_w] = (1, _v)
for _w, _v in _TEENS.items():
    _NUMWORDS[_w] = (1, _v)
for _w, _v in _TENS.items():
    _NUMWORDS[_w] = (1, _v)
for _w, _v in _MULTIPLIERS.items():
    _NUMWORDS[_w] = (_v, 0)

# Bare 0-10 words are left standalone-untouched (see module docstring).
_STANDALONE_UNAMBIGUOUS_MIN = 11

# Which category can grammatically precede which, when merging two adjacent
# number words into one compound (e.g. "twenty two" = S then U). Anything
# not listed here (e.g. U-then-U, as in digit-by-digit "two one five") is
# NOT merged - each word is left as its own independent run.
_VALID_ADJACENT = {
    ('S', 'U'),                                      # twenty two
    ('U', 'M'), ('T', 'M'), ('S', 'M'), ('M', 'M'),   # two/fifteen/twenty/hundred -> hundred/thousand/...
    ('M', 'U'), ('M', 'T'), ('M', 'S'),               # hundred twenty, hundred five, thousand two
}
# "and" only bridges two runs when sandwiched directly after a multiplier
# and before a unit/teen/tens word (e.g. "two hundred and five"), never
# between two bare units (e.g. NOT "one and two").
_VALID_AND_BRIDGE = {('M', 'U'), ('M', 'T'), ('M', 'S')}

_WORD_RE = re.compile(r"[A-Za-z]+(?:-[A-Za-z]+)*")


def _category(word):
    if word in _UNITS:
        return 'U'
    if word in _TEENS:
        return 'T'
    if word in _TENS:
        return 'S'
    if word in _MULTIPLIERS:
        return 'M'
    return None


def _subword_categories(token_lower):
    """For a (possibly hyphenated, e.g. 'twenty-two') token, return its list
    of (word, category) parts, or None if any part isn't a recognized number
    word, or if a hyphenated token's internal parts aren't a grammatically
    valid compound."""
    parts = token_lower.split('-')
    cats = []
    for p in parts:
        c = _category(p)
        if c is None:
            return None
        cats.append((p, c))
    for (_, c1), (_, c2) in zip(cats, cats[1:]):
        if (c1, c2) not in _VALID_ADJACENT:
            return None
    return cats


def _words_to_number(words):
    """Classic scale/increment accumulator (`words` excludes 'and')."""
    result = current = 0
    for w in words:
        scale, increment = _NUMWORDS[w]
        if scale > 1:
            current = max(1, current)
        current = current * scale + increment
        if scale > 100:
            result += current
            current = 0
    return result + current


def _run_words(matches, info, start, end):
    words = []
    for k in range(start, end + 1):
        w = matches[k].group().lower()
        if w == 'and':
            continue
        words.extend(p for p, _c in info[k])
    return words


def _run_has_multiplier(info, start, end):
    return any(c == 'M' for k in range(start, end + 1) for _, c in info[k])


def _merge_year_runs(matches, text, info, runs, run_values, run_has_multiplier):
    """Merge two adjacent non-multiplier runs that each read as a 10-99
    group (e.g. "nineteen" + "ninety nine") into one 4-digit year by
    concatenation (19|99 -> 1999), matching how English years are actually
    spoken - rather than leaving them as two separate digit strings ("19
    99"). Only merges across a single plain space (no "and"), and only
    when neither run contains a hundred/thousand/etc multiplier, so this
    never interferes with ordinary arithmetic compounds."""
    merged = []
    i = 0
    n = len(runs)
    while i < n:
        if i + 1 < n and not run_has_multiplier[i] and not run_has_multiplier[i + 1]:
            start1, end1 = runs[i]
            start2, end2 = runs[i + 1]
            gap = text[matches[end1].end():matches[start2].start()]
            if gap == ' ' and 11 <= run_values[i] <= 99 and 10 <= run_values[i + 1] <= 99:
                merged.append((start1, end2, run_values[i] * 100 + run_values[i + 1]))
                i += 2
                continue
        merged.append((*runs[i], run_values[i]))
        i += 1
    return merged


def convert_number_words_to_digits(text):
    """Replace unambiguous spelled-out English cardinal number phrases with
    digit strings (e.g. "two hundred" -> "200", "twenty two" -> "22",
    "two thousand twenty two" -> "2022"). See module docstring for the
    standalone-0-10 exception and the digit-by-digit-reading safeguard."""
    matches = list(_WORD_RE.finditer(text))
    if not matches:
        return text

    info = [_subword_categories(m.group().lower()) for m in matches]

    runs = []  # (start_idx, end_idx) inclusive, into `matches`/`info`
    i = 0
    n = len(matches)
    while i < n:
        if info[i] is None:
            i += 1
            continue
        j = i
        while True:
            nxt = j + 1
            if (nxt < n and matches[nxt].group().lower() == 'and'
                    and text[matches[j].end():matches[nxt].start()] == ' '):
                nxt2 = nxt + 1
                if (nxt2 < n and info[nxt2] is not None
                        and text[matches[nxt].end():matches[nxt2].start()] == ' '
                        and (info[j][-1][1], info[nxt2][0][1]) in _VALID_AND_BRIDGE):
                    j = nxt2
                    continue
                break
            if (nxt < n and info[nxt] is not None
                    and text[matches[j].end():matches[nxt].start()] == ' '
                    and (info[j][-1][1], info[nxt][0][1]) in _VALID_ADJACENT):
                j = nxt
                continue
            break
        runs.append((i, j))
        i = j + 1

    if not runs:
        return text

    run_values = [_words_to_number(_run_words(matches, info, s, e)) for s, e in runs]
    run_has_multiplier = [_run_has_multiplier(info, s, e) for s, e in runs]
    runs_with_values = _merge_year_runs(matches, text, info, runs, run_values, run_has_multiplier)

    out = []
    cursor = 0
    for start, end, value in runs_with_values:
        if start == end and value <= _STANDALONE_UNAMBIGUOUS_MIN - 1:
            continue  # bare standalone 0-10 word - ambiguous, leave as-is
        span_start = matches[start].start()
        span_end = matches[end].end()
        out.append(text[cursor:span_start])
        out.append(str(value))
        cursor = span_end
    out.append(text[cursor:])
    return ''.join(out)
