"""Local, dedicated grammar/fluency correction for English translation output.

BUG FOUND 2026-11-22 (user, after testing the Google Translate fallback wired
into /confidence the same day): "for the unnatural phrasing for problem #1, i
tested the app, and nothing changed. i want to integrate a new smart AI
system because google translate is limited for us, i also want it to be
better than google translate, what the system should do, is maybe
gramatically verify the english translation, and correct it the best
possible way". That earlier fallback never actually helped because
compute_roundtrip_confidence() (ai_backend_server.py) measures MEANING
preservation via back-translation word overlap, not fluency/grammaticality -
a literal-but-awkward IndicTrans2 translation that still preserves meaning
well scores just as high (or higher) than a more natural Google Translate
rephrasing, so the fallback almost never actually swapped anything in for
this specific complaint.

This module runs a dedicated, purpose-built Grammatical Error Correction
(GEC) model - pszemraj/grammar-synthesis-small (T5-small based, 77M params,
Apache-2.0 licensed: https://huggingface.co/pszemraj/grammar-synthesis-small)
- as a final polishing pass on whatever English translation is currently
"best" (after IndicTrans2 + ensemble retry + the optional Google fallback
have already run, see apply_grammar_correction() in ai_backend_server.py). A
genuinely fluency-focused correction, instead of hoping a meaning-
preservation metric happens to also catch unnatural phrasing.

The model card states it's trained specifically to correct grammar WITHOUT
semantically changing text that's already grammatically correct, so the
caller runs this largely unconditionally for target_lang == 'en' and only
uses compute_roundtrip_confidence afterward as a safety net (not a gate) to
reject the rare rewrite that drifts in meaning.

Uses `transformers`, already installed in production as a transitive
dependency of the NeMo RNNT / SpeechBrain install chain (see Dockerfile) -
no new runtime pip dependency is introduced by this module. Lazy-loaded on
first use, mirroring every other model in this project (see
get_parakeet_unified_model() in ai_backend_server.py) so Punjabi-target
traffic that never needs this pays nothing for it.

Never raises: any failure (missing model files, import error, generation
exception) is caught and reported as None/False, so a caller can always
safely keep whatever translation it already had.
"""
import os
import re
import threading

# Matches one "sentence" (text up to and including a run of .!? , or a
# trailing remainder with no terminal punctuation at all).
_SENTENCE_RE = re.compile(r'[^.!?]*[.!?]+|[^.!?]+$')

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', 'grammar-synthesis-small')
# Guards against a pathologically long single "sentence" blowing up
# generation time - ordinary spoken-utterance translations are nowhere near
# this long.
_MAX_INPUT_CHARS = 2000

_pipeline = None
_load_lock = threading.Lock()
_load_failed = False


def _load_pipeline():
    global _pipeline, _load_failed
    if _pipeline is not None or _load_failed:
        return _pipeline
    if os.environ.get('DISABLE_GRAMMAR_CORRECTION', '').strip() == '1':
        _load_failed = True
        return None
    with _load_lock:
        if _pipeline is None and not _load_failed:
            if not os.path.isdir(MODEL_DIR):
                print(f"\nℹ️ No grammar-correction model found at {MODEL_DIR} - skipping grammar correction pass.")
                _load_failed = True
                return None
            print("\n📥 Loading grammar-synthesis-small (first English grammar-correction request)...")
            try:
                from transformers import pipeline
                _pipeline = pipeline(
                    'text2text-generation',
                    model=MODEL_DIR,
                    tokenizer=MODEL_DIR,
                    device=-1,  # CPU only - this process already caps torch threads, see ai_backend_server.py
                )
                print("✅ Grammar-correction model loaded.")
            except Exception as e:
                print(f"   ⚠️ Failed to load grammar-correction model: {e}")
                _load_failed = True
                _pipeline = None
    return _pipeline


def is_available() -> bool:
    """Whether the grammar-correction model can be used (model files present,
    not disabled, loads successfully). Loads the model on first call."""
    return _load_pipeline() is not None


def _truncate_to_sentence_count(original, corrected):
    """LIVE-VERIFIED FINDING (2026-11-23): capping max_new_tokens relative to
    input length did NOT stop the model from appending an entire extra,
    invented sentence after a correct fix - all 3 live test cases produced
    byte-for-byte the same hallucinated addition as the uncapped version
    (e.g. "she go to school" -> "she goes to school. She went to school."),
    because the model only needed a handful of extra tokens to do it, well
    within any reasonable cap.

    This is a second, more direct mitigation: since the whole point of this
    pass is a 1:1 grammar fix (not adding new information), if the corrected
    text has MORE sentences than the original, it almost certainly tacked on
    an invented continuation - keep only the first len(original sentences)
    sentences of the correction.
    """
    orig_sentences = [s for s in _SENTENCE_RE.findall(original) if s.strip()]
    corr_sentences = [s for s in _SENTENCE_RE.findall(corrected) if s.strip()]
    if not orig_sentences or len(corr_sentences) <= len(orig_sentences):
        return corrected
    truncated = ''.join(corr_sentences[:len(orig_sentences)]).strip()
    return truncated or corrected


def correct_grammar(text):
    """Run the GEC pass over `text` (English only). Returns the corrected
    string, or None on any failure/unavailability/no-op - callers must treat
    this as best-effort and keep their existing translation when None comes
    back.

    LIVE-VERIFIED FAILURE MODE (2026-11-23): on short, single-sentence
    translations (the normal case for spoken-utterance input), this model
    consistently APPENDED an entire extra, invented sentence after an
    otherwise-correct fix (e.g. "she go to school" -> "she goes to school.
    She went to school." - 3/3 live tests showed this same pattern). Capping
    `max_new_tokens` relative to the input's own token count (rather than
    the previous flat 256) makes it much harder for the model to generate a
    whole unrelated extra clause - this is a generation-time mitigation,
    complementary to (not a replacement for) the post-hoc
    _looks_like_hallucinated_addition() safety net in ai_backend_server.py's
    apply_grammar_correction(), which still rejects anything that slips
    through.
    """
    if not text or not text.strip():
        return None
    pipe = _load_pipeline()
    if pipe is None:
        return None
    try:
        input_text = text[:_MAX_INPUT_CHARS]
        input_token_count = len(pipe.tokenizer(input_text)['input_ids'])
        max_new_tokens = min(200, input_token_count + 12)
        result = pipe(input_text, max_new_tokens=max_new_tokens, num_beams=2, truncation=True)
        corrected = result[0]['generated_text'].strip()
        if not corrected:
            return None
        corrected = _truncate_to_sentence_count(input_text, corrected)
        return corrected or None
    except Exception as e:
        print(f"   ⚠️ Grammar correction failed: {e}")
        return None
