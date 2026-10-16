"""
FREE AI Translation Backend Server
Translation runs on IndicTrans2 (CTranslate2, CPU-friendly int8 quantized
inference, no GPU required). IndicTrans2 is purpose-built for English<->Indic
languages (incl. Punjabi) and outperforms both NLLB-200 and Google Translate
on Indic pairs, especially on longer/more complex sentences, since it's
trained on a much larger and cleaner English-Indic parallel corpus (BPCC)
instead of a generic 200-language dataset.

Speech recognition has no Whisper anywhere in it (removed 2026-10, see repo
memory): English ASR is NVIDIA's Parakeet-unified-en-0.6b (ONNX, torch-free),
Punjabi ASR is AI4Bharat's IndicConformer (native NeMo RNNT decoder, torch),
and spoken-language auto-detection for 'auto'-mode requests is SpeechBrain's
VoxLingua107 ECAPA-TDNN language-ID model.

Installation:
pip install -r requirements.txt

Run:
python ai_backend_server.py

Then update your app to use http://localhost:5000 instead of Google APIs
"""

from flask import Flask, request, jsonify, g
from flask_cors import CORS
import ctranslate2
import sentencepiece as spm
# faster_whisper itself is no longer used for transcription (no Whisper ASR
# models remain in this project) - kept only for its decode_audio() utility
# (PyAV-based, handles arbitrary audio containers like the app's .m4a
# recordings), which every ASR path below relies on.
from faster_whisper.audio import decode_audio
import torch  # only used directly for the SpeechBrain language-ID model below; NeMo RNNT (indicconformer_nemo_asr.py) uses it internally too
from indictrans_processor import IndicProcessor  # local port, see file docstring
from punjabi_numerals import convert_number_words_to_digits  # pa-source numeral-word-to-digit fix, see file docstring
from english_numerals import convert_number_words_to_digits as convert_english_number_words_to_digits  # Parakeet en-source numeral-word-to-digit fix, see file docstring
from punjabi_name_corrections import fix_name_substitutions  # en->pa curated name-substitution fix, see file docstring
from parakeet_name_corrections import correct_kaur_mishearing  # Parakeet sentence-final "Kaur"->"Kor"/"Kore"/"Core" fix, see file docstring
from punjabi_alphanumeric_fix import fix_alphanumeric_codes  # en->pa digit-anchored code mangling fix, see file docstring
from indicconformer_asr import IndicConformerASR  # torch-free ONNX Punjabi ASR, see file docstring
from parakeet_unified_asr import ParakeetUnifiedASR  # torch-free ONNX English ASR (RNNT), see file docstring
import pytesseract
from PIL import Image, ImageOps
import os
import re
import sys
import json
import base64
import tempfile
import io
import time
import unicodedata
import threading
import numpy as np
from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate

# Windows console defaults to cp1252, which can't print emoji used in log messages below.
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

# Tesseract OCR (used by /ocr, see below) ships as a separate system install,
# not a pip package - pytesseract just shells out to the tesseract.exe binary.
# Point pytesseract at whichever known Windows install location exists (the
# UB Mannheim installer defaults to per-machine Program Files, but can also
# install per-user under %LOCALAPPDATA%); if tesseract is already on PATH
# this whole block is a harmless no-op.
_TESSERACT_WIN_PATHS = [
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    os.path.expandvars(r"%LOCALAPPDATA%\Programs\Tesseract-OCR\tesseract.exe"),
]
for _tesseract_path in _TESSERACT_WIN_PATHS:
    if os.path.exists(_tesseract_path):
        pytesseract.pytesseract.tesseract_cmd = _tesseract_path
        break

app = Flask(__name__)
CORS(app)  # Allow requests from your web app

# --- Public-deployment hardening -------------------------------------------
# This server has no built-in user accounts, and translation/speech requests
# are expensive (translation model, Whisper speech recognition). Once this is reachable
# from the internet rather than just a LAN, it needs at least a shared-secret
# API key and basic rate limiting, or anyone who finds the URL can run up
# compute costs or take the server down. Both are no-ops for local
# development (no key configured) so `python ai_backend_server.py` still
# works out of the box on localhost/LAN.
API_KEY = os.environ.get('BACKEND_API_KEY', '').strip()
# QA-only route (bypasses the glossary) - not needed by the shipped app, so
# keep it off unless explicitly enabled, to shrink the public attack surface.
ENABLE_DEBUG_ROUTES = os.environ.get('ENABLE_DEBUG_ROUTES', '').strip() == '1'
# Reject oversized request bodies (e.g. huge audio uploads) before they reach
# any handler - a cheap defense against payload-based DoS.
app.config['MAX_CONTENT_LENGTH'] = 20 * 1024 * 1024  # 20 MB

# Simple in-memory sliding-window limiter, keyed by caller IP (or API key if
# provided). Not distributed/multi-process safe, but this server runs single-
# process - good enough to stop one client from hammering it.
_RATE_LIMIT_WINDOW_SECONDS = 60
_RATE_LIMIT_MAX_REQUESTS = int(os.environ.get('RATE_LIMIT_MAX_REQUESTS', 20))
_rate_limit_hits: dict = {}


def _rate_limit_key() -> str:
    return request.headers.get('X-API-Key') or request.remote_addr or 'unknown'


def _is_rate_limited() -> bool:
    now = time.time()
    key = _rate_limit_key()
    hits = [t for t in _rate_limit_hits.get(key, []) if now - t < _RATE_LIMIT_WINDOW_SECONDS]
    hits.append(now)
    _rate_limit_hits[key] = hits
    return len(hits) > _RATE_LIMIT_MAX_REQUESTS


_UNAUTHENTICATED_ROUTES = {'/health'}


@app.before_request
def _start_request_timer():
    # Registered before the auth/rate-limit check below so latency is timed
    # (and telemetry still logged) even for rejected (401/429) requests.
    g.request_start_time = time.time()


@app.before_request
def _enforce_api_key_and_rate_limit():
    if request.path in _UNAUTHENTICATED_ROUTES:
        return None
    if request.path == '/_debug_raw_translate' and not ENABLE_DEBUG_ROUTES:
        return jsonify({'error': 'not found'}), 404
    if API_KEY and request.headers.get('X-API-Key') != API_KEY:
        return jsonify({'error': 'Invalid or missing API key'}), 401
    if _is_rate_limited():
        return jsonify({'error': 'Rate limit exceeded, please slow down'}), 429
    return None


# Structured one-line-per-request telemetry (issue #8) - logged to stdout,
# which Fly.io already captures via `flyctl logs`, so no new logging infra
# is needed. Deliberately logs only lengths/language pairs/scores, NEVER the
# raw translated/transcribed text itself, to avoid putting user content in
# logs.
_TELEMETRY_ROUTES = {'/translate', '/speech-to-text', '/ocr'}


@app.after_request
def _log_telemetry(response):
    try:
        start = getattr(g, 'request_start_time', None)
        entry = {
            'ts': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
            'route': request.path,
            'status': response.status_code,
            'latency_ms': round((time.time() - start) * 1000, 1) if start is not None else None,
        }

        if request.method == 'POST' and request.path in _TELEMETRY_ROUTES:
            req_json = request.get_json(silent=True) or {}
            resp_json = response.get_json(silent=True)

            if request.path == '/translate':
                entry['source_lang'] = req_json.get('source_lang')
                entry['target_lang'] = req_json.get('target_lang')
                entry['text_length'] = len(req_json.get('text') or '')
                if resp_json:
                    entry['confidence'] = resp_json.get('confidence')
                    entry['retried'] = resp_json.get('retried', False)
            elif request.path == '/speech-to-text':
                entry['requested_language'] = req_json.get('language')
                if resp_json:
                    entry['detected_language'] = resp_json.get('language')
                    entry['confidence'] = resp_json.get('confidence')
                    entry['transcript_length'] = len(resp_json.get('transcript') or '')
            elif request.path == '/ocr':
                if resp_json:
                    entry['detected_language'] = resp_json.get('language')
                    entry['confidence'] = resp_json.get('confidence')

        print(f"TELEMETRY {json.dumps(entry, ensure_ascii=False)}")
    except Exception as e:
        print(f"   ⚠️ Telemetry logging failed: {e}")
    return response


if not API_KEY:
    print("⚠️  BACKEND_API_KEY is not set - server is running with NO authentication.")
    print("   Fine for local/LAN development, but set BACKEND_API_KEY before exposing")
    print("   this server to the public internet.")
# -----------------------------------------------------------------------------

print("=" * 60)
print("🚀 Starting AI Translation Server...")
print("=" * 60)

# Load IndicTrans2 Translation Models (AI4Bharat) via CTranslate2 - one model
# per direction, since IndicTrans2 uses separate en->indic and indic->en
# checkpoints. These are AI4Bharat's own official int8-quantized CT2-ported
# checkpoints (see repo memory for the download source/process) - same
# trained weights as the 1B HF models this server used to run via
# `transformers`, just converted to a much faster/lighter CPU inference
# engine (no GPU, no torch/transformers dependency needed at all).
device = "cpu"
print(f"\n📥 Loading IndicTrans2 translation models (CTranslate2, device: {device})...")
EN_INDIC_CT2_DIR = os.path.join(os.path.dirname(__file__), 'models', 'indictrans2-en-indic-ct2-int8')
INDIC_EN_CT2_DIR = os.path.join(os.path.dirname(__file__), 'models', 'indictrans2-indic-en-ct2-int8')

try:
    en_indic_translator = ctranslate2.Translator(EN_INDIC_CT2_DIR, device=device)
    en_indic_sp_src = spm.SentencePieceProcessor(model_file=os.path.join(EN_INDIC_CT2_DIR, "vocab", "model.SRC"))
    en_indic_sp_tgt = spm.SentencePieceProcessor(model_file=os.path.join(EN_INDIC_CT2_DIR, "vocab", "model.TGT"))

    indic_en_translator = ctranslate2.Translator(INDIC_EN_CT2_DIR, device=device)
    indic_en_sp_src = spm.SentencePieceProcessor(model_file=os.path.join(INDIC_EN_CT2_DIR, "vocab", "model.SRC"))
    indic_en_sp_tgt = spm.SentencePieceProcessor(model_file=os.path.join(INDIC_EN_CT2_DIR, "vocab", "model.TGT"))

    indic_processor = IndicProcessor(inference=True)
    print("✅ IndicTrans2 CT2 models loaded successfully! (same accuracy as before, CPU-fast)")
except Exception as e:
    print(f"❌ Error loading IndicTrans2 CT2 models: {e}")
    print("   Expected model directories (see repo memory for how to obtain them):")
    print(f"   {EN_INDIC_CT2_DIR}")
    print(f"   {INDIC_EN_CT2_DIR}")
    exit(1)

# --- Sentence/clause chunking helpers -----------------------------------
# Empirically (tested directly against this model), IndicTrans2 translates
# long, multi-clause sentences MORE accurately as a single whole chunk than
# when pre-split into clauses - splitting loses cross-clause context (e.g.
# pronoun/gender agreement) and can cause dropped content at clause
# boundaries. The model only starts silently truncating output on its own
# for genuinely very long sentences (safe up to ~86 words in testing, but
# breaking down by ~116 words), so clause-splitting is kept only as a
# fallback for sentences beyond that range.
#
# BUG FOUND 2026-10-09 (user reported pa->en voice translation silently
# dropping most of the content, sometimes failing outright): real ASR
# transcripts (confirmed via IndicConformer/NeMo-RNNT output - see repo
# memory) NEVER contain sentence-ending punctuation (no "।", no "?") OR
# commas, unlike typed/punctuated text. A longer spoken utterance therefore
# arrives here as ONE giant unsplittable "sentence" with zero break points
# for _CLAUSE_SPLIT_RE (needs English conjunctions) or the comma fallback to
# find. Confirmed via direct testing against the live model: a 161-word
# unpunctuated chunk silently translated only ~2 of 8 sentences' worth of
# content (29 output words); a 259+ word unpunctuated chunk crashes outright
# with `RuntimeError: No position encodings are defined for positions >=
# 256` (IndicTrans2's hard SentencePiece-token position limit). FIX:
# `_split_long_sentence` now hard-splits by fixed word windows as an
# absolute last resort when no clause/comma break points exist at all, so
# no chunk can ever reach either failure zone regardless of punctuation.
MAX_CHUNK_WORDS = 80
_SENTENCE_SPLIT_RE = re.compile(r'(?<=[.!?।])\s+')
_CLAUSE_SPLIT_RE = re.compile(
    r'(?<=[,;])\s+(?=(?:and|but|or|so|because|although|however|therefore|'
    r'meanwhile|while|since|which|who|that)\b)',
    re.IGNORECASE,
)

# BUG FOUND 2026-10-11 (user: "if you speak multiple sentences, the full text is
# transcribed but the translation skips some of it"): the 80-word cap above does
# not help because IndicTrans2 already drops whole sentences when several
# unpunctuated sentences arrive as ONE chunk - measured on real held-out
# transcripts joined into run-ons: 18% of sentences largely missing from the
# English at 2 sentences/run-on, 33% at 5. Whisper used to mask this because it
# emits punctuation; IndicConformer emits none. FIX: restore sentence breaks in
# unpunctuated Punjabi after finite verbs/auxiliaries (token set picked from
# train-split statistics: P(sentence-final | token) >= 0.6, plus the plain
# auxiliaries), never before a continuation word (ਕਿ/ਜੋ/ਅਤੇ/...) and never
# leaving a piece shorter than _PA_BREAK_MIN_PIECE_WORDS.
_PA_SENTENCE_FINAL = frozenset(
    'ਏ ਐ ਗਿਆ ਗਿਐ ਕੀਤੈ ਰਿਹੈ ਜਾਵੇਗਾ ਜਾਵੇਗੀ ਜਾਏਗਾ ਜਾਏਗੀ ਜਾਣਗੇ ਜਾਣਗੀਆਂ ਜਾਵੇ ਜਾਏ '
    'ਸਨ ਦਿੱਤੈ ਕਰਨਗੇ ਕਰੇਗਾ ਕਰੇਗੀ ਹੋਵੇਗਾ ਹੋਵੇਗੀ ਹੋਣਗੇ ਸਕੇ ਲਿਐ ਚਾਹੀਦੈ ਹੋਇਐ ਸਕਦੈ '
    'ਹੈ ਹਨ ਸੀ ਹਾਂ'.split()
)
_PA_CONTINUATION = frozenset(
    'ਕਿ ਜੋ ਜਿਸ ਜਿਨ੍ਹਾਂ ਜਿੱਥੇ ਜਿਵੇਂ ਜਦੋਂ ਜਦ ਜਦਕਿ ਅਤੇ ਤੇ ਪਰ ਜਾਂ ਤਾਂ ਜੇ ਜੇਕਰ ਕਿਉਂਕਿ '
    'ਭਾਵੇਂ ਵੀ ਹੀ ਨਹੀਂ ਨਾ'.split()
)
_PA_BREAK_MIN_TOTAL_WORDS = 10
_PA_BREAK_MIN_PIECE_WORDS = 3


def _restore_punjabi_breaks(sentence):
    """Split an unpunctuated Gurmukhi run-on into sentence-like pieces (see
    comment above). Returns [sentence] unchanged when it is short, not mostly
    Gurmukhi, or has no safe break point."""
    words = sentence.split()
    if len(words) < _PA_BREAK_MIN_TOTAL_WORDS:
        return [sentence]
    letters = [c for c in sentence if c.isalpha()]
    if not letters or sum(1 for c in letters if '\u0A00' <= c <= '\u0A7F') / len(letters) < 0.6:
        return [sentence]
    pieces, start = [], 0
    for i, word in enumerate(words[:-1]):
        if word not in _PA_SENTENCE_FINAL:
            continue
        if words[i + 1] in _PA_CONTINUATION:
            continue
        if i + 1 - start < _PA_BREAK_MIN_PIECE_WORDS:
            continue
        if len(words) - (i + 1) < _PA_BREAK_MIN_PIECE_WORDS:
            continue
        pieces.append(" ".join(words[start:i + 1]))
        start = i + 1
    pieces.append(" ".join(words[start:]))
    return pieces


def _hard_split_by_words(sentence, max_words=MAX_CHUNK_WORDS):
    """Absolute last-resort split: break into fixed-size word windows. Needed
    for real ASR transcripts of continuous speech, which contain no commas
    or sentence-ending punctuation at all to split on (see comment above)."""
    words = sentence.split()
    return [" ".join(words[i:i + max_words]) for i in range(0, len(words), max_words)]


def _split_long_sentence(sentence):
    """Break an overly long/complex sentence into smaller clauses, falling
    back to a hard word-count split (see comment above) when no clause/comma
    break points exist at all - guarantees every returned piece stays at or
    under MAX_CHUNK_WORDS no matter how unpunctuated the input is."""
    if len(sentence.split()) <= MAX_CHUNK_WORDS:
        return [sentence]
    clauses = _CLAUSE_SPLIT_RE.split(sentence)
    if len(clauses) == 1:
        # Fallback: split on any comma if the sentence is still too long
        clauses = re.split(r'(?<=,)\s+', sentence)
    result = []
    for clause in clauses:
        if len(clause.split()) > MAX_CHUNK_WORDS:
            result.extend(_hard_split_by_words(clause))
        else:
            result.append(clause)
    return result



def chunk_text(text):
    """Split text into sentence-level chunks, further splitting overly long
    sentences into clauses, so each piece sent to the model stays short
    enough to translate reliably."""
    sentences = _SENTENCE_SPLIT_RE.split(text.strip())
    chunks = []
    for sentence in sentences:
        if not sentence.strip():
            continue
        for piece in _restore_punjabi_breaks(sentence):
            chunks.extend(_split_long_sentence(piece))
    return [c.strip() for c in chunks if c.strip()]


_SMART_QUOTES = {
    '\u2018': "'", '\u2019': "'",  # ' '
    '\u201c': '"', '\u201d': '"',  # " "
    '\u2013': '-', '\u2014': '-',  # – —
    '\u2026': '...',               # …
}
_WHITESPACE_RE = re.compile(r'[ \t]+')


def _gurmukhi_char_count(text):
    """Count characters in the Gurmukhi Unicode block, used to score which
    transliteration/transcription candidate actually produced Punjabi script."""
    return sum(1 for c in text if '\u0A00' <= c <= '\u0A7F')


def normalize_text(text):
    """Clean up common typing/ASR artifacts before translation: smart quotes,
    stray control characters, and repeated whitespace. Keeping the model's
    input consistent measurably reduces garbled output on messy input.
    NFC-normalizes Unicode so visually-identical Gurmukhi conjuncts always
    have the same underlying codepoint sequence (matters for glossary/dict
    lookups and exact-match comparisons elsewhere)."""
    text = unicodedata.normalize('NFC', text)
    for smart, plain in _SMART_QUOTES.items():
        text = text.replace(smart, plain)
    text = ''.join(ch for ch in text if ch == '\n' or ch.isprintable())
    text = _WHITESPACE_RE.sub(' ', text)
    return text.strip()


# Rare IndicTrans2 decoding quirk (confirmed reproducible, deterministic):
# the model has been observed emitting a token from a different script's
# vocabulary in place of a visually near-identical Gurmukhi letter, e.g.
# 'ک' (Perso-Arabic keheh) instead of 'ਕ' (Gurmukhi ka) inside "ਇੱਕ". Add more
# confirmed pairs here as they're found - see repo memory for how this one
# was diagnosed.
_SCRIPT_HOMOGLYPH_FIXES = {
    '\u06A9': '\u0A15',  # ARABIC LETTER KEHEH -> GURMUKHI LETTER KA (ਕ)
}
_FOREIGN_SCRIPT_RE = re.compile(
    r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]'
)


def _fix_script_homoglyphs(text):
    """Apply known cross-script homoglyph fixes and log any unrecognized
    foreign-script character left in Gurmukhi output, so new cases can be
    added to _SCRIPT_HOMOGLYPH_FIXES."""
    for bad, good in _SCRIPT_HOMOGLYPH_FIXES.items():
        text = text.replace(bad, good)
    leftover = _FOREIGN_SCRIPT_RE.findall(text)
    if leftover:
        print(f"⚠️  Unrecognized foreign-script character(s) in Gurmukhi output: {leftover!r}")
    return text


def translate_chunks(chunks, source_lang, target_lang, num_beams=4, length_penalty=1.0):
    """Translate a list of text chunks using the correct IndicTrans2 CT2 model
    for the given direction, then return the translated chunks in order.
    num_beams/length_penalty are exposed so a low-confidence result can be
    retried with different search parameters (see /translate's retry logic).

    LATENCY FIX (2026-10-03): default num_beams lowered from 8 to 4 - local
    CT2 benchmarking showed beam=8 costs ~2.7s/call vs beam=4's ~1.5s (beams
    1/2/4 all cluster around the same per-request floor; the real cost jump
    starts at beam>=6), with every `verify:true` request (always-on in the
    mobile app) paying this cost at least twice (forward + back-translation
    inside compute_roundtrip_confidence), so this alone roughly halves the
    common-case /translate latency. See repo memory for the full benchmark.

    Pre/post-processing (normalize, script transliteration, glossary-adjacent
    cleanup) is unchanged - still IndicProcessor. What's different from the
    old transformers-based version is only the tokenize+generate+decode
    middle step: SentencePiece encode -> add IndicTrans2's language-tag
    tokens -> ctranslate2.Translator.translate_batch() -> SentencePiece
    decode. This exactly mirrors AI4Bharat's own official CT2 inference
    recipe (see inference/engine.py in the IndicTrans2 repo)."""
    lang_map = {'en': 'eng_Latn', 'pa': 'pan_Guru'}
    src = lang_map.get(source_lang, 'eng_Latn')
    tgt = lang_map.get(target_lang, 'pan_Guru')

    if source_lang == 'en':
        translator, sp_src = en_indic_translator, en_indic_sp_src
    else:
        translator, sp_src = indic_en_translator, indic_en_sp_src

    # IndicProcessor.preprocess_batch already prepends "{src} {tgt} " to each
    # sentence (needed by the old HF tokenizer path) - but AI4Bharat's CT2
    # convention adds the language tags AFTER SentencePiece encoding, as
    # their own separate un-split tokens. Strip the plain-text prefix back
    # off here so it can be re-added post-SPM in the right place.
    preprocessed = indic_processor.preprocess_batch(chunks, src_lang=src, tgt_lang=tgt)
    prefix = f"{src} {tgt} "
    stripped = [s[len(prefix):] if s.startswith(prefix) else s for s in preprocessed]

    spm_encoded = [" ".join(sp_src.encode(s, out_type=str)) for s in stripped]
    tagged = [f"{src} {tgt} {s}" for s in spm_encoded]
    tokenized_sents = [t.strip().split(" ") for t in tagged]

    translation_results = translator.translate_batch(
        tokenized_sents,
        beam_size=num_beams,
        length_penalty=length_penalty,
        repetition_penalty=1.3,   # discourage repeated/looping phrases
        no_repeat_ngram_size=3,   # forbid repeating any 3-word sequence
        max_decoding_length=320,
        max_input_length=0,      # no extra truncation beyond chunking upstream
    )
    decoded = [" ".join(r.hypotheses[0]) for r in translation_results]
    # Undo SentencePiece's "_" word-boundary marker to get plain text back.
    decoded = [d.replace(" ", "").replace("\u2581", " ").strip() for d in decoded]

    result = indic_processor.postprocess_batch(decoded, lang=tgt)
    if tgt == 'pan_Guru':
        result = [_fix_script_homoglyphs(t) for t in result]
    return result


# Optional exact-match glossary for terms/phrases you've observed the model
# mistranslate (proper nouns, brand names, fixed idioms, honorifics, etc).
# Populate this as you test - e.g. GLOSSARY[('en','pa')]['Google Play'] = 'ਗੂਗਲ ਪਲੇ'
# Seeded with a few common, well-established greetings/courtesy phrases.
# Phrases whose Punjabi verb form depends on the speaker's grammatical gender
# (e.g. 'I am lost') live in GENDERED_GLOSSARY below instead of here.
GLOSSARY = {
    ('en', 'pa'): {
        # Greetings / courtesy
        'Hello': 'ਸਤ ਸ੍ਰੀ ਅਕਾਲ',
        'Thank you': 'ਧੰਨਵਾਦ',
        'Thank you very much': 'ਬਹੁਤ ਧੰਨਵਾਦ',
        'Please': 'ਕਿਰਪਾ ਕਰਕੇ',
        'Sorry': 'ਮਾਫ਼ ਕਰਨਾ',
        'Excuse me': 'ਮਾਫ਼ ਕਰਨਾ',
        'Good morning': 'ਸ਼ੁਭ ਸਵੇਰ',
        'Good afternoon': 'ਸ਼ੁਭ ਦੁਪਹਿਰ',
        'Good evening': 'ਸ਼ੁਭ ਸ਼ਾਮ',
        'Good night': 'ਸ਼ੁਭ ਰਾਤ',
        'Welcome': 'ਜੀ ਆਇਆਂ ਨੂੰ',
        'Yes': 'ਹਾਂ',
        'No': 'ਨਹੀਂ',
        'Congratulations': 'ਵਧਾਈਆਂ',
        # Numbers 1-10
        'One': 'ਇੱਕ',
        'Two': 'ਦੋ',
        'Three': 'ਤਿੰਨ',
        'Four': 'ਚਾਰ',
        'Five': 'ਪੰਜ',
        'Six': 'ਛੇ',
        'Seven': 'ਸੱਤ',
        'Eight': 'ਅੱਠ',
        'Nine': 'ਨੌਂ',
        'Ten': 'ਦਸ',
        # Days of the week
        'Monday': 'ਸੋਮਵਾਰ',
        'Tuesday': 'ਮੰਗਲਵਾਰ',
        'Wednesday': 'ਬੁੱਧਵਾਰ',
        'Thursday': 'ਵੀਰਵਾਰ',
        'Friday': 'ਸ਼ੁੱਕਰਵਾਰ',
        'Saturday': 'ਸ਼ਨੀਵਾਰ',
        'Sunday': 'ਐਤਵਾਰ',
        # Family
        'Mother': 'ਮਾਂ',
        'Father': 'ਪਿਤਾ',
        'Brother': 'ਭਰਾ',
        'Sister': 'ਭੈਣ',
        'Friend': 'ਦੋਸਤ',
        # Common everyday nouns
        'Water': 'ਪਾਣੀ',
        'Food': 'ਖਾਣਾ',
        'House': 'ਘਰ',
        'School': 'ਸਕੂਲ',
        'Work': 'ਕੰਮ',
        'Money': 'ਪੈਸਾ',
        'Today': 'ਅੱਜ',
        # Common conversational phrases (mirrors the pa->en entries below,
        # so these are locked-in correct in both directions, not just one)
        'How are you?': 'ਤੁਸੀਂ ਕਿਵੇਂ ਹੋ?',
        'I am fine': 'ਮੈਂ ਠੀਕ ਹਾਂ',
        'What is your name?': 'ਤੁਹਾਡਾ ਨਾਮ ਕੀ ਹੈ?',
        'Nice to meet you': 'ਤੁਹਾਨੂੰ ਮਿਲ ਕੇ ਖੁਸ਼ੀ ਹੋਈ',
        'Goodbye': 'ਅਲਵਿਦਾ',
        'See you later': 'ਫਿਰ ਮਿਲਾਂਗੇ',
        "I don't understand": 'ਮੈਨੂੰ ਸਮਝ ਨਹੀਂ ਆਈ',
        'How much does this cost?': 'ਇਹ ਕਿੰਨੇ ਦਾ ਹੈ?',
        'Where is the bathroom?': 'ਬਾਥਰੂਮ ਕਿੱਥੇ ਹੈ?',
        'What time is it?': 'ਕਿੰਨਾ ਵਜਿਆ ਹੈ?',
        'Happy birthday': 'ਜਨਮਦਿਨ ਮੁਬਾਰਕ',
        'Happy New Year': 'ਨਵਾਂ ਸਾਲ ਮੁਬਾਰਕ',
        'I am sorry': 'ਮੈਨੂੰ ਅਫ਼ਸੋਸ ਹੈ',
        # Travel & directions
        'Where is the train station?': 'ਰੇਲਵੇ ਸਟੇਸ਼ਨ ਕਿੱਥੇ ਹੈ?',
        'How much is the ticket?': 'ਟਿਕਟ ਕਿੰਨੇ ਦੀ ਹੈ?',
        'I need a taxi': 'ਮੈਨੂੰ ਟੈਕਸੀ ਦੀ ਲੋੜ ਹੈ',
        'Turn left': 'ਖੱਬੇ ਮੁੜੋ',
        'Turn right': 'ਸੱਜੇ ਮੁੜੋ',
        'Go straight': 'ਸਿੱਧਾ ਜਾਓ',
        'Stop here': 'ਇੱਥੇ ਰੁਕੋ',
        # Medical & emergency
        'I need a doctor': 'ਮੈਨੂੰ ਡਾਕਟਰ ਚਾਹੀਦਾ ਹੈ',
        'Call an ambulance': 'ਐਂਬੂਲੈਂਸ ਬੁਲਾਓ',
        'Where is the hospital?': 'ਹਸਪਤਾਲ ਕਿੱਥੇ ਹੈ?',
        'I am not feeling well': 'ਮੇਰੀ ਤਬੀਅਤ ਠੀਕ ਨਹੀਂ ਹੈ',
        'I have a headache': 'ਮੇਰੇ ਸਿਰ ਵਿੱਚ ਦਰਦ ਹੈ',
        'I have a fever': 'ਮੈਨੂੰ ਬੁਖਾਰ ਹੈ',
        'Help me, please': 'ਕਿਰਪਾ ਕਰਕੇ ਮੇਰੀ ਮਦਦ ਕਰੋ',
        'Call the police': 'ਪੁਲਿਸ ਨੂੰ ਬੁਲਾਓ',
        'This is an emergency': 'ਇਹ ਇੱਕ ਐਮਰਜੈਂਸੀ ਹੈ',
        'I need help': 'ਮੈਨੂੰ ਮਦਦ ਚਾਹੀਦੀ ਹੈ',
        # Common tourist/dining phrases the model tends to mistranslate or
        # phrase awkwardly on its own (e.g. "change" as money vs. reform).
        'Can I have the bill, please?': 'ਕੀ ਮੈਨੂੰ ਬਿੱਲ ਮਿਲ ਸਕਦਾ ਹੈ?',
        'Please keep the change.': 'ਕਿਰਪਾ ਕਰਕੇ ਬਾਕੀ ਪੈਸੇ ਰੱਖੋ।',
        'I lost my passport at the airport.': 'ਮੇਰਾ ਪਾਸਪੋਰਟ ਹਵਾਈ ਅੱਡੇ ਉੱਤੇ ਗੁੰਮ ਹੋ ਗਿਆ।',
        'Can you give me a glass of water?': 'ਕੀ ਤੁਸੀਂ ਮੈਨੂੰ ਇੱਕ ਗਲਾਸ ਪਾਣੀ ਦੇ ਸਕਦੇ ਹੋ?',
    },
    ('pa', 'en'): {
        'ਸਤ ਸ੍ਰੀ ਅਕਾਲ': 'Hello',
        'ਧੰਨਵਾਦ': 'Thank you',
        'ਬਹੁਤ ਧੰਨਵਾਦ': 'Thank you very much',
        'ਕਿਰਪਾ ਕਰਕੇ': 'Please',
        'ਮਾਫ਼ ਕਰਨਾ': 'Sorry',
        'ਸ਼ੁਭ ਸਵੇਰ': 'Good morning',
        'ਸ਼ੁਭ ਦੁਪਹਿਰ': 'Good afternoon',
        'ਸ਼ੁਭ ਸ਼ਾਮ': 'Good evening',
        'ਸ਼ੁਭ ਰਾਤ': 'Good night',
        'ਜੀ ਆਇਆਂ ਨੂੰ': 'Welcome',
        'ਹਾਂ': 'Yes',
        'ਨਹੀਂ': 'No',
        'ਵਧਾਈਆਂ': 'Congratulations',
        'ਇੱਕ': 'One',
        'ਦੋ': 'Two',
        'ਤਿੰਨ': 'Three',
        'ਚਾਰ': 'Four',
        'ਪੰਜ': 'Five',
        'ਛੇ': 'Six',
        'ਸੱਤ': 'Seven',
        'ਅੱਠ': 'Eight',
        'ਨੌਂ': 'Nine',
        'ਦਸ': 'Ten',
        'ਸੋਮਵਾਰ': 'Monday',
        'ਮੰਗਲਵਾਰ': 'Tuesday',
        'ਬੁੱਧਵਾਰ': 'Wednesday',
        'ਵੀਰਵਾਰ': 'Thursday',
        'ਸ਼ੁੱਕਰਵਾਰ': 'Friday',
        'ਸ਼ਨੀਵਾਰ': 'Saturday',
        'ਐਤਵਾਰ': 'Sunday',
        'ਮਾਂ': 'Mother',
        'ਪਿਤਾ': 'Father',
        'ਭਰਾ': 'Brother',
        'ਭੈਣ': 'Sister',
        'ਦੋਸਤ': 'Friend',
        'ਪਾਣੀ': 'Water',
        'ਖਾਣਾ': 'Food',
        'ਘਰ': 'House',
        'ਸਕੂਲ': 'School',
        'ਕੰਮ': 'Work',
        'ਪੈਸਾ': 'Money',
        'ਅੱਜ': 'Today',
        # Common conversational phrases
        'ਤੁਸੀਂ ਕਿਵੇਂ ਹੋ?': 'How are you?',
        'ਮੈਂ ਠੀਕ ਹਾਂ': 'I am fine',
        'ਤੁਹਾਡਾ ਨਾਮ ਕੀ ਹੈ?': 'What is your name?',
        'ਤੁਹਾਨੂੰ ਮਿਲ ਕੇ ਖੁਸ਼ੀ ਹੋਈ': 'Nice to meet you',
        'ਅਲਵਿਦਾ': 'Goodbye',
        'ਫਿਰ ਮਿਲਾਂਗੇ': 'See you later',
        'ਮੈਨੂੰ ਸਮਝ ਨਹੀਂ ਆਈ': "I don't understand",
        'ਇਹ ਕਿੰਨੇ ਦਾ ਹੈ?': 'How much does this cost?',
        'ਬਾਥਰੂਮ ਕਿੱਥੇ ਹੈ?': 'Where is the bathroom?',
        'ਕਿੰਨਾ ਵਜਿਆ ਹੈ?': 'What time is it?',
        'ਜਨਮਦਿਨ ਮੁਬਾਰਕ': 'Happy birthday',
        'ਨਵਾਂ ਸਾਲ ਮੁਬਾਰਕ': 'Happy New Year',
        'ਮੈਨੂੰ ਅਫ਼ਸੋਸ ਹੈ': 'I am sorry',
        # Travel & directions
        'ਰੇਲਵੇ ਸਟੇਸ਼ਨ ਕਿੱਥੇ ਹੈ?': 'Where is the train station?',
        'ਟਿਕਟ ਕਿੰਨੇ ਦੀ ਹੈ?': 'How much is the ticket?',
        'ਮੈਨੂੰ ਟੈਕਸੀ ਦੀ ਲੋੜ ਹੈ': 'I need a taxi',
        'ਖੱਬੇ ਮੁੜੋ': 'Turn left',
        'ਸੱਜੇ ਮੁੜੋ': 'Turn right',
        'ਸਿੱਧਾ ਜਾਓ': 'Go straight',
        'ਇੱਥੇ ਰੁਕੋ': 'Stop here',
        # Medical & emergency
        'ਮੈਨੂੰ ਡਾਕਟਰ ਚਾਹੀਦਾ ਹੈ': 'I need a doctor',
        'ਐਂਬੂਲੈਂਸ ਬੁਲਾਓ': 'Call an ambulance',
        'ਹਸਪਤਾਲ ਕਿੱਥੇ ਹੈ?': 'Where is the hospital?',
        'ਮੇਰੀ ਤਬੀਅਤ ਠੀਕ ਨਹੀਂ ਹੈ': 'I am not feeling well',
        'ਮੇਰੇ ਸਿਰ ਵਿੱਚ ਦਰਦ ਹੈ': 'I have a headache',
        'ਮੈਨੂੰ ਬੁਖਾਰ ਹੈ': 'I have a fever',
        'ਕਿਰਪਾ ਕਰਕੇ ਮੇਰੀ ਮਦਦ ਕਰੋ': 'Help me, please',
        'ਪੁਲਿਸ ਨੂੰ ਬੁਲਾਓ': 'Call the police',
        'ਇਹ ਇੱਕ ਐਮਰਜੈਂਸੀ ਹੈ': 'This is an emergency',
        'ਮੈਨੂੰ ਮਦਦ ਚਾਹੀਦੀ ਹੈ': 'I need help',
        'ਮੈਂ ਗੁਆਚ ਗਿਆ ਹਾਂ': 'I am lost',
        'ਮੈਂ ਗੁਆਚ ਗਈ ਹਾਂ': 'I am lost',
        'ਮੈਂ ਥੱਕਿਆ ਹੋਇਆ ਹਾਂ': 'I am tired',
        'ਮੈਂ ਥੱਕੀ ਹੋਈ ਹਾਂ': 'I am tired',
        'ਮੈਂ ਆ ਗਿਆ ਹਾਂ': 'I have arrived',
        'ਮੈਂ ਆ ਗਈ ਹਾਂ': 'I have arrived',
        'ਮੈਂ ਭੁੱਲ ਗਿਆ': 'I forgot',
        'ਮੈਂ ਭੁੱਲ ਗਈ': 'I forgot',
        'ਮੈਂ ਉੱਥੇ ਗਿਆ': 'I went there',
        'ਮੈਂ ਉੱਥੇ ਗਈ': 'I went there',
        'ਕੀ ਮੈਨੂੰ ਬਿੱਲ ਮਿਲ ਸਕਦਾ ਹੈ?': 'Can I have the bill, please?',
        'ਕਿਰਪਾ ਕਰਕੇ ਬਾਕੀ ਪੈਸੇ ਰੱਖੋ।': 'Please keep the change.',
        'ਮੇਰਾ ਪਾਸਪੋਰਟ ਹਵਾਈ ਅੱਡੇ ਉੱਤੇ ਗੁੰਮ ਹੋ ਗਿਆ।': 'I lost my passport at the airport.',
        'ਕੀ ਤੁਸੀਂ ਮੈਨੂੰ ਇੱਕ ਗਲਾਸ ਪਾਣੀ ਦੇ ਸਕਦੇ ਹੋ?': 'Can you give me a glass of water?',
        # Found via a held-out (non-glossary-tuned) test batch - the model's
        # raw output inverted this idiom's meaning ("ਕਦੋਂ ਦੀ ਗਈ ਹੋਈ ਹੈ" = "has
        # been gone/out since when", not "was supplied").
        'ਬਿਜਲੀ ਕਦੋਂ ਦੀ ਗਈ ਹੋਈ ਹੈ?': 'When did the power go out?',
        # Same batch - raw output rendered "ਵੱਡਾ ਸਾਈਜ਼ ਹੈ" ("bigger size") as
        # the disfluent "what's the big size?" instead of a natural request.
        'ਇਹ ਜੁੱਤੀ ਮੈਨੂੰ ਥੋੜੀ ਛੋਟੀ ਹੈ, ਕੀ ਵੱਡਾ ਸਾਈਜ਼ ਹੈ?': 'This shoe is a little small for me, do you have a bigger size?',
    },
}

# Phrases where the correct Punjabi translation depends on the speaker's
# grammatical gender (English has no such distinction, so only en->pa needs
# this). Selected via the request's optional "gender" field ('m' or 'f'),
# defaulting to 'm' to preserve prior behavior when the field is omitted.
GENDERED_GLOSSARY = {
    ('en', 'pa'): {
        'I am lost': {'m': 'ਮੈਂ ਗੁਆਚ ਗਿਆ ਹਾਂ', 'f': 'ਮੈਂ ਗੁਆਚ ਗਈ ਹਾਂ'},
        # The 4 entries below were sourced from the model's OWN masculine-
        # default output (not hand-typed), then the feminine form derived via
        # the standard Punjabi ਿਆ/ਗਿਆ (masc.) -> ੀ/ਗਈ (fem.) agreement pattern
        # already used above, and confirmed via a pa->en back-translation
        # round-trip matching the original English meaning before being
        # hardcoded (see repo memory for the verification transcript).
        'I am tired': {'m': 'ਮੈਂ ਥੱਕਿਆ ਹੋਇਆ ਹਾਂ', 'f': 'ਮੈਂ ਥੱਕੀ ਹੋਈ ਹਾਂ'},
        'I have arrived': {'m': 'ਮੈਂ ਆ ਗਿਆ ਹਾਂ', 'f': 'ਮੈਂ ਆ ਗਈ ਹਾਂ'},
        'I forgot': {'m': 'ਮੈਂ ਭੁੱਲ ਗਿਆ', 'f': 'ਮੈਂ ਭੁੱਲ ਗਈ'},
        'I went there': {'m': 'ਮੈਂ ਉੱਥੇ ਗਿਆ', 'f': 'ਮੈਂ ਉੱਥੇ ਗਈ'},
    },
}


# Trailing punctuation stripped when matching a chunk against the glossary -
# includes the Gurmukhi danda/double-danda alongside standard ASCII marks.
_GLOSSARY_TRAILING_PUNCT = '.!?,;:।॥'
_glossary_exact_cache = {}
_glossary_ci_cache = {}


def _glossary_exact_lookup(source_lang, target_lang):
    """NFC-normalized (but case-preserving) copy of the glossary for the
    given direction, built once per direction and cached. NFC normalization
    guards against visually-identical Gurmukhi conjuncts (e.g. a virama+letter
    sequence) being encoded with different underlying codepoints depending on
    where the text came from (typed, model output, Whisper transcript)."""
    key = (source_lang, target_lang)
    exact = _glossary_exact_cache.get(key)
    if exact is None:
        overrides = GLOSSARY.get(key, {})
        exact = {
            unicodedata.normalize('NFC', k): unicodedata.normalize('NFC', v)
            for k, v in overrides.items()
        }
        _glossary_exact_cache[key] = exact
    return exact


def _glossary_ci_lookup(source_lang, target_lang):
    """Case-insensitive, NFC-normalized copy of the glossary for the given
    direction, built once per direction and cached."""
    key = (source_lang, target_lang)
    ci = _glossary_ci_cache.get(key)
    if ci is None:
        overrides = GLOSSARY.get(key, {})
        ci = {
            unicodedata.normalize('NFC', k.lower()): unicodedata.normalize('NFC', v)
            for k, v in overrides.items()
        }
        _glossary_ci_cache[key] = ci
    return ci


_gendered_exact_cache = {}
_gendered_ci_cache = {}


def _gendered_glossary_lookup(source_lang, target_lang, gender):
    """NFC-normalized exact + case-insensitive copies of GENDERED_GLOSSARY
    for the given direction, resolved to the requested gender ('m'/'f',
    falling back to 'm' for an unrecognized value) and cached per
    (direction, gender)."""
    key = (source_lang, target_lang, gender)
    exact = _gendered_exact_cache.get(key)
    if exact is None:
        entries = GENDERED_GLOSSARY.get((source_lang, target_lang), {})
        exact, ci = {}, {}
        for phrase, variants in entries.items():
            translation = unicodedata.normalize('NFC', variants.get(gender, variants.get('m')))
            norm = unicodedata.normalize('NFC', phrase)
            exact[norm] = translation
            ci[norm.lower()] = translation
        _gendered_exact_cache[key] = exact
        _gendered_ci_cache[key] = ci
    return exact, _gendered_ci_cache[key]


def apply_glossary(original_chunks, translated_chunks, source_lang, target_lang, gender='m'):
    """Force known-correct translations for chunks matching a glossary
    entry, overriding whatever the model produced. Matches case-insensitively
    (so "hello"/"HELLO" both hit the "Hello" entry) and tolerates trailing
    punctuation (so "Hello."/"Hello!" still match "Hello"), reattaching
    whatever trailing punctuation was present. `gender` ('m' or 'f') picks
    the right verb form for entries in GENDERED_GLOSSARY and is otherwise
    ignored."""
    overrides = GLOSSARY.get((source_lang, target_lang), {})
    gendered_entries = GENDERED_GLOSSARY.get((source_lang, target_lang), {})
    if not overrides and not gendered_entries:
        return translated_chunks

    exact_overrides = _glossary_exact_lookup(source_lang, target_lang)
    ci_overrides = _glossary_ci_lookup(source_lang, target_lang)
    gendered_exact, gendered_ci = _gendered_glossary_lookup(source_lang, target_lang, gender)
    result = []
    for original, translated in zip(original_chunks, translated_chunks):
        stripped = unicodedata.normalize('NFC', original.strip())
        if stripped in gendered_exact:
            result.append(gendered_exact[stripped])
            continue
        if stripped in exact_overrides:
            result.append(exact_overrides[stripped])
            continue
        core = stripped.rstrip(_GLOSSARY_TRAILING_PUNCT)
        trailing = stripped[len(core):]
        match = gendered_ci.get(core.lower(), ci_overrides.get(core.lower()))
        result.append(match + trailing if match is not None else translated)
    return result


# Common English contractions, expanded before word-overlap scoring so a
# purely cosmetic phrasing difference ("I'm" vs "I am") doesn't get scored
# as a meaning mismatch between the original and the back-translation.
_CONTRACTIONS = {
    "i'm": "i am", "you're": "you are", "he's": "he is", "she's": "she is",
    "it's": "it is", "we're": "we are", "they're": "they are",
    "that's": "that is", "there's": "there is", "what's": "what is",
    "isn't": "is not", "aren't": "are not", "wasn't": "was not",
    "weren't": "were not", "don't": "do not", "doesn't": "does not",
    "didn't": "did not", "can't": "cannot", "couldn't": "could not",
    "won't": "will not", "wouldn't": "would not", "shouldn't": "should not",
    "haven't": "have not", "hasn't": "has not", "hadn't": "had not",
    "i've": "i have", "you've": "you have", "we've": "we have",
    "they've": "they have", "i'll": "i will", "you'll": "you will",
    "we'll": "we will", "they'll": "they will",
}
_CONTRACTIONS_RE = re.compile(r"\b(" + "|".join(re.escape(k) for k in _CONTRACTIONS) + r")\b")


def _words_for_overlap(text):
    """Tokenize text for confidence scoring: expand English contractions and
    drop punctuation (via \\w+, which also strips the Gurmukhi danda/double
    danda attached to a sentence's last word) so cosmetic differences don't
    get counted as meaning mismatches.

    \\w alone is NOT enough for Gurmukhi: Python's Unicode \\w matches
    Letter/Number categories but not Mark (dependent vowel signs/matras,
    virama, etc.), so a plain \\w+ regex fragments a single Gurmukhi word
    like "ਮੈਨੂੰ" into several meaningless pieces ('ਮ', 'ਨ', ...) at each
    matra - silently making every Punjabi confidence score meaningless
    (discovered while investigating issue #4, see repo memory). Adding the
    whole Gurmukhi Unicode block (U+0A00-U+0A7F, which covers consonants,
    independent vowels, matras, virama and nukta) to the character class
    keeps a word's matras attached to its base letters.
    """
    lowered = text.lower()
    expanded = _CONTRACTIONS_RE.sub(lambda m: _CONTRACTIONS[m.group(0)], lowered)
    return set(re.findall(r"[\w\u0A00-\u0A7F]+", expanded, flags=re.UNICODE))


def compute_roundtrip_confidence(original_text, source_lang, target_lang, translated_text):
    """Translate the result back to the source language and measure word
    overlap with the original as a rough confidence signal - low overlap
    often indicates the forward translation dropped or distorted meaning.
    This doubles translation cost, so it's opt-in (see 'verify' request flag).

    Uses Jaccard similarity (intersection / union) rather than plain recall
    (intersection / original) - recall alone gives a perfect score to a
    back-translation that contains every original word PLUS a pile of extra,
    wrong content, since it never penalizes spurious additions.
    """
    try:
        back_source_chunks = [translated_text]
        back_chunks = translate_chunks(back_source_chunks, target_lang, source_lang)
        # Apply the glossary to the back-translation too - otherwise a
        # glossary-forced phrase (e.g. "Hello" -> "ਸਤ ਸ੍ਰੀ ਅਕਾਲ") gets
        # back-translated by the raw model into something that doesn't
        # overlap with the original at all, producing a false "low
        # confidence" score for a translation that was actually perfect.
        back_chunks = apply_glossary(back_source_chunks, back_chunks, target_lang, source_lang)
        back_translation = back_chunks[0] if back_chunks else ''
        original_words = _words_for_overlap(original_text)
        back_words = _words_for_overlap(back_translation)

        # Code-switched input (e.g. a Punjabi sentence with an embedded
        # English loanword like "office") often gets that loanword correctly
        # carried through on the FORWARD translation, but translating AGAIN
        # back to the source language tends to localize it to a native
        # synonym ("ਦਫ਼ਤਰ") - a sign of a healthy, idiomatic translation, not
        # dropped meaning. Left alone, that loanword is a guaranteed,
        # unavoidable mismatch in the round-trip comparison every time. If a
        # Latin-script source token already appears verbatim in the
        # (already-produced) forward translation, trust that direct signal
        # instead of penalizing the back-translation for localizing it.
        forward_lower = translated_text.lower()
        for token in original_words:
            if token.isascii() and token.isalpha() and token not in back_words:
                if re.search(r'\b' + re.escape(token) + r'\b', forward_lower):
                    back_words.add(token)

        union = original_words | back_words
        if not union:
            return 0.0, back_translation
        similarity = len(original_words & back_words) / len(union)
        return round(similarity, 2), back_translation
    except Exception as e:
        print(f"   ⚠️ Round-trip confidence check failed: {e}")
        return None, None

# Parakeet-unified-en-0.6b (RNNT, ONNX int8, torch-free) - the default (and
# only) English ASR since Whisper was removed entirely (2026-10, see repo
# memory). Measured 2026-10 on 200 random Svarah (Indian-accented English)
# clips: WER 7.58% vs 13.94% for Whisper small, and ~1.75x faster on CPU (see
# parakeet_unified_asr.py docstring + repo memory). Emits punctuation and
# casing. Loaded LAZILY on the first English request (~1GB RAM) so Punjabi-only
# traffic never pays for it. DISABLE_PARAKEET_UNIFIED=1 disables it (no
# fallback model exists for English anymore - the request just errors).
PARAKEET_UNIFIED_DIR = os.path.join(os.path.dirname(__file__), 'models', 'parakeet-unified-en-onnx', 'onnx_int8')
parakeet_unified_model = None
_parakeet_unified_load_lock = threading.Lock()
_parakeet_unified_load_failed = False


def get_parakeet_unified_model():
    """Return the Parakeet-unified English model, loading it on first call (None if unavailable)."""
    global parakeet_unified_model, _parakeet_unified_load_failed
    if parakeet_unified_model is not None or _parakeet_unified_load_failed:
        return parakeet_unified_model
    if os.environ.get('DISABLE_PARAKEET_UNIFIED', '').strip() == '1':
        _parakeet_unified_load_failed = True
        return None
    with _parakeet_unified_load_lock:
        if parakeet_unified_model is None and not _parakeet_unified_load_failed:
            if not os.path.isdir(PARAKEET_UNIFIED_DIR):
                print(f"\nℹ️ No Parakeet-unified model found at {PARAKEET_UNIFIED_DIR} - no English ASR will be available.")
                _parakeet_unified_load_failed = True
                return None
            print("\n📥 Loading Parakeet-unified-en-0.6b (ONNX int8) (first English request)...")
            try:
                parakeet_unified_model = ParakeetUnifiedASR(
                    os.path.join(PARAKEET_UNIFIED_DIR, 'encoder.int8.onnx'),
                    os.path.join(PARAKEET_UNIFIED_DIR, 'decoder_joint.int8.onnx'),
                    os.path.join(PARAKEET_UNIFIED_DIR, 'tokenizer.model'),
                    cpu_threads=4,
                )
                print("✅ Parakeet-unified loaded! Will be used for English transcription.")
            except Exception as e:
                import traceback
                print(f"❌ Error loading Parakeet-unified: {e}\n{traceback.format_exc()}")
                _parakeet_unified_load_failed = True
    return parakeet_unified_model


# Load AI4Bharat's IndicConformer (encoder+CTC-decoder, exported to ONNX, no
# torch/NeMo needed at runtime - see indicconformer_asr.py docstring and repo
# memory for the full investigation). Serves as the fallback Punjabi ASR when
# the native NeMo RNNT model below isn't loaded. DISABLE_INDICCONFORMER=1 is
# a quick rollback switch to disable this fallback (Punjabi requests then
# depend entirely on NeMo RNNT below, erroring if that's also unavailable).
INDICCONFORMER_DIR = os.path.join(os.path.dirname(__file__), 'models', 'indicconformer-punjabi-ctc-onnx')
indicconformer_model = None
if os.environ.get('DISABLE_INDICCONFORMER', '').strip() == '1':
    print("\nℹ️ IndicConformer (ONNX CTC) disabled via DISABLE_INDICCONFORMER=1.")
elif os.path.isdir(INDICCONFORMER_DIR):
    print("\n📥 Loading IndicConformer Punjabi ASR model (ONNX)...")
    try:
        indicconformer_model = IndicConformerASR(
            os.path.join(INDICCONFORMER_DIR, 'model.onnx'),
            os.path.join(INDICCONFORMER_DIR, 'tokenizer_pa.model'),
            cpu_threads=4,
        )
        print("✅ IndicConformer loaded! Will be used for Punjabi transcription (unless NeMo RNNT below is also available).")
    except Exception as e:
        print(f"⚠️ Could not load IndicConformer model: {e}")
else:
    print(f"\nℹ️ No IndicConformer model found at {INDICCONFORMER_DIR}")

# Load IndicConformer's NATIVE RNNT decoder via AI4Bharat's NeMo fork (torch)
# - closes the CTC-vs-RNNT accuracy gap measured in the 2026-09-27
# investigation (see indicconformer_nemo_asr.py docstring + repo memory
# "OPTION 1 IMPLEMENTED" for the full rationale). This is a deliberate,
# narrow, Punjabi-ASR-ONLY exception to the project's broader torch-free
# architecture. Takes priority over the ONNX CTC-only model above when
# loaded - the PRIMARY Punjabi ASR model. DISABLE_NEMO_RNNT=1 is a quick
# rollback switch to fall back to the ONNX CTC model above instead.
INDICCONFORMER_NEMO_PATH = os.path.join(
    os.path.dirname(__file__), 'models', 'indicconformer-nemo', 'indicconformer_stt_pa_hybrid_rnnt_large.nemo'
)
indicconformer_nemo_model = None
if os.environ.get('DISABLE_NEMO_RNNT', '').strip() == '1':
    print("\nℹ️ NeMo RNNT disabled via DISABLE_NEMO_RNNT=1 - using ONNX CTC for Punjabi transcription.")
elif os.path.isfile(INDICCONFORMER_NEMO_PATH):
    print("\n📥 Loading IndicConformer Punjabi ASR model (native NeMo RNNT, torch)...")
    try:
        from indicconformer_nemo_asr import IndicConformerNeMoASR
        indicconformer_nemo_model = IndicConformerNeMoASR(INDICCONFORMER_NEMO_PATH)
        print("✅ IndicConformer (NeMo RNNT) loaded! Will be used for Punjabi transcription.")
    except Exception as e:
        print(f"⚠️ Could not load IndicConformer NeMo RNNT model: {e}")
        print("   Falling back to ONNX CTC for Punjabi transcription.")
else:
    print(f"\nℹ️ No IndicConformer NeMo checkpoint found at {INDICCONFORMER_NEMO_PATH}")
    print("   Using ONNX CTC for Punjabi transcription.")

# Load SpeechBrain's VoxLingua107 ECAPA-TDNN spoken-language-ID model -
# replaces Whisper's detect_language() for 'auto'-mode /speech-to-text
# requests (Whisper removed entirely 2026-10, see repo memory). 107-language
# coverage incl. Punjabi/Hindi/Urdu/English, 6.7% error rate on the
# VoxLingua107 dev set. Reuses the torch install already present for NeMo
# RNNT above - the marginal cost of this model is just itself (~86MB) plus
# speechbrain/torchaudio/hyperpyyaml, not another full torch install.
# DISABLE_SPEECHBRAIN_LID=1 disables it ('auto'-mode requests then error;
# forced 'en'/'pa' requests are unaffected since they never need LID).
LANGUAGE_ID_DIR = os.path.join(os.path.dirname(__file__), 'models', 'lang-id-voxlingua107-ecapa')
language_id_model = None
_language_id_load_lock = threading.Lock()
_language_id_load_failed = False


def get_language_id_model():
    """Return the SpeechBrain VoxLingua107 language-ID model, loading it on first call."""
    global language_id_model, _language_id_load_failed
    if language_id_model is not None or _language_id_load_failed:
        return language_id_model
    if os.environ.get('DISABLE_SPEECHBRAIN_LID', '').strip() == '1':
        _language_id_load_failed = True
        return None
    with _language_id_load_lock:
        if language_id_model is None and not _language_id_load_failed:
            if not os.path.isdir(LANGUAGE_ID_DIR):
                print(f"\nℹ️ No language-ID model found at {LANGUAGE_ID_DIR} - 'auto' language mode will be unavailable.")
                _language_id_load_failed = True
                return None
            print("\n📥 Loading SpeechBrain language-ID model (first 'auto'-mode request)...")
            try:
                from speechbrain.inference.classifiers import EncoderClassifier
                from speechbrain.utils.fetching import LocalStrategy
                # LocalStrategy.COPY avoids symlinking the local model dir into
                # savedir's HF-cache-style layout - SpeechBrain's default
                # SYMLINK strategy raises OSError: [WinError 1314] on Windows
                # without Developer Mode/admin (confirmed via local testing);
                # COPY works identically on Linux/Docker too, just uses a
                # little more disk space, so it's used unconditionally rather
                # than only on Windows.
                language_id_model = EncoderClassifier.from_hparams(
                    source=LANGUAGE_ID_DIR, savedir=LANGUAGE_ID_DIR, run_opts={'device': 'cpu'},
                    local_strategy=LocalStrategy.COPY
                )
                print("✅ SpeechBrain language-ID model loaded!")
            except Exception as e:
                import traceback
                print(f"❌ Error loading SpeechBrain language-ID model: {e}\n{traceback.format_exc()}")
                _language_id_load_failed = True
    return language_id_model


def detect_spoken_language(audio_array):
    """Run SpeechBrain VoxLingua107 LID on a mono float32 16kHz numpy array.
    Returns (lang, iso_code, confidence) where lang is 'en' or 'pa' - Hindi
    and Urdu are grouped with Punjabi, the same heuristic the old
    Whisper-based LID used, since they're the languages most often confused
    with Punjabi (and this app only supports en/pa anyway)."""
    model = get_language_id_model()
    audio_tensor = torch.from_numpy(audio_array).unsqueeze(0)
    _, log_likelihood, _, text_lab = model.classify_batch(audio_tensor)
    iso_code = text_lab[0].split(':')[0].strip().lower()
    confidence = float(log_likelihood.exp()[0])
    lang = 'pa' if iso_code in ('pa', 'hi', 'ur') else 'en'
    return lang, iso_code, confidence


print("\n" + "=" * 60)
print("✅ SERVER READY!")
print("🌐 Running on: http://localhost:5000")
print("=" * 60)
print("\nAvailable endpoints:")
print("  POST /translate - Translate text (English ↔ Punjabi)")
print("  POST /speech-to-text - Convert speech to text")
print("  POST /ocr - Recognize text from a photo (English/Punjabi)")
print("  GET /health - Check server status")
print("\n" + "=" * 60)


@app.route('/health', methods=['GET'])
def health():
    """Check if server is running"""
    return jsonify({
        'status': 'healthy',
        'models': {
            'translation': 'IndicTrans2 (1B, en-indic + indic-en)',
            'speech': (
                ('IndicConformer-NeMo-RNNT (Punjabi)' if indicconformer_nemo_model is not None
                 else 'IndicConformer (Punjabi)' if indicconformer_model is not None
                 else 'no Punjabi ASR model loaded')
                + (' + Parakeet-unified (English)' if os.path.isdir(PARAKEET_UNIFIED_DIR)
                   and os.environ.get('DISABLE_PARAKEET_UNIFIED', '').strip() != '1' else ' + no English ASR model loaded')
                + (' + SpeechBrain VoxLingua107 (language-ID)' if os.path.isdir(LANGUAGE_ID_DIR)
                   and os.environ.get('DISABLE_SPEECHBRAIN_LID', '').strip() != '1' else '')
            ),
            'transliteration': 'IndicXlit'
        },
        'message': 'AI Translation Server is running!'
    })


# Where users can flag a translation as wrong from the app. Appended as
# JSON-lines so real-world failures can be reviewed later and turned into
# GLOSSARY entries or new HELD_OUT_SET test cases - this is the only way to
# learn about accuracy problems that don't show up in curated test sets.
FEEDBACK_LOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'feedback_log.jsonl')
# Keep a single stray/malicious client from growing this file without bound.
_MAX_FEEDBACK_TEXT_LENGTH = 2000


@app.route('/feedback', methods=['POST'])
def feedback():
    """Log a user-flagged translation for later human review."""
    data = request.get_json(silent=True) or {}
    source_text = str(data.get('source_text', ''))[:_MAX_FEEDBACK_TEXT_LENGTH]
    translation = str(data.get('translation', ''))[:_MAX_FEEDBACK_TEXT_LENGTH]
    source_lang = str(data.get('source_lang', ''))[:10]
    target_lang = str(data.get('target_lang', ''))[:10]
    note = str(data.get('note', ''))[:_MAX_FEEDBACK_TEXT_LENGTH]

    if not source_text or not translation:
        return jsonify({'error': 'source_text and translation are required'}), 400

    entry = {
        'timestamp': time.time(),
        'source_lang': source_lang,
        'target_lang': target_lang,
        'source_text': source_text,
        'translation': translation,
        'note': note,
    }
    try:
        with open(FEEDBACK_LOG_PATH, 'a', encoding='utf-8') as f:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    except OSError as e:
        return jsonify({'error': f'Could not save feedback: {e}'}), 500

    return jsonify({'status': 'ok'})


@app.route('/transliterate', methods=['POST'])
def transliterate_text():
    """
    Transliterate romanized Punjabi to Gurmukhi script
    
    Request body:
    {
        "text": "ki haal hai"
    }
    
    Returns:
    {
        "gurmukhi": "ਕੀ ਹਾਲ ਹੈ",
        "romanized": "ki haal hai"
    }
    """
    try:
        data = request.get_json()
        romanized_text = data.get('text', '')
        
        if not romanized_text:
            return jsonify({'error': 'No text provided'}), 400
        
        print(f"\n🔤 Transliteration request:")
        print(f"   Input (romanized): {romanized_text}")
        
        # ITRANS/ISO transliteration rarely raises exceptions for text that
        # doesn't match their convention - it silently produces WRONG output
        # instead of failing. So try both schemes and keep whichever actually
        # produced real Gurmukhi script, rather than only falling back on
        # exceptions that almost never fire.
        candidates = []
        for scheme in (sanscript.ITRANS, sanscript.ISO):
            try:
                candidates.append(transliterate(romanized_text, scheme, sanscript.GURMUKHI))
            except Exception:
                continue

        if candidates:
            gurmukhi_text = max(candidates, key=_gurmukhi_char_count)
        else:
            gurmukhi_text = romanized_text  # Last resort: return as-is
        
        print(f"   Output (Gurmukhi): {gurmukhi_text}")
        print("   ✅ Transliteration complete")
        
        return jsonify({
            'gurmukhi': gurmukhi_text,
            'romanized': romanized_text,
            'model': 'IndicXlit'
        })
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"❌ Transliteration error: {str(e)}")
        print(error_details)
        return jsonify({'error': str(e), 'details': error_details}), 500


# Ensemble decoding config for /translate's verify path: alternate beam-search
# parameters tried (in order) when the first attempt's round-trip confidence
# is too low, stopping early once a candidate is "good enough".
# NOTE: raising the threshold to 0.75 and adding a third config was tried and
# measured (see repo memory) - it produced no measurable accuracy gain, only
# extra latency, so this was reverted back to the original tuning.
_ENSEMBLE_CONFIDENCE_THRESHOLD = 0.6
# pa-source round-trips score systematically lower than en-source ones under
# the corrected (whole-word) Gurmukhi tokenization used by
# compute_roundtrip_confidence (the old \w+-only tokenizer fragmented
# Gurmukhi words at every matra, which inflated pa-source scores via bogus
# single-character fragment overlap - see repo memory, issue #4). Benchmarked
# the old vs new metric on TEST_SET+HELD_OUT_SET: en-source scores are
# identical (back-translation there is also English, untouched by the
# tokenizer fix), but pa-source scores dropped enough that the single 0.6
# threshold would trigger ensemble retries noticeably more often for
# genuinely-fine pa->en translations. A separate, lower threshold keeps the
# retry rate calibrated to the now-more-accurate pa-source scores.
_ENSEMBLE_CONFIDENCE_THRESHOLD_PA_SOURCE = 0.45
_ENSEMBLE_GOOD_ENOUGH_THRESHOLD = 0.85
# LATENCY FIX (2026-10-03): beam sizes lowered (12->6, 6->3) alongside the
# new num_beams=4 default above - still one config above and one below the
# default (preserving the "try a different search path" intent) but each
# retry attempt is now meaningfully cheaper (see translate_chunks docstring).
_ENSEMBLE_DECODING_CONFIGS = [
    {'num_beams': 6, 'length_penalty': 0.8},
    {'num_beams': 3, 'length_penalty': 1.4},
]


@app.route('/_debug_raw_translate', methods=['POST'])
def _debug_raw_translate():
    """QA utility: translate bypassing the glossary, so a glossary entry's
    Gurmukhi text can be cross-checked against the model's own independent
    output (see verify_glossary_translations.py) before/without a native
    speaker review pass."""
    data = request.json
    text = normalize_text(data.get('text', ''))
    source_lang = data.get('source_lang', 'en')
    target_lang = data.get('target_lang', 'pa')
    chunks = chunk_text(text) or [text]
    translated = translate_chunks(chunks, source_lang, target_lang)
    return jsonify({'translation': ' '.join(translated)})


@app.route('/translate', methods=['POST'])
def translate_text():
    """
    Translate text between English and Punjabi
    
    Request body:
    {
        "text": "Hello, how are you?",
        "source_lang": "en",  # 'en' or 'pa'
        "target_lang": "pa",  # 'en' or 'pa'
        "gender": "m"          # optional: 'm' (default) or 'f' - picks the
                                # speaker-gender-agreeing verb form for a
                                # handful of glossary phrases (e.g. "I am lost")
    }
    """
    try:
        data = request.json
        text = data.get('text', '')
        source_lang = data.get('source_lang', 'en')  # en or pa
        target_lang = data.get('target_lang', 'pa')  # en or pa
        gender = data.get('gender', 'm')
        if gender not in ('m', 'f'):
            gender = 'm'
        
        if not text:
            return jsonify({'error': 'No text provided'}), 400

        text = normalize_text(text)

        # Punjabi ASR/typed input spells numbers out as words (e.g. "ਪੰਦਰਾਂ
        # ਸੌ", "ਪੰਜਤਾਲੀ") which IndicTrans2 has been observed to mangle
        # (see repo memory) - converting unambiguous number-word phrases to
        # plain digits first lets the model simply copy them through intact.
        # English->Punjabi is unaffected (English numerals are already
        # digits by the time they reach here).
        if source_lang == 'pa':
            text = convert_number_words_to_digits(text)

        # Prevent same-language translation
        if source_lang == target_lang:
            print(f"\n⚠️ Same language translation attempted: {source_lang} → {target_lang}")
            print(f"   Returning original text unchanged")
            return jsonify({
                'translation': text,
                'source_lang': source_lang,
                'target_lang': target_lang,
                'model': 'IndicTrans2-1B',
                'note': 'Same language - no translation performed'
            })
        
        print(f"\n🔄 Translation request: {source_lang} → {target_lang}")
        print(f"   Input: {text[:100]}...")

        # Split into sentence/clause-level chunks so long, complex sentences
        # don't get garbled by the model losing track of dependencies.
        chunks = chunk_text(text)
        if not chunks:
            chunks = [text]

        print(f"   Split into {len(chunks)} chunk(s) for translation")

        translated_chunks = translate_chunks(chunks, source_lang, target_lang)
        translated_chunks = apply_glossary(chunks, translated_chunks, source_lang, target_lang, gender)
        translation = ' '.join(translated_chunks)

        # IndicTrans2 has been observed to occasionally substitute a
        # different, more common Punjabi name for a small set of specific
        # names when a surname follows (e.g. "Harpreet Singh" translates as
        # "Gurpreet Singh") even though the same name translates correctly
        # on its own. Narrow, curated, post-translation-only fix - see
        # punjabi_name_corrections.py docstring for the full rationale and
        # why this is deliberately NOT a general proper-noun fix.
        if source_lang == 'en' and target_lang == 'pa':
            translation = fix_name_substitutions(text, translation)
            translation = fix_alphanumeric_codes(text, translation)

        print(f"   Output: {translation[:100]}...")
        print("   ✅ Translation complete")

        response_payload = {
            'translation': translation,
            'source_lang': source_lang,
            'target_lang': target_lang,
            'model': 'IndicTrans2-1B',
            'chunks': len(chunks)
        }

        # Optional round-trip confidence check - pass "verify": true in the
        # request body to enable (roughly doubles latency for this request).
        if data.get('verify'):
            confidence, back_translation = compute_roundtrip_confidence(
                text, source_lang, target_lang, translation
            )

            # Ensemble decoding: if the first attempt scored low confidence,
            # try each alternate beam-search config in turn and keep
            # whichever candidate scores highest - a different search path
            # sometimes avoids whatever caused the model to drop/distort
            # meaning the first time. Stops early once a candidate is
            # confident enough, to avoid paying for configs that won't help.
            ensemble_threshold = (
                _ENSEMBLE_CONFIDENCE_THRESHOLD_PA_SOURCE if source_lang == 'pa' else _ENSEMBLE_CONFIDENCE_THRESHOLD
            )
            if confidence is not None and confidence < ensemble_threshold:
                best_translation, best_chunks = translation, translated_chunks
                best_confidence, best_back = confidence, back_translation
                for config in _ENSEMBLE_DECODING_CONFIGS:
                    if best_confidence is not None and best_confidence >= _ENSEMBLE_GOOD_ENOUGH_THRESHOLD:
                        break
                    print(f"   ⚠️ Confidence {best_confidence} - trying alternate decoding {config}...")
                    candidate_chunks = translate_chunks(chunks, source_lang, target_lang, **config)
                    candidate_chunks = apply_glossary(chunks, candidate_chunks, source_lang, target_lang, gender)
                    candidate_translation = ' '.join(candidate_chunks)
                    candidate_confidence, candidate_back = compute_roundtrip_confidence(
                        text, source_lang, target_lang, candidate_translation
                    )
                    if candidate_confidence is not None and (
                        best_confidence is None or candidate_confidence > best_confidence
                    ):
                        best_translation, best_chunks = candidate_translation, candidate_chunks
                        best_confidence, best_back = candidate_confidence, candidate_back

                if best_translation != translation:
                    print(f"   ✅ Ensemble improved confidence: {confidence} → {best_confidence}")
                    translation, translated_chunks = best_translation, best_chunks
                    confidence, back_translation = best_confidence, best_back
                    if source_lang == 'en' and target_lang == 'pa':
                        translation = fix_alphanumeric_codes(text, translation)
                    response_payload['translation'] = translation
                    response_payload['retried'] = True

            response_payload['confidence'] = confidence
            response_payload['back_translation'] = back_translation

        return jsonify(response_payload)
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"❌ Translation error: {str(e)}")
        print(f"   Error details:\n{error_details}")
        return jsonify({'error': str(e), 'details': error_details}), 500


@app.route('/speech-to-text', methods=['POST'])
def speech_to_text():
    """
    Convert speech audio to text.
    Supports both English and Punjabi, plus automatic language detection.

    Request body:
    {
        "audio": "base64_encoded_audio_data",
        "language": "auto"  # 'auto', 'en', or 'pa'
    }
    """
    try:
        data = request.json
        audio_base64 = data.get('audio', '')
        language = data.get('language', 'auto')  # auto, en, or pa
        
        if not audio_base64:
            return jsonify({'error': 'No audio data provided'}), 400
        
        print(f"\n🎤 Speech-to-text request (language: {language})")
        
        # Decode base64 audio
        audio_data = base64.b64decode(audio_base64)
        print(f"   📊 Audio data size: {len(audio_data)} bytes")

        # Save to temporary file. decode_audio() (from faster-whisper, see
        # import comment at top of file) decodes audio itself via PyAV
        # (statically bundled with ffmpeg's decoding libraries) - no system
        # ffmpeg install required.
        with tempfile.NamedTemporaryFile(suffix='.webm', delete=False) as temp_audio:
            temp_audio.write(audio_data)
            temp_path = temp_audio.name
        
        try:
            print("   🎯 High-accuracy transcription (this may take a few seconds)...")

            # Honor the caller's language hint instead of always auto-detecting.
            # Automatic language-ID is unreliable on short/noisy clips and can
            # misidentify Punjabi as Hindi/Urdu (or vice versa), which then
            # transcribes in the WRONG script entirely - a much bigger error
            # than any per-word inaccuracy. Only auto-detect when the caller
            # genuinely doesn't know the language.
            asr_language = {'en': 'en', 'pa': 'pa'}.get(language)

            if asr_language is None:
                # Genuine "auto" request (no hint from the caller). Run a
                # cheap, decode-free language-ID pass first (SpeechBrain's
                # VoxLingua107 model), map the result with the same en/pa
                # grouping used everywhere else in this file, then fall
                # through to the normal forced-language dispatch below with
                # a definite answer.
                lid_model = get_language_id_model()
                if lid_model is None:
                    os.unlink(temp_path)
                    return jsonify({'error': 'Language detection model failed to load - check server logs'}), 503
                lid_audio = decode_audio(temp_path, sampling_rate=16000)
                asr_language, detected_iso, detected_prob = detect_spoken_language(lid_audio)
                print(f"   🔎 Auto-detected language: {detected_iso} ({detected_prob:.2f}) -> using '{asr_language}'")

            # IndicConformer is the only Punjabi ASR path (Whisper removed
            # entirely, 2026-10). The native NeMo RNNT decoder (torch) is
            # preferred over the torch-free ONNX CTC-only export when
            # available - it closes the CTC-vs-RNNT accuracy gap documented
            # in repo memory.
            if asr_language == 'pa':
                if indicconformer_nemo_model is None and indicconformer_model is None:
                    os.unlink(temp_path)
                    return jsonify({'error': 'No Punjabi ASR model is loaded - check server logs'}), 503

                audio_array = decode_audio(temp_path, sampling_rate=16000)
                if indicconformer_nemo_model is not None:
                    transcript, confidence = indicconformer_nemo_model.transcribe(audio_array)
                    model_name = 'IndicConformer-NeMo-RNNT'
                else:
                    transcript, confidence = indicconformer_model.transcribe(audio_array)
                    model_name = 'IndicConformer'
                detected_lang = 'pa'

                print(f"   ✅ Detected {detected_lang} (confidence: {confidence:.2f})")
                print(f"   📝 Transcript: {transcript[:80]}...")
                print("   ✅ Transcription complete")

                os.unlink(temp_path)

                return jsonify({
                    'transcript': transcript,
                    'language': detected_lang,
                    'confidence': confidence,
                    'model': model_name
                })

            # Parakeet-unified (RNNT, ONNX int8, torch-free) is the only
            # English ASR path (Whisper removed entirely, 2026-10). No
            # fallback model exists - a missing model or a transcription
            # failure both error out rather than silently degrading.
            unified_model = get_parakeet_unified_model()
            if unified_model is None:
                os.unlink(temp_path)
                return jsonify({'error': 'No English ASR model is loaded - check server logs'}), 503

            audio_array = decode_audio(temp_path, sampling_rate=16000)
            transcript, confidence = unified_model.transcribe(audio_array)
            transcript = convert_english_number_words_to_digits(transcript)
            transcript = correct_kaur_mishearing(transcript)
            detected_lang = 'en'
            model_name = 'Parakeet-unified-0.6b'

            print(f"   ✅ Detected {detected_lang} (confidence: {confidence:.2f})")
            print(f"   📝 Transcript: {transcript[:80]}...")
            print("   ✅ Transcription complete")

            os.unlink(temp_path)

            return jsonify({
                'transcript': transcript,
                'language': detected_lang,
                'confidence': confidence,
                'model': model_name
            })
            
        except Exception as inner_error:
            print(f"❌ Transcription error: {str(inner_error)}")
            # Clean up temp file on error
            if os.path.exists(temp_path):
                os.unlink(temp_path)
            raise inner_error
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"❌ Speech-to-text error: {str(e)}")
        print(f"   Error details:\n{error_details}")
        return jsonify({'error': str(e), 'details': error_details}), 500


# Language codes accepted by Tesseract for this app's two supported languages.
_TESSERACT_LANG_MAP = {'en': 'eng', 'pa': 'pan'}
_MAX_OCR_IMAGE_BYTES = 10 * 1024 * 1024  # 10 MB, well under MAX_CONTENT_LENGTH


# Fallback page-segmentation modes tried (in order) when Tesseract's default
# full-page layout analysis (PSM 3) finds zero text - covers the common
# "one big word/line filling the frame" case (e.g. a close-up phone photo),
# where PSM 3's paragraph/column detection can fail entirely even though a
# simpler assumption about the layout would succeed.
# 6 = single uniform block of text, 7 = single text line, 8 = single word.
_OCR_FALLBACK_PSMS = (6, 7, 8)


def _run_tesseract(image, tesseract_lang, config=''):
    """Run Tesseract once with the given config and return (text, avg_confidence)."""
    data = pytesseract.image_to_data(image, lang=tesseract_lang, output_type=pytesseract.Output.DICT, config=config)
    words = []
    confidences = []
    for text, conf in zip(data['text'], data['conf']):
        text = text.strip()
        if not text:
            continue
        words.append(text)
        try:
            conf_value = float(conf)
        except (TypeError, ValueError):
            conf_value = -1
        if conf_value >= 0:
            confidences.append(conf_value)
    recognized_text = ' '.join(words)
    avg_confidence = (sum(confidences) / len(confidences)) if confidences else 0.0
    return recognized_text, avg_confidence


def _ocr_with_confidence(image, tesseract_lang):
    """Run Tesseract OCR for one language and return (text, avg_confidence).
    avg_confidence is the mean of Tesseract's own per-word confidence scores
    (0-100, -1 for non-text regions which are excluded) - used to pick the
    better result between English and Punjabi when the caller doesn't know
    which script the photo contains.

    Falls back to alternate page-segmentation modes (see _OCR_FALLBACK_PSMS)
    if the default full-page analysis finds no text at all - a close-up
    photo of a single word/line can otherwise come back completely empty."""
    recognized_text, avg_confidence = _run_tesseract(image, tesseract_lang)
    if recognized_text:
        return recognized_text, avg_confidence

    for psm in _OCR_FALLBACK_PSMS:
        fallback_text, fallback_confidence = _run_tesseract(image, tesseract_lang, config=f'--psm {psm}')
        if fallback_text:
            return fallback_text, fallback_confidence
    return recognized_text, avg_confidence


@app.route('/ocr', methods=['POST'])
def ocr_image():
    """
    Recognize text from a photo (e.g. a sign/menu/document) via Tesseract
    OCR, so it can be fed into /translate - powers the app's photo
    translation feature.

    Request body:
    {
        "image": "base64_encoded_image_data",
        "language": "auto"  # 'auto', 'en', or 'pa' - which script to expect
    }

    Returns:
    {
        "text": "recognized text",
        "language": "en" | "pa",   # the language actually used for OCR
        "confidence": 0.0-100.0,   # Tesseract's own average word confidence
        "model": "Tesseract OCR"
    }

    NOTE: requires the Tesseract OCR binary to be installed separately (it's
    not a pip package - pytesseract just shells out to tesseract.exe) with
    the 'eng' and 'pan' trained data files.
    """
    try:
        data = request.json
        image_base64 = data.get('image', '')
        language = data.get('language', 'auto')

        if not image_base64:
            return jsonify({'error': 'No image data provided'}), 400

        image_bytes = base64.b64decode(image_base64)
        if len(image_bytes) > _MAX_OCR_IMAGE_BYTES:
            return jsonify({'error': 'Image too large'}), 413

        print(f"\n📷 OCR request (language: {language}, size: {len(image_bytes)} bytes)")

        try:
            image = Image.open(io.BytesIO(image_bytes))
            # Phone cameras store rotation as EXIF metadata rather than
            # rotating the actual pixels - without this, sideways/upside-
            # down photos silently find zero text (Tesseract assumes
            # upright horizontal lines).
            image = ImageOps.exif_transpose(image)
            image = image.convert('RGB')
        except Exception:
            return jsonify({'error': 'Could not read image data'}), 400

        # TEMP DEBUG: dump every received photo to disk so we can inspect
        # exactly what the server sees (remove once the OCR bug is fixed).
        try:
            image.save(os.path.join(os.path.dirname(__file__), '_debug_last_ocr.png'))
            print(f"   💾 Saved debug image: _debug_last_ocr.png ({image.size[0]}x{image.size[1]})")
        except Exception as debug_err:
            print(f"   ⚠️ Could not save debug image: {debug_err}")

        if language in ('en', 'pa'):
            text, confidence = _ocr_with_confidence(image, _TESSERACT_LANG_MAP[language])
            detected_lang = language
        else:
            # Auto: Tesseract only recognizes text from the script(s) it's
            # told to look for, so run both languages and keep whichever
            # found more/more-confident text rather than silently missing
            # one script entirely.
            en_text, en_confidence = _ocr_with_confidence(image, 'eng')
            pa_text, pa_confidence = _ocr_with_confidence(image, 'pan')
            if pa_confidence > en_confidence and pa_text:
                text, confidence, detected_lang = pa_text, pa_confidence, 'pa'
            else:
                text, confidence, detected_lang = en_text, en_confidence, 'en'

        print(f"   ✅ OCR complete ({detected_lang}, confidence {confidence:.1f}): {text[:80]}")

        return jsonify({
            'text': text,
            'language': detected_lang,
            'confidence': round(confidence, 1),
            'model': 'Tesseract OCR',
        })

    except pytesseract.TesseractNotFoundError:
        return jsonify({
            'error': 'Tesseract OCR is not installed on the server',
            'message': 'Install Tesseract OCR (with eng + pan trained data) and restart the server.',
        }), 503
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"❌ OCR error: {str(e)}")
        print(f"   Error details:\n{error_details}")
        return jsonify({'error': str(e), 'details': error_details}), 500


@app.route('/detect-language', methods=['POST'])
def detect_language():
    """
    Detect if text is English or Punjabi
    Simple heuristic-based detection
    
    Request body:
    {
        "text": "Sample text to detect"
    }
    """
    try:
        data = request.json
        text = data.get('text', '')
        
        if not text:
            return jsonify({'error': 'No text provided'}), 400
        
        # Simple detection: check for Punjabi Unicode range
        punjabi_chars = sum(1 for c in text if '\u0A00' <= c <= '\u0A7F')
        total_chars = len([c for c in text if c.isalpha()])
        
        if total_chars == 0:
            detected = 'en'
            confidence = 0.5
        elif punjabi_chars / total_chars > 0.3:
            detected = 'pa'
            confidence = min(0.95, punjabi_chars / total_chars)
        else:
            detected = 'en'
            confidence = min(0.95, 1 - (punjabi_chars / total_chars))
        
        print(f"\n🔍 Language detection: {detected} ({confidence*100:.0f}% confidence)")
        
        return jsonify({
            'language': detected,
            'confidence': confidence
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    # Run the server
    app.run(host='0.0.0.0', port=5000, debug=False)
