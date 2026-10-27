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
from punjabi_numerals import convert_number_words_to_digits, detect_unreliable_numeral  # pa-source numeral-word-to-digit fix + ASR digit-corruption flag, see file docstring
from english_numerals import convert_number_words_to_digits as convert_english_number_words_to_digits  # Parakeet en-source numeral-word-to-digit fix, see file docstring
from punjabi_name_corrections import fix_name_substitutions, detect_dropped_names  # en->pa curated name-substitution fix + dropped-name flag, see file docstring
from word_sense_disambiguation import disambiguate_ambiguous_words  # en-source lexical-ambiguity fix (e.g. "fall" season vs verb), see file docstring
from beam_sensitive_patterns import translate_chunks_with_beam_overrides  # known beam-search-artifact fix, see file docstring
from parakeet_name_corrections import correct_kaur_mishearing  # Parakeet sentence-final "Kaur"->"Kor"/"Kore"/"Core" fix, see file docstring
from punjabi_alphanumeric_fix import fix_alphanumeric_codes  # en->pa digit-anchored code mangling fix, see file docstring
from indicconformer_asr import IndicConformerASR  # torch-free ONNX Punjabi ASR, see file docstring
from parakeet_unified_asr import ParakeetUnifiedASR  # torch-free ONNX English ASR (RNNT), see file docstring
import google_translate_fallback  # optional, opt-in Google Translate fallback for /translate's verify path, see file docstring
import google_tts_fallback  # optional, opt-in Google Cloud TTS fallback for the web build's /tts route, see file docstring
import pytesseract
from ocr_processor import TESSERACT_LANG_MAP, ocr_with_confidence  # torch-free Tesseract OCR wrapper, see file docstring
import document_translator  # "recreate document with translated text" pipeline, see file docstring
import pymupdf  # PDF text-layer/page inspection for /translate-document, see document_translator.py docstring
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
# QA/operator-only routes - not called by the shipped app, kept behind
# ENABLE_DEBUG_ROUTES to shrink the public attack surface (see _debug_raw_translate
# and /_debug_feedback_log).
_DEBUG_ONLY_ROUTES = {'/_debug_raw_translate', '/_debug_feedback_log'}


@app.before_request
def _start_request_timer():
    # Registered before the auth/rate-limit check below so latency is timed
    # (and telemetry still logged) even for rejected (401/429) requests.
    g.request_start_time = time.time()


@app.before_request
def _enforce_api_key_and_rate_limit():
    # Browsers send an unauthenticated OPTIONS preflight before the real
    # cross-origin request - it never carries the custom X-API-Key header,
    # so rejecting it here would make flask_cors's own CORS headers never
    # reach the browser (a non-2xx preflight response fails CORS entirely,
    # regardless of what headers are attached to it). The real request right
    # after this preflight is still fully subject to the checks below.
    if request.method == 'OPTIONS':
        return None
    if request.path in _UNAUTHENTICATED_ROUTES:
        return None
    if request.path in _DEBUG_ONLY_ROUTES and not ENABLE_DEBUG_ROUTES:
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
if google_translate_fallback.is_configured():
    print("ℹ️  GOOGLE_TRANSLATE_API_KEY is set - /translate's verify path may fall back")
    print("   to Google Translate when it scores more confident than IndicTrans2.")
if google_tts_fallback.is_configured():
    print("ℹ️  GOOGLE_TRANSLATE_API_KEY is also used by /tts (Google Cloud Text-to-Speech)")
    print("   for the web build, which has no local Punjabi voice otherwise.")
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


# BUG FOUND 2026-10-18 (user: pa->en voice translation "doesn't include
# punctuation after user takes pauses"): each chunk produced by chunk_text()
# above (incl. the pause-driven splits from _restore_punjabi_breaks(), which
# exist specifically because unpunctuated Punjabi ASR output has no other
# sentence-boundary signal) is translated INDEPENDENTLY, then the results
# were simply ' '.join()'d - IndicTrans2 doesn't reliably add its own
# trailing period when the source chunk had no punctuation to begin with, so
# multiple translated sentences ran together with no boundary at all in the
# English output. Only affects pa->en (source chunks are unpunctuated
# Gurmukhi); en->pa input is typed/punctuated already, so IndicTrans2 output
# punctuation there was never the problem.
_SENTENCE_END_PUNCT = ('.', '!', '?', '."', '!"', '?"', ',', ';', ':')


def _join_translated_chunks(translated_chunks, target_lang):
    """Join translated chunks into the final text, inserting a period after
    any chunk (other than the last) that doesn't already end in terminal
    punctuation - so internally-split pauses/sentences come back out as
    separate English sentences instead of one run-on."""
    if target_lang != 'en' or len(translated_chunks) <= 1:
        return ' '.join(translated_chunks)
    pieces = []
    for i, chunk in enumerate(translated_chunks):
        chunk = chunk.strip()
        if i < len(translated_chunks) - 1 and chunk and not chunk.endswith(_SENTENCE_END_PUNCT):
            chunk += '.'
        pieces.append(chunk)
    return ' '.join(pieces)


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
#
# Found 2026-10-25 while generating new glossary candidates: IndicProcessor's
# output consistently spells several Gurmukhi "nukta" letters as a base
# letter + separate combining nukta (U+0A3C) - e.g. 'ਡ' + '਼' instead of the
# dedicated Gurmukhi codepoint 'ੜ' - rather than the precomposed letter.
# Visually identical in most fonts, but NOT fixed by NFC normalization:
# confirmed via unicodedata that Gurmukhi nukta letters are canonical
# decompositions excluded from NFC recomposition (Unicode's Composition
# Exclusion Table), so this needs the same explicit replace-based fix as the
# Arabic-keheh case above. Affects common, everyday words (ਜ਼ਰੂਰ, ਆਵਾਜ਼, ਤੇਜ਼,
# ਸੜਕ, ਫ਼ੋਨ, etc.), not just the new glossary entries - fixing it here
# benefits all pa-output translations, not only glossary-matched ones.
_SCRIPT_HOMOGLYPH_FIXES = {
    '\u06A9': '\u0A15',          # ARABIC LETTER KEHEH -> GURMUKHI LETTER KA (ਕ)
    '\u0A16\u0A3C': '\u0A59',    # ਖ + nukta -> GURMUKHI LETTER KHHA (ਖ਼)
    '\u0A17\u0A3C': '\u0A5A',    # ਗ + nukta -> GURMUKHI LETTER GHHA (ਗ਼)
    '\u0A1C\u0A3C': '\u0A5B',    # ਜ + nukta -> GURMUKHI LETTER ZA (ਜ਼)
    '\u0A21\u0A3C': '\u0A5C',    # ਡ + nukta -> GURMUKHI LETTER RRA (ੜ)
    '\u0A2B\u0A3C': '\u0A5E',    # ਫ + nukta -> GURMUKHI LETTER FA (ਫ਼)
    '\u0A32\u0A3C': '\u0A33',    # ਲ + nukta -> GURMUKHI LETTER LLA (ਲ਼)
    '\u0A38\u0A3C': '\u0A36',    # ਸ + nukta -> GURMUKHI LETTER SHA (ਸ਼)
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
        # "fall" (season) vs "fall" (to drop) word-sense ambiguity is now
        # handled generally, pre-translation, by disambiguate_ambiguous_words()
        # in word_sense_disambiguation.py - no per-sentence entry needed here.
        #
        # 2026-10-25 EXPANSION: ~175 more entries, generated via the raw model
        # itself (never hand-typed) then round-trip back-translation checked,
        # same discipline as the GENDERED_GLOSSARY entries above - see repo
        # memory for the full generation/review methodology. A handful of
        # the model's own raw outputs were manually corrected where verified
        # wrong (e.g. 'Nineteen' - model gave the ordinal "ਉੱਨੀਵੀਂ" instead of
        # the cardinal "ਉੱਨੀ"); several candidates were deliberately EXCLUDED
        # as too ambiguous to safely hardcode (e.g. 'Light' collides with the
        # existing ਬੱਤੀ homophone bug; 'Which' needs gender agreement; 'Second'
        # is ordinal/time-unit ambiguous; 'Shop' got the "shopping" sense, not
        # "store"; 'Million' has no single correct word under India's
        # lakh/crore numbering system).
        # Numbers (words) 11-100
        'Eleven': 'ਗਿਆਰਾਂ',
        'Twelve': 'ਬਾਰਾਂ',
        'Thirteen': 'ਤੇਰਾਂ',
        'Fourteen': 'ਚੌਦਾਂ',
        'Fifteen': 'ਪੰਦਰਾਂ',
        'Sixteen': 'ਸੋਲਾਂ',
        'Seventeen': 'ਸਤਾਰਾਂ',
        'Eighteen': 'ਅਠਾਰਾਂ',
        'Nineteen': 'ਉੱਨੀ',
        'Twenty': 'ਵੀਹ',
        'Thirty': 'ਤੀਹ',
        'Forty': 'ਚਾਲੀ',
        'Fifty': 'ਪੰਜਾਹ',
        'Sixty': 'ਸੱਠ',
        'Seventy': 'ਸੱਤਰ',
        'Eighty': 'ਅੱਸੀ',
        'Ninety': 'ਨੱਬੇ',
        'Hundred': 'ਸੌ',
        'Thousand': 'ਹਜ਼ਾਰ',
        'Zero': 'ਜ਼ੀਰੋ',
        # Months
        'January': 'ਜਨਵਰੀ',
        'February': 'ਫਰਵਰੀ',
        'March': 'ਮਾਰਚ',
        'April': 'ਅਪ੍ਰੈਲ',
        'May': 'ਮਈ',
        'June': 'ਜੂਨ',
        'July': 'ਜੁਲਾਈ',
        'August': 'ਅਗਸਤ',
        'September': 'ਸਤੰਬਰ',
        'October': 'ਅਕਤੂਬਰ',
        'November': 'ਨਵੰਬਰ',
        'December': 'ਦਸੰਬਰ',
        # Time words
        'Now': 'ਹੁਣ',
        'Later': 'ਬਾਅਦ ਵਿੱਚ',
        # 'ਕੱਲ੍ਹ' genuinely means BOTH "yesterday" and "tomorrow" in Punjabi
        # depending on context (same ambiguity as Hindi "kal") - mapped
        # en->pa only for 'Yesterday' (the pa->en reverse entry below
        # resolves to "Tomorrow" only, so adding a reverse 'Yesterday' entry
        # too would just silently overwrite it).
        'Yesterday': 'ਕੱਲ੍ਹ',
        'Tomorrow': 'ਕੱਲ੍ਹ',
        'Tonight': 'ਅੱਜ ਰਾਤ',
        'Morning': 'ਸਵੇਰ',
        'Afternoon': 'ਦੁਪਹਿਰ',
        'Evening': 'ਸ਼ਾਮ',
        'Night': 'ਰਾਤ',
        'Week': 'ਹਫ਼ਤਾ',
        'Month': 'ਮਹੀਨਾ',
        'Year': 'ਸਾਲ',
        'Hour': 'ਘੰਟਾ',
        'Minute': 'ਮਿੰਟ',
        # Family (additional) - English kinship terms are broader than
        # Punjabi's (which distinguishes paternal/maternal side, e.g.
        # grandfather = ਦਾਦਾ paternal vs ਨਾਨਾ maternal) - each entry below is
        # the single most common default sense, not a precise match for
        # every family structure.
        'Grandmother': 'ਦਾਦੀ ਮਾਂ',
        'Grandfather': 'ਦਾਦਾ ਜੀ',
        'Daughter': 'ਧੀ',
        'Son': 'ਪੁੱਤਰ',
        'Husband': 'ਪਤੀ',
        'Wife': 'ਪਤਨੀ',
        'Aunt': 'ਮਾਸੀ',
        'Uncle': 'ਚਾਚਾ ਜੀ',
        'Cousin': 'ਚਚੇਰਾ ਭਰਾ',
        'Children': 'ਬੱਚੇ',
        'Family': 'ਪਰਿਵਾਰ',
        'Grandson': 'ਪੋਤਾ',
        'Granddaughter': 'ਪੋਤੀ',
        # Colors
        'Red': 'ਲਾਲ',
        'Blue': 'ਨੀਲਾ',
        'Green': 'ਹਰਾ',
        'Yellow': 'ਪੀਲਾ',
        'Black': 'ਕਾਲਾ',
        'White': 'ਚਿੱਟਾ',
        'Orange': 'ਸੰਤਰਾ',
        'Purple': 'ਜਾਮਨੀ',
        'Pink': 'ਗੁਲਾਬੀ',
        'Brown': 'ਬਰਾਊਨ',
        'Gray': 'ਸਲੇਟੀ',
        # Common adjectives - masculine citation form (standard
        # lexicographic convention for a Punjabi adjective listed in
        # isolation); first-person sentences that must actually agree with
        # the speaker's gender live in GENDERED_GLOSSARY instead (e.g. the
        # standalone 'Hungry' word below vs. the full sentence 'I am hungry').
        'Big': 'ਵੱਡਾ',
        'Small': 'ਛੋਟਾ',
        'Hot': 'ਗਰਮ',
        'Cold': 'ਠੰਢ',
        'Good': 'ਚੰਗਾ',
        'Bad': 'ਮਾੜਾ',
        'Fast': 'ਤੇਜ਼',
        'Slow': 'ਹੌਲੀ',
        'Expensive': 'ਮਹਿੰਗਾ',
        'Cheap': 'ਸਸਤਾ',
        'Beautiful': 'ਸੁੰਦਰ',
        'New': 'ਨਵਾਂ',
        'Old': 'ਪੁਰਾਣਾ',
        'Happy': 'ਖੁਸ਼',
        'Sad': 'ਉਦਾਸ',
        'Tired': 'ਥੱਕਿਆ ਹੋਇਆ',
        'Hungry': 'ਭੁੱਖਾ',
        'Thirsty': 'ਪਿਆਸਾ',
        'Clean': 'ਸਾਫ਼-ਸੁਥਰਾ',
        'Dirty': 'ਗੰਦਾ',
        'Easy': 'ਆਸਾਨ',
        'Difficult': 'ਮੁਸ਼ਕਿਲ',
        'Strong': 'ਮਜ਼ਬੂਤ',
        'Weak': 'ਕਮਜ਼ੋਰ',
        'Tall': 'ਲੰਬਾ',
        'Heavy': 'ਭਾਰੀ',
        'Loud': 'ਉੱਚੀ ਆਵਾਜ਼',
        'Quiet': 'ਚੁੱਪ',
        # Common everyday nouns
        'Car': 'ਕਾਰ',
        'Bus': 'ਬੱਸ',
        'Train': 'ਟ੍ਰੇਨ',
        'Phone': 'ਫ਼ੋਨ',
        'Book': 'ਕਿਤਾਬ',
        'Pen': 'ਪੈਨ',
        'Bag': 'ਬੈਗ',
        'Shirt': 'ਸ਼ਰਟ',
        'Shoes': 'ਜੁੱਤੇ',
        'Bed': 'ਬਿਸਤਰਾ',
        'Chair': 'ਚੇਅਰ',
        'Table': 'ਟੇਬਲ',
        'Door': 'ਦਰਵਾਜ਼ਾ',
        'Window': 'ਵਿੰਡੋ',
        'Key': 'ਕੁੰਜੀ',
        'Bottle': 'ਬੋਤਲ',
        'Road': 'ਸੜਕ',
        'City': 'ਸ਼ਹਿਰ',
        'Village': 'ਪਿੰਡ',
        'Market': 'ਬਾਜ਼ਾਰ',
        'Office': 'ਦਫ਼ਤਰ',
        'Letter': 'ਚਿੱਠੀ',
        'Paper': 'ਕਾਗਜ਼',
        'Clothes': 'ਕੱਪੜੇ',
        'Umbrella': 'ਛਤਰੀ',
        'Fan': 'ਪੱਖਾ',
        'Computer': 'ਕੰਪਿਊਟਰ',
        # Body parts
        'Head': 'ਸਿਰ',
        'Eye': 'ਅੱਖ',
        'Ear': 'ਕੰਨ',
        'Nose': 'ਨੱਕ',
        'Mouth': 'ਮੂੰਹ',
        'Hand': 'ਹੱਥ',
        'Leg': 'ਲੱਤ',
        'Foot': 'ਪੈਰ',
        'Hair': 'ਵਾਲ',
        'Stomach': 'ਪੇਟ',
        'Back': 'ਪਿੱਠ',
        'Arm': 'ਬਾਂਹ',
        'Finger': 'ਉਂਗਲ',
        'Heart': 'ਦਿਲ',
        # Weather
        'Rain': 'ਮੀਂਹ',
        'Sun': 'ਸੂਰਜ',
        'Wind': 'ਹਵਾ',
        'Snow': 'ਬਰਫਬਾਰੀ',
        'Storm': 'ਤੂਫਾਨ',
        'Fog': 'ਧੁੰਦ',
        'Cloud': 'ਬਾਦਲ',
        # Question words
        'What': 'ਕੀ',
        'Where': 'ਕਿੱਥੇ',
        'When': 'ਜਦੋਂ',
        'Why': 'ਕਿਉਂ',
        'How': 'ਕਿਵੇਂ',
        'Who': 'ਕੌਣ',
        # Additional common phrases (non-gendered; gendered first-person
        # phrases like 'I am hungry'/'I love you' live in GENDERED_GLOSSARY)
        'What is this?': 'ਇਹ ਕੀ ਹੈ?',
        'I am happy': 'ਮੈਂ ਖੁਸ਼ ਹਾਂ।',
        'I am sad': 'ਮੈਂ ਦੁਖੀ ਹਾਂ।',
        'It is raining': 'ਮੀਂਹ ਪੈ ਰਿਹਾ ਹੈ।',
        'It is very hot today': 'ਅੱਜ ਬਹੁਤ ਗਰਮੀ ਹੈ।',
        'It is very cold today': 'ਅੱਜ ਬਹੁਤ ਠੰਢ ਹੈ।',
        'What is the weather today?': 'ਅੱਜ ਮੌਸਮ ਕੀ ਹੈ?',
        'Can you help me?': 'ਕੀ ਤੁਸੀਂ ਮੇਰੀ ਮਦਦ ਕਰ ਸਕਦੇ ਹੋ?',
        'Where are you from?': 'ਤੁਸੀਂ ਕਿੱਥੋਂ ਹੋ?',
        'Can you speak slowly?': 'ਕੀ ਤੁਸੀਂ ਹੌਲੀ ਬੋਲ ਸਕਦੇ ਹੋ?',
        'Can you repeat that?': 'ਕੀ ਤੁਸੀਂ ਇਸ ਨੂੰ ਦੁਹਰਾ ਸਕਦੇ ਹੋ?',
        'This is delicious': 'ਇਹ ਸੁਆਦੀ ਹੈ',
        'What is the time now?': 'ਹੁਣ ਸਮਾਂ ਕੀ ਹੈ?',
        "Let's go": 'ਚੱਲੀਏ।',
        'Wait for me': 'ਮੇਰੀ ਉਡੀਕ ਕਰੋ।',
        'Come here': 'ਇੱਥੇ ਆਓ।',
        'Sit down': 'ਬੈਠ ਜਾਓ।',
        'Stand up': 'ਖੜ੍ਹੋ।',
        'Open the door': 'ਦਰਵਾਜ਼ਾ ਖੋਲ੍ਹੋ।',
        'Close the door': 'ਦਰਵਾਜ਼ਾ ਬੰਦ ਕਰੋ।',
        'Take this': 'ਇਸ ਨੂੰ ਲਓ',
        # 2026-10-25 EXPANSION #2: same generation/review discipline as the
        # first expansion above (raw model output, pa->en round-trip
        # checked, manual linguistic review on top of chrF). Several raw
        # outputs were manually corrected to a less ambiguous word/sense
        # (e.g. 'Run' - model's raw ਚਲਾਓ means "drive/operate", not "run on
        # foot"; 'Field' - model's raw ਫੀਲਡ is just the English loanword, not
        # the common agricultural sense ਖੇਤ relevant to Punjab; 'Star' -
        # model's raw ਸਟਾਰ is the "celebrity" loanword sense, not astronomical
        # ਤਾਰਾ; 'Doctor' - model appended a redundant "ਡਾ." abbreviation).
        # Deliberately EXCLUDED: 'Want'/'Need' (no stable citation-form verb,
        # raw output was a conjugated clause), 'Live' (raw ਲਾਈਵ is the
        # English "live broadcast" loanword, not "to reside"), 'Clean'
        # (verb sense duplicates the existing adjective entry), 'Dress' (raw
        # output collided with the existing generic 'Clothes' entry), 'Farm'
        # (raw ਫਾਰਮ is a genuine homophone risk with "form", confirmed by its
        # own back-translation mistranslating it as "Form"), 'Bored' (no
        # correction found - ਅੱਕਿਆ ਹੋਇਆ failed its own round-trip check,
        # back-translating as "stared at").
        # Common verbs (citation/imperative form)
        'Eat': 'ਖਾਓ।',
        'Drink': 'ਪੀਓ।',
        'Sleep': 'ਸੌਂਵੋ।',
        'See': 'ਵੇਖੋ।',
        'Give': 'ਦਿਓ।',
        'Buy': 'ਖ਼ਰੀਦੋ',
        'Sell': 'ਵੇਚੋ',
        'Read': 'ਪੜ੍ਹੋ।',
        'Write': 'ਲਿਖੋ।',
        'Speak': 'ਬੋਲੋ।',
        'Listen': 'ਸੁਣੋ।',
        'Understand': 'ਸਮਝੋ।',
        'Think': 'ਸੋਚੋ।',
        'Play': 'ਖੇਡੋ',
        'Run': 'ਦੌੜੋ',
        'Walk': 'ਸੈਰ ਕਰੋ।',
        'Drive': 'ਡਰਾਈਵ ਕਰੋ',
        'Cook': 'ਖਾਣਾ ਬਣਾਓ।',
        'Wash': 'ਧੋਵੋ',
        'Look': 'ਦੇਖੋ।',
        'Find': 'ਲੱਭੋ',
        'Start': 'ਸ਼ੁਰੂ ਕਰੋ',
        'Finish': 'ਖਤਮ ਕਰੋ',
        'Pay': 'ਭੁਗਤਾਨ ਕਰੋ',
        'Send': 'ਭੇਜੋ।',
        'Remember': 'ਯਾਦ ਰੱਖੋ',
        # Professions
        'Teacher': 'ਅਧਿਆਪਕ',
        'Doctor': 'ਡਾਕਟਰ',
        'Engineer': 'ਇੰਜੀਨੀਅਰ',
        'Farmer': 'ਕਿਸਾਨ',
        'Driver': 'ਡਰਾਈਵਰ',
        'Police officer': 'ਪੁਲਿਸ ਅਧਿਕਾਰੀ',
        'Nurse': 'ਨਰਸ',
        'Lawyer': 'ਵਕੀਲ',
        'Student': 'ਵਿਦਿਆਰਥੀ',
        # Clothing (additional)
        'Pants': 'ਪੈਂਟ',
        'Turban': 'ਪੱਗ',
        'Sweater': 'ਸਵੈਟਰ',
        'Jacket': 'ਜੈਕੇਟ',
        'Socks': 'ਜੁਰਾਬਾਂ',
        'Belt': 'ਬੈਲਟ',
        # Food & kitchen
        'Milk': 'ਦੁੱਧ',
        'Tea': 'ਚਾਹ',
        'Sugar': 'ਸ਼ੂਗਰ',
        'Salt': 'ਲੂਣ',
        'Bread': 'ਰੋਟੀ',
        'Rice': 'ਚਾਵਲ',
        'Vegetables': 'ਸਬਜ਼ੀਆਂ',
        'Fruit': 'ਫਲ',
        'Meat': 'ਮੀਟ',
        'Egg': 'ਅੰਡਾ',
        'Oil': 'ਤੇਲ',
        'Spoon': 'ਚਮਚ',
        'Fork': 'ਫੋਰਕ',
        'Knife': 'ਚਾਕੂ',
        'Plate': 'ਪਲੇਟ',
        'Cup': 'ਕੱਪ',
        # Transportation (additional)
        'Bicycle': 'ਸਾਈਕਲ',
        'Motorcycle': 'ਮੋਟਰ ਸਾਈਕਲ',
        'Airplane': 'ਹਵਾਈ ਜਹਾਜ਼',
        'Bridge': 'ਪੁਲ',
        # Technology
        'Internet': 'ਇੰਟਰਨੈੱਟ',
        'Password': 'ਪਾਸਵਰਡ',
        'Email': 'ਈਮੇਲ',
        'Message': 'ਸੁਨੇਹਾ',
        'Battery': 'ਬੈਟਰੀ',
        'Charger': 'ਚਾਰਜਰ',
        # Nature & farming
        'Tree': 'ਰੁੱਖ',
        'Flower': 'ਫੁੱਲ',
        'Mountain': 'ਪਹਾੜ',
        'River': 'ਨਦੀ',
        'Field': 'ਖੇਤ',
        'Crop': 'ਫਸਲ',
        'Wheat': 'ਕਣਕ',
        'Forest': 'ਜੰਗਲ',
        'Sky': 'ਅਸਮਾਨ',
        'Star': 'ਤਾਰਾ',
        'Moon': 'ਚੰਦ',
        # Emotions (additional) - masculine citation form, same convention
        # as the existing Happy/Sad/Tired/Hungry adjective entries above.
        'Scared': 'ਡਰਿਆ ਹੋਇਆ',
        'Surprised': 'ਹੈਰਾਨ',
        'Worried': 'ਚਿੰਤਤ',
        'Excited': 'ਉਤਸ਼ਾਹਿਤ',
        'Angry': 'ਗੁੱਸੇ ਵਿੱਚ',
        # More common phrases (non-gendered; gendered ones are in
        # GENDERED_GLOSSARY below)
        'What do you do for work?': 'ਤੁਸੀਂ ਕੰਮ ਲਈ ਕੀ ਕਰਦੇ ਹੋ?',
        'Nice weather today': 'ਅੱਜ ਮੌਸਮ ਸੁਹਾਵਣਾ ਹੈ।',
        'Can I get some water?': 'ਕੀ ਮੈਂ ਪਾਣੀ ਲਿਆ ਸਕਦਾ ਹਾਂ?',
        'What would you like to eat?': 'ਤੁਸੀਂ ਕੀ ਖਾਣਾ ਚਾਹੁੰਦੇ ਹੋ?',
        'See you tomorrow': 'ਕੱਲ੍ਹ ਮਿਲਾਂਗੇ।',
        'Have a safe trip': 'ਸੁਰੱਖਿਅਤ ਯਾਤਰਾ ਕਰੋ।',
        'Can you show me the way?': 'ਕੀ ਤੁਸੀਂ ਮੈਨੂੰ ਰਾਹ ਦਿਖਾ ਸਕਦੇ ਹੋ?',
        # 2026-10-25 EXPANSION #3: same generation/review discipline as the
        # prior two expansions (raw model output, round-trip back-
        # translation checked, manual linguistic review on top of chrF -
        # several raw chrF scores below are misleadingly low due to valid
        # case/synonym/plural variance, not actual errors, e.g. 'Exam' ->
        # ਪ੍ਰੀਖਿਆ back-translated as "The test" scored chrF 0.0 but is
        # correct). Manually corrected where the raw output was genuinely
        # wrong (verified via a second round-trip check before hardcoding):
        # 'Sofa'/'Mirror'/'Pillow' (raw model gave an oblique/plural form,
        # not the citation singular), 'Candle' (raw ਦੀਵਾ means "lamp/diya",
        # not "candle" - corrected to ਮੋਮਬੱਤੀ), 'Guru' (raw output appended
        # an unnecessary ਜੀ honorific). Deliberately EXCLUDED: 'Sold out'
        # (raw ਵਿਕ ਗਿਆ is gender-marked past tense with no safe gender-
        # neutral citation form, same risk class as the 'Want'/'Need'
        # exclusions from expansion #2).
        # Animals
        'Dog': 'ਕੁੱਤਾ',
        'Cat': 'ਬਿੱਲੀ',
        'Cow': 'ਗਾਂ',
        'Horse': 'ਘੋੜਾ',
        'Buffalo': 'ਮੱਝ',
        'Bird': 'ਪੰਛੀ',
        'Fish': 'ਮੱਛੀ',
        'Snake': 'ਸੱਪ',
        'Elephant': 'ਹਾਥੀ',
        'Monkey': 'ਬਾਂਦਰ',
        # Household items
        'Sofa': 'ਸੋਫਾ',
        'Mirror': 'ਸ਼ੀਸ਼ਾ',
        'Clock': 'ਘੜੀ',
        'Pillow': 'ਸਿਰਹਾਣਾ',
        'Blanket': 'ਕੰਬਲ',
        'Towel': 'ਤੌਲੀਆ',
        'Soap': 'ਸਾਬਣ',
        'Broom': 'ਝਾੜੂ',
        'Bucket': 'ਬਾਲਟੀ',
        'Candle': 'ਮੋਮਬੱਤੀ',
        # Religious & cultural
        'Gurdwara': 'ਗੁਰਦਵਾਰਾ',
        'Temple': 'ਮੰਦਰ',
        'Prayer': 'ਪ੍ਰਾਰਥਨਾ',
        'Festival': 'ਤਿਉਹਾਰ',
        'Wedding': 'ਵਿਆਹ',
        'Blessing': 'ਅਸ਼ੀਰਵਾਦ',
        'God': 'ਰੱਬ',
        'Guru': 'ਗੁਰੂ',
        # Safety & disaster
        'Fire': 'ਅੱਗ',
        'Earthquake': 'ਭੁਚਾਲ',
        'Accident': 'ਦੁਰਘਟਨਾ',
        'Theft': 'ਚੋਰੀ',
        'Flood': 'ਹੜ੍ਹ',
        'Smoke': 'ਧੂੰਆਂ',
        # Shopping & retail
        'Discount': 'ਛੋਟ',
        'Receipt': 'ਰਸੀਦ',
        'Cash': 'ਨਕਦ',
        'Credit card': 'ਕ੍ਰੈਡਿਟ ਕਾਰਡ',
        'Size': 'ਅਕਾਰ',
        'Price': 'ਕੀਮਤ',
        'Free': 'ਮੁਫ਼ਤ',
        # Exclamations
        'Wow': 'ਵਾਹ',
        'Really?': 'ਅਸਲ ਵਿੱਚ?',
        'Of course': 'ਬੇਸ਼ਕ',
        'No problem': 'ਕੋਈ ਸਮੱਸਿਆ ਨਹੀਂ',
        'Maybe': 'ਸ਼ਾਇਦ',
        'Perhaps': 'ਸ਼ਾਇਦ',
        # School & education
        'Homework': 'ਹੋਮਵਰਕ',
        'Exam': 'ਪ੍ਰੀਖਿਆ',
        'Class': 'ਕਲਾਸ',
        'Lesson': 'ਸਬਕ',
        'Pencil': 'ਪੈਨਸਿਲ',
        'Notebook': 'ਨੋਟਬੁੱਕ',
        'Classroom': 'ਕਲਾਸਰੂਮ',
        # Sports
        'Cricket': 'ਕ੍ਰਿਕਟ',
        'Football': 'ਫੁੱਟਬਾਲ',
        'Game': 'ਖੇਡ',
        'Team': 'ਟੀਮ',
        'Match': 'ਮੈਚ',
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
        # 2026-10-25 EXPANSION (mirrors the en->pa additions above)
        'ਗਿਆਰਾਂ': 'Eleven',
        'ਬਾਰਾਂ': 'Twelve',
        'ਤੇਰਾਂ': 'Thirteen',
        'ਚੌਦਾਂ': 'Fourteen',
        'ਪੰਦਰਾਂ': 'Fifteen',
        'ਸੋਲਾਂ': 'Sixteen',
        'ਸਤਾਰਾਂ': 'Seventeen',
        'ਅਠਾਰਾਂ': 'Eighteen',
        'ਉੱਨੀ': 'Nineteen',
        'ਵੀਹ': 'Twenty',
        'ਤੀਹ': 'Thirty',
        'ਚਾਲੀ': 'Forty',
        'ਪੰਜਾਹ': 'Fifty',
        'ਸੱਠ': 'Sixty',
        'ਸੱਤਰ': 'Seventy',
        'ਅੱਸੀ': 'Eighty',
        'ਨੱਬੇ': 'Ninety',
        'ਸੌ': 'Hundred',
        'ਹਜ਼ਾਰ': 'Thousand',
        'ਜ਼ੀਰੋ': 'Zero',
        'ਜਨਵਰੀ': 'January',
        'ਫਰਵਰੀ': 'February',
        'ਮਾਰਚ': 'March',
        'ਅਪ੍ਰੈਲ': 'April',
        'ਮਈ': 'May',
        'ਜੂਨ': 'June',
        'ਜੁਲਾਈ': 'July',
        'ਅਗਸਤ': 'August',
        'ਸਤੰਬਰ': 'September',
        'ਅਕਤੂਬਰ': 'October',
        'ਨਵੰਬਰ': 'November',
        'ਦਸੰਬਰ': 'December',
        'ਹੁਣ': 'Now',
        'ਬਾਅਦ ਵਿੱਚ': 'Later',
        'ਕੱਲ੍ਹ': 'Tomorrow',  # ambiguous (also means "yesterday") - see en->pa note above
        'ਅੱਜ ਰਾਤ': 'Tonight',
        'ਸਵੇਰ': 'Morning',
        'ਦੁਪਹਿਰ': 'Afternoon',
        'ਸ਼ਾਮ': 'Evening',
        'ਰਾਤ': 'Night',
        'ਹਫ਼ਤਾ': 'Week',
        'ਮਹੀਨਾ': 'Month',
        'ਸਾਲ': 'Year',
        'ਘੰਟਾ': 'Hour',
        'ਮਿੰਟ': 'Minute',
        'ਦਾਦੀ ਮਾਂ': 'Grandmother',
        'ਦਾਦਾ ਜੀ': 'Grandfather',
        'ਧੀ': 'Daughter',
        'ਪੁੱਤਰ': 'Son',
        'ਪਤੀ': 'Husband',
        'ਪਤਨੀ': 'Wife',
        'ਮਾਸੀ': 'Aunt',
        'ਚਾਚਾ ਜੀ': 'Uncle',
        'ਚਚੇਰਾ ਭਰਾ': 'Cousin',
        'ਬੱਚੇ': 'Children',
        'ਪਰਿਵਾਰ': 'Family',
        'ਪੋਤਾ': 'Grandson',
        'ਪੋਤੀ': 'Granddaughter',
        'ਲਾਲ': 'Red',
        'ਨੀਲਾ': 'Blue',
        'ਹਰਾ': 'Green',
        'ਪੀਲਾ': 'Yellow',
        'ਕਾਲਾ': 'Black',
        'ਚਿੱਟਾ': 'White',
        'ਸੰਤਰਾ': 'Orange',
        'ਜਾਮਨੀ': 'Purple',
        'ਗੁਲਾਬੀ': 'Pink',
        'ਬਰਾਊਨ': 'Brown',
        'ਸਲੇਟੀ': 'Gray',
        'ਵੱਡਾ': 'Big',
        'ਛੋਟਾ': 'Small',
        'ਗਰਮ': 'Hot',
        'ਠੰਢ': 'Cold',
        'ਚੰਗਾ': 'Good',
        'ਮਾੜਾ': 'Bad',
        'ਤੇਜ਼': 'Fast',
        'ਹੌਲੀ': 'Slow',
        'ਮਹਿੰਗਾ': 'Expensive',
        'ਸਸਤਾ': 'Cheap',
        'ਸੁੰਦਰ': 'Beautiful',
        'ਨਵਾਂ': 'New',
        'ਪੁਰਾਣਾ': 'Old',
        'ਖੁਸ਼': 'Happy',
        'ਉਦਾਸ': 'Sad',
        'ਥੱਕਿਆ ਹੋਇਆ': 'Tired',
        'ਭੁੱਖਾ': 'Hungry',
        'ਪਿਆਸਾ': 'Thirsty',
        'ਸਾਫ਼-ਸੁਥਰਾ': 'Clean',
        'ਗੰਦਾ': 'Dirty',
        'ਆਸਾਨ': 'Easy',
        'ਮੁਸ਼ਕਿਲ': 'Difficult',
        'ਮਜ਼ਬੂਤ': 'Strong',
        'ਕਮਜ਼ੋਰ': 'Weak',
        'ਲੰਬਾ': 'Tall',
        'ਭਾਰੀ': 'Heavy',
        'ਉੱਚੀ ਆਵਾਜ਼': 'Loud',
        'ਚੁੱਪ': 'Quiet',
        'ਕਾਰ': 'Car',
        'ਬੱਸ': 'Bus',
        'ਟ੍ਰੇਨ': 'Train',
        'ਫ਼ੋਨ': 'Phone',
        'ਕਿਤਾਬ': 'Book',
        'ਪੈਨ': 'Pen',
        'ਬੈਗ': 'Bag',
        'ਸ਼ਰਟ': 'Shirt',
        'ਜੁੱਤੇ': 'Shoes',
        'ਬਿਸਤਰਾ': 'Bed',
        'ਚੇਅਰ': 'Chair',
        'ਟੇਬਲ': 'Table',
        'ਦਰਵਾਜ਼ਾ': 'Door',
        'ਵਿੰਡੋ': 'Window',
        'ਕੁੰਜੀ': 'Key',
        'ਬੋਤਲ': 'Bottle',
        'ਸੜਕ': 'Road',
        'ਸ਼ਹਿਰ': 'City',
        'ਪਿੰਡ': 'Village',
        'ਬਾਜ਼ਾਰ': 'Market',
        'ਦਫ਼ਤਰ': 'Office',
        'ਚਿੱਠੀ': 'Letter',
        'ਕਾਗਜ਼': 'Paper',
        'ਕੱਪੜੇ': 'Clothes',
        'ਛਤਰੀ': 'Umbrella',
        'ਪੱਖਾ': 'Fan',
        'ਕੰਪਿਊਟਰ': 'Computer',
        'ਸਿਰ': 'Head',
        'ਅੱਖ': 'Eye',
        'ਕੰਨ': 'Ear',
        'ਨੱਕ': 'Nose',
        'ਮੂੰਹ': 'Mouth',
        'ਹੱਥ': 'Hand',
        'ਲੱਤ': 'Leg',
        'ਪੈਰ': 'Foot',
        'ਵਾਲ': 'Hair',
        'ਪੇਟ': 'Stomach',
        'ਪਿੱਠ': 'Back',
        'ਬਾਂਹ': 'Arm',
        'ਉਂਗਲ': 'Finger',
        'ਦਿਲ': 'Heart',
        'ਮੀਂਹ': 'Rain',
        'ਸੂਰਜ': 'Sun',
        'ਹਵਾ': 'Wind',
        'ਬਰਫਬਾਰੀ': 'Snow',
        'ਤੂਫਾਨ': 'Storm',
        'ਧੁੰਦ': 'Fog',
        'ਬਾਦਲ': 'Cloud',
        'ਕੀ': 'What',
        'ਕਿੱਥੇ': 'Where',
        'ਜਦੋਂ': 'When',
        'ਕਿਉਂ': 'Why',
        'ਕਿਵੇਂ': 'How',
        'ਕੌਣ': 'Who',
        'ਇਹ ਕੀ ਹੈ?': 'What is this?',
        'ਮੈਂ ਖੁਸ਼ ਹਾਂ।': 'I am happy',
        'ਮੈਂ ਦੁਖੀ ਹਾਂ।': 'I am sad',
        'ਮੀਂਹ ਪੈ ਰਿਹਾ ਹੈ।': 'It is raining',
        'ਅੱਜ ਬਹੁਤ ਗਰਮੀ ਹੈ।': 'It is very hot today',
        'ਅੱਜ ਬਹੁਤ ਠੰਢ ਹੈ।': 'It is very cold today',
        'ਅੱਜ ਮੌਸਮ ਕੀ ਹੈ?': 'What is the weather today?',
        'ਕੀ ਤੁਸੀਂ ਮੇਰੀ ਮਦਦ ਕਰ ਸਕਦੇ ਹੋ?': 'Can you help me?',
        'ਤੁਸੀਂ ਕਿੱਥੋਂ ਹੋ?': 'Where are you from?',
        'ਕੀ ਤੁਸੀਂ ਹੌਲੀ ਬੋਲ ਸਕਦੇ ਹੋ?': 'Can you speak slowly?',
        'ਕੀ ਤੁਸੀਂ ਇਸ ਨੂੰ ਦੁਹਰਾ ਸਕਦੇ ਹੋ?': 'Can you repeat that?',
        'ਇਹ ਸੁਆਦੀ ਹੈ': 'This is delicious',
        'ਹੁਣ ਸਮਾਂ ਕੀ ਹੈ?': 'What is the time now?',
        'ਚੱਲੀਏ।': "Let's go",
        'ਮੇਰੀ ਉਡੀਕ ਕਰੋ।': 'Wait for me',
        'ਇੱਥੇ ਆਓ।': 'Come here',
        'ਬੈਠ ਜਾਓ।': 'Sit down',
        'ਖੜ੍ਹੋ।': 'Stand up',
        'ਦਰਵਾਜ਼ਾ ਖੋਲ੍ਹੋ।': 'Open the door',
        'ਦਰਵਾਜ਼ਾ ਬੰਦ ਕਰੋ।': 'Close the door',
        'ਇਸ ਨੂੰ ਲਓ': 'Take this',
        # Gendered phrases (en->pa gender-selected via GENDERED_GLOSSARY below;
        # both m/f Punjabi forms map back to the same English here, since the
        # pa->en reverse direction doesn't need a gender choice).
        'ਮੈਂ ਭੁੱਖਾ ਹਾਂ।': 'I am hungry',
        'ਮੈਂ ਭੁੱਖੀ ਹਾਂ।': 'I am hungry',
        'ਮੈਂ ਪਿਆਸਾ ਹਾਂ।': 'I am thirsty',
        'ਮੈਂ ਪਿਆਸੀ ਹਾਂ।': 'I am thirsty',
        'ਮੈਂ ਪਸੰਦ ਕਰਦਾ ਹਾਂ': 'I like it',
        'ਮੈਂ ਪਸੰਦ ਕਰਦੀ ਹਾਂ': 'I like it',
        "ਮੈਂ ਪਸੰਦ ਨਹੀਂ ਕਰਦਾ": "I don't like it",
        "ਮੈਂ ਪਸੰਦ ਨਹੀਂ ਕਰਦੀ": "I don't like it",
        'ਮੈਂ ਤੁਹਾਨੂੰ ਪਿਆਰ ਕਰਦਾ ਹਾਂ।': 'I love you',
        'ਮੈਂ ਤੁਹਾਨੂੰ ਪਿਆਰ ਕਰਦੀ ਹਾਂ।': 'I love you',
        'ਮੈਂ ਪੰਜਾਬੀ ਸਿੱਖ ਰਿਹਾ ਹਾਂ।': 'I am learning Punjabi',
        'ਮੈਂ ਪੰਜਾਬੀ ਸਿੱਖ ਰਹੀ ਹਾਂ।': 'I am learning Punjabi',
        'ਮੈਂ ਪੰਜਾਬੀ ਚੰਗੀ ਤਰ੍ਹਾਂ ਨਹੀਂ ਬੋਲਦਾ।': "I don't speak Punjabi well",
        'ਮੈਂ ਪੰਜਾਬੀ ਚੰਗੀ ਤਰ੍ਹਾਂ ਨਹੀਂ ਬੋਲਦੀ।': "I don't speak Punjabi well",
        'ਮੈਂ ਬਾਅਦ ਵਿੱਚ ਵਾਪਸ ਆਵਾਂਗਾ': 'I will come back later',
        'ਮੈਂ ਬਾਅਦ ਵਿੱਚ ਵਾਪਸ ਆਵਾਂਗੀ': 'I will come back later',
        # 2026-10-25 EXPANSION #2 (mirrors the en->pa additions above)
        'ਖਾਓ।': 'Eat',
        'ਪੀਓ।': 'Drink',
        'ਸੌਂਵੋ।': 'Sleep',
        'ਵੇਖੋ।': 'See',
        'ਦਿਓ।': 'Give',
        'ਖ਼ਰੀਦੋ': 'Buy',
        'ਵੇਚੋ': 'Sell',
        'ਪੜ੍ਹੋ।': 'Read',
        'ਲਿਖੋ।': 'Write',
        'ਬੋਲੋ।': 'Speak',
        'ਸੁਣੋ।': 'Listen',
        'ਸਮਝੋ।': 'Understand',
        'ਸੋਚੋ।': 'Think',
        'ਖੇਡੋ': 'Play',
        'ਦੌੜੋ': 'Run',
        'ਸੈਰ ਕਰੋ।': 'Walk',
        'ਡਰਾਈਵ ਕਰੋ': 'Drive',
        'ਖਾਣਾ ਬਣਾਓ।': 'Cook',
        'ਧੋਵੋ': 'Wash',
        'ਦੇਖੋ।': 'Look',
        'ਲੱਭੋ': 'Find',
        'ਸ਼ੁਰੂ ਕਰੋ': 'Start',
        'ਖਤਮ ਕਰੋ': 'Finish',
        'ਭੁਗਤਾਨ ਕਰੋ': 'Pay',
        'ਭੇਜੋ।': 'Send',
        'ਯਾਦ ਰੱਖੋ': 'Remember',
        'ਅਧਿਆਪਕ': 'Teacher',
        'ਡਾਕਟਰ': 'Doctor',
        'ਇੰਜੀਨੀਅਰ': 'Engineer',
        'ਕਿਸਾਨ': 'Farmer',
        'ਡਰਾਈਵਰ': 'Driver',
        'ਪੁਲਿਸ ਅਧਿਕਾਰੀ': 'Police officer',
        'ਨਰਸ': 'Nurse',
        'ਵਕੀਲ': 'Lawyer',
        'ਵਿਦਿਆਰਥੀ': 'Student',
        'ਪੈਂਟ': 'Pants',
        'ਪੱਗ': 'Turban',
        'ਸਵੈਟਰ': 'Sweater',
        'ਜੈਕੇਟ': 'Jacket',
        'ਜੁਰਾਬਾਂ': 'Socks',
        'ਬੈਲਟ': 'Belt',
        'ਦੁੱਧ': 'Milk',
        'ਚਾਹ': 'Tea',
        'ਸ਼ੂਗਰ': 'Sugar',
        'ਲੂਣ': 'Salt',
        'ਰੋਟੀ': 'Bread',
        'ਚਾਵਲ': 'Rice',
        'ਸਬਜ਼ੀਆਂ': 'Vegetables',
        'ਫਲ': 'Fruit',
        'ਮੀਟ': 'Meat',
        'ਅੰਡਾ': 'Egg',
        'ਤੇਲ': 'Oil',
        'ਚਮਚ': 'Spoon',
        'ਫੋਰਕ': 'Fork',
        'ਚਾਕੂ': 'Knife',
        'ਪਲੇਟ': 'Plate',
        'ਕੱਪ': 'Cup',
        'ਸਾਈਕਲ': 'Bicycle',
        'ਮੋਟਰ ਸਾਈਕਲ': 'Motorcycle',
        'ਹਵਾਈ ਜਹਾਜ਼': 'Airplane',
        'ਪੁਲ': 'Bridge',
        'ਇੰਟਰਨੈੱਟ': 'Internet',
        'ਪਾਸਵਰਡ': 'Password',
        'ਈਮੇਲ': 'Email',
        'ਸੁਨੇਹਾ': 'Message',
        'ਬੈਟਰੀ': 'Battery',
        'ਚਾਰਜਰ': 'Charger',
        'ਰੁੱਖ': 'Tree',
        'ਫੁੱਲ': 'Flower',
        'ਪਹਾੜ': 'Mountain',
        'ਨਦੀ': 'River',
        'ਖੇਤ': 'Field',
        'ਫਸਲ': 'Crop',
        'ਕਣਕ': 'Wheat',
        'ਜੰਗਲ': 'Forest',
        'ਅਸਮਾਨ': 'Sky',
        'ਤਾਰਾ': 'Star',
        'ਚੰਦ': 'Moon',
        'ਡਰਿਆ ਹੋਇਆ': 'Scared',
        'ਹੈਰਾਨ': 'Surprised',
        'ਚਿੰਤਤ': 'Worried',
        'ਉਤਸ਼ਾਹਿਤ': 'Excited',
        'ਗੁੱਸੇ ਵਿੱਚ': 'Angry',
        'ਤੁਸੀਂ ਕੰਮ ਲਈ ਕੀ ਕਰਦੇ ਹੋ?': 'What do you do for work?',
        'ਅੱਜ ਮੌਸਮ ਸੁਹਾਵਣਾ ਹੈ।': 'Nice weather today',
        'ਕੀ ਮੈਂ ਪਾਣੀ ਲਿਆ ਸਕਦਾ ਹਾਂ?': 'Can I get some water?',
        'ਤੁਸੀਂ ਕੀ ਖਾਣਾ ਚਾਹੁੰਦੇ ਹੋ?': 'What would you like to eat?',
        'ਕੱਲ੍ਹ ਮਿਲਾਂਗੇ।': 'See you tomorrow',
        'ਸੁਰੱਖਿਅਤ ਯਾਤਰਾ ਕਰੋ।': 'Have a safe trip',
        'ਕੀ ਤੁਸੀਂ ਮੈਨੂੰ ਰਾਹ ਦਿਖਾ ਸਕਦੇ ਹੋ?': 'Can you show me the way?',
        # Gendered phrases (2026-10-25 EXPANSION #2) - both m/f forms map
        # back to the same English, same pattern as the existing gendered
        # reverse entries above.
        'ਮੈਂ ਤੁਹਾਨੂੰ ਫ਼ੋਨ ਕਰਾਂਗਾ।': 'I will call you',
        'ਮੈਂ ਤੁਹਾਨੂੰ ਫ਼ੋਨ ਕਰਾਂਗੀ।': 'I will call you',
        'ਮੈਂ ਕੰਮ ਕਰਨ ਜਾ ਰਿਹਾ ਹਾਂ': 'I am going to work',
        'ਮੈਂ ਕੰਮ ਕਰਨ ਜਾ ਰਹੀ ਹਾਂ': 'I am going to work',
        'ਮੈਂ ਜਲਦੀ ਵਾਪਸ ਆਵਾਂਗਾ।': 'I will be right back',
        'ਮੈਂ ਜਲਦੀ ਵਾਪਸ ਆਵਾਂਗੀ।': 'I will be right back',
        # 2026-10-25 EXPANSION #3 (reverse entries - see en->pa side above
        # for corrections/exclusions). 'ਸ਼ਾਇਦ' only gets ONE reverse mapping
        # ('Maybe', the more common/casual choice) since 'Maybe' and
        # 'Perhaps' both map to it on the en->pa side - can't reverse-map
        # the same Punjabi key to two different English values.
        'ਕੁੱਤਾ': 'Dog',
        'ਬਿੱਲੀ': 'Cat',
        'ਗਾਂ': 'Cow',
        'ਘੋੜਾ': 'Horse',
        'ਮੱਝ': 'Buffalo',
        'ਪੰਛੀ': 'Bird',
        'ਮੱਛੀ': 'Fish',
        'ਸੱਪ': 'Snake',
        'ਹਾਥੀ': 'Elephant',
        'ਬਾਂਦਰ': 'Monkey',
        'ਸੋਫਾ': 'Sofa',
        'ਸ਼ੀਸ਼ਾ': 'Mirror',
        'ਘੜੀ': 'Clock',
        'ਸਿਰਹਾਣਾ': 'Pillow',
        'ਕੰਬਲ': 'Blanket',
        'ਤੌਲੀਆ': 'Towel',
        'ਸਾਬਣ': 'Soap',
        'ਝਾੜੂ': 'Broom',
        'ਬਾਲਟੀ': 'Bucket',
        'ਮੋਮਬੱਤੀ': 'Candle',
        'ਗੁਰਦਵਾਰਾ': 'Gurdwara',
        'ਮੰਦਰ': 'Temple',
        'ਪ੍ਰਾਰਥਨਾ': 'Prayer',
        'ਤਿਉਹਾਰ': 'Festival',
        'ਵਿਆਹ': 'Wedding',
        'ਅਸ਼ੀਰਵਾਦ': 'Blessing',
        'ਰੱਬ': 'God',
        'ਗੁਰੂ': 'Guru',
        'ਅੱਗ': 'Fire',
        'ਭੁਚਾਲ': 'Earthquake',
        'ਦੁਰਘਟਨਾ': 'Accident',
        'ਚੋਰੀ': 'Theft',
        'ਹੜ੍ਹ': 'Flood',
        'ਧੂੰਆਂ': 'Smoke',
        'ਛੋਟ': 'Discount',
        'ਰਸੀਦ': 'Receipt',
        'ਨਕਦ': 'Cash',
        'ਕ੍ਰੈਡਿਟ ਕਾਰਡ': 'Credit card',
        'ਅਕਾਰ': 'Size',
        'ਕੀਮਤ': 'Price',
        'ਮੁਫ਼ਤ': 'Free',
        'ਵਾਹ': 'Wow',
        'ਅਸਲ ਵਿੱਚ?': 'Really?',
        'ਬੇਸ਼ਕ': 'Of course',
        'ਕੋਈ ਸਮੱਸਿਆ ਨਹੀਂ': 'No problem',
        'ਸ਼ਾਇਦ': 'Maybe',
        'ਹੋਮਵਰਕ': 'Homework',
        'ਪ੍ਰੀਖਿਆ': 'Exam',
        'ਕਲਾਸ': 'Class',
        'ਸਬਕ': 'Lesson',
        'ਪੈਨਸਿਲ': 'Pencil',
        'ਨੋਟਬੁੱਕ': 'Notebook',
        'ਕਲਾਸਰੂਮ': 'Classroom',
        'ਕ੍ਰਿਕਟ': 'Cricket',
        'ਫੁੱਟਬਾਲ': 'Football',
        'ਖੇਡ': 'Game',
        'ਟੀਮ': 'Team',
        'ਮੈਚ': 'Match',
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
        # 2026-10-25 EXPANSION - same sourcing/verification discipline as above
        'I am hungry': {'m': 'ਮੈਂ ਭੁੱਖਾ ਹਾਂ।', 'f': 'ਮੈਂ ਭੁੱਖੀ ਹਾਂ।'},
        'I am thirsty': {'m': 'ਮੈਂ ਪਿਆਸਾ ਹਾਂ।', 'f': 'ਮੈਂ ਪਿਆਸੀ ਹਾਂ।'},
        'I like it': {'m': 'ਮੈਂ ਪਸੰਦ ਕਰਦਾ ਹਾਂ', 'f': 'ਮੈਂ ਪਸੰਦ ਕਰਦੀ ਹਾਂ'},
        "I don't like it": {'m': 'ਮੈਂ ਪਸੰਦ ਨਹੀਂ ਕਰਦਾ', 'f': 'ਮੈਂ ਪਸੰਦ ਨਹੀਂ ਕਰਦੀ'},
        'I love you': {'m': 'ਮੈਂ ਤੁਹਾਨੂੰ ਪਿਆਰ ਕਰਦਾ ਹਾਂ।', 'f': 'ਮੈਂ ਤੁਹਾਨੂੰ ਪਿਆਰ ਕਰਦੀ ਹਾਂ।'},
        'I am learning Punjabi': {'m': 'ਮੈਂ ਪੰਜਾਬੀ ਸਿੱਖ ਰਿਹਾ ਹਾਂ।', 'f': 'ਮੈਂ ਪੰਜਾਬੀ ਸਿੱਖ ਰਹੀ ਹਾਂ।'},
        "I don't speak Punjabi well": {'m': 'ਮੈਂ ਪੰਜਾਬੀ ਚੰਗੀ ਤਰ੍ਹਾਂ ਨਹੀਂ ਬੋਲਦਾ।', 'f': 'ਮੈਂ ਪੰਜਾਬੀ ਚੰਗੀ ਤਰ੍ਹਾਂ ਨਹੀਂ ਬੋਲਦੀ।'},
        'I will come back later': {'m': 'ਮੈਂ ਬਾਅਦ ਵਿੱਚ ਵਾਪਸ ਆਵਾਂਗਾ', 'f': 'ਮੈਂ ਬਾਅਦ ਵਿੱਚ ਵਾਪਸ ਆਵਾਂਗੀ'},
        # 2026-10-25 EXPANSION #2 - same sourcing/verification discipline
        'I will call you': {'m': 'ਮੈਂ ਤੁਹਾਨੂੰ ਫ਼ੋਨ ਕਰਾਂਗਾ।', 'f': 'ਮੈਂ ਤੁਹਾਨੂੰ ਫ਼ੋਨ ਕਰਾਂਗੀ।'},
        'I am going to work': {'m': 'ਮੈਂ ਕੰਮ ਕਰਨ ਜਾ ਰਿਹਾ ਹਾਂ', 'f': 'ਮੈਂ ਕੰਮ ਕਰਨ ਜਾ ਰਹੀ ਹਾਂ'},
        'I will be right back': {'m': 'ਮੈਂ ਜਲਦੀ ਵਾਪਸ ਆਵਾਂਗਾ।', 'f': 'ਮੈਂ ਜਲਦੀ ਵਾਪਸ ਆਵਾਂਗੀ।'},
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

        # F1 (2*intersection / (|A|+|B|)) instead of plain Jaccard
        # (intersection/union): a correct translation that needs MORE native
        # words to express something the original said in fewer (idioms,
        # code-switched loanwords expanding into a multi-word native phrase
        # on the back-translation, e.g. "best friend" -> "ਸਭ ਤੋਂ ਚੰਗਾ ਦੋਸਤ")
        # inflates Jaccard's union denominator and gets penalized for being
        # longer, even though nothing is wrong (see repo memory, issue #2/#4
        # investigation). F1 is mathematically always >= Jaccard for the same
        # inputs, so this consistently gives correct-but-longer paraphrases a
        # fairer score, while still measurably penalizing a back-translation
        # padded with unrelated extra words (unlike plain recall, which the
        # original Jaccard choice deliberately avoided for that reason - F1's
        # precision term still drags the score down when |B| grows without
        # new overlap).
        intersection = len(original_words & back_words)
        if not original_words and not back_words:
            return 0.0, back_translation
        similarity = 2 * intersection / (len(original_words) + len(back_words))
        return round(similarity, 2), back_translation
    except Exception as e:
        print(f"   ⚠️ Round-trip confidence check failed: {e}")
        return None, None


def verify_and_improve_translation(text, source_lang, target_lang, chunks, translation, gender='m'):
    """Round-trip-score `translation` and, if confidence is below threshold,
    try each alternate beam-search config in `_ENSEMBLE_DECODING_CONFIGS`,
    keeping whichever candidate scores highest - a different search path
    sometimes avoids whatever caused the model to drop/distort meaning the
    first time. Stops early once a candidate is confident enough, to avoid
    paying for configs that won't help. Shared by /translate's `verify` path
    and /confidence, so the latter's background check can also upgrade a
    bad translation instead of only ever flagging it.
    Returns (translation, confidence, back_translation, retried)."""
    confidence, back_translation = compute_roundtrip_confidence(
        text, source_lang, target_lang, translation
    )
    retried = False
    ensemble_threshold = (
        _ENSEMBLE_CONFIDENCE_THRESHOLD_PA_SOURCE if source_lang == 'pa' else _ENSEMBLE_CONFIDENCE_THRESHOLD
    )
    if confidence is not None and confidence < ensemble_threshold:
        best_translation = translation
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
                best_translation = candidate_translation
                best_confidence, best_back = candidate_confidence, candidate_back

        if best_translation != translation:
            print(f"   ✅ Ensemble improved confidence: {confidence} → {best_confidence}")
            translation = best_translation
            confidence, back_translation = best_confidence, best_back
            if source_lang == 'en' and target_lang == 'pa':
                translation = fix_alphanumeric_codes(text, translation)
            retried = True

    return translation, confidence, back_translation, retried



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


LID_WINDOW_SECONDS = 3.0


def detect_spoken_language(audio_array, sample_rate=16000):
    """Run SpeechBrain VoxLingua107 LID on a mono float32 16kHz numpy array.
    Classifies in fixed-length windows rather than the whole clip at once - a
    single whole-clip classification is fragile for code-switched speech: a
    long enough embedded English clause can flip the ENTIRE verdict to 'en'
    (confirmed empirically - a clip that's ~45% English by duration flipped
    to 'en' at only 0.44 confidence), which then sends the Punjabi portions
    through the English-only model, which can't transcribe Punjabi at all.
    So the rule is asymmetric on purpose: if ANY window looks like Punjabi/
    Hindi/Urdu, treat the WHOLE clip as Punjabi - IndicConformer/NeMo RNNT
    tolerates occasional embedded English words reasonably well (see repo
    notes, 2026-10-19 real-audio re-test), but Parakeet (English-only) has no
    ability to transcribe Punjabi content at all, so a false positive for
    'pa' is far cheaper than a false positive for 'en'.
    Returns (lang, iso_code, confidence) - iso_code/confidence are the
    highest-confidence window's values, for logging only; the actual `lang`
    decision follows the any-window-is-Punjabi rule above, not just the top
    window."""
    model = get_language_id_model()
    window_len = int(LID_WINDOW_SECONDS * sample_rate)
    windows = [audio_array[i:i + window_len] for i in range(0, len(audio_array), window_len)]
    windows = [w for w in windows if len(w) >= sample_rate] or [audio_array]

    results = []
    for w in windows:
        audio_tensor = torch.from_numpy(w).unsqueeze(0)
        _, log_likelihood, _, text_lab = model.classify_batch(audio_tensor)
        iso_code = text_lab[0].split(':')[0].strip().lower()
        confidence = float(log_likelihood.exp()[0])
        results.append((iso_code, confidence))

    lang = 'pa' if any(iso in ('pa', 'hi', 'ur') for iso, _ in results) else 'en'
    top_iso, top_confidence = max(results, key=lambda r: r[1])
    return lang, top_iso, top_confidence


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
# Lives under models/ (the same persistent Fly Volume download_models.py
# uses - see fly.toml's [[mounts]]), NOT next to this script - the script's
# own directory is the container's ephemeral filesystem and gets wiped on
# every deploy, which was silently discarding every flagged translation.
FEEDBACK_LOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', 'feedback_log.jsonl')
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
        os.makedirs(os.path.dirname(FEEDBACK_LOG_PATH), exist_ok=True)
        with open(FEEDBACK_LOG_PATH, 'a', encoding='utf-8') as f:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    except OSError as e:
        return jsonify({'error': f'Could not save feedback: {e}'}), 500

    return jsonify({'status': 'ok'})


@app.route('/_debug_feedback_log', methods=['GET'])
def _debug_feedback_log():
    """Operator-only: read back flagged translations for human review.

    Was previously unreachable without `flyctl ssh console` + manually
    catting the file off the deployed machine - this is the "is anyone
    actually reviewing this" gap, now closed. Gated behind
    ENABLE_DEBUG_ROUTES (see _DEBUG_ONLY_ROUTES) since it's an operator tool,
    not something the shipped app itself calls.

    Query params: ?limit=N (most recent N entries, default 200).
    """
    limit = request.args.get('limit', 200, type=int)
    entries = []
    try:
        with open(FEEDBACK_LOG_PATH, encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except FileNotFoundError:
        return jsonify({'count': 0, 'entries': []})

    entries.sort(key=lambda e: e.get('timestamp', 0), reverse=True)
    entries = entries[:limit]
    return jsonify({'count': len(entries), 'entries': entries})


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


def translate_plain_text(text, source_lang, target_lang, gender='m'):
    """Core text translation (chunking + IndicTrans2 + glossary + name/
    alphanumeric fixes), WITHOUT /translate's optional round-trip-
    verification/ensemble-retry mode. Factored out so /translate-document
    (document recreation) can translate each detected text block/line using
    the exact same pipeline as /translate, without duplicating that route's
    logic inline or paying for verify-mode's extra latency per block."""
    if not text or not text.strip():
        return ''
    text = normalize_text(text)
    if source_lang == 'pa':
        text = convert_number_words_to_digits(text)
    if source_lang == 'en' and target_lang == 'pa':
        text = disambiguate_ambiguous_words(text)
    if source_lang == target_lang:
        return text
    chunks = chunk_text(text)
    if not chunks:
        chunks = [text]
    translated_chunks = translate_chunks_with_beam_overrides(chunks, source_lang, target_lang, translate_chunks)
    translated_chunks = apply_glossary(chunks, translated_chunks, source_lang, target_lang, gender)
    translation = ' '.join(translated_chunks)
    if source_lang == 'en' and target_lang == 'pa':
        translation = fix_name_substitutions(text, translation)
        translation = fix_alphanumeric_codes(text, translation)
    return translation


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

        # Rewrite known-ambiguous English words (e.g. "fall") to an
        # unambiguous synonym when context indicates which sense is meant,
        # before the model ever sees the ambiguous word - see
        # word_sense_disambiguation.py for why this is general rather than
        # a fixed phrase list.
        if source_lang == 'en' and target_lang == 'pa':
            text = disambiguate_ambiguous_words(text)

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

        translated_chunks = translate_chunks_with_beam_overrides(chunks, source_lang, target_lang, translate_chunks)
        translated_chunks = apply_glossary(chunks, translated_chunks, source_lang, target_lang, gender)
        translation = _join_translated_chunks(translated_chunks, target_lang)

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

        # A curated name present in the source but missing (not just
        # substituted) from the output can't be safely auto-corrected (see
        # punjabi_name_corrections.py), so flag it instead - same role
        # numeral_uncertain plays for ASR digit corruption.
        if source_lang == 'en' and target_lang == 'pa' and detect_dropped_names(text, translation):
            response_payload['name_uncertain'] = True
            print("   ⚠️ A curated name may have been dropped from the translation - flagging name_uncertain")

        # Optional round-trip confidence check - pass "verify": true in the
        # request body to enable (roughly doubles latency for this request).
        if data.get('verify'):
            translation, confidence, back_translation, retried = verify_and_improve_translation(
                text, source_lang, target_lang, chunks, translation, gender
            )
            response_payload['translation'] = translation
            response_payload['confidence'] = confidence
            response_payload['back_translation'] = back_translation
            if retried:
                response_payload['retried'] = True

            # Optional Google Translate fallback (inert unless
            # GOOGLE_TRANSLATE_API_KEY is set - see google_translate_fallback.py
            # docstring). Scored with the exact same round-trip-confidence
            # metric as IndicTrans2's own result, so the swap only happens
            # when Google's translation is actually more accurate for this
            # specific request - never preferred by default.
            if confidence is not None and google_translate_fallback.is_configured():
                google_translation = google_translate_fallback.translate_via_google(text, source_lang, target_lang)
                if google_translation:
                    google_confidence, google_back_translation = compute_roundtrip_confidence(
                        text, source_lang, target_lang, google_translation
                    )
                    if google_confidence is not None and google_confidence > confidence:
                        print(f"   ✅ Google Translate fallback scored higher ({google_confidence} > {confidence}) - using it")
                        response_payload['translation'] = google_translation
                        response_payload['confidence'] = google_confidence
                        response_payload['back_translation'] = google_back_translation
                        response_payload['fallback_used'] = 'google_translate'
                        response_payload['indictrans2_confidence'] = confidence

        return jsonify(response_payload)
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"❌ Translation error: {str(e)}")
        print(f"   Error details:\n{error_details}")
        return jsonify({'error': str(e), 'details': error_details}), 500


@app.route('/confidence', methods=['POST'])
def confidence():
    """Score an ALREADY-PRODUCED translation's round-trip confidence, and -
    unlike the first version of this endpoint - also attempt the SAME
    ensemble re-decoding `/translate`'s verify path uses if that score comes
    back low, so a background caller can get a genuinely better translation
    back, not just a warning about a bad one. Lets a caller get the fast,
    non-blocking single-pass translation first (see /translate) and run this
    heavier check afterwards in the background, instead of paying verify
    mode's full latency up front - added because the mobile app's real-time
    flow found that cost (18-27s on a physical device) unacceptable and had
    disabled verify mode entirely, silently making its own low-confidence-
    warning UI dead code.

    Request body: {"text", "source_lang", "target_lang", "translation", "gender"}
    Response: {"confidence": 0.0-1.0 or null, "translation"?, "retried"?}
    ("translation"/"retried" are only present if ensemble retry found and
    used a better candidate than the one passed in.)
    """
    try:
        data = request.json
        text = data.get('text', '')
        source_lang = data.get('source_lang', 'en')
        target_lang = data.get('target_lang', 'pa')
        translation = data.get('translation', '')
        gender = data.get('gender', 'm')
        if gender not in ('m', 'f'):
            gender = 'm'
        if not text or not translation or source_lang == target_lang:
            return jsonify({'confidence': None})
        chunks = chunk_text(text) or [text]
        improved_translation, score, _back_translation, retried = verify_and_improve_translation(
            text, source_lang, target_lang, chunks, translation, gender
        )
        response_payload = {'confidence': score}
        if retried:
            response_payload['translation'] = improved_translation
            response_payload['retried'] = True
        return jsonify(response_payload)
    except Exception as e:
        print(f"❌ Confidence check error: {str(e)}")
        return jsonify({'confidence': None, 'error': str(e)}), 500


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

                # Known ASR failure mode (confirmed 2026-10-17, see repo
                # memory + punjabi_numerals.py docstring): a literal digit in
                # the RAW transcript is abnormal (this ASR never emits
                # digits on its own) and signals the model has mis-heard a
                # complex/compound spoken number. No safe text-level fix
                # exists - just flag it so callers can warn the user.
                response = {
                    'transcript': transcript,
                    'language': detected_lang,
                    'confidence': confidence,
                    'model': model_name
                }
                if detect_unreliable_numeral(transcript):
                    response['numeral_uncertain'] = True
                    print("   ⚠️ Digit(s) found in raw Punjabi ASR transcript - flagging numeral_uncertain")
                return jsonify(response)

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


_MAX_OCR_IMAGE_BYTES = 10 * 1024 * 1024  # 10 MB, well under MAX_CONTENT_LENGTH

# Fonts bundled for /translate-document's redrawn translated text - need an
# actual TTF per script since Pillow/PyMuPDF can't render Gurmukhi (or any
# non-Latin script) with a built-in default font. Keyed by TARGET language
# (the script the redrawn/reinserted text will be in), not source.
_DOCUMENT_FONT_PATHS = {
    'en': os.path.join(os.path.dirname(__file__), 'fonts', 'NotoSans-Regular.ttf'),
    'pa': os.path.join(os.path.dirname(__file__), 'fonts', 'NotoSansGurmukhi-Regular.ttf'),
}
_MAX_DOCUMENT_BYTES = 15 * 1024 * 1024  # 15 MB decoded, under MAX_CONTENT_LENGTH (base64 adds ~33%)
_MAX_TTS_TEXT_CHARS = 2000


@app.route('/tts', methods=['POST'])
def text_to_speech():
    """
    Synthesize speech audio for text via Google Cloud TTS - only needed by
    the web build, for languages the browser's own SpeechSynthesis engine
    has no local voice for (chiefly Punjabi, see google_tts_fallback.py's
    docstring). Native Android builds use react-native-tts's on-device
    voices directly and never call this.

    Request body:
    {
        "text": "...",
        "language": "pa"  # 'en' or 'pa'
    }
    """
    data = request.json
    text = (data.get('text') or '').strip()
    language = data.get('language', 'pa')

    if not text:
        return jsonify({'error': 'No text provided'}), 400
    if len(text) > _MAX_TTS_TEXT_CHARS:
        return jsonify({'error': 'Text too long'}), 413
    if language not in ('en', 'pa'):
        return jsonify({'error': 'Unsupported language'}), 400
    if not google_tts_fallback.is_configured():
        return jsonify({'error': 'Cloud TTS is not configured on this server'}), 503

    audio_content = google_tts_fallback.synthesize(text, language)
    if audio_content is None:
        return jsonify({'error': 'Cloud TTS request failed'}), 502
    return jsonify({'audioContent': audio_content})


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
            text, confidence = ocr_with_confidence(image, TESSERACT_LANG_MAP[language])
            detected_lang = language
        else:
            # Auto: Tesseract only recognizes text from the script(s) it's
            # told to look for, so run both languages and keep whichever
            # found more/more-confident text rather than silently missing
            # one script entirely.
            en_text, en_confidence = ocr_with_confidence(image, 'eng')
            pa_text, pa_confidence = ocr_with_confidence(image, 'pan')
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


_GURMUKHI_RANGE = range(0x0A00, 0x0A80)


def _detect_script_lang(text):
    """Cheap, torch-free script detection from a text sample: Gurmukhi
    unicode block presence means Punjabi, otherwise assume English. Used
    only to resolve 'auto' source_lang for /translate-document's PDF
    text-layer path, where a real text sample is already available for free
    (no need to run OCR confidence scoring like the image path does)."""
    return 'pa' if any(ord(ch) in _GURMUKHI_RANGE for ch in text) else 'en'


def _detect_image_source_lang(image):
    """Same auto-detection approach as /ocr: run Tesseract for both
    languages and keep whichever found more/more-confident text."""
    en_text, en_confidence = ocr_with_confidence(image, 'eng')
    pa_text, pa_confidence = ocr_with_confidence(image, 'pan')
    if pa_confidence > en_confidence and pa_text:
        return 'pa'
    return 'en'


@app.route('/translate-document', methods=['POST'])
def translate_document():
    """
    Recreate a photographed/scanned document or PDF with its text replaced
    by the translation, preserving approximate position/size/color (image
    path) or the exact original font size/color (PDF text-layer path). See
    document_translator.py's module docstring for the full pipeline
    explanation and its honest fidelity limits - this is OCR/CV-based
    reconstruction, not true document-structure editing, for scanned pages.

    Request body:
    {
        "file": "base64_encoded_file_data",
        "file_type": "image" | "pdf",   # optional - auto-detected from the
                                         # decoded bytes if omitted
        "source_lang": "auto",          # 'auto', 'en', or 'pa'
        "target_lang": "pa",            # 'en' or 'pa'
        "gender": "m"                   # optional: 'm' (default) or 'f'
    }

    Returns:
    {
        "output_type": "image" | "pdf",
        "data": "base64_encoded_result",
        "source_lang": "en" | "pa",
        "target_lang": "en" | "pa"
    }
    """
    try:
        data = request.json
        file_base64 = data.get('file', '')
        file_type = data.get('file_type', '')
        source_lang = data.get('source_lang', 'auto')
        target_lang = data.get('target_lang', 'pa')
        gender = data.get('gender', 'm')
        if gender not in ('m', 'f'):
            gender = 'm'
        if target_lang not in ('en', 'pa'):
            return jsonify({'error': "target_lang must be 'en' or 'pa'"}), 400

        if not file_base64:
            return jsonify({'error': 'No file data provided'}), 400

        file_bytes = base64.b64decode(file_base64)
        if len(file_bytes) > _MAX_DOCUMENT_BYTES:
            return jsonify({'error': 'File too large'}), 413

        if not file_type:
            file_type = 'pdf' if file_bytes[:4] == b'%PDF' else 'image'
        if file_type not in ('image', 'pdf'):
            return jsonify({'error': "file_type must be 'image' or 'pdf'"}), 400

        font_path = _DOCUMENT_FONT_PATHS[target_lang]
        print(f"\n📄 Document translation request ({file_type}, source={source_lang}, "
              f"target={target_lang}, size={len(file_bytes)} bytes)")

        if file_type == 'image':
            try:
                image = Image.open(io.BytesIO(file_bytes))
                # See /ocr's identical handling - phone cameras store
                # rotation as EXIF metadata, not rotated pixels.
                image = ImageOps.exif_transpose(image)
                image = image.convert('RGB')
            except Exception:
                return jsonify({'error': 'Could not read image data'}), 400

            resolved_source_lang = source_lang if source_lang in ('en', 'pa') else _detect_image_source_lang(image)
            tesseract_lang = TESSERACT_LANG_MAP[resolved_source_lang]
            translate_fn = lambda text: translate_plain_text(text, resolved_source_lang, target_lang, gender)

            recreated = document_translator.recreate_image_with_translated_text(
                image, tesseract_lang, font_path, translate_fn
            )
            buf = io.BytesIO()
            recreated.save(buf, format='PNG')
            result_base64 = base64.b64encode(buf.getvalue()).decode('ascii')

            print(f"   ✅ Document recreation complete ({resolved_source_lang} → {target_lang})")
            return jsonify({
                'output_type': 'image',
                'data': result_base64,
                'source_lang': resolved_source_lang,
                'target_lang': target_lang,
            })

        else:  # file_type == 'pdf'
            try:
                probe_doc = pymupdf.open(stream=file_bytes, filetype='pdf')
            except Exception:
                return jsonify({'error': 'Could not read PDF data'}), 400

            if source_lang in ('en', 'pa'):
                resolved_source_lang = source_lang
            else:
                sample_text = probe_doc[0].get_text().strip() if len(probe_doc) else ''
                if len(sample_text) >= 10:
                    resolved_source_lang = _detect_script_lang(sample_text)
                else:
                    # No real text layer to sample from (scanned first
                    # page) - rasterize just that page to decide the
                    # script via OCR confidence, same as the image path.
                    zoom = 150 / 72.0
                    pix = probe_doc[0].get_pixmap(matrix=pymupdf.Matrix(zoom, zoom))
                    sample_image = Image.frombytes('RGB', (pix.width, pix.height), pix.samples)
                    resolved_source_lang = _detect_image_source_lang(sample_image)
            probe_doc.close()

            tesseract_lang = TESSERACT_LANG_MAP[resolved_source_lang]
            translate_fn = lambda text: translate_plain_text(text, resolved_source_lang, target_lang, gender)

            result_bytes = document_translator.recreate_pdf_with_translated_text(
                file_bytes, tesseract_lang, font_path, translate_fn
            )
            result_base64 = base64.b64encode(result_bytes).decode('ascii')

            print(f"   ✅ Document recreation complete ({resolved_source_lang} → {target_lang})")
            return jsonify({
                'output_type': 'pdf',
                'data': result_base64,
                'source_lang': resolved_source_lang,
                'target_lang': target_lang,
            })

    except pytesseract.TesseractNotFoundError:
        return jsonify({
            'error': 'Tesseract OCR is not installed on the server',
            'message': 'Install Tesseract OCR (with eng + pan trained data) and restart the server.',
        }), 503
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"❌ Document translation error: {str(e)}")
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
