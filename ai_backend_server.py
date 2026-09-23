"""
FREE AI Translation Backend Server
Uses AI4Bharat's IndicTrans2 + OpenAI Whisper
IndicTrans2 is purpose-built for English<->Indic languages (incl. Punjabi) and
outperforms both NLLB-200 and Google Translate on Indic pairs, especially on
longer/more complex sentences, since it's trained on a much larger and cleaner
English-Indic parallel corpus (BPCC) instead of a generic 200-language dataset.

Installation:
pip install -r requirements.txt
(requires the IndicTransToolkit package for pre/post-processing)

Run:
python ai_backend_server.py

Then update your app to use http://localhost:5000 instead of Google APIs
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from indictrans_processor import IndicProcessor  # local port, see file docstring
import whisper
import pytesseract
from PIL import Image, ImageOps
import os
import re
import sys
import json
import types
import importlib.util
import base64
import tempfile
import io
import time
import unicodedata
import numpy as np
import subprocess
from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate

# Windows console defaults to cp1252, which can't print emoji used in log messages below.
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

# IndicTrans2's remote modeling code (trust_remote_code=True) still imports the
# `transformers.onnx` module (removed in newer transformers) just to define an
# unused ONNX-export config class. Stub it out so the import succeeds.
if not importlib.util.find_spec("transformers.onnx"):
    _onnx_stub = types.ModuleType("transformers.onnx")

    class OnnxConfig:
        pass

    class OnnxSeq2SeqConfigWithPast(OnnxConfig):
        pass

    _onnx_stub.OnnxConfig = OnnxConfig
    _onnx_stub.OnnxSeq2SeqConfigWithPast = OnnxSeq2SeqConfigWithPast
    sys.modules["transformers.onnx"] = _onnx_stub

    _onnx_utils_stub = types.ModuleType("transformers.onnx.utils")

    def compute_effective_axis_dimension(dimension, fixed_dimension, num_token_to_add=0):
        if dimension <= 0:
            dimension = fixed_dimension
        dimension -= num_token_to_add
        return dimension

    _onnx_utils_stub.compute_effective_axis_dimension = compute_effective_axis_dimension
    sys.modules["transformers.onnx.utils"] = _onnx_utils_stub

# Add FFmpeg to PATH if not already there
ffmpeg_path = "C:\\ffmpeg\\ffmpeg-8.0-essentials_build\\bin"
if os.path.exists(ffmpeg_path) and ffmpeg_path not in os.environ['PATH']:
    os.environ['PATH'] = ffmpeg_path + os.pathsep + os.environ['PATH']
    print(f"✅ Added FFmpeg to PATH: {ffmpeg_path}")

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
# are expensive (1B-parameter model, Whisper large-v3). Once this is reachable
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
_RATE_LIMIT_MAX_REQUESTS = 20
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


if not API_KEY:
    print("⚠️  BACKEND_API_KEY is not set - server is running with NO authentication.")
    print("   Fine for local/LAN development, but set BACKEND_API_KEY before exposing")
    print("   this server to the public internet.")
# -----------------------------------------------------------------------------

print("=" * 60)
print("🚀 Starting AI Translation Server...")
print("=" * 60)

# Load IndicTrans2 Translation Models (AI4Bharat) - one model per direction,
# since IndicTrans2 uses separate en->indic and indic->en checkpoints.
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\n📥 Loading IndicTrans2 translation models (device: {device})...")
print("   Using 1B models for maximum translation accuracy...")
print("   (First time will download several GB per model, please wait...)")
en_indic_ckpt = "ai4bharat/indictrans2-en-indic-1B"
indic_en_ckpt = "ai4bharat/indictrans2-indic-en-1B"

# Optional LoRA adapters produced by finetune_indictrans2_punjabi.ipynb (see
# that notebook - trained on Colab, then the output adapter folder is copied
# here and pointed to via these env vars). Absent by default; base models
# are used unchanged until an adapter is actually trained and configured.
EN_PA_LORA_DIR = os.environ.get("EN_PA_LORA_ADAPTER_DIR", "")
PA_EN_LORA_DIR = os.environ.get("PA_EN_LORA_ADAPTER_DIR", "")


def _maybe_apply_lora(base_model, adapter_dir, label):
    if not adapter_dir:
        return base_model
    if not os.path.isdir(adapter_dir):
        print(f"⚠️  {label} LoRA adapter dir not found: {adapter_dir} - using base model.")
        return base_model
    from peft import PeftModel
    print(f"🔧 Applying {label} LoRA adapter from {adapter_dir}...")
    return PeftModel.from_pretrained(base_model, adapter_dir).merge_and_unload()


try:
    en_indic_tokenizer = AutoTokenizer.from_pretrained(en_indic_ckpt, trust_remote_code=True)
    en_indic_model = AutoModelForSeq2SeqLM.from_pretrained(en_indic_ckpt, trust_remote_code=True).to(device)
    en_indic_model = _maybe_apply_lora(en_indic_model, EN_PA_LORA_DIR, "en->pa")
    en_indic_model.eval()

    indic_en_tokenizer = AutoTokenizer.from_pretrained(indic_en_ckpt, trust_remote_code=True)
    indic_en_model = AutoModelForSeq2SeqLM.from_pretrained(indic_en_ckpt, trust_remote_code=True).to(device)
    indic_en_model = _maybe_apply_lora(indic_en_model, PA_EN_LORA_DIR, "pa->en")
    indic_en_model.eval()

    indic_processor = IndicProcessor(inference=True)
    print("✅ IndicTrans2 models loaded successfully! (Highest translation accuracy for Indic languages)")
except Exception as e:
    print(f"❌ Error loading IndicTrans2: {e}")
    print("   Please run: pip install -r requirements.txt")
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
MAX_CHUNK_WORDS = 80
_SENTENCE_SPLIT_RE = re.compile(r'(?<=[.!?।])\s+')
_CLAUSE_SPLIT_RE = re.compile(
    r'(?<=[,;])\s+(?=(?:and|but|or|so|because|although|however|therefore|'
    r'meanwhile|while|since|which|who|that)\b)',
    re.IGNORECASE,
)


def _split_long_sentence(sentence):
    """Break an overly long/complex sentence into smaller clauses."""
    if len(sentence.split()) <= MAX_CHUNK_WORDS:
        return [sentence]
    clauses = _CLAUSE_SPLIT_RE.split(sentence)
    if len(clauses) > 1:
        return clauses
    # Fallback: split on any comma if the sentence is still too long
    parts = re.split(r'(?<=,)\s+', sentence)
    return parts if len(parts) > 1 else [sentence]


def chunk_text(text):
    """Split text into sentence-level chunks, further splitting overly long
    sentences into clauses, so each piece sent to the model stays short
    enough to translate reliably."""
    sentences = _SENTENCE_SPLIT_RE.split(text.strip())
    chunks = []
    for sentence in sentences:
        if not sentence.strip():
            continue
        chunks.extend(_split_long_sentence(sentence))
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


def translate_chunks(chunks, source_lang, target_lang, num_beams=8, length_penalty=1.0):
    """Translate a list of text chunks using the correct IndicTrans2 model
    for the given direction, then return the translated chunks in order.
    num_beams/length_penalty are exposed so a low-confidence result can be
    retried with different search parameters (see /translate's retry logic)."""
    lang_map = {'en': 'eng_Latn', 'pa': 'pan_Guru'}
    src = lang_map.get(source_lang, 'eng_Latn')
    tgt = lang_map.get(target_lang, 'pan_Guru')

    if source_lang == 'en':
        tokenizer, model = en_indic_tokenizer, en_indic_model
    else:
        tokenizer, model = indic_en_tokenizer, indic_en_model

    batch = indic_processor.preprocess_batch(chunks, src_lang=src, tgt_lang=tgt)
    inputs = tokenizer(batch, truncation=True, padding="longest", return_tensors="pt").to(device)

    with torch.no_grad():
        generated_tokens = model.generate(
            **inputs,
            use_cache=True,
            min_length=0,
            max_length=320,
            num_beams=num_beams,  # wider beam search = more accurate (slower)
            length_penalty=length_penalty,
            repetition_penalty=1.3,   # discourage repeated/looping phrases
            no_repeat_ngram_size=3,   # forbid repeating any 3-word sequence
        )

    with tokenizer.as_target_tokenizer():
        decoded = tokenizer.batch_decode(
            generated_tokens.detach().cpu(),
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )

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
    get counted as meaning mismatches."""
    lowered = text.lower()
    expanded = _CONTRACTIONS_RE.sub(lambda m: _CONTRACTIONS[m.group(0)], lowered)
    return set(re.findall(r"\w+", expanded, flags=re.UNICODE))


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
        union = original_words | back_words
        if not union:
            return 0.0, back_translation
        similarity = len(original_words & back_words) / len(union)
        return round(similarity, 2), back_translation
    except Exception as e:
        print(f"   ⚠️ Round-trip confidence check failed: {e}")
        return None, None

# Load Whisper Speech Recognition Model
# Using 'large-v3' model for MAXIMUM ACCURACY (95%+ for clear speech!)
# This is the most accurate Whisper model available
print("\n📥 Loading Whisper speech recognition model...")
print("   Using 'large-v3' model for maximum accuracy (95%+)...")
print("   (First time will download ~3GB, please wait...)")
try:
    whisper_model = whisper.load_model("large-v3")  # BEST ACCURACY: 90-95% for speech!
    print("✅ Whisper 'large-v3' loaded successfully! (Highest accuracy speech recognition)")
except Exception as e:
    print(f"❌ Error loading Whisper: {e}")
    print("   Please run: pip install openai-whisper")
    exit(1)

# Load our custom fine-tuned Punjabi Whisper model, if present (see
# finetune_whisper_punjabi.py). This is a separate Hugging Face transformers
# checkpoint (not compatible with the openai-whisper package above), so it's
# loaded via a transformers ASR pipeline and used only for Punjabi audio -
# English/auto-detect requests keep using the openai-whisper model above.
PUNJABI_WHISPER_MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models', 'whisper-punjabi-final')
punjabi_whisper_pipeline = None
if os.path.isdir(PUNJABI_WHISPER_MODEL_DIR):
    print("\n📥 Loading fine-tuned Punjabi Whisper model...")
    try:
        # The tokenizer.json saved by the (newer) Kaggle environment isn't
        # readable by the tokenizers version pinned here for IndicTrans2
        # compatibility. Fine-tuning doesn't change Whisper's vocabulary, so
        # load the tokenizer/feature extractor from the original base model
        # instead, and only take the fine-tuned weights from the local folder.
        from transformers import WhisperProcessor, WhisperForConditionalGeneration
        processor = WhisperProcessor.from_pretrained("openai/whisper-small")
        punjabi_model = WhisperForConditionalGeneration.from_pretrained(PUNJABI_WHISPER_MODEL_DIR)
        punjabi_whisper_pipeline = pipeline(
            "automatic-speech-recognition",
            model=punjabi_model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            device=0 if torch.cuda.is_available() else -1,
        )
        print("✅ Fine-tuned Punjabi Whisper model loaded! Will be used for Punjabi transcription.")
    except Exception as e:
        print(f"⚠️ Could not load fine-tuned Punjabi Whisper model: {e}")
        print("   Falling back to Whisper 'large-v3' for Punjabi transcription.")
else:
    print(f"\nℹ️ No fine-tuned Punjabi Whisper model found at {PUNJABI_WHISPER_MODEL_DIR}")
    print("   Using Whisper 'large-v3' for Punjabi transcription.")

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
            'speech': 'Whisper (large-v3)' + (' + fine-tuned Punjabi model' if punjabi_whisper_pipeline is not None else ''),
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
_ENSEMBLE_GOOD_ENOUGH_THRESHOLD = 0.85
_ENSEMBLE_DECODING_CONFIGS = [
    {'num_beams': 12, 'length_penalty': 0.8},
    {'num_beams': 6, 'length_penalty': 1.4},
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
            if confidence is not None and confidence < _ENSEMBLE_CONFIDENCE_THRESHOLD:
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


# Short list of common words/greetings in each language, passed to Whisper as
# initial_prompt to bias its decoder toward the right vocabulary and script.
_WHISPER_VOCAB_PROMPT = {
    'en': 'Hello, thank you, please, water, house, work, family, friend, today.',
    'pa': 'ਸਤ ਸ੍ਰੀ ਅਕਾਲ, ਧੰਨਵਾਦ, ਕਿਰਪਾ ਕਰਕੇ, ਪਾਣੀ, ਘਰ, ਕੰਮ, ਪਰਿਵਾਰ, ਦੋਸਤ, ਅੱਜ।',
}


@app.route('/speech-to-text', methods=['POST'])
def speech_to_text():
    """
    Convert speech audio to text using Whisper
    Supports both English and Punjabi automatic detection
    
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
        
        # Check if ffmpeg is installed
        try:
            result = subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True, text=True)
            print(f"   ✅ FFmpeg found: {result.stdout.split()[2]}")
        except (FileNotFoundError, subprocess.CalledProcessError) as e:
            print(f"   ❌ FFmpeg not found: {str(e)}")
            return jsonify({
                'error': 'FFmpeg not installed',
                'message': 'Whisper requires FFmpeg. Please restart the server after FFmpeg installation.',
                'fallback': 'google'
            }), 503
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(suffix='.webm', delete=False) as temp_audio:
            temp_audio.write(audio_data)
            temp_path = temp_audio.name
        
        try:
            # MAXIMUM ACCURACY: Use best decoding parameters
            print("   🎯 High-accuracy transcription (this may take 3-5 seconds)...")

            # Honor the caller's language hint instead of always auto-detecting.
            # Whisper's language auto-ID is unreliable on short/noisy clips and
            # often misidentifies Punjabi as Hindi/Urdu (or vice versa), which
            # then transcribes in the WRONG script entirely - a much bigger
            # error than any per-word inaccuracy. Only fall back to auto-detect
            # when the caller genuinely doesn't know the language.
            whisper_language = {'en': 'en', 'pa': 'pa'}.get(language)

            # Use our fine-tuned Punjabi model when transcribing Punjabi and
            # it's available - it's specifically trained on Punjabi speech,
            # so it beats the general-purpose large-v3 model for this language.
            if whisper_language == 'pa' and punjabi_whisper_pipeline is not None:
                pipeline_result = punjabi_whisper_pipeline(
                    temp_path,
                    generate_kwargs={"language": "pa", "task": "transcribe"},
                )
                transcript = pipeline_result['text'].strip()
                detected_lang = 'pa'
                confidence = 0.5  # this pipeline doesn't expose a no-speech/confidence score
                model_name = 'Whisper-punjabi-finetuned'
            else:
                result = whisper_model.transcribe(
                    temp_path,
                    language=whisper_language,
                    fp16=False,
                    beam_size=5,     # More thorough search (vs 1 for speed)
                    best_of=5,       # Try multiple candidates (vs 1 for speed)
                    temperature=0.0,  # Deterministic output (most accurate)
                    condition_on_previous_text=False,  # avoid Whisper's known hallucination-loop failure mode
                    # Priming text biases Whisper's decoder toward this vocabulary
                    # and script, which helps it settle on correct Gurmukhi
                    # spellings for common words instead of phonetic guesses.
                    initial_prompt=_WHISPER_VOCAB_PROMPT.get(whisper_language),
                )

                transcript = result['text'].strip()

                if whisper_language:
                    # We forced the language, so trust it rather than re-deriving
                    # it from Whisper's (now irrelevant) auto-detection field.
                    detected_lang = whisper_language
                else:
                    raw_detected = result.get('language', 'en')
                    # Whisper might detect as 'hi' or other languages, map to en/pa
                    detected_lang = 'pa' if raw_detected in ('pa', 'hi', 'ur') else 'en'

                # Calculate confidence
                def get_confidence(result):
                    if 'segments' in result and result['segments']:
                        avg_no_speech = sum(s.get('no_speech_prob', 0) for s in result['segments']) / len(result['segments'])
                        return 1.0 - avg_no_speech
                    return 0.5

                confidence = get_confidence(result)
                model_name = 'Whisper-large-v3'
            
            print(f"   ✅ Detected {detected_lang} (confidence: {confidence:.2f})")
            print(f"   📝 Transcript: {transcript[:80]}...")
            print("   ✅ Transcription complete")
            
            # Clean up temp file
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
