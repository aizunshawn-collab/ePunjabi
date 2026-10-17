"""
Tesseract OCR wrapper - torch-free, no heavy ML model loading, so it can be
imported standalone (e.g. by a benchmark script) without pulling in
IndicTrans2/Parakeet/NeMo. Used by ai_backend_server.py's /ocr route.

Tesseract OCR itself ships as a separate system install, not a pip package -
pytesseract just shells out to the tesseract.exe/tesseract binary.
"""
import os

import pytesseract

# Point pytesseract at whichever known Windows install location exists (the
# UB Mannheim installer defaults to per-machine Program Files, but can also
# install per-user under %LOCALAPPDATA%); if tesseract is already on PATH
# this whole block is a harmless no-op. Linux/production (Dockerfile installs
# tesseract-ocr via apt, which lands on PATH) never hits this at all.
_TESSERACT_WIN_PATHS = [
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    os.path.expandvars(r"%LOCALAPPDATA%\Programs\Tesseract-OCR\tesseract.exe"),
]
for _tesseract_path in _TESSERACT_WIN_PATHS:
    if os.path.exists(_tesseract_path):
        pytesseract.pytesseract.tesseract_cmd = _tesseract_path
        break

# Language codes accepted by Tesseract for this app's two supported languages.
TESSERACT_LANG_MAP = {'en': 'eng', 'pa': 'pan'}

# Fallback page-segmentation modes tried (in order) when Tesseract's default
# full-page layout analysis (PSM 3) finds zero text - covers the common
# "one big word/line filling the frame" case (e.g. a close-up phone photo),
# where PSM 3's paragraph/column detection can fail entirely even though a
# simpler assumption about the layout would succeed.
# 6 = single uniform block of text, 7 = single text line, 8 = single word.
OCR_FALLBACK_PSMS = (6, 7, 8)


def run_tesseract(image, tesseract_lang, config=''):
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


def ocr_with_confidence(image, tesseract_lang):
    """Run Tesseract OCR for one language and return (text, avg_confidence).
    avg_confidence is the mean of Tesseract's own per-word confidence scores
    (0-100, -1 for non-text regions which are excluded) - used to pick the
    better result between English and Punjabi when the caller doesn't know
    which script the photo contains.

    Falls back to alternate page-segmentation modes (see OCR_FALLBACK_PSMS)
    if the default full-page analysis finds no text at all - a close-up
    photo of a single word/line can otherwise come back completely empty."""
    recognized_text, avg_confidence = run_tesseract(image, tesseract_lang)
    if recognized_text:
        return recognized_text, avg_confidence

    for psm in OCR_FALLBACK_PSMS:
        fallback_text, fallback_confidence = run_tesseract(image, tesseract_lang, config=f'--psm {psm}')
        if fallback_text:
            return fallback_text, fallback_confidence
    return recognized_text, avg_confidence
