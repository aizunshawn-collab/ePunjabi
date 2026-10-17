"""
Accuracy, confidence, and latency benchmark for the /ocr endpoint's Tesseract
OCR pipeline (ocr_processor.py) - the backend for the app's "scan a photo"
translation feature.

Test images are SYNTHESIZED (rendered text, not real camera photos) under
two condition profiles per test string:
  - "clean":      sharp rendered text, high contrast, no noise - an upper
                  bound / sanity-check condition (closer to a flat scanned
                  document than a phone photo).
  - "photo_like": the same text with rotation, heavy Gaussian blur, reduced
                  contrast/resolution (downscale+upscale), sensor noise, and
                  low-quality JPEG re-compression - a rough proxy for a
                  less-than-ideal handheld phone photo (slight motion blur,
                  distance, low light).

CAVEAT (same spirit as evaluate_translation_accuracy.py /
evaluate_parakeet_unified_accuracy.py's caveats): this measures Tesseract's
lexical accuracy under controlled synthetic degradation, NOT full real-world
photo robustness (no real glare, perspective warp, complex/cluttered
backgrounds, varied lighting, or genuine camera lens blur) - still a much
more repeatable, comparable signal than anecdotal impressions, and the gap
between "clean" and "photo_like" numbers below is itself informative.

Standalone - imports ocr_processor.py directly (torch-free, no heavy
IndicTrans2/Parakeet/NeMo model loading), so this runs in a couple of
seconds of setup rather than needing the full backend server up.

Requires a Gurmukhi-capable font for the Punjabi test images (Windows ships
no such font by default) - set the GURMUKHI_FONT_PATH env var to one
(e.g. a locally downloaded Noto Sans Gurmukhi .ttf); Punjabi cases are
skipped with a warning if not found.

Usage: python evaluate_ocr_accuracy.py
"""
import io
import os
import time

import jiwer
import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont

from ocr_processor import TESSERACT_LANG_MAP, ocr_with_confidence

GURMUKHI_FONT_PATH = os.environ.get(
    "GURMUKHI_FONT_PATH",
    os.path.join(os.environ.get("TEMP", ""), "ocr_test_fonts", "NotoSansGurmukhi.ttf"),
)
ENGLISH_FONT_PATH = r"C:\Windows\Fonts\arial.ttf"

# (lang, reference_text) - short phrases representative of signs/menus/docs,
# this app's actual "scan a photo" use case.
TEST_SET = [
    ("en", "Welcome to Delhi Airport"),
    ("en", "Exit this way"),
    ("en", "Menu: Chicken Curry - Rs. 250"),
    ("en", "Platform 4 - Train to Amritsar"),
    ("en", "No smoking in this area"),
    ("en", "Open 9 AM to 10 PM"),
    ("en", "Please wear a mask"),
    ("en", "Room 204"),
    ("pa", "ਸਤ ਸ੍ਰੀ ਅਕਾਲ ਜੀ"),
    ("pa", "ਬਾਹਰ ਜਾਣ ਦਾ ਰਸਤਾ"),
    ("pa", "ਕਿਰਪਾ ਕਰਕੇ ਕਤਾਰ ਵਿੱਚ ਖੜ੍ਹੇ ਰਹੋ"),
    ("pa", "ਹਸਪਤਾਲ ਇਸ ਪਾਸੇ ਹੈ"),
]


def render_clean(text, font_path, font_size=40):
    font = ImageFont.truetype(font_path, font_size)
    dummy = Image.new("RGB", (1, 1))
    bbox = ImageDraw.Draw(dummy).textbbox((0, 0), text, font=font)
    width, height = bbox[2] - bbox[0] + 60, bbox[3] - bbox[1] + 60
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.text((30 - bbox[0], 30 - bbox[1]), text, font=font, fill="black")
    return image


def render_photo_like(text, font_path, font_size=40):
    image = render_clean(text, font_path, font_size)
    image = image.rotate(-6, expand=True, fillcolor="white")
    image = image.filter(ImageFilter.GaussianBlur(radius=2.2))
    image = ImageEnhance.Contrast(image).enhance(0.55)
    image = ImageEnhance.Brightness(image).enhance(1.15)

    # Simulate a lower-effective-resolution camera shot (common for a photo
    # taken from a bit further away) by downscaling then upscaling back.
    w, h = image.size
    image = image.resize((max(1, w // 3), max(1, h // 3)), Image.BILINEAR)
    image = image.resize((w, h), Image.BILINEAR)

    # Sensor/low-light noise.
    arr = np.asarray(image).astype(np.int16)
    noise = np.random.default_rng(0).normal(0, 14, arr.shape)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    image = Image.fromarray(arr)

    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=35)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def cer(reference, hypothesis):
    if not reference.strip():
        return 0.0 if not hypothesis.strip() else 1.0
    return jiwer.cer(reference, hypothesis)


def run_condition(label, render_fn):
    print(f"\n{'=' * 70}\nCONDITION: {label}\n{'=' * 70}")
    rows = []
    for lang, ref_text in TEST_SET:
        if lang == "pa" and not os.path.exists(GURMUKHI_FONT_PATH):
            print(f"[skip] no Gurmukhi font at {GURMUKHI_FONT_PATH}: {ref_text}")
            continue
        font_path = ENGLISH_FONT_PATH if lang == "en" else GURMUKHI_FONT_PATH
        image = render_fn(ref_text, font_path)

        t0 = time.perf_counter()
        text, confidence = ocr_with_confidence(image, TESSERACT_LANG_MAP[lang])
        elapsed_ms = (time.perf_counter() - t0) * 1000

        # Auto-detect check: run both languages, see if the higher-confidence
        # one matches the true language (mirrors /ocr's 'auto' mode logic).
        en_text, en_conf = ocr_with_confidence(image, "eng")
        pa_text, pa_conf = ocr_with_confidence(image, "pan")
        auto_detected = "pa" if (pa_conf > en_conf and pa_text) else "en"
        auto_correct = auto_detected == lang

        error_rate = cer(ref_text, text) * 100
        rows.append((lang, ref_text, text, confidence, elapsed_ms, error_rate, auto_correct))

        print(f"[{lang}] ref : {ref_text}")
        print(f"     hyp : {text}")
        print(f"     CER {error_rate:5.1f}%  conf {confidence:5.1f}  {elapsed_ms:6.0f}ms  "
              f"auto->{auto_detected}{'' if auto_correct else '  <-- WRONG'}")

    if not rows:
        return rows

    confidences = [r[3] for r in rows]
    latencies = [r[4] for r in rows]
    error_rates = [r[5] for r in rows]
    auto_accuracy = sum(r[6] for r in rows) / len(rows) * 100

    print(f"\n--- {label} summary ({len(rows)} cases) ---")
    print(f"Mean CER:        {np.mean(error_rates):.1f}%")
    print(f"Median CER:      {np.median(error_rates):.1f}%")
    print(f"Mean confidence: {np.mean(confidences):.1f}")
    print(f"Mean latency:    {np.mean(latencies):.0f}ms")
    print(f"Auto-detect accuracy: {auto_accuracy:.0f}%")
    return rows


def main():
    run_condition("clean", render_clean)
    run_condition("photo_like", render_photo_like)


if __name__ == "__main__":
    main()
