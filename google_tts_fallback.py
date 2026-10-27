"""Optional Google Cloud Text-to-Speech fallback for the web build, used
only when the browser itself has no local voice for the requested language
installed (this is always true for Punjabi on essentially every desktop/
mobile browser today - see web/shims/tts.ts, which calls /tts below only
after confirming no local 'pa' voice exists via speechSynthesis.getVoices()).
Native Android builds never call this; they use the device's own
react-native-tts voices.

Disabled by default - only active when GOOGLE_TRANSLATE_API_KEY is set
(reused as-is: Cloud Translation and Cloud Text-to-Speech are both billed
through the same Google Cloud API key once both APIs are enabled on the
project - see google_translate_fallback.py for the sibling Translate
fallback this mirrors).

This module never raises: any failure (missing key, API not enabled,
network error, bad auth/quota, unexpected response shape) is caught and
reported as None, so a caller can always safely treat that as "fall back to
something else" rather than a hard error.
"""
import json
import os
import urllib.error
import urllib.request

API_KEY = os.environ.get('GOOGLE_TRANSLATE_API_KEY', '').strip()
_API_URL = 'https://texttospeech.googleapis.com/v1/text:synthesize'
_TIMEOUT_SECONDS = 15

# Wavenet voices - same ones this project used before migrating to the
# current IndicTrans2/react-native-tts architecture (see
# BILINGUAL_UI_UPDATES.md / google-tts-voice.html). Only a female voice is
# wired up for now (matches the old app's own default) - gender selection
# can be added later if needed.
_VOICE_NAMES = {
    'pa': 'pa-IN-Wavenet-A',
    'en': 'en-US-Wavenet-F',
}
_LANGUAGE_CODES = {
    'pa': 'pa-IN',
    'en': 'en-US',
}


def is_configured() -> bool:
    return bool(API_KEY)


def synthesize(text, language):
    """Returns base64-encoded MP3 audio for `text`, or None on any failure.
    `language` must be 'en' or 'pa'."""
    if not API_KEY or not text:
        return None
    voice_name = _VOICE_NAMES.get(language)
    language_code = _LANGUAGE_CODES.get(language)
    if not voice_name or not language_code:
        return None
    body = json.dumps({
        'input': {'text': text},
        'voice': {'languageCode': language_code, 'name': voice_name},
        'audioConfig': {'audioEncoding': 'MP3'},
    }).encode('utf-8')
    req = urllib.request.Request(
        f'{_API_URL}?key={API_KEY}',
        data=body,
        headers={'Content-Type': 'application/json'},
        method='POST',
    )
    try:
        with urllib.request.urlopen(req, timeout=_TIMEOUT_SECONDS) as resp:
            data = json.loads(resp.read().decode('utf-8'))
        audio = data.get('audioContent')
        return audio if isinstance(audio, str) else None
    except urllib.error.HTTPError as e:
        print(f"   ⚠️ Google Cloud TTS request failed: {e.code} {e.read().decode('utf-8', errors='replace')}")
        return None
    except (urllib.error.URLError, json.JSONDecodeError, OSError, ValueError) as e:
        print(f"   ⚠️ Google Cloud TTS request failed: {e}")
        return None
