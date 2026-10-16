"""
Accuracy (WER) + speed benchmark for NVIDIA Parakeet-unified-en-0.6b (RNNT, ONNX,
torch-free) - the sole production English ASR model (Whisper has been fully
removed from the backend, so there is no longer a second English ASR model to
compare against; see repo memory for the historical Parakeet-vs-Whisper numbers
that justified the switch).

Test audio is synthesized via gTTS (clean, single-speaker TTS) for a diverse,
reproducible set of reference sentences covering greetings, numbers, proper
nouns, directions, medical/emergency phrases, and longer complex sentences -
the kind of content this app's users actually say.

CAVEAT (same spirit as evaluate_translation_accuracy.py's caveats): gTTS audio
is clean synthetic speech, not real noisy mic input - this measures Parakeet's
lexical/vocabulary accuracy under controlled conditions, not real-world
robustness to accents/noise/microphone artifacts. Still a much more rigorous,
repeatable signal than anecdotal impressions.

Usage: python evaluate_parakeet_unified_accuracy.py
"""
import os
import tempfile
import time

import jiwer
import numpy as np
from faster_whisper.audio import decode_audio
from gtts import gTTS

from parakeet_unified_asr import ParakeetUnifiedASR

PARAKEET_UNIFIED_DIR = os.path.join(os.path.dirname(__file__), "models", "parakeet-unified-en-onnx", "onnx_int8")

# Diverse English reference sentences - common app-relevant content
# (greetings, numbers, names, directions, medical/emergency, longer/complex).
TEST_SET = [
    "Hello, how are you today?",
    "Thank you very much for your help.",
    "What is your name?",
    "Can you give me a glass of water, please?",
    "I need a doctor right away.",
    "Call an ambulance, this is an emergency.",
    "Where is the nearest train station?",
    "My phone number is nine one five, two two three, four four five six.",
    "I was born on the fourteenth of March, nineteen ninety two.",
    "Harpreet Singh and Jaspreet Kaur are going to the market.",
    "Turn left at the next traffic light and then go straight.",
    "I lost my passport at the airport yesterday afternoon.",
    "Can I have the bill, please?",
    "The weather department predicted heavy rain this weekend.",
    "My brother lives in Delhi because he has a good job there.",
    "Please keep the change and have a wonderful day.",
    "The shop closes at eight o'clock in the evening.",
    "I am really looking forward to seeing you again soon.",
    "She said that the meeting has been moved to next Tuesday.",
    "This book was absolutely fascinating from start to finish.",
    # Extra: Indian proper nouns in English speech - very common for this
    # app's bilingual Punjabi-English users, worth checking in isolation.
    "Gurpreet Kaur called Manpreet Singh about the wedding.",
    "Amandeep and Simran went to visit their uncle in Ludhiana.",
    "Rajinder Kaur works at the hospital in Jalandhar.",
    "Baljeet Singh will pick you up from Amritsar airport.",
    "Please tell Navjot that the parcel arrived safely.",
]


def synthesize_clip(text, out_path):
    gTTS(text=text, lang="en").save(out_path)
    return decode_audio(out_path, sampling_rate=16000).astype(np.float32)


def normalize(text):
    return jiwer.Compose([
        jiwer.ToLowerCase(),
        jiwer.RemovePunctuation(),
        jiwer.RemoveMultipleSpaces(),
        jiwer.Strip(),
    ])(text)


def main():
    print("Loading Parakeet-unified-en-0.6b (ONNX)...")
    parakeet = ParakeetUnifiedASR(
        os.path.join(PARAKEET_UNIFIED_DIR, "encoder.int8.onnx"),
        os.path.join(PARAKEET_UNIFIED_DIR, "decoder_joint.int8.onnx"),
        os.path.join(PARAKEET_UNIFIED_DIR, "tokenizer.model"),
        cpu_threads=4,
    )

    wers, times = [], []

    print(f"\nSynthesizing + transcribing {len(TEST_SET)} test sentences via gTTS...\n")
    with tempfile.TemporaryDirectory() as tmp_dir:
        for i, ref_text in enumerate(TEST_SET):
            mp3_path = os.path.join(tmp_dir, f"clip_{i}.mp3")
            audio = synthesize_clip(ref_text, mp3_path)

            t0 = time.perf_counter()
            pk_text, pk_conf = parakeet.transcribe(audio)
            pk_time = (time.perf_counter() - t0) * 1000

            ref_norm = normalize(ref_text)
            pk_wer = jiwer.wer(ref_norm, normalize(pk_text)) * 100

            wers.append(pk_wer)
            times.append(pk_time)

            print(f"[{i + 1}/{len(TEST_SET)}] ref: {ref_text}")
            print(f"    Parakeet ({pk_time:6.0f}ms, WER {pk_wer:5.1f}%, conf {pk_conf:.2f}): {pk_text}")
            print()

    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Model':<12} {'Mean WER':>10} {'Median WER':>12} {'Mean latency':>14}")
    print(f"{'Parakeet':<12} {np.mean(wers):>9.1f}% {np.median(wers):>11.1f}% {np.mean(times):>12.0f}ms")


if __name__ == "__main__":
    main()
