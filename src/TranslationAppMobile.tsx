import React, {useState, useEffect, useMemo, useRef} from 'react';
import {
  StyleSheet,
  Text,
  View,
  TouchableOpacity,
  ScrollView,
  Alert,
  ActivityIndicator,
  PermissionsAndroid,
  Platform,
  TextInput,
  Modal,
  StatusBar,
} from 'react-native';
import {SafeAreaView} from 'react-native-safe-area-context';
import Icon from 'react-native-vector-icons/MaterialIcons';
import Voice from '@react-native-voice/voice';
import Tts from 'react-native-tts';
import AudioRecorderPlayer from 'react-native-audio-recorder-player';
import RNFS from 'react-native-fs';
import AsyncStorage from '@react-native-async-storage/async-storage';
import {launchCamera, launchImageLibrary, Asset} from 'react-native-image-picker';

// Local self-hosted AI backend (IndicTrans2 + Whisper, see ai_backend_server.py).
// This app relies entirely on this backend for translation/detection - there
// is no Google Translate fallback, so the backend must be running and
// reachable for translation to work.
// Fill this in with the deployed public backend URL (e.g.
// 'https://your-app-name.fly.dev' - see fly.toml) once it's hosted somewhere
// reachable from anywhere, not just a local network. Leave blank while still
// developing against a local/LAN backend - the Settings-screen override
// below always takes priority, so this only matters as the out-of-the-box
// default for a fresh install.
const PRODUCTION_BACKEND_URL = '';
// NOTE: On the Android emulator, "localhost" refers to the emulator itself,
// not your computer, so 10.0.2.2 (the documented emulator alias for the host
// machine's localhost) is used as the dev default there. On a REAL physical
// device, 10.0.2.2/localhost will NOT work - the backend server binds to all
// interfaces (0.0.0.0), so a phone on the same Wi-Fi network can reach it via
// the host computer's LAN IP instead, or (once deployed) via
// PRODUCTION_BACKEND_URL above from anywhere. Since a LAN IP varies per
// network and can't be baked into the app at build time, it's also
// user-configurable at runtime from the Settings screen (see
// BACKEND_URL_STORAGE_KEY below).
const DEFAULT_BACKEND_URL =
  PRODUCTION_BACKEND_URL || (Platform.OS === 'android' ? 'http://10.0.2.2:5000' : 'http://localhost:5000');
const BACKEND_URL_STORAGE_KEY = 'ai_backend_url';

let currentBackendUrl = DEFAULT_BACKEND_URL;

const getBackendUrl = (): string => currentBackendUrl;

const persistBackendUrl = async (url: string): Promise<void> => {
  currentBackendUrl = url;
  try {
    await AsyncStorage.setItem(BACKEND_URL_STORAGE_KEY, url);
  } catch {
    // Non-fatal: the URL still takes effect for this session even if it
    // can't be persisted for next launch.
  }
};

const loadPersistedBackendUrl = async (): Promise<string> => {
  try {
    const stored = await AsyncStorage.getItem(BACKEND_URL_STORAGE_KEY);
    if (stored) {
      currentBackendUrl = stored;
    }
  } catch {
    // Fall back to the default set above.
  }
  return currentBackendUrl;
};

// Shared secret the backend can require via BACKEND_API_KEY once it's
// exposed to the public internet (see ai_backend_server.py) - empty by
// default, which matches the backend's own no-auth local/LAN dev mode.
const BACKEND_API_KEY_STORAGE_KEY = 'ai_backend_api_key';

let currentApiKey = '';

const getApiKey = (): string => currentApiKey;

const persistApiKey = async (key: string): Promise<void> => {
  currentApiKey = key;
  try {
    await AsyncStorage.setItem(BACKEND_API_KEY_STORAGE_KEY, key);
  } catch {
    // Non-fatal, same as persistBackendUrl above.
  }
};

const loadPersistedApiKey = async (): Promise<string> => {
  try {
    const stored = await AsyncStorage.getItem(BACKEND_API_KEY_STORAGE_KEY);
    if (stored) {
      currentApiKey = stored;
    }
  } catch {
    // Fall back to the empty default set above.
  }
  return currentApiKey;
};

const backendHeaders = (extra?: Record<string, string>): Record<string, string> => {
  const headers: Record<string, string> = {...extra};
  if (currentApiKey) {
    headers['X-API-Key'] = currentApiKey;
  }
  return headers;
};
// Language detection itself is a cheap Unicode-range heuristic (no model),
// so it's always fast - but this is the FIRST backend call translateAndSpeak
// makes, so on a cold (scaled-to-zero) worker it's also the one that has to
// wait out the worker's cold-start/model-load time before anything responds.
// Keep this at least as generous as the expected cold-start time so a
// sleeping worker isn't mistaken for the backend being unreachable.
const AI_BACKEND_DETECT_TIMEOUT_MS = 45000;
// A single verify=true call already does 2 translation passes (forward +
// back-translation), and low-confidence results trigger up to 2 more
// ensemble-decoding retries on top of that - each pass on CPU-only 1B
// IndicTrans2 models can take 30-45s alone, so the worst case can approach
// 2-3 minutes. Measured a single short-sentence verify=true call taking
// ~44s in practice, so 60s was cutting it too close and caused real
// AbortError timeouts - keep this generous.
const AI_BACKEND_TRANSLATE_TIMEOUT_MS = 150000;
// Whisper large-v3 with beam_size=5/best_of=5 on CPU is considerably slower
// than a single-pass translation, so this gets a longer budget.
const AI_BACKEND_SPEECH_TIMEOUT_MS = 90000;
// Used only to decide, before recording starts, whether the backend is up -
// kept short so the app doesn't make the user wait to find out it needs to
// fall back to on-device recognition.
const AI_BACKEND_HEALTH_TIMEOUT_MS = 2500;
// Tesseract OCR runs twice (once per language) when the caller doesn't know
// which script the photo contains, so give it a generous budget.
const AI_BACKEND_OCR_TIMEOUT_MS = 30000;

// Where the recorded voice clip is written before being uploaded. Living in
// the app's private cache dir means no extra storage permission is needed.
const VOICE_RECORDING_PATH = `${RNFS.CachesDirectoryPath}/voice_input.m4a`;

// Single shared recorder instance for the app's lifetime.
const audioRecorderPlayer = new AudioRecorderPlayer();

const fetchWithTimeout = (url: string, options: RequestInit, timeoutMs: number): Promise<Response> => {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), timeoutMs);
  return fetch(url, {...options, signal: controller.signal}).finally(() => clearTimeout(timeoutId));
};

// Quick pre-flight check used before recording starts, so we can decide
// upfront whether to use the backend (better accuracy, esp. for Punjabi) or
// fall back to on-device recognition, rather than discovering the backend is
// unreachable only after already recording audio.
const isBackendReachable = async (): Promise<boolean> => {
  try {
    const response = await fetchWithTimeout(`${getBackendUrl()}/health`, {method: 'GET'}, AI_BACKEND_HEALTH_TIMEOUT_MS);
    return response.ok;
  } catch {
    return false;
  }
};

// Detect language via the local AI backend's heuristic detector.
const detectLanguageLocalAI = async (text: string): Promise<string> => {
  const response = await fetchWithTimeout(
    `${getBackendUrl()}/detect-language`,
    {
      method: 'POST',
      headers: backendHeaders({'Content-Type': 'application/json'}),
      body: JSON.stringify({text}),
    },
    AI_BACKEND_DETECT_TIMEOUT_MS,
  );

  if (!response.ok) {
    throw new Error(`AI backend detect-language failed: ${response.status}`);
  }

  const data = await response.json();
  if (!data.language) {
    throw new Error('AI backend returned no detected language');
  }
  return data.language;
};

type LocalTranslationResult = {
  translation: string;
  confidence: number | null;
};

// Translate via the local AI backend (IndicTrans2). Throws if the backend is
// unreachable or returns an error. Passing verify=true also runs the
// backend's opt-in round-trip confidence check (translates the result back
// to the source language and compares word overlap) so the app can flag
// translations that are likely inaccurate - at the cost of roughly doubling
// how long the request takes.
const translateLocalAI = async (
  text: string,
  sourceLang: string,
  targetLang: string,
  verify: boolean = false,
): Promise<LocalTranslationResult> => {
  const response = await fetchWithTimeout(
    `${getBackendUrl()}/translate`,
    {
      method: 'POST',
      headers: backendHeaders({'Content-Type': 'application/json'}),
      body: JSON.stringify({text, source_lang: sourceLang, target_lang: targetLang, verify}),
    },
    AI_BACKEND_TRANSLATE_TIMEOUT_MS,
  );

  if (!response.ok) {
    throw new Error(`AI backend translate failed: ${response.status}`);
  }

  const data = await response.json();
  if (!data.translation) {
    throw new Error('AI backend returned no translation');
  }
  return {
    translation: data.translation,
    confidence: typeof data.confidence === 'number' ? data.confidence : null,
  };
};

type LocalTranscriptionResult = {
  transcript: string;
  language: string;
};

// Transcribe a recorded audio clip via the local AI backend's Whisper
// endpoint - meaningfully more accurate for Punjabi than Android's on-device
// recognizer, which has inconsistent/weak Punjabi language support.
const transcribeLocalAI = async (
  audioBase64: string,
  language: 'auto' | 'en' | 'pa',
): Promise<LocalTranscriptionResult> => {
  const response = await fetchWithTimeout(
    `${getBackendUrl()}/speech-to-text`,
    {
      method: 'POST',
      headers: backendHeaders({'Content-Type': 'application/json'}),
      body: JSON.stringify({audio: audioBase64, language}),
    },
    AI_BACKEND_SPEECH_TIMEOUT_MS,
  );

  if (!response.ok) {
    throw new Error(`AI backend speech-to-text failed: ${response.status}`);
  }

  const data = await response.json();
  if (typeof data.transcript !== 'string') {
    throw new Error('AI backend returned no transcript');
  }
  return {
    transcript: data.transcript,
    language: data.language,
  };
};

type LocalOcrResult = {
  text: string;
  language: string;
};

// Recognize text from a photo (e.g. a sign/menu/document) via the local AI
// backend's Tesseract OCR endpoint, so it can be fed into translateAndSpeak -
// powers the "scan a photo" translation feature.
const scanPhotoLocalAI = async (
  imageBase64: string,
  language: 'auto' | 'en' | 'pa',
): Promise<LocalOcrResult> => {
  const response = await fetchWithTimeout(
    `${getBackendUrl()}/ocr`,
    {
      method: 'POST',
      headers: backendHeaders({'Content-Type': 'application/json'}),
      body: JSON.stringify({image: imageBase64, language}),
    },
    AI_BACKEND_OCR_TIMEOUT_MS,
  );

  if (!response.ok) {
    throw new Error(`AI backend OCR failed: ${response.status}`);
  }

  const data = await response.json();
  if (typeof data.text !== 'string') {
    throw new Error('AI backend returned no OCR text');
  }
  return {
    text: data.text,
    language: data.language,
  };
};

// Lets a user flag a translation they believe is wrong so it can be reviewed
// later and turned into a GLOSSARY fix or a new test case - the only way to
// learn about real-world accuracy problems beyond the curated test set.
// Best-effort: failures are surfaced to the caller but shouldn't be treated
// as fatal by callers, since this is a "nice to have" not core functionality.
const sendFeedback = async (
  sourceText: string,
  translation: string,
  sourceLang: string,
  targetLang: string,
): Promise<void> => {
  const response = await fetchWithTimeout(
    `${getBackendUrl()}/feedback`,
    {
      method: 'POST',
      headers: backendHeaders({'Content-Type': 'application/json'}),
      body: JSON.stringify({
        source_text: sourceText,
        translation,
        source_lang: sourceLang,
        target_lang: targetLang,
      }),
    },
    AI_BACKEND_DETECT_TIMEOUT_MS,
  );

  if (!response.ok) {
    throw new Error(`AI backend feedback failed: ${response.status}`);
  }
};

type UiLang = 'en' | 'pa';

const LANGUAGE_NAME_MAP: Record<UiLang, Record<string, string>> = {
  en: {
    en: 'English',
    pa: 'Punjabi',
  },
  pa: {
    en: 'ਅੰਗਰੇਜ਼ੀ',
    pa: 'ਪੰਜਾਬੀ',
  },
};

interface UiStrings {
  appTitle: string;
  topBarSubtitle: string;
  openMenu: string;
  menuHeader: string;
  menuHome: string;
  menuSettings: string;
  menuAbout: string;
  ready: string;
  listening: string;
  inputting: string;
  speakNow: string;
  tapAndSpeak: string;
  stopRecording: string;
  translatingBtn: string;
  startVoice: string;
  typePlaceholder: string;
  translateBtn: string;
  typeInstead: string;
  speakInstead: string;
  back: string;
  detectedSuffix: string;
  translationLabel: string;
  listeningPunjabi: string;
  listeningEnglish: string;
  voiceStartFailed: string;
  processingSpeech: string;
  detectingLanguage: string;
  translationErrorTitle: string;
  translationErrorMessage: string;
  serverWakingUpTitle: string;
  serverWakingUpMessage: string;
  speakingPunjabi: string;
  punjabiComplete: string;
  micPermissionTitle: string;
  micPermissionMessage: string;
  askLater: string;
  cancel: string;
  ok: string;
  permissionDeniedTitle: string;
  permissionDeniedMessage: string;
  speechFailedFallback: string;
  sttSectionTitle: string;
  sttHint: string;
  autoAdaptive: string;
  englishOnly: string;
  punjabiOnly: string;
  voiceSelectionTitle: string;
  femalePremium: string;
  malePremium: string;
  testVoiceTitle: string;
  serverSectionTitle: string;
  serverHint: string;
  serverPlaceholder: string;
  apiKeyPlaceholder: string;
  serverSaveButton: string;
  serverTestingStatus: string;
  serverConnectedStatus: string;
  serverUnreachableStatus: string;
  featuresTitle: string;
  feature1: string;
  feature2: string;
  feature3: string;
  feature4: string;
  feature5: string;
  translating: (src: string, tgt: string) => string;
  speaking: (tgt: string) => string;
  speechError: (msg: string) => string;
  translationFailed: (err: string) => string;
  ttsFailed: (err: string) => string;
  lowConfidenceWarning: (score: string) => string;
  flagButton: string;
  flagSent: string;
  flagFailed: string;
  scanPhoto: string;
  scanningPhoto: string;
  choosePhotoSourceTitle: string;
  cameraOption: string;
  galleryOption: string;
  cameraPermissionTitle: string;
  cameraPermissionMessage: string;
  galleryPermissionTitle: string;
  galleryPermissionMessage: string;
  ocrNoTextFound: string;
  ocrFailed: (err: string) => string;
}

const UI_STRINGS: Record<UiLang, UiStrings> = {
  en: {
    appTitle: 'ePunjabi',
    topBarSubtitle: 'AI TRANSLATION ENGINE',
    openMenu: 'Open menu',
    menuHeader: 'Menu',
    menuHome: 'Translate',
    menuSettings: 'Voice Settings & Test',
    menuAbout: 'App Features',
    ready: 'Ready for voice translation',
    listening: 'Listening...',
    inputting: 'Inputting...',
    speakNow: 'Speak now in English or Punjabi...',
    tapAndSpeak: 'SmartDetection: speak English or Punjabi',
    stopRecording: 'Stop Recording',
    translatingBtn: 'Translating...',
    startVoice: 'Start Voice Translation',
    typePlaceholder: 'Type in English or Punjabi...',
    translateBtn: 'Translate',
    typeInstead: 'Type Instead',
    speakInstead: 'Speak Instead',
    back: 'Back',
    detectedSuffix: '(detected)',
    translationLabel: 'Translation',
    listeningPunjabi: 'Listening in Punjabi...',
    listeningEnglish: 'Listening in English...',
    voiceStartFailed: 'Failed to start voice recognition',
    processingSpeech: 'Processing your speech...',
    detectingLanguage: 'Detecting language...',
    translationErrorTitle: 'Translation Error',
    translationErrorMessage: 'Failed to translate. Please try again.',
    serverWakingUpTitle: 'Server Waking Up',
    serverWakingUpMessage: 'The translation server was asleep and is starting up. Please try again in a moment.',
    speakingPunjabi: 'Speaking Punjabi...',
    punjabiComplete: 'Punjabi translation complete!',
    micPermissionTitle: 'Microphone Permission',
    micPermissionMessage: 'This app needs access to your microphone for voice translation',
    askLater: 'Ask Me Later',
    cancel: 'Cancel',
    ok: 'OK',
    permissionDeniedTitle: 'Permission Denied',
    permissionDeniedMessage: 'Microphone permission is required for voice translation',
    speechFailedFallback: 'Speech recognition failed',
    sttSectionTitle: 'Speech Recognition Language',
    sttHint:
      'Android can only listen for one language at a time. "Auto" guesses based on the last detected language — pick a specific one if it keeps mishearing you.',
    autoAdaptive: 'Auto (adaptive)',
    englishOnly: 'English Only',
    punjabiOnly: 'Punjabi Only',
    voiceSelectionTitle: 'Voice Selection',
    femalePremium: 'Female (Premium)',
    malePremium: 'Male (Premium)',
    testVoiceTitle: 'Test Punjabi Voice',
    serverSectionTitle: 'Translation Server Address',
    serverHint:
      'This app needs the AI translation server running on a computer (see ai_backend_server.py). On a real phone, enter that computer\'s LAN IP address and make sure the phone is on the same Wi-Fi network, e.g. http://192.168.1.50:5000',
    serverPlaceholder: 'http://192.168.1.50:5000',
    apiKeyPlaceholder: 'API key (leave blank if server has none)',
    serverSaveButton: 'Save & Test Connection',
    serverTestingStatus: 'Testing connection...',
    serverConnectedStatus: 'Connected to translation server',
    serverUnreachableStatus: 'Could not reach server at this address',
    featuresTitle: 'App Features',
    feature1: 'Voice recognition in English',
    feature2: 'Real-time translation to Punjabi',
    feature3: 'Authentic Punjabi pronunciation',
    feature4: 'Native Android app experience',
    feature5: 'Secure Google Cloud integration',
    translating: (src, tgt) => `Translating ${src} → ${tgt}...`,
    speaking: tgt => `Speaking ${tgt}...`,
    speechError: msg => `Error: ${msg}`,
    translationFailed: err => `Translation failed: ${err}`,
    ttsFailed: err => `TTS failed: ${err}`,
    lowConfidenceWarning: score => `This translation scored low-confidence (${score}) - please double check it.`,
    flagButton: 'Report this translation',
    flagSent: 'Thanks - this translation was flagged for review.',
    flagFailed: 'Could not send feedback, please check your connection.',
    scanPhoto: 'Scan Photo',
    scanningPhoto: 'Reading text from photo...',
    choosePhotoSourceTitle: 'Scan Photo',
    cameraOption: 'Take Photo',
    galleryOption: 'Choose from Gallery',
    cameraPermissionTitle: 'Camera Permission',
    cameraPermissionMessage: 'This app needs access to your camera to scan and translate text from photos',
    galleryPermissionTitle: 'Photo Gallery Permission',
    galleryPermissionMessage: 'This app needs access to your photos to scan and translate text from a picture',
    ocrNoTextFound: 'No text found in the photo. Try again with clearer lighting/focus.',
    ocrFailed: err => `Could not read text from photo: ${err}`,
  },
  pa: {
    appTitle: 'ePunjabi',
    topBarSubtitle: 'AI ਅਨੁਵਾਦ ਇੰਜਿਨ',
    openMenu: 'ਮੀਨੂ ਖੋਲ੍ਹੋ',
    menuHeader: 'ਮੀਨੂ',
    menuHome: 'ਅਨੁਵਾਦ ਕਰੋ',
    menuSettings: 'ਆਵਾਜ਼ ਸੈਟਿੰਗਾਂ ਅਤੇ ਟੈਸਟ',
    menuAbout: 'ਐਪ ਵਿਸ਼ੇਸ਼ਤਾਵਾਂ',
    ready: 'ਵੌਇਸ ਅਨੁਵਾਦ ਲਈ ਤਿਆਰ',
    listening: 'ਸੁਣ ਰਿਹਾ ਹੈ...',
    inputting: 'ਦਰਜ ਹੋ ਰਿਹਾ ਹੈ...',
    speakNow: 'ਹੁਣ ਅੰਗਰੇਜ਼ੀ ਜਾਂ ਪੰਜਾਬੀ ਵਿੱਚ ਬੋਲੋ...',
    tapAndSpeak: 'ਸਮਾਰਟ ਪਛਾਣ: ਅੰਗਰੇਜ਼ੀ ਜਾਂ ਪੰਜਾਬੀ ਬੋਲੋ',
    stopRecording: 'ਰਿਕਾਰਡਿੰਗ ਬੰਦ ਕਰੋ',
    translatingBtn: 'ਅਨੁਵਾਦ ਹੋ ਰਿਹਾ ਹੈ...',
    startVoice: 'ਵੌਇਸ ਅਨੁਵਾਦ ਸ਼ੁਰੂ ਕਰੋ',
    typePlaceholder: 'ਅੰਗਰੇਜ਼ੀ ਜਾਂ ਪੰਜਾਬੀ ਵਿੱਚ ਲਿਖੋ...',
    translateBtn: 'ਅਨੁਵਾਦ ਕਰੋ',
    typeInstead: 'ਟਾਈਪ ਕਰੋ',
    speakInstead: 'ਬੋਲੋ',
    back: 'ਪਿੱਛੇ',
    detectedSuffix: '(ਪਛਾਣਿਆ ਗਿਆ)',
    translationLabel: 'ਅਨੁਵਾਦ',
    listeningPunjabi: 'ਪੰਜਾਬੀ ਵਿੱਚ ਸੁਣ ਰਿਹਾ ਹੈ...',
    listeningEnglish: 'ਅੰਗਰੇਜ਼ੀ ਵਿੱਚ ਸੁਣ ਰਿਹਾ ਹੈ...',
    voiceStartFailed: 'ਵੌਇਸ ਪਛਾਣ ਸ਼ੁਰੂ ਕਰਨ ਵਿੱਚ ਅਸਫਲ',
    processingSpeech: 'ਤੁਹਾਡੀ ਆਵਾਜ਼ ਦੀ ਪ੍ਰਕਿਰਿਆ ਹੋ ਰਹੀ ਹੈ...',
    detectingLanguage: 'ਭਾਸ਼ਾ ਦੀ ਪਛਾਣ ਹੋ ਰਹੀ ਹੈ...',
    translationErrorTitle: 'ਅਨੁਵਾਦ ਗਲਤੀ',
    translationErrorMessage: 'ਅਨੁਵਾਦ ਕਰਨ ਵਿੱਚ ਅਸਫਲ। ਕਿਰਪਾ ਕਰਕੇ ਦੁਬਾਰਾ ਕੋਸ਼ਿਸ਼ ਕਰੋ।',
    serverWakingUpTitle: 'ਸਰਵਰ ਚਾਲੂ ਹੋ ਰਿਹਾ ਹੈ',
    serverWakingUpMessage: 'ਅਨੁਵਾਦ ਸਰਵਰ ਸੁੱਤਾ ਹੋਇਆ ਸੀ ਅਤੇ ਹੁਣ ਚਾਲੂ ਹੋ ਰਿਹਾ ਹੈ। ਕਿਰਪਾ ਕਰਕੇ ਥੋੜ੍ਹੀ ਦੇਰ ਵਿੱਚ ਦੁਬਾਰਾ ਕੋਸ਼ਿਸ਼ ਕਰੋ।',
    speakingPunjabi: 'ਪੰਜਾਬੀ ਬੋਲ ਰਿਹਾ ਹੈ...',
    punjabiComplete: 'ਪੰਜਾਬੀ ਅਨੁਵਾਦ ਪੂਰਾ ਹੋ ਗਿਆ!',
    micPermissionTitle: 'ਮਾਈਕ੍ਰੋਫ਼ੋਨ ਦੀ ਇਜਾਜ਼ਤ',
    micPermissionMessage: 'ਇਸ ਐਪ ਨੂੰ ਵੌਇਸ ਅਨੁਵਾਦ ਲਈ ਤੁਹਾਡੇ ਮਾਈਕ੍ਰੋਫ਼ੋਨ ਦੀ ਲੋੜ ਹੈ',
    askLater: 'ਬਾਅਦ ਵਿੱਚ ਪੁੱਛੋ',
    cancel: 'ਰੱਦ ਕਰੋ',
    ok: 'ਠੀਕ ਹੈ',
    permissionDeniedTitle: 'ਇਜਾਜ਼ਤ ਨਾਮਨਜ਼ੂਰ',
    permissionDeniedMessage: 'ਵੌਇਸ ਅਨੁਵਾਦ ਲਈ ਮਾਈਕ੍ਰੋਫ਼ੋਨ ਦੀ ਇਜਾਜ਼ਤ ਲੋੜੀਂਦੀ ਹੈ',
    speechFailedFallback: 'ਸਪੀਚ ਪਛਾਣ ਅਸਫਲ ਹੋਈ',
    sttSectionTitle: 'ਸਪੀਚ ਪਛਾਣ ਭਾਸ਼ਾ',
    sttHint:
      'ਐਂਡਰਾਇਡ ਇੱਕ ਸਮੇਂ ਵਿੱਚ ਸਿਰਫ਼ ਇੱਕ ਭਾਸ਼ਾ ਹੀ ਸੁਣ ਸਕਦਾ ਹੈ। "ਆਟੋ" ਆਖਰੀ ਪਛਾਣੀ ਗਈ ਭਾਸ਼ਾ ਦੇ ਆਧਾਰ ਤੇ ਅੰਦਾਜ਼ਾ ਲਗਾਉਂਦਾ ਹੈ — ਜੇ ਇਹ ਗਲਤ ਸਮਝਦਾ ਰਹੇ ਤਾਂ ਕੋਈ ਖਾਸ ਭਾਸ਼ਾ ਚੁਣੋ।',
    autoAdaptive: 'ਆਟੋ (ਅਨੁਕੂਲ)',
    englishOnly: 'ਸਿਰਫ਼ ਅੰਗਰੇਜ਼ੀ',
    punjabiOnly: 'ਸਿਰਫ਼ ਪੰਜਾਬੀ',
    voiceSelectionTitle: 'ਆਵਾਜ਼ ਦੀ ਚੋਣ',
    femalePremium: 'ਔਰਤ (ਪ੍ਰੀਮੀਅਮ)',
    malePremium: 'ਮਰਦ (ਪ੍ਰੀਮੀਅਮ)',
    testVoiceTitle: 'ਪੰਜਾਬੀ ਆਵਾਜ਼ ਦੀ ਜਾਂਚ ਕਰੋ',
    serverSectionTitle: 'ਅਨੁਵਾਦ ਸਰਵਰ ਪਤਾ',
    serverHint:
      'ਇਸ ਐਪ ਨੂੰ ਕਿਸੇ ਕੰਪਿਊਟਰ ਤੇ ਚੱਲ ਰਹੇ AI ਅਨੁਵਾਦ ਸਰਵਰ ਦੀ ਲੋੜ ਹੈ (ai_backend_server.py ਦੇਖੋ)। ਅਸਲੀ ਫ਼ੋਨ ਤੇ, ਉਸ ਕੰਪਿਊਟਰ ਦਾ LAN IP ਪਤਾ ਦਰਜ ਕਰੋ ਅਤੇ ਯਕੀਨੀ ਬਣਾਓ ਕਿ ਫ਼ੋਨ ਉਸੇ ਵਾਈ-ਫਾਈ ਨੈੱਟਵਰਕ ਤੇ ਹੈ, ਜਿਵੇਂ http://192.168.1.50:5000',
    serverPlaceholder: 'http://192.168.1.50:5000',
    apiKeyPlaceholder: 'API ਕੁੰਜੀ (ਜੇ ਸਰਵਰ ਤੇ ਕੋਈ ਨਹੀਂ ਹੈ ਤਾਂ ਖਾਲੀ ਛੱਡੋ)',
    serverSaveButton: 'ਸੰਭਾਲੋ ਅਤੇ ਕਨੈਕਸ਼ਨ ਜਾਂਚੋ',
    serverTestingStatus: 'ਕਨੈਕਸ਼ਨ ਦੀ ਜਾਂਚ ਹੋ ਰਹੀ ਹੈ...',
    serverConnectedStatus: 'ਅਨੁਵਾਦ ਸਰਵਰ ਨਾਲ ਕਨੈਕਟ ਹੋ ਗਿਆ',
    serverUnreachableStatus: 'ਇਸ ਪਤੇ ਤੇ ਸਰਵਰ ਤੱਕ ਪਹੁੰਚ ਨਹੀਂ ਹੋ ਸਕੀ',
    featuresTitle: 'ਐਪ ਵਿਸ਼ੇਸ਼ਤਾਵਾਂ',
    feature1: 'ਅੰਗਰੇਜ਼ੀ ਵਿੱਚ ਆਵਾਜ਼ ਦੀ ਪਛਾਣ',
    feature2: 'ਪੰਜਾਬੀ ਵਿੱਚ ਰੀਅਲ-ਟਾਈਮ ਅਨੁਵਾਦ',
    feature3: 'ਅਸਲ ਪੰਜਾਬੀ ਉਚਾਰਨ',
    feature4: 'ਨੇਟਿਵ ਐਂਡਰਾਇਡ ਐਪ ਅਨੁਭਵ',
    feature5: 'ਸੁਰੱਖਿਅਤ ਗੂਗਲ ਕਲਾਉਡ ਏਕੀਕਰਨ',
    translating: (src, tgt) => `${src} ਤੋਂ ${tgt} ਵਿੱਚ ਅਨੁਵਾਦ ਹੋ ਰਿਹਾ ਹੈ...`,
    speaking: tgt => `${tgt} ਬੋਲ ਰਿਹਾ ਹੈ...`,
    speechError: msg => `ਗਲਤੀ: ${msg}`,
    translationFailed: err => `ਅਨੁਵਾਦ ਅਸਫਲ: ${err}`,
    ttsFailed: err => `TTS ਅਸਫਲ: ${err}`,
    lowConfidenceWarning: score => `ਇਸ ਅਨੁਵਾਦ ਦਾ ਭਰੋਸਾ ਘੱਟ ਹੈ (${score}) - ਕਿਰਪਾ ਕਰਕੇ ਦੁਬਾਰਾ ਜਾਂਚ ਕਰੋ।`,
    flagButton: 'ਇਸ ਅਨੁਵਾਦ ਦੀ ਰਿਪੋਰਟ ਕਰੋ',
    flagSent: 'ਧੰਨਵਾਦ - ਇਹ ਅਨੁਵਾਦ ਸਮੀਖਿਆ ਲਈ ਭੇਜ ਦਿੱਤਾ ਗਿਆ ਹੈ।',
    flagFailed: 'ਫੀਡਬੈਕ ਭੇਜੀ ਨਹੀਂ ਜਾ ਸਕੀ, ਕਿਰਪਾ ਕਰਕੇ ਆਪਣਾ ਕਨੈਕਸ਼ਨ ਜਾਂਚੋ।',
    scanPhoto: 'ਫੋਟੋ ਸਕੈਨ ਕਰੋ',
    scanningPhoto: 'ਫੋਟੋ ਤੋਂ ਲਿਖਤ ਪੜ੍ਹੀ ਜਾ ਰਹੀ ਹੈ...',
    choosePhotoSourceTitle: 'ਫੋਟੋ ਸਕੈਨ ਕਰੋ',
    cameraOption: 'ਫੋਟੋ ਖਿੱਚੋ',
    galleryOption: 'ਗੈਲਰੀ ਤੋਂ ਚੁਣੋ',
    cameraPermissionTitle: 'ਕੈਮਰਾ ਇਜਾਜ਼ਤ',
    cameraPermissionMessage: 'ਇਸ ਐਪ ਨੂੰ ਫੋਟੋਆਂ ਤੋਂ ਲਿਖਤ ਸਕੈਨ ਅਤੇ ਅਨੁਵਾਦ ਕਰਨ ਲਈ ਤੁਹਾਡੇ ਕੈਮਰੇ ਦੀ ਲੋੜ ਹੈ',
    galleryPermissionTitle: 'ਫੋਟੋ ਗੈਲਰੀ ਇਜਾਜ਼ਤ',
    galleryPermissionMessage: 'ਇਸ ਐਪ ਨੂੰ ਤਸਵੀਰ ਤੋਂ ਲਿਖਤ ਸਕੈਨ ਅਤੇ ਅਨੁਵਾਦ ਕਰਨ ਲਈ ਤੁਹਾਡੀਆਂ ਫੋਟੋਆਂ ਤੱਕ ਪਹੁੰਚ ਦੀ ਲੋੜ ਹੈ',
    ocrNoTextFound: 'ਫੋਟੋ ਵਿੱਚ ਕੋਈ ਲਿਖਤ ਨਹੀਂ ਮਿਲੀ। ਸਾਫ਼ ਰੌਸ਼ਨੀ/ਫੋਕਸ ਨਾਲ ਦੁਬਾਰਾ ਕੋਸ਼ਿਸ਼ ਕਰੋ।',
    ocrFailed: err => `ਫੋਟੋ ਤੋਂ ਲਿਖਤ ਨਹੀਂ ਪੜ੍ਹੀ ਜਾ ਸਕੀ: ${err}`,
  },
};

// Feature icons shown on the About screen, parallel to feature1..feature5.
const FEATURE_ICONS = ['mic', 'translate', 'record-voice-over', 'smartphone', 'lock'];

// ---- Theme (light/dark) -----------------------------------------------
// A small token palette instead of hardcoded hex values throughout the
// stylesheet, so the whole UI can flip between a clean light look and a
// deep, glow-accented dark/"futuristic AI" look via one flag.
interface ThemeColors {
  background: string;
  surface: string;
  surfaceAlt: string;
  border: string;
  textPrimary: string;
  textSecondary: string;
  accent: string;
  accentSoft: string;
  success: string;
  successSoft: string;
  danger: string;
  dangerSoft: string;
  warning: string;
  overlay: string;
}

const lightTheme: ThemeColors = {
  background: '#F1F3FB',
  surface: '#FFFFFF',
  surfaceAlt: '#F5F6FC',
  border: '#E3E6F3',
  textPrimary: '#14172B',
  textSecondary: '#666C85',
  accent: '#4285F4',
  accentSoft: '#E8F0FE',
  success: '#0FA968',
  successSoft: '#E7F9F1',
  danger: '#E5484D',
  dangerSoft: '#FCEAEB',
  warning: '#DC8A00',
  overlay: 'rgba(12,14,26,0.45)',
};

const darkTheme: ThemeColors = {
  background: '#080A14',
  surface: '#131629',
  surfaceAlt: '#191D34',
  border: '#272C48',
  textPrimary: '#EEF0FA',
  textSecondary: '#9298B8',
  accent: '#5B9DFF',
  accentSoft: '#1C3559',
  success: '#2FD98A',
  successSoft: '#123527',
  danger: '#FF6B70',
  dangerSoft: '#3A1E22',
  warning: '#FFC15E',
  overlay: 'rgba(0,0,0,0.65)',
};

const DARK_MODE_STORAGE_KEY = 'ui_dark_mode';

const TranslationAppMobile = (): JSX.Element => {
  const [inputText, setInputText] = useState('');
  const [translatedText, setTranslatedText] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [isListening, setIsListening] = useState(false);
  const [isProcessingSpeech, setIsProcessingSpeech] = useState(false);
  const [status, setStatus] = useState('Ready for voice translation');
  const [selectedVoice, setSelectedVoice] = useState('pa-IN-Wavenet-A');
  const [inputMode, setInputMode] = useState<'voice' | 'text'>('voice');
  const [typedText, setTypedText] = useState('');
  const [inputLanguage, setInputLanguage] = useState('en');
  const [outputLanguage, setOutputLanguage] = useState('pa');
  const [sttLanguageMode, setSttLanguageMode] = useState<'auto' | 'en-US' | 'pa-IN'>('auto');
  const [menuVisible, setMenuVisible] = useState(false);
  const [activeScreen, setActiveScreen] = useState<'home' | 'settings' | 'about'>('home');
  const [uiLanguage, setUiLanguage] = useState<UiLang>('en');
  const [serverUrlDraft, setServerUrlDraft] = useState(DEFAULT_BACKEND_URL);
  const [apiKeyDraft, setApiKeyDraft] = useState('');
  const [serverStatus, setServerStatus] = useState<'idle' | 'testing' | 'connected' | 'unreachable'>('idle');
  const [feedbackState, setFeedbackState] = useState<'idle' | 'sending' | 'sent'>('idle');
  const [isScanningPhoto, setIsScanningPhoto] = useState(false);
  const [isDarkMode, setIsDarkMode] = useState(false);
  const uiLanguageRef = useRef<UiLang>('en');

  const theme = isDarkMode ? darkTheme : lightTheme;
  const styles = useMemo(() => createStyles(theme), [theme]);

  const toggleDarkMode = () => {
    setIsDarkMode(prev => {
      const next = !prev;
      AsyncStorage.setItem(DARK_MODE_STORAGE_KEY, next ? '1' : '0').catch(() => {
        // Non-fatal: preference just won't persist across app restarts.
      });
      return next;
    });
  };

  useEffect(() => {
    uiLanguageRef.current = uiLanguage;
  }, [uiLanguage]);

  useEffect(() => {
    AsyncStorage.getItem(DARK_MODE_STORAGE_KEY)
      .then(stored => {
        if (stored) {
          setIsDarkMode(stored === '1');
        }
      })
      .catch(() => {
        // Fall back to the light-theme default set above.
      });
  }, []);

  const LANGUAGE_NAMES = LANGUAGE_NAME_MAP[uiLanguage];
  const T = UI_STRINGS[uiLanguage];

  const toggleUiLanguage = () => {
    setUiLanguage(prev => {
      const next: UiLang = prev === 'en' ? 'pa' : 'en';
      if (!isLoading && !isListening && !translatedText) {
        setStatus(UI_STRINGS[next].ready);
      }
      return next;
    });
  };

  const menuItems: {key: typeof activeScreen; label: string; icon: string}[] = [
    {key: 'home', label: T.menuHome, icon: 'translate'},
    {key: 'settings', label: T.menuSettings, icon: 'tune'},
    {key: 'about', label: T.menuAbout, icon: 'auto-awesome'},
  ];

  const navigateTo = (screen: typeof activeScreen) => {
    setActiveScreen(screen);
    setMenuVisible(false);
  };

  useEffect(() => {
    requestMicrophonePermission();
    initializeVoice();
    initializeTts();

    loadPersistedBackendUrl().then(async url => {
      setServerUrlDraft(url);
      setServerStatus('testing');
      const reachable = await isBackendReachable();
      setServerStatus(reachable ? 'connected' : 'unreachable');
    });
    loadPersistedApiKey().then(setApiKeyDraft);

    return () => {
      Voice.destroy().then(Voice.removeAllListeners);
      Tts.removeAllListeners('tts-start');
      Tts.removeAllListeners('tts-finish');
      Tts.removeAllListeners('tts-cancel');
    };
  }, []);

  const handleSaveServerUrl = async () => {
    const trimmed = serverUrlDraft.trim().replace(/\/+$/, '');
    if (!trimmed) {
      return;
    }
    setServerUrlDraft(trimmed);
    await persistBackendUrl(trimmed);
    await persistApiKey(apiKeyDraft.trim());
    setServerStatus('testing');
    const reachable = await isBackendReachable();
    setServerStatus(reachable ? 'connected' : 'unreachable');
  };

  const initializeTts = () => {
    Tts.setDefaultLanguage('pa-IN').catch(() => {
      // Falls back to the device default if pa-IN voice data isn't installed
    });
    Tts.addEventListener('tts-start', () => setStatus(UI_STRINGS[uiLanguageRef.current].speakingPunjabi));
    Tts.addEventListener('tts-finish', () => setStatus(UI_STRINGS[uiLanguageRef.current].punjabiComplete));
    Tts.addEventListener('tts-cancel', () => setStatus(UI_STRINGS[uiLanguageRef.current].ready));
    // TEMP DIAGNOSTIC: log all installed voices so we can see which Punjabi
    // voice variants are available on this device/emulator.
    Tts.getInitStatus()
      .then(() => Tts.voices())
      .then((v: any) => console.log('TTS_VOICES_DEBUG', JSON.stringify(v)))
      .catch((e: any) => console.log('TTS_VOICES_DEBUG_ERROR', e));
  };

  const requestMicrophonePermission = async () => {
    if (Platform.OS === 'android') {
      try {
        const strings = UI_STRINGS[uiLanguageRef.current];
        const granted = await PermissionsAndroid.request(
          PermissionsAndroid.PERMISSIONS.RECORD_AUDIO,
          {
            title: strings.micPermissionTitle,
            message: strings.micPermissionMessage,
            buttonNeutral: strings.askLater,
            buttonNegative: strings.cancel,
            buttonPositive: strings.ok,
          },
        );
        if (granted !== PermissionsAndroid.RESULTS.GRANTED) {
          Alert.alert(strings.permissionDeniedTitle, strings.permissionDeniedMessage);
        }
      } catch (err) {
        console.warn(err);
      }
    }
  };

  // Requested lazily (only when the user taps "Scan Photo"), unlike the
  // microphone permission above which is requested eagerly on launch -
  // camera use is occasional/optional, so there's no need to prompt upfront.
  const requestCameraPermission = async (): Promise<boolean> => {
    if (Platform.OS !== 'android') {
      return true;
    }
    try {
      const strings = UI_STRINGS[uiLanguageRef.current];
      const granted = await PermissionsAndroid.request(
        PermissionsAndroid.PERMISSIONS.CAMERA,
        {
          title: strings.cameraPermissionTitle,
          message: strings.cameraPermissionMessage,
          buttonNeutral: strings.askLater,
          buttonNegative: strings.cancel,
          buttonPositive: strings.ok,
        },
      );
      if (granted !== PermissionsAndroid.RESULTS.GRANTED) {
        Alert.alert(strings.permissionDeniedTitle, strings.permissionDeniedMessage);
        return false;
      }
      return true;
    } catch (err) {
      console.warn(err);
      return false;
    }
  };

  // Requested lazily (only when the user picks "Gallery" from the Scan Photo
  // menu). Android 13+ uses READ_MEDIA_IMAGES; older versions use
  // READ_EXTERNAL_STORAGE (declared with maxSdkVersion=32 in the manifest).
  const requestGalleryPermission = async (): Promise<boolean> => {
    if (Platform.OS !== 'android') {
      return true;
    }
    try {
      const strings = UI_STRINGS[uiLanguageRef.current];
      const permission =
        Platform.Version >= 33
          ? PermissionsAndroid.PERMISSIONS.READ_MEDIA_IMAGES
          : PermissionsAndroid.PERMISSIONS.READ_EXTERNAL_STORAGE;
      const granted = await PermissionsAndroid.request(permission, {
        title: strings.galleryPermissionTitle,
        message: strings.galleryPermissionMessage,
        buttonNeutral: strings.askLater,
        buttonNegative: strings.cancel,
        buttonPositive: strings.ok,
      });
      if (granted !== PermissionsAndroid.RESULTS.GRANTED) {
        Alert.alert(strings.permissionDeniedTitle, strings.permissionDeniedMessage);
        return false;
      }
      return true;
    } catch (err) {
      console.warn(err);
      return false;
    }
  };

  const initializeVoice = () => {
    Voice.onSpeechStart = () => {
      setStatus(UI_STRINGS[uiLanguageRef.current].listening);
    };

    Voice.onSpeechEnd = () => {
      // Auto-detected end of speech: recording has already stopped, so the
      // button should no longer say "Stop Recording" while we transcribe.
      setIsListening(false);
      setIsProcessingSpeech(true);
      setStatus(UI_STRINGS[uiLanguageRef.current].inputting);
    };

    Voice.onSpeechResults = (e) => {
      console.log('SPEECH_RESULTS_DEBUG', JSON.stringify(e));
      setIsProcessingSpeech(false);
      if (e.value && e.value[0]) {
        const spokenText = e.value[0];
        setInputText(spokenText);
        translateAndSpeak(spokenText);
      } else {
        setStatus(UI_STRINGS[uiLanguageRef.current].speechError('no speech recognized'));
      }
    };

    Voice.onSpeechError = (e) => {
      const strings = UI_STRINGS[uiLanguageRef.current];
      setStatus(strings.speechError(e.error?.message || strings.speechFailedFallback));
      setIsListening(false);
      setIsProcessingSpeech(false);
    };
  };

  // Fallback path: Android's on-device speech recognizer. Used only when the
  // local AI backend isn't reachable (see startVoiceRecognition below).
  const startOnDeviceVoiceRecognition = async () => {
    try {
      setIsListening(true);
      setIsProcessingSpeech(false);

      // Adaptive locale: when set to Auto, use whichever language was
      // detected most recently (defaults to English on first use). This
      // improves recognition accuracy since Android's on-device recognizer
      // only supports one locale per session.
      const locale =
        sttLanguageMode === 'auto'
          ? inputLanguage === 'pa'
            ? 'pa-IN'
            : 'en-US'
          : sttLanguageMode;

      setStatus(
        locale === 'pa-IN'
          ? T.listeningPunjabi
          : T.listeningEnglish,
      );
      // NOTE: passing EXTRA_SPEECH_INPUT_COMPLETE_SILENCE_LENGTH_MILLIS /
      // EXTRA_SPEECH_INPUT_POSSIBLY_COMPLETE_SILENCE_LENGTH_MILLIS with a
      // large value (e.g. 5000ms) causes this device's recognizer to create
      // a second, empty "final" segment after the real one, which then
      // overwrites the real transcript before onResults() fires (results
      // come back empty). Use the engine's own defaults instead.
      await Voice.start(locale);
    } catch (error) {
      console.error('Voice start error:', error);
      setStatus(T.voiceStartFailed);
      setIsListening(false);
    }
  };

  const stopOnDeviceVoiceRecognition = async () => {
    try {
      await Voice.stop();
      setIsListening(false);
      setStatus(T.processingSpeech);
    } catch (error) {
      console.error('Voice stop error:', error);
      setIsListening(false);
    }
  };

  // Which engine is handling the recording currently in progress - decided
  // fresh each time recording starts (see startVoiceRecognition).
  const recordingModeRef = useRef<'backend' | 'ondevice'>('ondevice');

  // Primary path: record raw audio and transcribe it via the local AI
  // backend's Whisper model, which is meaningfully more accurate for Punjabi
  // than Android's built-in recognizer. Falls back to on-device recognition
  // if the backend isn't reachable, decided upfront via a quick health check
  // so we never end up having recorded audio we can't send anywhere.
  const startVoiceRecognition = async () => {
    setInputText('');
    setTranslatedText('');
    setIsProcessingSpeech(false);

    const backendUp = await isBackendReachable();
    if (!backendUp) {
      recordingModeRef.current = 'ondevice';
      await startOnDeviceVoiceRecognition();
      return;
    }

    try {
      recordingModeRef.current = 'backend';
      setIsListening(true);
      setStatus(T.listening);
      await audioRecorderPlayer.startRecorder(VOICE_RECORDING_PATH);
    } catch (error) {
      console.error('Backend audio recording failed to start, falling back to on-device voice:', error);
      recordingModeRef.current = 'ondevice';
      setIsListening(false);
      await startOnDeviceVoiceRecognition();
    }
  };

  const stopVoiceRecognition = async () => {
    if (recordingModeRef.current === 'ondevice') {
      await stopOnDeviceVoiceRecognition();
      return;
    }

    try {
      await audioRecorderPlayer.stopRecorder();
      setIsListening(false);
      setIsProcessingSpeech(true);
      setStatus(T.processingSpeech);

      const audioBase64 = await RNFS.readFile(VOICE_RECORDING_PATH, 'base64');

      // Whisper's own language auto-detection is unreliable for Punjabi
      // (often confuses it with Hindi/Urdu), so never send a bare 'auto' -
      // always resolve to a concrete guess, same adaptive heuristic the
      // on-device path uses (last detected language) when set to Auto.
      const languageHint: 'en' | 'pa' =
        sttLanguageMode === 'pa-IN'
          ? 'pa'
          : sttLanguageMode === 'en-US'
            ? 'en'
            : inputLanguage === 'pa'
              ? 'pa'
              : 'en';

      const {transcript} = await transcribeLocalAI(audioBase64, languageHint);
      setIsProcessingSpeech(false);

      if (transcript.trim()) {
        setInputText(transcript);
        translateAndSpeak(transcript);
      } else {
        setStatus(T.speechError('no speech recognized'));
      }
    } catch (error) {
      console.error('Backend transcription failed:', error);
      setIsProcessingSpeech(false);
      setStatus(T.speechError(String(error)));
    }
  };

  const toggleInputMode = () => {
    setInputMode(prev => (prev === 'voice' ? 'text' : 'voice'));
    setInputText('');
    setTranslatedText('');
    setTypedText('');
    setStatus(T.ready);
  };

  const submitTypedText = () => {
    const text = typedText.trim();
    if (!text) {
      return;
    }
    setInputText(text);
    translateAndSpeak(text);
  };

  const resetToButtons = () => {
    setInputText('');
    setTranslatedText('');
    setTypedText('');
    setFeedbackState('idle');
    setStatus(T.ready);
  };

  // Runs OCR on a captured/picked photo, then feeds the recognized text into
  // the same translateAndSpeak flow used by voice/typed input.
  const handlePickedPhoto = async (asset: Asset | undefined) => {
    if (!asset?.base64) {
      return;
    }
    setInputText('');
    setTranslatedText('');
    setIsScanningPhoto(true);
    setStatus(T.scanningPhoto);
    try {
      const {text} = await scanPhotoLocalAI(asset.base64, 'auto');
      const recognizedText = text.trim();
      if (!recognizedText) {
        setStatus(T.ocrNoTextFound);
        return;
      }
      setIsScanningPhoto(false);
      translateAndSpeak(recognizedText);
    } catch (error) {
      console.error('OCR error:', error);
      setStatus(T.ocrFailed(String(error)));
    } finally {
      setIsScanningPhoto(false);
    }
  };

  const scanPhotoFromCamera = async () => {
    const granted = await requestCameraPermission();
    if (!granted) {
      return;
    }
    const response = await launchCamera({mediaType: 'photo', includeBase64: true, quality: 0.8});
    if (response.didCancel || response.errorCode) {
      return;
    }
    await handlePickedPhoto(response.assets?.[0]);
  };

  const scanPhotoFromGallery = async () => {
    const granted = await requestGalleryPermission();
    if (!granted) {
      return;
    }
    const response = await launchImageLibrary({mediaType: 'photo', includeBase64: true, quality: 0.8});
    if (response.didCancel || response.errorCode) {
      return;
    }
    await handlePickedPhoto(response.assets?.[0]);
  };

  const startPhotoScan = () => {
    Alert.alert(T.choosePhotoSourceTitle, undefined, [
      {text: T.cameraOption, onPress: () => scanPhotoFromCamera()},
      {text: T.galleryOption, onPress: () => scanPhotoFromGallery()},
      {text: T.cancel, style: 'cancel'},
    ]);
  };

  const handleFlagTranslation = async () => {
    setFeedbackState('sending');
    try {
      await sendFeedback(inputText, translatedText, inputLanguage, outputLanguage);
      setFeedbackState('sent');
    } catch (error) {
      console.error('Feedback error:', error);
      setFeedbackState('idle');
      Alert.alert(T.flagFailed);
    }
  };

  const translateAndSpeak = async (text: string) => {
    setIsLoading(true);
    setFeedbackState('idle');
    setStatus(T.detectingLanguage);

    try {
      // Auto-detect whether the input is English or Punjabi, then translate
      // into the other language automatically, entirely via the local AI
      // backend (IndicTrans2). No Google Translate fallback - the backend
      // must be running and reachable for this to work.
      let sourceLang = await detectLanguageLocalAI(text);
      // Normalize to just 'en'/'pa' - the detector can return other
      // language codes (e.g. 'hi') which this app only handles as English.
      sourceLang = sourceLang === 'pa' ? 'pa' : 'en';
      const targetLang = sourceLang === 'pa' ? 'en' : 'pa';
      setInputLanguage(sourceLang);
      setOutputLanguage(targetLang);

      setStatus(T.translating(LANGUAGE_NAMES[sourceLang], LANGUAGE_NAMES[targetLang]));

      // verify=true asks the backend to also translate the result back to
      // the source language and score word overlap, so we can flag
      // likely-inaccurate translations to the user. Costs roughly double
      // the translation time (see AI_BACKEND_TRANSLATE_TIMEOUT_MS above).
      const {translation: translated, confidence} = await translateLocalAI(
        text,
        sourceLang,
        targetLang,
        true,
      );

      setTranslatedText(translated);
      setStatus(T.speaking(LANGUAGE_NAMES[targetLang]));

      // Speak using on-device TTS in the correct target language
      await speakTranslation(translated, targetLang);

      // Flag translations the round-trip check scored as likely inaccurate,
      // without blocking the UI or overwriting the translated text itself.
      if (confidence !== null && confidence < 0.5) {
        console.log(`Low-confidence translation (score: ${confidence}):`, text, '->', translated);
        setStatus(T.lowConfidenceWarning(String(confidence)));
      }

    } catch (error: any) {
      console.error('Translation error:', error);
      setStatus(T.translationFailed(String(error)));
      // A timed-out fetch (AbortError) almost always means the backend was
      // asleep and still starting up, not a real failure - tell the user to
      // just retry shortly instead of showing a generic error.
      if (error?.name === 'AbortError') {
        Alert.alert(T.serverWakingUpTitle, T.serverWakingUpMessage);
      } else {
        Alert.alert(T.translationErrorTitle, T.translationErrorMessage);
      }
    } finally {
      setIsLoading(false);
    }
  };

  const speakTranslation = async (text: string, languageCode: string) => {
    try {
      const ttsLocale = languageCode === 'pa' ? 'pa-IN' : 'en-US';
      await Tts.setDefaultLanguage(ttsLocale);
      // TEMP DIAGNOSTIC: this embedded/neural Punjabi voice
      // (pa-in-x-pac-lstm-embedded) may not tolerate pitch/rate shifting on
      // this emulator, so use the engine's natural defaults for now.
      await Tts.setDefaultPitch(1.0);
      await Tts.setDefaultRate(1.0, true);
      console.log('TTS_SPEAK_DEBUG', ttsLocale, text);
      const result = await Tts.speak(text);
      console.log('TTS_SPEAK_RESULT', JSON.stringify(result));
    } catch (error) {
      console.error('TTS error:', error);
      setStatus(T.ttsFailed(String(error)));
    }
  };

  const testPhrases = [
    {punjabi: 'ਸਤ ਸ੍ਰੀ ਅਕਾਲ', english: 'Hello'},
    {punjabi: 'ਤੁਸੀ ਕਿਵੇਂ ਹੋ?', english: 'How are you?'},
    {punjabi: 'ਧੰਨਵਾਦ ਜੀ', english: 'Thank you'},
    {punjabi: 'ਮੈਂ ਠੀਕ ਹਾਂ', english: 'I am fine'},
  ];

  return (
    <SafeAreaView style={styles.container}>
      <StatusBar
        barStyle={isDarkMode ? 'light-content' : 'dark-content'}
        backgroundColor={theme.surface}
      />
      <View style={styles.topBar}>
        <TouchableOpacity
          style={styles.menuButton}
          onPress={() => setMenuVisible(true)}
          accessibilityLabel={T.openMenu}>
          <Icon name="menu" size={26} color={theme.accent} />
        </TouchableOpacity>
        <View style={styles.topBarTitleWrap} pointerEvents="none">
          <Icon name="graphic-eq" size={16} color={theme.accent} style={styles.topBarTitleIcon} />
          <View style={styles.topBarTitleColumn}>
            <Text style={styles.topBarTitle}>{T.appTitle}</Text>
            <Text style={styles.topBarSubtitle}>{T.topBarSubtitle}</Text>
          </View>
        </View>
        <View style={styles.topBarActions}>
          <TouchableOpacity style={styles.themeToggleButton} onPress={toggleDarkMode}>
            <Icon name={isDarkMode ? 'light-mode' : 'dark-mode'} size={18} color={theme.accent} />
          </TouchableOpacity>
          <TouchableOpacity style={styles.langToggleButton} onPress={toggleUiLanguage}>
            <Text style={styles.langToggleText}>{uiLanguage === 'en' ? 'ਪੰ' : 'EN'}</Text>
          </TouchableOpacity>
        </View>
        <View style={styles.topBarAccentLine} />
      </View>

      <Modal
        visible={menuVisible}
        animationType="fade"
        transparent
        onRequestClose={() => setMenuVisible(false)}>
        <TouchableOpacity
          style={styles.menuOverlay}
          activeOpacity={1}
          onPress={() => setMenuVisible(false)}>
          <View style={styles.menuDrawer}>
            <Text style={styles.menuHeader}>{T.menuHeader}</Text>
            {menuItems.map(item => (
              <TouchableOpacity
                key={item.key}
                style={[styles.menuItem, activeScreen === item.key && styles.menuItemActive]}
                onPress={() => navigateTo(item.key)}>
                <Icon
                  name={item.icon}
                  size={20}
                  color={activeScreen === item.key ? theme.accent : theme.textSecondary}
                  style={styles.menuItemIcon}
                />
                <Text
                  style={[
                    styles.menuItemText,
                    activeScreen === item.key && styles.menuItemTextActive,
                  ]}>
                  {item.label}
                </Text>
              </TouchableOpacity>
            ))}
          </View>
        </TouchableOpacity>
      </Modal>

      <ScrollView contentContainerStyle={styles.scrollContent}>
        {activeScreen === 'home' && (
          <>
            {!translatedText && (
              <View style={styles.voiceSection}>
                {inputMode === 'voice' ? (
                  <>
                    <View style={styles.voiceIconRing}>
                      <TouchableOpacity
                        style={[styles.voiceIconButton, isListening && styles.voiceIconButtonActive]}
                        onPress={isListening ? stopVoiceRecognition : startVoiceRecognition}
                        disabled={isLoading || isProcessingSpeech}>
                        {isLoading || isProcessingSpeech ? (
                          <ActivityIndicator color="#ffffff" size="large" />
                        ) : (
                          <Icon name={isListening ? 'mic' : 'mic-none'} size={48} color="#ffffff" />
                        )}
                      </TouchableOpacity>
                    </View>

                    <Text style={styles.voiceHint}>
                      {isListening
                        ? T.speakNow
                        : T.tapAndSpeak}
                    </Text>
                  </>
                ) : (
                  <>
                    <TextInput
                      style={styles.typedInput}
                      placeholder={T.typePlaceholder}
                      placeholderTextColor={theme.textSecondary}
                      value={typedText}
                      onChangeText={setTypedText}
                      editable={!isLoading}
                      multiline
                    />
                    <TouchableOpacity
                      style={[styles.voiceButton, isLoading && styles.voiceButtonActive]}
                      onPress={submitTypedText}
                      disabled={isLoading || !typedText.trim()}>
                      <Icon name="translate" size={18} color="#ffffff" style={styles.iconInline} />
                      <Text style={styles.voiceButtonText}>
                        {isLoading ? T.translatingBtn : T.translateBtn}
                      </Text>
                    </TouchableOpacity>
                  </>
                )}

                <TouchableOpacity style={styles.modeToggle} onPress={toggleInputMode}>
                  <Icon
                    name={inputMode === 'voice' ? 'keyboard' : 'mic'}
                    size={16}
                    color={theme.accent}
                    style={styles.iconInline}
                  />
                  <Text style={styles.modeToggleText}>
                    {inputMode === 'voice' ? T.typeInstead : T.speakInstead}
                  </Text>
                </TouchableOpacity>

                <TouchableOpacity
                  style={styles.modeToggle}
                  onPress={startPhotoScan}
                  disabled={isLoading || isListening || isProcessingSpeech || isScanningPhoto}>
                  <Icon name="photo-camera" size={16} color={theme.accent} style={styles.iconInline} />
                  <Text style={styles.modeToggleText}>{T.scanPhoto}</Text>
                </TouchableOpacity>
              </View>
            )}

            {!!translatedText && (
              <View style={styles.backButtonWrapper}>
                <TouchableOpacity style={styles.backButton} onPress={resetToButtons}>
                  <Icon name="arrow-back" size={16} color={theme.accent} style={styles.iconInline} />
                  <Text style={styles.backButtonText}>{T.back}</Text>
                </TouchableOpacity>
              </View>
            )}

            <View style={styles.statusCard}>
              <View
                style={[
                  styles.statusDot,
                  {
                    backgroundColor: isListening
                      ? theme.danger
                      : isLoading || isProcessingSpeech || isScanningPhoto
                        ? theme.warning
                        : theme.success,
                  },
                ]}
              />
              <Text style={styles.statusText}>{status}</Text>
              {(isLoading || isScanningPhoto) && (
                <ActivityIndicator size="small" color={theme.accent} style={styles.loader} />
              )}
            </View>

            {!!translatedText && (
              <>
                <View style={styles.translationSection}>
                  <View style={styles.textBox}>
                    <View style={styles.textBoxTitleRow}>
                      <Icon name="record-voice-over" size={16} color={theme.accent} style={styles.iconInline} />
                      <Text style={styles.textBoxTitle}>
                        {LANGUAGE_NAMES[inputLanguage]} {T.detectedSuffix}
                      </Text>
                    </View>
                    <Text style={styles.textContent}>{inputText}</Text>
                  </View>

                  <View style={styles.textBox}>
                    <View style={styles.textBoxTitleRow}>
                      <Icon name="translate" size={16} color={theme.accent} style={styles.iconInline} />
                      <Text style={styles.textBoxTitle}>
                        {LANGUAGE_NAMES[outputLanguage]} {T.translationLabel}
                      </Text>
                    </View>
                    <Text style={[styles.textContent, styles.punjabiText]}>{translatedText}</Text>
                  </View>
                </View>

                <TouchableOpacity
                  style={styles.flagButton}
                  onPress={handleFlagTranslation}
                  disabled={feedbackState !== 'idle'}
                  accessibilityLabel={T.flagButton}>
                  <Icon name="flag" size={14} color={theme.danger} style={styles.iconInline} />
                  <Text style={styles.flagButtonText}>
                    {feedbackState === 'sent' ? T.flagSent : feedbackState === 'sending' ? '...' : T.flagButton}
                  </Text>
                </TouchableOpacity>
              </>
            )}
          </>
        )}

        {activeScreen === 'settings' && (
          <>
            <View style={styles.voiceSelector}>
              <View style={styles.sectionTitleRow}>
                <Icon name="dns" size={16} color={theme.textSecondary} style={styles.iconInline} />
                <Text style={styles.sectionTitle}>{T.serverSectionTitle}</Text>
              </View>
              <Text style={styles.settingHint}>{T.serverHint}</Text>
              <TextInput
                style={styles.serverInput}
                placeholder={T.serverPlaceholder}
                placeholderTextColor={theme.textSecondary}
                value={serverUrlDraft}
                onChangeText={setServerUrlDraft}
                autoCapitalize="none"
                autoCorrect={false}
                keyboardType="url"
              />
              <TextInput
                style={styles.serverInput}
                placeholder={T.apiKeyPlaceholder}
                placeholderTextColor={theme.textSecondary}
                value={apiKeyDraft}
                onChangeText={setApiKeyDraft}
                autoCapitalize="none"
                autoCorrect={false}
                secureTextEntry
              />
              <TouchableOpacity style={styles.serverSaveButton} onPress={handleSaveServerUrl}>
                <Icon name="save" size={16} color="#ffffff" style={styles.iconInline} />
                <Text style={styles.serverSaveButtonText}>{T.serverSaveButton}</Text>
              </TouchableOpacity>
              {serverStatus !== 'idle' && (
                <View style={styles.serverStatusRow}>
                  <Icon
                    name={
                      serverStatus === 'testing'
                        ? 'sync'
                        : serverStatus === 'connected'
                          ? 'check-circle'
                          : 'error-outline'
                    }
                    size={14}
                    color={
                      serverStatus === 'connected'
                        ? theme.success
                        : serverStatus === 'unreachable'
                          ? theme.danger
                          : theme.textSecondary
                    }
                    style={styles.iconInline}
                  />
                  <Text
                    style={[
                      styles.serverStatusText,
                      serverStatus === 'connected' && styles.serverStatusConnected,
                      serverStatus === 'unreachable' && styles.serverStatusUnreachable,
                    ]}>
                    {serverStatus === 'testing'
                      ? T.serverTestingStatus
                      : serverStatus === 'connected'
                        ? T.serverConnectedStatus
                        : T.serverUnreachableStatus}
                  </Text>
                </View>
              )}
            </View>

            <View style={styles.voiceSelector}>
              <View style={styles.sectionTitleRow}>
                <Icon name="record-voice-over" size={16} color={theme.textSecondary} style={styles.iconInline} />
                <Text style={styles.sectionTitle}>{T.sttSectionTitle}</Text>
              </View>
              <Text style={styles.settingHint}>
                {T.sttHint}
              </Text>
              <TouchableOpacity
                style={[styles.voiceOption, sttLanguageMode === 'auto' && styles.voiceOptionSelected]}
                onPress={() => setSttLanguageMode('auto')}>
                <Icon name="shuffle" size={16} color={theme.accent} style={styles.iconInline} />
                <Text style={styles.voiceOptionText}>{T.autoAdaptive}</Text>
              </TouchableOpacity>
              <TouchableOpacity
                style={[styles.voiceOption, sttLanguageMode === 'en-US' && styles.voiceOptionSelected]}
                onPress={() => setSttLanguageMode('en-US')}>
                <Icon name="language" size={16} color={theme.accent} style={styles.iconInline} />
                <Text style={styles.voiceOptionText}>{T.englishOnly}</Text>
              </TouchableOpacity>
              <TouchableOpacity
                style={[styles.voiceOption, sttLanguageMode === 'pa-IN' && styles.voiceOptionSelected]}
                onPress={() => setSttLanguageMode('pa-IN')}>
                <Icon name="language" size={16} color={theme.accent} style={styles.iconInline} />
                <Text style={styles.voiceOptionText}>{T.punjabiOnly}</Text>
              </TouchableOpacity>
            </View>

            <View style={styles.voiceSelector}>
              <View style={styles.sectionTitleRow}>
                <Icon name="graphic-eq" size={16} color={theme.textSecondary} style={styles.iconInline} />
                <Text style={styles.sectionTitle}>{T.voiceSelectionTitle}</Text>
              </View>
              <TouchableOpacity
                style={[styles.voiceOption, selectedVoice === 'pa-IN-Wavenet-A' && styles.voiceOptionSelected]}
                onPress={() => setSelectedVoice('pa-IN-Wavenet-A')}>
                <Icon name="person" size={16} color={theme.accent} style={styles.iconInline} />
                <Text style={styles.voiceOptionText}>{T.femalePremium}</Text>
              </TouchableOpacity>
              <TouchableOpacity
                style={[styles.voiceOption, selectedVoice === 'pa-IN-Wavenet-B' && styles.voiceOptionSelected]}
                onPress={() => setSelectedVoice('pa-IN-Wavenet-B')}>
                <Icon name="person" size={16} color={theme.accent} style={styles.iconInline} />
                <Text style={styles.voiceOptionText}>{T.malePremium}</Text>
              </TouchableOpacity>
            </View>

            <View style={styles.testSection}>
              <View style={styles.sectionTitleRow}>
                <Icon name="science" size={16} color={theme.textSecondary} style={styles.iconInline} />
                <Text style={styles.sectionTitle}>{T.testVoiceTitle}</Text>
              </View>
              <View style={styles.phrasesGrid}>
                {testPhrases.map((phrase, index) => (
                  <TouchableOpacity
                    key={index}
                    style={styles.phraseButton}
                    onPress={() => speakTranslation(phrase.punjabi, 'pa')}>
                    <Text style={styles.phraseText}>{phrase.punjabi}</Text>
                  </TouchableOpacity>
                ))}
              </View>
            </View>
          </>
        )}

        {activeScreen === 'about' && (
          <View style={styles.features}>
            <View style={styles.sectionTitleRow}>
              <Icon name="auto-awesome" size={16} color={theme.textSecondary} style={styles.iconInline} />
              <Text style={styles.sectionTitle}>{T.featuresTitle}</Text>
            </View>
            {[T.feature1, T.feature2, T.feature3, T.feature4, T.feature5].map((feature, index) => (
              <View key={index} style={styles.featureRow}>
                <Icon name={FEATURE_ICONS[index]} size={16} color={theme.accent} style={styles.iconInline} />
                <Text style={styles.featureItem}>{feature}</Text>
              </View>
            ))}
          </View>
        )}
      </ScrollView>
    </SafeAreaView>
  );
};

const createStyles = (theme: ThemeColors) => StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: theme.background,
  },
  scrollContent: {
    padding: 16,
    flexGrow: 1,
    justifyContent: 'center',
  },
  topBar: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    backgroundColor: theme.surface,
    paddingHorizontal: 12,
    paddingVertical: 14,
    borderBottomWidth: 1,
    borderBottomColor: theme.border,
    elevation: 3,
    position: 'relative',
  },
  topBarAccentLine: {
    position: 'absolute',
    left: 0,
    right: 0,
    bottom: -1,
    height: 2,
    backgroundColor: theme.accent,
    opacity: 0.6,
  },
  menuButton: {
    padding: 6,
    borderRadius: 10,
  },
  menuButtonSpacer: {
    width: 36,
  },
  topBarActions: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
  },
  themeToggleButton: {
    width: 36,
    height: 36,
    borderRadius: 18,
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: theme.accentSoft,
    borderWidth: 1,
    borderColor: theme.border,
  },
  langToggleButton: {
    minWidth: 36,
    paddingVertical: 6,
    paddingHorizontal: 10,
    borderRadius: 14,
    backgroundColor: theme.accentSoft,
    alignItems: 'center',
  },
  langToggleText: {
    fontSize: 14,
    fontWeight: 'bold',
    color: theme.accent,
    letterSpacing: 0.5,
  },
  topBarTitleWrap: {
    position: 'absolute',
    left: 0,
    right: 0,
    top: 0,
    bottom: 0,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    transform: [{translateX: -15}],
  },
  topBarTitleColumn: {
    alignItems: 'center',
  },
  topBarTitleIcon: {
    marginRight: 6,
  },
  topBarTitle: {
    fontSize: 17,
    fontWeight: 'bold',
    color: theme.textPrimary,
    letterSpacing: 0.3,
  },
  topBarSubtitle: {
    fontSize: 9,
    fontWeight: '700',
    color: theme.accent,
    letterSpacing: 1.5,
    marginTop: 2,
    opacity: 0.85,
  },
  menuOverlay: {
    flex: 1,
    backgroundColor: theme.overlay,
  },
  menuDrawer: {
    width: 250,
    height: '100%',
    backgroundColor: theme.surface,
    paddingTop: 60,
    paddingHorizontal: 18,
    borderRightWidth: 1,
    borderRightColor: theme.border,
    elevation: 8,
  },
  menuHeader: {
    fontSize: 20,
    fontWeight: 'bold',
    color: theme.textPrimary,
    marginBottom: 16,
    letterSpacing: 0.3,
  },
  menuItem: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingVertical: 14,
    borderBottomWidth: 1,
    borderBottomColor: theme.border,
  },
  menuItemIcon: {
    marginRight: 12,
  },
  menuItemActive: {
    borderBottomColor: theme.accent,
  },
  menuItemText: {
    fontSize: 16,
    color: theme.textPrimary,
  },
  menuItemTextActive: {
    color: theme.accent,
    fontWeight: 'bold',
  },
  voiceSection: {
    alignItems: 'center',
    marginVertical: 20,
  },
  iconInline: {
    marginRight: 8,
  },
  voiceButton: {
    flexDirection: 'row',
    backgroundColor: theme.success,
    paddingVertical: 18,
    paddingHorizontal: 30,
    borderRadius: 28,
    elevation: 5,
    minWidth: 250,
    alignItems: 'center',
    justifyContent: 'center',
  },
  voiceButtonActive: {
    backgroundColor: theme.danger,
  },
  voiceButtonText: {
    color: '#ffffff',
    fontSize: 17,
    fontWeight: 'bold',
    letterSpacing: 0.3,
  },
  voiceIconRing: {
    width: 140,
    height: 140,
    borderRadius: 70,
    borderWidth: 1.5,
    borderColor: theme.accent,
    opacity: 0.9,
    alignItems: 'center',
    justifyContent: 'center',
  },
  voiceIconButton: {
    backgroundColor: theme.accent,
    width: 116,
    height: 116,
    borderRadius: 58,
    elevation: 10,
    alignItems: 'center',
    justifyContent: 'center',
    borderWidth: 4,
    borderColor: theme.accentSoft,
    shadowColor: theme.accent,
    shadowOffset: {width: 0, height: 6},
    shadowOpacity: 0.45,
    shadowRadius: 14,
  },
  voiceIconButtonActive: {
    backgroundColor: theme.danger,
  },
  voiceHint: {
    marginTop: 14,
    fontSize: 12,
    color: theme.textSecondary,
    textAlign: 'center',
  },
  typedInput: {
    width: '100%',
    minHeight: 80,
    backgroundColor: theme.surface,
    borderRadius: 14,
    borderWidth: 1.5,
    borderColor: theme.border,
    padding: 12,
    fontSize: 16,
    color: theme.textPrimary,
    marginBottom: 16,
    textAlignVertical: 'top',
  },
  modeToggle: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    marginTop: 16,
    paddingVertical: 10,
    paddingHorizontal: 20,
    borderRadius: 20,
    backgroundColor: theme.surfaceAlt,
    borderWidth: 1,
    borderColor: theme.border,
  },
  modeToggleText: {
    fontSize: 13,
    color: theme.accent,
    fontWeight: '600',
  },
  backButtonWrapper: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  backButton: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    alignSelf: 'center',
    paddingVertical: 12,
    paddingHorizontal: 22,
    borderRadius: 20,
    backgroundColor: theme.accentSoft,
  },
  backButtonText: {
    fontSize: 16,
    color: theme.accent,
    fontWeight: '600',
  },
  flagButton: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    alignSelf: 'center',
    paddingVertical: 8,
    paddingHorizontal: 16,
    borderRadius: 16,
    backgroundColor: theme.dangerSoft,
    marginTop: 4,
    marginBottom: 10,
  },
  flagButtonText: {
    fontSize: 13,
    color: theme.danger,
    fontWeight: '600',
  },
  statusCard: {
    backgroundColor: theme.surface,
    padding: 15,
    borderRadius: 14,
    marginVertical: 10,
    elevation: 2,
    borderWidth: 1,
    borderColor: theme.border,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
  },
  statusDot: {
    width: 9,
    height: 9,
    borderRadius: 5,
    marginRight: 10,
  },
  statusText: {
    fontSize: 14,
    color: theme.textPrimary,
    flex: 1,
  },
  loader: {
    marginLeft: 10,
  },
  translationSection: {
    marginVertical: 15,
  },
  textBox: {
    backgroundColor: theme.surface,
    padding: 15,
    borderRadius: 16,
    marginVertical: 8,
    elevation: 2,
    borderWidth: 1.5,
    borderColor: theme.border,
    borderLeftWidth: 3,
    borderLeftColor: theme.accent,
  },
  textBoxTitleRow: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 8,
  },
  textBoxTitle: {
    fontSize: 15,
    fontWeight: 'bold',
    color: theme.accent,
  },
  textContent: {
    fontSize: 16,
    lineHeight: 24,
    color: theme.textPrimary,
  },
  punjabiText: {
    fontSize: 18,
  },
  voiceSelector: {
    backgroundColor: theme.surface,
    padding: 15,
    borderRadius: 16,
    marginVertical: 10,
    elevation: 2,
    borderWidth: 1,
    borderColor: theme.border,
    borderLeftWidth: 3,
    borderLeftColor: theme.accent,
  },
  sectionTitleRow: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 12,
  },
  sectionTitle: {
    fontSize: 13,
    fontWeight: 'bold',
    color: theme.textSecondary,
    letterSpacing: 1.2,
    textTransform: 'uppercase',
  },
  settingHint: {
    fontSize: 12,
    color: theme.textSecondary,
    marginBottom: 12,
    lineHeight: 18,
  },
  voiceOption: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: theme.surfaceAlt,
    padding: 12,
    borderRadius: 10,
    marginVertical: 5,
    borderWidth: 1.5,
    borderColor: theme.border,
  },
  voiceOptionSelected: {
    borderColor: theme.accent,
    backgroundColor: theme.accentSoft,
  },
  voiceOptionText: {
    fontSize: 14,
    color: theme.textPrimary,
  },
  serverInput: {
    width: '100%',
    backgroundColor: theme.surfaceAlt,
    borderRadius: 10,
    borderWidth: 1.5,
    borderColor: theme.border,
    padding: 10,
    fontSize: 14,
    color: theme.textPrimary,
    marginBottom: 10,
  },
  serverSaveButton: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: theme.accent,
    paddingVertical: 12,
    borderRadius: 10,
    shadowColor: theme.accent,
    shadowOffset: {width: 0, height: 4},
    shadowOpacity: 0.35,
    shadowRadius: 8,
    elevation: 4,
  },
  serverSaveButtonText: {
    color: '#ffffff',
    fontSize: 14,
    fontWeight: '600',
  },
  serverStatusRow: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    marginTop: 10,
  },
  serverStatusText: {
    fontSize: 13,
    color: theme.textSecondary,
    textAlign: 'center',
  },
  serverStatusConnected: {
    color: theme.success,
    fontWeight: '600',
  },
  serverStatusUnreachable: {
    color: theme.danger,
    fontWeight: '600',
  },
  testSection: {
    backgroundColor: theme.surface,
    padding: 15,
    borderRadius: 16,
    marginVertical: 10,
    elevation: 2,
    borderWidth: 1,
    borderColor: theme.border,
    borderLeftWidth: 3,
    borderLeftColor: theme.accent,
  },
  phrasesGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 8,
  },
  phraseButton: {
    backgroundColor: theme.accentSoft,
    paddingVertical: 10,
    paddingHorizontal: 15,
    borderRadius: 20,
    marginVertical: 4,
    borderWidth: 1,
    borderColor: theme.border,
  },
  phraseText: {
    fontSize: 16,
    color: theme.accent,
  },
  features: {
    backgroundColor: theme.surface,
    padding: 15,
    borderRadius: 16,
    marginVertical: 10,
    elevation: 2,
    borderWidth: 1,
    borderColor: theme.border,
    borderLeftWidth: 3,
    borderLeftColor: theme.accent,
  },
  featureRow: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingVertical: 8,
    borderBottomWidth: 1,
    borderBottomColor: theme.border,
  },
  featureItem: {
    fontSize: 14,
    color: theme.textPrimary,
    flexShrink: 1,
  },
});

export default TranslationAppMobile;
