// Web replacement for '@react-native-voice/voice', backed by the browser's
// SpeechRecognition API. Mirrors just the subset of the real library's API
// surface TranslationAppMobile.tsx actually uses: settable on* callback
// properties (not addEventListener - that's how the real library works too)
// plus start/stop/destroy/removeAllListeners.
type SpeechResultsEvent = {value?: string[]};
type SpeechErrorEvent = {error?: {message?: string}};

const SpeechRecognitionCtor: typeof window.SpeechRecognition | undefined =
  (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition;

class VoiceWeb {
  onSpeechStart: (() => void) | null = null;
  onSpeechEnd: (() => void) | null = null;
  onSpeechResults: ((e: SpeechResultsEvent) => void) | null = null;
  onSpeechError: ((e: SpeechErrorEvent) => void) | null = null;

  private recognition: InstanceType<typeof window.SpeechRecognition> | null = null;

  async start(locale: string): Promise<void> {
    if (!SpeechRecognitionCtor) {
      throw new Error('Speech recognition is not supported in this browser');
    }
    this.recognition = new SpeechRecognitionCtor();
    this.recognition.lang = locale;
    this.recognition.interimResults = false;
    this.recognition.maxAlternatives = 1;
    this.recognition.onstart = () => this.onSpeechStart?.();
    this.recognition.onend = () => this.onSpeechEnd?.();
    this.recognition.onresult = (event: any) => {
      const transcript = event.results?.[0]?.[0]?.transcript;
      this.onSpeechResults?.({value: transcript ? [transcript] : []});
    };
    this.recognition.onerror = (event: any) => {
      this.onSpeechError?.({error: {message: event.error}});
    };
    this.recognition.start();
  }

  async stop(): Promise<void> {
    this.recognition?.stop();
  }

  async destroy(): Promise<void> {
    if (this.recognition) {
      this.recognition.onend = null;
      this.recognition.stop();
      this.recognition = null;
    }
  }

  removeAllListeners = (): void => {
    this.onSpeechStart = null;
    this.onSpeechEnd = null;
    this.onSpeechResults = null;
    this.onSpeechError = null;
  };
}

export default new VoiceWeb();
